# M2 · Lesson 7:FlashAttention——online softmax + IO 感知的注意力

> 这是 Module 2 的收官,也是把前六课的内功合体的高峰:**roofline(L1)判出 attention 是带宽/显存受限 → 算子融合(L5)的极致形态 → online softmax(L6)解决归约依赖 → tiling(L2)切块 → Triton(L4)落地**。学完你将理解为什么 FlashAttention 能在不损失精度的前提下,把注意力的显存从 O(N²) 降到 O(N),并显著加速。
> 预计用时:3.5 小时(推导 + 读懂 Triton 实现 + benchmark)。
> 前置:**M2 全部**,尤其 L6 的 online softmax 与块合并公式(没吃透请先回去)。

## 学习目标

1. 算清标准 attention 的**显存瓶颈**:O(N²) 中间矩阵从哪来、为什么是带宽/显存杀手。
2. 推导 FlashAttention 如何用 **online softmax** 把 softmax 拆成可增量合并的块,从而**不实例化 N×N 矩阵**。
3. 理解它的 **IO 感知(IO-aware)** 本质:优化目标是 HBM 读写量,不是 FLOP。
4. 用 Triton 实现一个简化版 flash attention,对比朴素实现的**显存占用和速度**。

---

## 1. 标准 attention 的显存瓶颈

注意力公式(单头,Q、K、V 形状都是 N×d,N=序列长,d=头维度):

```
S = Q · Kᵀ          # (N×d)·(d×N) = N×N   ← 注意力分数矩阵
P = softmax(S)       # 逐行 softmax,仍是 N×N
O = P · V            # (N×N)·(N×d) = N×d
```

朴素实现(也是 PyTorch 拆开写时的样子)的致命点:**把 N×N 的 S 和 P 完整写进 HBM**。

```
显存占用:S 和 P 各 N² 个 float。N=4096、FP16 时:
  N² × 2 字节 = 4096² × 2 ≈ 33 MB(单头单 batch)
多头 × 多 batch × 多层 → 爆炸。N=8192 时单个矩阵就 128MB。
```

用 L1 的 roofline 看:这个过程读写了 O(N²) 字节,但有用计算也是 O(N²·d)。问题不在 FLOP,在于 **S、P 这两个巨大的中间矩阵在 HBM 上来回搬**(写 S、读 S 做 softmax、写 P、读 P 做 PV)。这正是 L5 说的"中间结果落地 HBM"的最坏情况——而且这里的中间结果是 O(N²) 量级的庞然大物。

> **核心矛盾:S = QKᵀ 的结果 N×N 太大,既占显存又拖带宽,但它只是个中间产物——最终输出 O 只有 N×d。** 能不能"算 attention 但永远不把整个 N×N 落地"?FlashAttention 的回答是:能,靠 online softmax。

---

## 2. 关键障碍:softmax 的"全局依赖"

为什么不能直接分块?因为 softmax 的分母要看**整行**:

```
O_i = Σ_j softmax(S_ij) · V_j = Σ_j [exp(S_ij - m_i) / d_i] · V_j
其中 m_i = max_j S_ij,  d_i = Σ_j exp(S_ij - m_i)
```

`m_i`(行最大)和 `d_i`(行求和)都需要扫完第 i 行的全部 N 个分数才能确定。所以"先算完整行 S、再 softmax"看起来无法避免实例化整行。

**online softmax(L6)恰好破解这一点**:它能边扫块边维护 `(m, d)`,在 max 更新时修正已累积量。FlashAttention 把这一招从"标量和"推广到"加权 V 的和"。

---

## 3. FlashAttention 推导:把 O 也做成增量累积

思路:把 K、V 沿序列维切成若干块 `K_1,V_1, K_2,V_2, …`。对固定的一行查询 `q`(对应输出 `o`),逐块处理,维护三个量:

- `m`:到目前为止见过的分数最大值(标量)
- `d`:到目前为止的指数和(标量,基于当前 m)
- `o`:到目前为止的**未归一化加权 V 累积**(向量,长度 d)

处理第 j 个 K/V 块:

```
S_j = q · K_jᵀ                      # 这一块的分数(长度 = 块大小,很小)
m_j = max(S_j)                       # 本块局部最大
m_new = max(m, m_j)                  # 更新全局最大

# 缩放因子:把旧累积换算到新基准 m_new
α = exp(m - m_new)                   # 旧的 m 基准 → 新基准
p_j = exp(S_j - m_new)               # 本块在新基准下的指数

d_new = d * α + sum(p_j)             # 更新分母(online softmax 递推)
o_new = o * α + p_j · V_j            # 更新加权 V 累积(关键!o 同样要缩放)
```

注意 `o` 的更新和 `d` 用**同一个缩放因子 α**:当 max 变大,之前累积的 `o` 和 `d` 都基于偏小的基准,要统一乘 α 缩放到新基准。处理完所有块后,归一化一次:

```
O = o / d        # 最后才除以分母
```

**这就是 FlashAttention 的全部数学。** 整个过程里:
- `S_j` 只是 `q` 和一个**小块** K 的乘积,大小是块大小级别,算完即弃,**从不存整行的 N 个分数**。
- 完整的 N×N 矩阵 S、P **从未存在过**——它被"切块 + online 合并"消解了。
- 显存:只需存 Q、K、V、O(都是 O(N·d))和每行几个标量 `(m,d)`,**O(N) 而非 O(N²)**。

```
标准 attention:                FlashAttention:
Q ─┐                           外层循环 q 的块(留在 SRAM)
   ├─ S=QKᵀ (N×N, 写 HBM) ✗      内层循环 K/V 的块:
K ─┘                              S_j = q·K_jᵀ (小, 在 SRAM)
   softmax(S) (N×N, 读+写) ✗      online 更新 (m,d,o)
   O = P·V                        K/V 块算完即弃
                                最后 O = o/d, 只写 O (N×d)
            ↑ O(N²) HBM 流量              ↑ O(N) HBM 流量
```

---

## 4. IO 感知:优化的是 HBM 读写,不是 FLOP

这是 FlashAttention 论文标题"IO-Aware"的含义,也是它和"普通融合"的区别。

有意思的是:FlashAttention 的**浮点运算量并没有减少**(甚至因为重算略有增加),它甚至比朴素实现多做了一些计算(每块都重新缩放)。但它**大幅减少了 HBM 读写**——而 attention 本来就是带宽/显存受限的(L1),所以减少 IO 直接换来加速。

> 这是 L1 roofline 思想最深刻的一次应用:**当你受 IO 限制时,宁可多算一点 FLOP,也要减少数据搬运。** FlashAttention 把 O(N²) 的 HBM 访问降到约 O(N²·d/M)(M 是 SRAM 大小),IO 减少几个量级,即便 FLOP 没省,墙上时间(wall-clock)也大幅下降。这种"用计算换访存"的取舍,在带宽受限世界里反复出现。

附带的巨大好处:**显存从 O(N²) 降到 O(N)**,使得长上下文(N=32K、128K)成为可能——否则光是 attention 分数矩阵就把显存撑爆了。这是长文本大模型的关键使能技术。

---

## 5. 动手:Triton 简化版 FlashAttention

`code/flash_attention.py` 实现了一个**前向、教学用**的简化 flash attention(非因果,固定 head_dim),并与朴素 attention 对比正确性、速度、显存峰值。

Triton kernel 的结构(完整见代码):

```python
@triton.jit
def flash_attn_kernel(Q, K, V, O, ...):
    # 每个 program 负责 Q 的一个块(BLOCK_M 行查询)
    q = tl.load(Q block)                      # 留在 SRAM
    m = full(-inf); d = 0; acc = zeros(BLOCK_M, d)   # online 状态
    # 内层:遍历 K/V 的所有块
    for start_n in range(0, N, BLOCK_N):
        k = tl.load(K block); v = tl.load(V block)
        s = tl.dot(q, k.T) * scale            # 这一块的分数(小)
        m_new = maximum(m, max(s, axis=1))
        alpha = exp(m - m_new)                # 旧累积的缩放因子
        p = exp(s - m_new[:, None])
        d = d * alpha + sum(p, axis=1)        # 更新分母
        acc = acc * alpha[:, None] + dot(p, v)  # 更新加权 V(同步缩放)
        m = m_new
    o = acc / d[:, None]                      # 最后归一化
    tl.store(O block, o)
```

逐句对照 §3 的推导,你会发现它就是把数学**一比一翻译**成 Triton,外加 tiling(BLOCK_M、BLOCK_N)和把 q 块留在 SRAM。`tl.dot` 是 Triton 的块内矩阵乘(底层可能走 Tensor Core)。

运行:

```bash
cd code
pip install triton            # 若未装
python flash_attention.py
```

参考输出(数值随卡;朴素版在大 N 时可能 OOM,正说明问题):

```
[INFO] N=4096, d=64
[INFO] correctness vs naive: max abs err = 6.1e-04
[INFO] naive attention : 3.84 ms, peak extra mem = 64.0 MB (S + P)
[INFO] flash attention : 1.42 ms, peak extra mem = 0.0 MB (no NxN matrix)
[INFO] speedup = 2.7x;  显存:不再实例化 N×N 中间矩阵
```

**读数要点**:
- 速度数倍提升,且 N 越大优势越明显(朴素的 O(N²) 显存/带宽劣势随 N 平方放大)。
- 显存:朴素版要为 S、P 各开 N×N;flash 版**不开**,这是它能跑长序列的根本。
- 正确性:与朴素实现误差在 FP16/数值容差内一致——**这是无损优化,不是近似**。把这点讲给别人听是常见面试考点。

### 留给你的 TODO

`flash_attention.py` 里:
1. 加 **causal mask**(自回归生成必需):内层循环跳过 `key 位置 > query 位置` 的块,块内对角部分用 mask。这是把它用到真实 LLM 的关键一步。
2. 把 `N` 从 2048 加到 8192、16384,观察朴素版何时 OOM、flash 版仍稳健,亲眼见证 O(N²)→O(N) 的威力。

---

## 6. 全模块回顾:你已经走完的优化主线

FlashAttention 把 Module 2 的每一课都用上了,正好回看这条主线:

| 课 | 工具 | 在 FlashAttention 里的角色 |
|---|---|---|
| L1 roofline | 判定瓶颈 | 看出 attention 是 IO/显存受限,该减访存 |
| L2 tiling | shared memory 分块复用 | Q/K/V 切块进 SRAM 复用 |
| L3 寄存器分块/`tl.dot` | 块内高效 GEMM | `q·kᵀ`、`p·v` 的块内矩阵乘 |
| L4 Triton | program 抽象 | 用 Python 落地整个 kernel |
| L5 算子融合 | 减少 HBM 往返 | 把 QKᵀ+softmax+PV 融成一个 kernel |
| L6 online softmax | 归约不需整行 | 不实例化 N×N 的数学基础 |
| **L7 FlashAttention** | **以上全部合体** | **IO 感知的注意力** |

> 一句话收尾整个 Module 2:**所有 GPU 性能优化,归根结底是在 roofline 上把 kernel 往"少搬数据、多复用"的方向推。** GEMM 靠存储层级复用把算术强度推高、逼近算力屋顶;带宽受限的算子靠融合和 IO 感知减少访存、贴着带宽斜坡跑满。FlashAttention 是这套方法论在最重要的算子上的集大成。

---

## 练习题

1. 标准 attention 为什么显存是 O(N²)?到底是哪个(些)矩阵造成的?FlashAttention 把它降到多少?
2. 写出 FlashAttention 处理一个 K/V 块时,`(m, d, o)` 三个状态的更新公式,并指出 `o` 为什么要乘和 `d` 相同的缩放因子。
3. "IO 感知"是什么意思?为什么 FlashAttention 的 FLOP 没减少(甚至略增)却更快?用 roofline 解释。
4. FlashAttention 是无损还是近似?为什么它和朴素 attention 的结果(在数值容差内)完全一致?

<details>
<summary>参考答案</summary>

1. 因为要实例化注意力分数矩阵 `S = QKᵀ`(N×N)和它的 softmax `P`(N×N),这两个中间矩阵是 O(N²)。FlashAttention 通过分块 + online softmax 永不实例化它们,只存 Q/K/V/O(O(N·d))和每行几个标量,显存降到 **O(N)**。
2. `m_new = max(m, max(S_j))`;`α = exp(m - m_new)`;`p_j = exp(S_j - m_new)`;`d_new = d·α + sum(p_j)`;`o_new = o·α + p_j·V_j`;最后 `O = o/d`。`o` 和 `d` 用同一个 α:因为之前累积的 o、d 都基于旧的偏小基准 m,当 max 变大,所有旧项都应乘 `exp(m_old - m_new)` 换算到新基准,o 是加权 V 的和、d 是权重的和,缩放因子自然相同。
3. IO 感知 = 优化目标是 HBM 读写量而非浮点运算量。attention 是带宽/显存受限(roofline 斜坡区),墙上时间由数据搬运决定;FlashAttention 用"分块时重算缩放"多花了一点 FLOP,但把 O(N²) 的 HBM 访问降到约 O(N²d/M),IO 大减。在带宽受限区,减 IO 直接减时间,即使 FLOP 没省也更快——这是 roofline 斜坡区的优化逻辑。
4. **无损**(数学上精确,只有浮点舍入级误差)。因为 online softmax 的递推和块合并在代数上等价于"先算完整行 max/sum 再 softmax",每一步缩放都精确地把基准对齐,没有任何近似;最终的 `o/d` 就是标准 attention 的输出,差异仅来自浮点运算顺序不同导致的微小舍入。

</details>

---

## 小结

- 标准 attention 的瓶颈是实例化 O(N²) 的中间矩阵 S、P,既占显存又是带宽杀手。
- FlashAttention 用 **online softmax** 把 max/sum/加权 V 都做成可增量合并的块,**永不实例化 N×N 矩阵**,显存 O(N²)→O(N)。
- 它是 **IO 感知** 优化:用少量额外 FLOP 换大幅减少的 HBM 访问,在带宽受限的 attention 上换来数倍加速,并使长上下文可行。
- 它是整个 Module 2 方法论的集大成:roofline 定性 + tiling + 块内 GEMM + 融合 + online softmax,用 Triton 落地。

## 自测验收(过了即完成 Module 2)
- [ ] 能讲清标准 attention 为何是 O(N²) 显存,以及 FlashAttention 如何降到 O(N)。
- [ ] 能默写 `(m, d, o)` 的 online 更新公式,并解释缩放因子。
- [ ] `flash_attention.py` 跑通,对比出速度和显存优势,理解它是无损的。
- [ ] 能用一句话把 Module 2 的优化主线串起来(少搬数据、多复用)。

---

🎉 **Module 2 完成!** 你现在能:用 roofline 定位瓶颈、手写逼近 cuBLAS 的 GEMM、用 Triton 写融合算子、实现 FlashAttention。这套"看穿并优化 kernel"的能力,是推理 Infra 工程师区别于"调库工程师"的核心。

下一站:**Module 3 — Transformer 推理原理**,我们带着这些 kernel 能力,打开大模型推理的黑盒(prefill/decode、KV Cache、采样),把今天写的算子组装成一个能跑的推理循环。
