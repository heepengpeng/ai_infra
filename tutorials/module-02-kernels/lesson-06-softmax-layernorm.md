# M2 · Lesson 6:Softmax 与 LayerNorm 的高效实现

> Softmax 和 LayerNorm 是 Transformer 里出现频率最高的两个"归约型"算子。它们看着简单,却藏着两个关键点:**数值稳定性**(写错了会 NaN)和**归约策略**(两遍 vs online,直接决定能不能省一趟显存)。本课把这两点彻底讲透——其中的 **online softmax** 正是下一课 FlashAttention 的数学心脏。
> 预计用时:3 小时。
> 前置:M2 L4(Triton)、L5(融合);M1 L7(reduce 归约)。

## 学习目标

1. 理解 softmax 为什么必须**减最大值**,否则数值溢出。
2. 掌握 softmax 的**两遍法 vs online(一遍)法**,并能推导 online 的递推公式。
3. 理解 LayerNorm 的数学,以及用 **Welford / 两遍** 求均值方差。
4. 用 Triton 实现 fused softmax 与 fused LayerNorm,benchmark 对比 PyTorch。

---

## 1. Softmax 的数值稳定性:为什么要减 max

定义:`softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`。

问题:FP32 能表示的最大值约 `3.4e38`,而 `exp(89) ≈ 4.4e38` 就溢出成 `inf` 了。注意力分数经常上百,`exp` 直接爆。

救法:利用 softmax 的**平移不变性**——给所有 `x_i` 同时减一个常数 `c`,结果不变:

```
exp(x_i - c) / Σ_j exp(x_j - c)
 = [exp(x_i)/exp(c)] / [Σ_j exp(x_j)/exp(c)]
 = exp(x_i) / Σ_j exp(x_j)        ← exp(c) 上下约掉
```

取 `c = max(x)`,则所有指数 `≤ exp(0) = 1`,绝不溢出;最大那项恰好是 1。

> **铁律:任何 softmax 实现,第一步永远是减去该行最大值。** 这不是优化,是正确性底线。L4 的 fused-softmax 代码里那句 `x = x - tl.max(x, axis=0)` 就是它。

---

## 2. 两遍法 vs online 法

"减 max"带来一个工程难题:**你得先知道整行的 max,才能开始算 exp**。这意味着要扫数据不止一遍。

### 两遍法(safe softmax,3-pass 朴素版)

最直白的实现要扫 3 遍数据:

```
pass 1: m = max_i(x_i)                     # 求最大值
pass 2: d = Σ_i exp(x_i - m)               # 求归一化分母
pass 3: out_i = exp(x_i - m) / d           # 求输出
```

每遍都要把整行从显存读一次(若行装不进片上的话)。3 遍 = 3 次读。对带宽受限算子,这很贵。

> 注:如果一行能整个装进 SRAM(L4 的情形),其实读一次进片上、3 遍都在片上扫,显存只读一遍。两遍/online 的真正意义在于**行装不下、必须分块**时——这正是 attention 的 N 很大时的处境。

### Online softmax:一遍求出 max 和 sum

能不能**边扫边更新 max 和 sum,只扫一遍**?能。难点是:sum 依赖 max,但你扫到一半时 max 还可能变大。online softmax 的诀窍是**在 max 更新时,顺手修正已累计的 sum**。

设我们逐个(或逐块)处理元素,维护两个量:当前最大值 `m`、当前(基于 m 的)指数和 `d`。来一个新值 `x`:

```
m_new = max(m, x)
# 旧的 d 是以旧 m_old 为基准算的,现在基准变了,要乘 exp(m_old - m_new) 修正
d_new = d * exp(m_old - m_new) + exp(x - m_new)
```

**这就是 online softmax 的核心递推。** 关键那个修正因子 `exp(m_old - m_new)`:当 max 变大(`m_new > m_old`),它 < 1,把之前累积的、基于偏小基准的 sum"缩放"到新基准上。一遍扫完,`m` 是真 max、`d` 是真分母,再扫一遍(或用已存的)算输出即可。

把它从"逐元素"推广到"逐块",就是 FlashAttention 的并行归约形式。两个块 `(m_A, d_A)` 和 `(m_B, d_B)` 合并:

```
m = max(m_A, m_B)
d = d_A * exp(m_A - m) + d_B * exp(m_B - m)
```

> 记住这个合并公式。下一课 FlashAttention 就是把序列切成块,每块算局部 `(m, d)` 和局部加权 V,再用这个公式**增量合并**,从而**永远不需要把整行 attention 分数实例化出来**。online softmax 是它能做到"不落地大矩阵"的根本原因。

`code/online_softmax.py` 用纯 PyTorch/NumPy 实现了 online 递推,并和标准 softmax 对拍,帮你把公式跑通、建立信心(先在 CPU/简单张量上理解算法,再上 Triton)。

---

## 3. LayerNorm:数学与归约

LayerNorm 对每一行(最后一维 D)做标准化:

```
μ = (1/D) Σ_i x_i                       # 均值
σ² = (1/D) Σ_i (x_i - μ)²               # 方差
y_i = (x_i - μ) / sqrt(σ² + ε) · γ_i + β_i   # 标准化 + 仿射(γ,β 可学习)
```

它也是归约型(要先求整行的 μ 和 σ²),同样面临"扫几遍"的问题:

- **两遍法**:pass 1 求 μ,pass 2 求 σ²(需要 μ),pass 3 标准化。
- **一遍法(Welford 或 sum/sum² 技巧)**:同时累加 `Σx` 和 `Σx²`,则 `σ² = E[x²] - (E[x])²`,一遍扫完得到 μ 和 σ²。注意 `E[x²]-E[x]²` 在数值上可能有抵消误差,生产级用更稳的 **Welford 在线算法**;教学里 sum/sum² 够用且更直观。

> 和 softmax 对照:softmax 的归约是 `(max, sum_exp)`,LayerNorm 的归约是 `(sum, sum_sq)`。两者结构同源——**都是"先归约出统计量,再逐元素变换"**,所以融合方式也一样:整行进片上,归约 + 变换一气呵成,只读一遍写一遍(L5 的融合思想)。

RMSNorm(LLaMA 等用)是 LayerNorm 的简化:不减均值,只用均方根 `sqrt(mean(x²)+ε)` 归一化,少一个归约量,更快。理解了 LayerNorm,RMSNorm 顺手就会。

---

## 4. 动手:Triton fused softmax 与 fused LayerNorm

代码:`code/triton_softmax.py`(L4 已给,可复用)与本课 `code/triton_layernorm.py`。

LayerNorm 的 Triton kernel(一个 program 处理一行,完整见代码):

```python
@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, stride, N, eps,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    # 一遍求 mean 与 var(sum / sum-of-squares 技巧)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)        # 中心化,越界位置置 0 不参与
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask)
    b = tl.load(b_ptr + cols, mask=mask)
    y = xc * rstd * w + b                     # 标准化 + 仿射,全在片上
    tl.store(out_ptr + row * stride + cols, y, mask=mask)
```

整行驻留片上,均值、方差、标准化、仿射全部一个 kernel 搞定——这是 L5 融合思想在归约算子上的直接应用。

运行:

```bash
cd code
python triton_layernorm.py
python online_softmax.py     # 验证 online 递推 == 标准 softmax
```

LayerNorm 参考输出(数值随卡):

```
[INFO] correctness vs torch.layer_norm: max abs err = 4.7e-06
[INFO] shape = (8192, 4096)
[INFO] torch.layer_norm : 0.41 ms, 640 GB/s
[INFO] triton fused     : 0.27 ms, 972 GB/s
[INFO] speedup = 1.5x
```

加速来源依旧是**减少显存往返**(PyTorch 内部仍可能多趟),并把带宽吃得更满。

### 留给你的 TODO

1. `triton_layernorm.py`:在 kernel 里把 mean/var 改用更稳的写法,并实现 **RMSNorm**(去掉减均值),对比速度差异。
2. `online_softmax.py`:把逐元素递推改成**逐块**递推(用 §2 的块合并公式),为下一课 FlashAttention 预热。

---

## 5. 接回推理

- 这些归约算子在每个 Transformer block 里都出现多次(注意力前后各一个 Norm),融合实现直接影响端到端延迟。
- decode 阶段是带宽受限,Norm 这种逐元素+归约算子若不融合,会有可观的访存浪费;vLLM/TensorRT-LLM 都有手写的 fused RMSNorm(+ 残差融合)kernel。
- 最重要的:**online softmax 是 FlashAttention 的数学基础**。你现在把块合并公式推熟,下一课就能看懂"为什么 attention 可以不实例化 N×N 矩阵"。

---

## 练习题

1. 不减 max 的 softmax 在什么输入下会出问题?减 max 为什么不改变结果?
2. 写出 online softmax 处理新元素 `x` 时,`(m, d)` 的更新公式,并解释修正因子 `exp(m_old - m_new)` 的物理意义。
3. 两个块 `(m_A,d_A)`、`(m_B,d_B)` 怎么合并成整体的 `(m,d)`?
4. LayerNorm 的"一遍法"用 `σ² = E[x²] - E[x]²`,它有什么数值隐患?生产级怎么解决?

<details>
<summary>参考答案</summary>

1. 当 `x_i` 较大(如注意力分数上百),`exp(x_i)` 溢出成 inf,inf/inf = NaN。减 max 利用平移不变性:分子分母同除 `exp(max)`,结果不变,但所有指数 ≤ 1,不溢出。
2. `m_new = max(m_old, x)`;`d_new = d_old * exp(m_old - m_new) + exp(x - m_new)`。修正因子把"之前以 m_old 为基准累积的和"换算到"新基准 m_new",因为当最大值变大时,旧的每一项 `exp(x_i - m_old)` 都应变成 `exp(x_i - m_new) = exp(x_i - m_old)·exp(m_old - m_new)`,统一乘这个因子即可。
3. `m = max(m_A, m_B)`;`d = d_A·exp(m_A - m) + d_B·exp(m_B - m)`。即各块按自身基准的和,统一缩放到全局基准 m 再相加。
4. `E[x²]-E[x]²` 当两项都很大且接近时会发生**灾难性抵消**(catastrophic cancellation),丢失有效位甚至得到负方差。生产级用 **Welford 在线算法**(逐元素更新均值和 M2 累积量,数值稳定),或两遍法(先算准 μ 再算 σ²)。

</details>

---

## 小结

- softmax 必须**减每行 max**(平移不变性)保证不溢出,这是正确性底线。
- **online softmax**:用递推 `d_new = d·exp(m_old-m_new) + exp(x-m_new)` 边扫边修正,可一遍求出 max 和 sum;块合并公式是 FlashAttention 的核心。
- LayerNorm 同为归约型(`sum`/`sum²` 求 μ、σ²),一遍法有抵消隐患,生产用 Welford;RMSNorm 是其简化。
- Triton fused 实现把"整行进片上、归约+变换一气呵成",靠减少访存获得加速。

## 自测验收
- [ ] 能解释并手推 softmax 减 max 的正确性。
- [ ] 能默写 online softmax 的递推与块合并公式。
- [ ] `triton_layernorm.py` 与 `online_softmax.py` 跑通,结果与参考一致。
- [ ] 能说清 LayerNorm 一遍法的数值隐患与解决办法。

下一课:**Lesson 7 — FlashAttention**,把 online softmax + tiling + IO 感知合体,实现一个不落地 N×N 矩阵的注意力,这是整个 Module 2 的收官与最高峰。
