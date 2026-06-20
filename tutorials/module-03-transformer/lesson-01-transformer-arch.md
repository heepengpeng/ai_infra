# M3 · Lesson 1:Transformer 结构(推理视角)

> 本课用「推理工程师的眼睛」重新看一遍 decoder-only Transformer。你大概率在框架层用过它,但这次我们要拆到**每个子模块的计算量、参数量、访存特性**,为后面 prefill/decode、KV Cache、roofline 分析打地基。
> 预计用时:2.5 小时(阅读 + 手搭最小 GPT block)。
> 前置:M1 全部(尤其 L1 roofline 直觉)、熟悉 PyTorch 张量操作、线性代数(矩阵乘维度变换)。
> 环境:本课代码纯 CPU 即可跑(只搭结构、看 shape 与 FLOPs,不训练)。

## 学习目标

学完本课你应该能回答:
1. decoder-only Transformer 一次前向,数据的 shape 如何流动?每一步在算什么?
2. embedding、Attention、FFN、RoPE、RMSNorm、residual 各自的**参数量**和**计算量**公式是什么?
3. 哪些子模块是**计算密集(compute-bound)**,哪些是**访存密集(memory-bound)**?为什么?
4. 为什么现代大模型(LLaMA/Qwen 系)用 RMSNorm + RoPE + SwiGLU,而不是原始 Transformer 的 LayerNorm + 绝对位置编码?

---

## 1. 先建立全局地图:一个 token 的旅程

你做过模型迁移,见过 ONNX / MindIR 的计算图,知道 Transformer 是「一堆相同的 block 叠起来」。但迁移时你关心的是算子能不能对齐;现在我们关心的是**每个算子在推理时贵在哪**。

先看一张总图。假设词表大小 `V`,隐藏维度 `d`(也叫 `hidden_size` / `d_model`),层数 `L`,输入序列长度 `S`:

```
token ids  [S]
   │  Embedding 查表(lookup)
   ▼
hidden     [S, d]
   │
   ├──────────────  L 个相同的 Decoder Block 串联  ──────────────┐
   │  每个 block:                                                │
   │    x → RMSNorm → Multi-Head Self-Attention → +residual      │
   │    x → RMSNorm → FFN(SwiGLU)              → +residual      │
   └─────────────────────────────────────────────────────────────┘
   ▼
hidden     [S, d]
   │  最后一层 RMSNorm
   ▼
   │  LM Head(线性投影到词表)  [d, V]
   ▼
logits     [S, V]
```

**关键认知**:整个网络的「骨架」就是 `L` 个**结构完全相同**的 block 串联。block 内部只有两个核心子层——**Attention** 和 **FFN**——其余(Norm、residual、RoPE)都是围绕这两个核心的「配件」。推理优化 90% 的精力,花在这两个子层上。

> 记住这个比例感:对 7B 这种典型模型,**FFN 的参数量约为 Attention 的 2 倍**,但 **Attention 在长序列时计算量随 \(S^2\) 爆炸**。两者的瓶颈位置完全不同,后面会反复用到。

---

## 2. Embedding:一次查表,几乎零计算

输入是 `S` 个 token id(整数)。Embedding 层本质是一张大小为 `[V, d]` 的查找表 `W_emb`,第 `i` 个 token 直接取出第 `id_i` 行:

\[
\text{hidden}[i] = W_{\text{emb}}[\text{id}_i]
\]

- **参数量**:\(V \times d\)。对 Qwen2-0.5B,`V≈151936`、`d=896`,光 embedding 就有 ~1.36 亿参数,占了总参数相当一部分——**小模型的词表embedding 占比惊人**。
- **计算量**:几乎为 0(只是按索引取行,没有乘加)。
- **访存特性**:**纯访存**。它是一次 gather,从大表里随机取 `S` 行。

很多模型把 embedding 表和最后的 LM Head **权重共享(weight tying)**,即 `W_emb` 转置后复用为输出投影,省一份 `V×d` 参数。迁移时如果发现「输入输出共用一个权重」,就是这个原因。

---

## 3. 自注意力:Transformer 的灵魂(也是长序列的瓶颈)

### 3.1 单头注意力回顾

对输入 `X ∈ [S, d]`,先用三个权重矩阵投影出 Query、Key、Value:

\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V \quad (W_* \in \mathbb{R}^{d \times d})
\]

然后做缩放点积注意力:

\[
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
\]

其中 `M` 是**因果掩码(causal mask)**:一个上三角为 \(-\infty\) 的矩阵,保证位置 `i` 只能看到 `≤ i` 的位置——这正是「自回归」的数学体现,也是后面 KV Cache 能成立的根本前提。

```
    QKᵀ 得到 [S, S] 的注意力分数矩阵,再加因果掩码:

          k0   k1   k2   k3
    q0  [  •    -∞   -∞   -∞ ]    ← q0 只能看 k0
    q1  [  •    •    -∞   -∞ ]    ← q1 能看 k0,k1
    q2  [  •    •    •    -∞ ]
    q3  [  •    •    •    •  ]    ← q3 能看全部历史
```

### 3.2 多头(Multi-Head):把 d 切成 h 份并行

把 `d` 维切成 `h` 个头,每个头维度 `head_dim = d / h`,各自独立做注意力,最后拼回来再过一个输出投影 `W_O ∈ [d, d]`:

```
Q,K,V: [S, d]  --reshape-->  [h, S, head_dim]   (h 个头各算各的)
每个头: softmax(qkᵀ/√head_dim) v  →  [h, S, head_dim]
拼接 concat  →  [S, d]  --W_O-->  [S, d]
```

> 直觉:多头 = 让模型在**不同子空间**关注不同的关系模式(有的头管语法、有的头管指代)。从工程看,它只是把一个大矩阵乘拆成 `h` 个小的并行批量矩阵乘(bmm),不增加总计算量。

### 3.3 计算量与参数量

设 batch=1。一个注意力子层:

| 步骤 | 计算量(FLOPs,乘加按 2 算) | 说明 |
|---|---|---|
| QKV 投影 | \(3 \times 2 S d^2\) | 三个 `[S,d]×[d,d]` |
| \(QK^\top\) | \(2 S^2 d\) | 随 \(S^2\) 增长 |
| 加权 \(\cdot V\) | \(2 S^2 d\) | 随 \(S^2\) 增长 |
| 输出投影 \(W_O\) | \(2 S d^2\) | |

- **参数量**:\(4 d^2\)(\(W_Q, W_K, W_V, W_O\) 各 \(d^2\))。
- **关键结论**:**投影部分随 \(d^2\)、序列长度线性;但 \(QK^\top\) 和 \(\cdot V\) 随 \(S^2\) 增长。** 当序列很短时,瓶颈是投影矩阵乘(compute-bound,大 GEMM);当序列很长(几千、几万 token)时,\(S^2\) 项主导,attention 本身变成大头——这就是 FlashAttention(M2 L7)要解决的问题。

### 3.4 GQA:为 KV Cache 而生的优化

现代模型(Qwen2、LLaMA-2 70B)多用 **GQA(Grouped-Query Attention)**:Query 仍有 `h` 个头,但 Key/Value 只有 `h_kv`(`< h`)个头,多个 Q 头**共享**一组 KV。

为什么?因为 **KV Cache 的显存占用正比于 KV 头数**(下节 Lesson 3 详算)。GQA 用很小的精度代价,把 KV Cache 砍到 `h_kv / h`。这是一个纯粹「为推理访存」做的结构设计,体现了「结构服务于推理」的现代趋势。记住这个伏笔。

---

## 4. FFN:参数大户与计算大户

每个 block 的第二个子层是 FFN(也叫 MLP)。原始 Transformer 是两层带 ReLU 的全连接:

\[
\text{FFN}(x) = W_2 \,\sigma(W_1 x),\quad W_1 \in \mathbb{R}^{d \times d_{ff}},\; W_2 \in \mathbb{R}^{d_{ff} \times d}
\]

通常 `d_ff = 4d`(中间「升维」4 倍再降回来)。现代模型用 **SwiGLU**(门控),需要三个矩阵 `W_gate, W_up, W_down`:

\[
\text{FFN}_{\text{SwiGLU}}(x) = W_{\text{down}}\big(\,\text{SiLU}(W_{\text{gate}}\, x) \odot (W_{\text{up}}\, x)\,\big)
\]

为了让总参数量与 4d 的版本接近,SwiGLU 的中间维度通常取 `d_ff ≈ (8/3)d`(约 2.67d)。

- **参数量**:原始版 \(2 \cdot d \cdot d_{ff} = 8d^2\);SwiGLU 版 \(3 \cdot d \cdot d_{ff} \approx 8d^2\)。**约为 Attention 投影参数(\(4d^2\))的 2 倍**。
- **计算量**:\(\approx 2 \times (\text{参数量}) \times S = 16 S d^2\) 量级,**随 \(S\) 线性、随 \(d^2\)**,没有 \(S^2\) 项。
- **访存特性**:典型的**大 GEMM**,prefill 时是 compute-bound;decode 时(S=1)变成「瘦高矩阵 × 大权重」,瓶颈转为读权重的带宽。

> 整体记忆:**FFN 是参数和算力的大头(尤其短序列),Attention 的 \(S^2\) 项在长序列才接管瓶颈。** 这决定了不同场景的优化重点。

---

## 5. RMSNorm 与 residual:便宜但不可省

### 5.1 RMSNorm vs LayerNorm

原始 Transformer 用 **LayerNorm**:减均值、除标准差、再缩放平移。现代模型几乎都换成了 **RMSNorm**:

\[
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot g
\]

区别:RMSNorm **去掉了减均值和偏置**,只做「按均方根缩放」。

为什么换?**RMSNorm 更省**:少了求均值这一步、少了一组偏置参数,而效果几乎无损。对推理来说,Norm 是**访存密集**的逐元素操作(计算量极小,但要把整个 `[S,d]` 读一遍写一遍),省一点是一点。这也是 M2 L6「Norm 高效实现 / 算子融合」的对象。

### 5.2 residual:梯度高速公路

每个子层都包了一个残差连接 `x = x + Sublayer(Norm(x))`。注意现代模型用的是 **Pre-Norm**(先 Norm 再进子层),比原始的 Post-Norm 更易训练、更稳定。

```
Pre-Norm(现代主流):              Post-Norm(原始 Transformer):
  x ──┬─────────────► +           x ──┬──► Sublayer ──► + ──► Norm
      │               ▲               │                 ▲
      └─Norm─Sublayer─┘               └─────────────────┘
```

- **参数量**:RMSNorm 每个仅 `d` 个(缩放向量 `g`),全模型可忽略。
- **访存特性**:Norm 与 residual 加法都是 memory-bound 的逐元素操作。单看很便宜,但 `L` 层累积起来访存次数可观,是算子融合的常客。

---

## 6. RoPE:把位置「转」进 Q 和 K

原始 Transformer 在 embedding 上加一个绝对位置向量。现代模型用 **RoPE(旋转位置编码)**:不加额外向量,而是在 Attention 内部,对 Q、K 的每一对维度做一个**与位置相关的旋转**。

直觉:把 `head_dim` 维向量两两配对成复平面上的点,位置为 `m` 的 token,其第 `j` 对维度旋转角度 \(m\theta_j\)。两个 token 做点积时,结果只依赖它们的**相对距离**——这就天然编码了相对位置。

```
位置 m 的 query 第 j 对维度:
  [x_2j, x_2j+1] 旋转 mθ_j 度
旋转矩阵:
  [cos mθ  -sin mθ] [x_2j  ]
  [sin mθ   cos mθ] [x_2j+1]
```

为什么推理工程师要在意 RoPE?三点:
1. **它在 KV Cache 里很关键**:K 在写入 cache **之前**就已经施加了 RoPE(带上了它自己的绝对位置),所以缓存的 K 复用时位置信息是对的——这是 KV Cache 能正确工作的细节之一。
2. **长上下文外推**靠改 RoPE(NTK / YaRN 等),都是调 \(\theta\) 的 base。
3. **计算量**:逐元素的乘加,极小,memory-bound,通常与 QKV 投影融合。

> 一句话总结现代结构选型:**LayerNorm→RMSNorm、绝对位置→RoPE、ReLU-FFN→SwiGLU,这三个替换的共同目标是「同等效果下更省、更稳、更利于长序列与推理」。**

---

## 7. 汇总:谁计算密集,谁访存密集

把全部子模块按 roofline 视角归类(以 batch=1、prefill 大 `S` 为例):

| 子模块 | 参数量 | 计算量量级 | prefill 瓶颈 | decode(S=1)瓶颈 |
|---|---|---|---|---|
| Embedding | \(Vd\) | ~0 | 访存(gather) | 访存 |
| QKV / O 投影 | \(4d^2\) | \(O(Sd^2)\) | 计算(大 GEMM) | 访存(读权重) |
| \(QK^\top\)、\(\cdot V\) | 0 | \(O(S^2 d)\) | 长序列时计算 | 访存(读 KV Cache) |
| FFN | \(\approx 8d^2\) | \(O(Sd^2)\) | 计算(大 GEMM) | 访存(读权重) |
| RMSNorm / residual / RoPE | \(\approx 0\) | ~0 | 访存 | 访存 |

**核心结论(背下来)**:
- **prefill 阶段**(一次性处理长 prompt):大矩阵乘占主导,**算力受限(compute-bound)**。
- **decode 阶段**(逐 token,S=1):没有大矩阵乘,主要时间花在「把全部权重 + KV Cache 从显存读出来用一次」,**带宽受限(memory-bound)**。

这正好呼应 M1 L1 的 roofline 伏笔,也是 Lesson 2 的主线。

---

## 8. 动手实验

代码见 `code/mini_gpt_block.py`。我们用 PyTorch 手搭一个**最小但结构正确**的 decoder block(含 RMSNorm + RoPE + GQA-ready 的 MHA + SwiGLU FFN),并打印各子模块的参数量与一次前向的 shape 流动。

核心片段(完整见文件):

```python
class DecoderBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + residual:注意是先 norm 再进子层,结果加回原始 x
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

运行:

```bash
cd code
python mini_gpt_block.py
```

你会看到:
1. 每个子模块的参数量,以及 FFN ≈ 2× Attention 的比例验证。
2. 一次前向中 `[B, S, d]` 在各步的 shape。
3. 一个小实验:固定 `d`,把 `S` 翻倍,观察 attention 分数矩阵 `[S, S]` 内存随 \(S^2\) 膨胀。

### 实验任务
- **必做**:跑通脚本,核对打印的参数量与公式 \(4d^2\)(attn)、\(\approx 8d^2\)(ffn)是否吻合。
- **必做**:补全文件里标注 `TODO` 的 `apply_rope` 函数(给 Q/K 施加旋转),用提供的断言验证正确性。
- **选做**:把 `n_kv_heads` 从等于 `n_heads` 改成它的一半(GQA),观察 K/V 投影参数量减半。

---

## 练习题

1. Qwen2-0.5B:`d=896`、`L=24`、`n_heads=14`、`n_kv_heads=2`、`d_ff=4864`、`V=151936`。请估算:(a) 单层 attention 投影参数量(注意 GQA);(b) 单层 FFN(SwiGLU)参数量;(c) embedding 参数量。
2. 为什么 decode 阶段即使你的 GPU 算力(TFLOPS)很高,也跑不快?用本课的 roofline 结论解释。
3. 如果把序列从 512 加到 8192(16×),attention 的 \(QK^\top\) 计算量增长几倍?FFN 计算量增长几倍?这说明长上下文场景该优先优化谁?
4. RMSNorm 相比 LayerNorm 省了哪两样东西?为什么对推理(尤其 decode)有意义?

<details>
<summary>参考答案(想完再看)</summary>

1. (a) GQA 下 Q 投影 \(d\times d = 896^2\),K、V 投影各 \(d \times (n_{kv}\cdot head\_dim) = 896 \times (2 \times 64)=896\times128\),O 投影 \(896^2\)。合计 \(2\times896^2 + 2\times896\times128 \approx 1.835M\)。(对比非 GQA 的 \(4\times896^2\approx3.2M\),省了约 43%。)
   (b) SwiGLU 三个矩阵:\(3 \times d \times d_{ff} = 3\times896\times4864 \approx 13.07M\)。约为 attention 的 7 倍?注意这里 GQA 把 attn 压小了,非 GQA 时 FFN/attn≈4。
   (c) \(V\times d = 151936\times896 \approx 136M\)。**单看一层,FFN+attn 才 ~15M,而 embedding 一项就 136M**——这正是小模型词表占比巨大的现象。
2. decode 时 S=1,没有大矩阵乘可喂饱算力;每生成 1 个 token 要把全部权重(几 GB)从显存读一遍,计算强度极低,落在 roofline 的「带宽受限斜坡」上。算力再高也用不上,瓶颈是显存带宽。
3. \(QK^\top\) 随 \(S^2\),16× 序列 → 256× 计算;FFN 随 \(S\),→ 16×。长上下文场景 attention 的平方项爆炸,应优先优化 attention(FlashAttention、KV Cache、稀疏注意力)。
4. 省了:(1) 减均值的统计计算,(2) 偏置参数(及其加法)。Norm 是 memory-bound 操作,decode 阶段对延迟敏感,少读写、少计算都直接降延迟;`L` 层累积收益可观。

</details>

---

## 小结

- decoder-only Transformer = embedding + `L` 个相同 block + 末层 Norm + LM Head;block 内核心是 **Attention** 和 **FFN** 两个子层。
- **参数量**:FFN ≈ 2× Attention 投影;小模型 embedding 占比惊人。
- **计算量**:投影与 FFN 随 \(Sd^2\);attention 的 \(QK^\top/\cdot V\) 随 \(S^2\),长序列才接管瓶颈。
- **roofline 归类**:prefill = compute-bound(大 GEMM);decode = memory-bound(读权重 + KV Cache)。
- 现代结构选型(**RMSNorm / RoPE / SwiGLU / GQA**)的统一动机是「同效更省、更稳、更利于长序列与推理」。

## 自测验收(过了再进 Lesson 2)
- [ ] 能默画出一个 token 从 id 到 logits 的完整 shape 流动图。
- [ ] 能写出 attention 与 FFN 的参数量、计算量量级公式,并说清谁随 \(S\)、谁随 \(S^2\)。
- [ ] 能解释为什么 prefill 是计算受限、decode 是带宽受限。
- [ ] `mini_gpt_block.py` 跑通,补全的 `apply_rope` 通过断言。
- [ ] 能说清 GQA 为什么能省 KV Cache、RoPE 为什么对 KV Cache 正确性重要。

下一课:**Lesson 2 — 自回归生成:prefill vs decode**,我们用代码亲手测出这两个阶段的耗时差异,看清「decode 为什么慢」。
