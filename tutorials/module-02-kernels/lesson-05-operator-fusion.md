# M2 · Lesson 5:算子融合——把多趟显存往返压成一趟

> 上一课的 fused-softmax 让你尝到了"融合"的甜头:快,不是因为算得多,而是因为**搬得少**。本课把算子融合上升为方法论:它为什么能加速(两个独立来源)、什么算子值得融、怎么动手融。我们以 Transformer FFN 里最常见的 **bias + GELU** 为例,手动融合并 benchmark。
> 这是带宽受限算子优化的"第一工具",也是推理框架(TensorRT、vLLM)里 fusion pass 的核心思想。
> 预计用时:2.5 小时。
> 前置:M2 L1(roofline、带宽受限)、L4(Triton 基础)。

## 学习目标

1. 说清算子融合加速的**两个独立来源**:减少 kernel 启动开销 + 减少 global memory 往返。
2. 判断哪些算子适合融合(带宽受限的逐元素/归约链)。
3. 用 Triton 手写 **fused bias+GELU**,与"PyTorch 逐算子"和"`torch.compile`"对比。
4. 把融合和推理框架里的 fusion pass、以及后续 FlashAttention 联系起来。

---

## 1. 未融合时,代价藏在哪

考虑 FFN 里一段常见计算(GEMM 之后):

```python
y = x + bias        # 广播加 bias
y = gelu(y)         # 激活
```

PyTorch 默认怎么跑?**两个(及以上)独立 kernel**:

```
kernel 1 (加 bias): 读 x(HBM) → +bias → 写 tmp(HBM)
kernel 2 (gelu):    读 tmp(HBM) → gelu → 写 y(HBM)
```

注意中间结果 `tmp`:它被**写回 HBM 又立刻读回来**,纯属浪费。每个元素的真实显存流量:

```
未融合:读 x + 写 tmp + 读 tmp + 写 y = 4 次 HBM 访问 / 元素
融合后:读 x + 写 y               = 2 次 HBM 访问 / 元素
```

**访存量直接砍半。** 对带宽受限算子(GELU、bias 都是逐元素,AI 极低,L1 结论),访存量减半 ≈ 耗时减半。这是融合加速的**第一个来源:减少 global memory 往返**。

> 大数据类比:这就是 Spark 的**算子链(pipelining)**——`map(f).map(g)` 不会把中间结果落盘,而是一条记录连着过 f 和 g。不融合的逐算子执行,相当于每个 map 之间都 `cache` 到磁盘再读,IO 翻倍。GPU 上"落盘"换成"写回 HBM",道理一模一样。

---

## 2. 第二个来源:kernel 启动开销

每次启动一个 CUDA kernel,CPU 都要发射一条命令给 GPU,这有固定开销(几微秒级)。单看一次不多,但:

- 大模型一层有几十个小算子(bias、激活、残差、scale、dropout……),几十层叠起来上千次启动。
- decode 阶段每生成 1 个 token 就要把整个网络跑一遍,kernel 启动次数 × token 数,累积惊人。
- 当每个 kernel 本身很小(decode 时 batch=1,算子很小),**启动开销甚至能超过 kernel 实际计算时间**,GPU 大量时间在"等下一条命令"而非干活。

融合把 N 个 kernel 合成 1 个,启动开销降到 1/N。这是融合加速的**第二个来源:减少 kernel 启动次数**。

```
未融合:  [launch][k1 算] [launch][k2 算] [launch][k3 算]   ← 启动间隙 GPU 空转
融合:    [launch][   k1+k2+k3 一起算   ]                    ← 一次启动,连续干活
```

> 两个来源的相对重要性:**大张量(prefill/训练)**主要省的是访存往返(来源一);**小张量(decode、小 batch)**启动开销占比大,来源二更关键。CUDA Graph 是另一种专治"启动开销"的技术(M4 会遇到),而算子融合两个一起省。

---

## 3. 什么算子值得融合

判断标准(全部用 L1 的 roofline 视角):

✅ **值得融**:
- 一连串**带宽受限**的逐元素算子(bias、激活、scale、残差加、dropout)——它们各自都在带宽斜坡上,融合直接减访存。
- 逐元素 + 归约的组合(LayerNorm、softmax)——见 L6。
- **GEMM 的 epilogue**:把 GEMM 之后的 bias+激活直接焊在 GEMM kernel 尾部(cuBLAS/CUTLASS 的 epilogue fusion),数据还在寄存器/shared 里就处理完,连写回都省了。

❌ **不太值得 / 难融**:
- 已经是算力受限的大 GEMM,本身访存占比小,融个 bias 收益有限(但 epilogue 融合仍有意义)。
- 两个算子之间有**全局依赖/形状变化**(如需要跨整个张量归约再广播)——能融但要用 online/分块技巧(L6/L7)。

> 一句话:**融合主要救"带宽受限"的算子;算力受限的算子融合收益小。** 所以融合前先用 roofline 判定——又回到 L1 那把尺子。

---

## 4. 动手:fused bias + GELU

GELU(Gaussian Error Linear Unit)的常用 tanh 近似:

```
GELU(x) ≈ 0.5 · x · (1 + tanh[ √(2/π) · (x + 0.044715·x³) ])
```

我们要算 `gelu(x + bias)`,bias 沿最后一维广播(每列一个 bias,Linear 层的典型形状)。

完整代码见 `code/fused_bias_gelu.py`。Triton kernel:

```python
@triton.jit
def bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, n_rows, n_cols,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // tl.cdiv(n_cols, BLOCK_SIZE)
    col_block = pid % tl.cdiv(n_cols, BLOCK_SIZE)
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    offs = row * n_cols + col_offsets

    x = tl.load(x_ptr + offs, mask=mask)
    bias = tl.load(bias_ptr + col_offsets, mask=mask)  # 沿列广播
    z = x + bias                                       # 第 1 步:加 bias
    # 第 2 步:GELU(tanh 近似)。全部在寄存器里完成,不落 HBM。
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (z + 0.044715 * z * z * z)
    gelu = 0.5 * z * (1.0 + tl_tanh(inner))
    tl.store(out_ptr + offs, gelu, mask=mask)          # 只写一次
```

关键就在于:`x + bias` 的中间结果 `z` 留在**寄存器**里,直接喂给 GELU,**从不写回 HBM**。一次读 x、一次读 bias、一次写 out,干净利落。

运行:

```bash
cd code
python fused_bias_gelu.py
```

参考输出(数值随卡):

```
[INFO] correctness vs torch: max abs err = 3.1e-07
[INFO] shape = (8192, 8192)
[INFO] torch (2 kernels) : 1.42 ms, 369 GB/s
[INFO] triton fused      : 0.78 ms, 672 GB/s
[INFO] torch.compile     : 0.81 ms, 647 GB/s
[INFO] speedup fused/eager = 1.8x
```

**读数要点**:
- 融合版接近 2 倍加速,基本兑现了"访存减半"的理论预期。
- `torch.compile` 也能自动融合到差不多的水平——这说明**编译器能自动做的融合,你不必手写**;手写 Triton 的价值在于编译器搞不定的复杂融合(如 FlashAttention)。
- 看 GB/s:融合版更接近你 L1 测的峰值带宽,说明它把带宽吃得更满。

### 留给你的 TODO

`fused_bias_gelu.py` 里:
1. 把 GELU 换成 SiLU/Swish(`x·sigmoid(x)`),验证融合同样有效(LLaMA 系用的就是 SiLU)。
2. 实现一个"未融合"的 Triton 两 kernel 版本(bias 一个 kernel、gelu 一个),亲手测出它比融合版慢,体会中间结果落地 HBM 的代价。

---

## 5. 接回推理框架

你刚手动做的事,推理框架在**编译期自动**做,叫 **fusion pass**:

- **TensorRT / TensorRT-LLM**:图优化阶段把 conv+bias+relu、gemm+bias+gelu 等模式匹配并融合成单个 kernel。
- **PyTorch `torch.compile`(Inductor)**:自动把逐元素算子链融合,生成 Triton kernel(对,它后端就是 Triton!)。
- **vLLM / 手写引擎**:对热点路径(如 RMSNorm、RoPE、激活)写专门的融合 kernel。

更进一步,**FlashAttention 本质就是一次"超级融合"**:把 `QK^T → scale → softmax → ×V` 这一长串(还包括一个巨大的中间矩阵)融进一个 kernel,中间结果永不落 HBM。L7 你会看到,它用的正是本课"中间结果留在片上"的思想,只是要配合 online softmax 解决归约依赖。

> 记住这条优化主线:**带宽受限 → 减少访存 → 算子融合 → 极致形态就是 FlashAttention。** 本课是这条线的方法论,下两课是它在 softmax/LayerNorm/attention 上的具体兑现。

---

## 练习题

1. 列出 `gelu(x+bias)` 未融合(2 kernel)和融合(1 kernel)各自每元素的 HBM 访问次数,算理论加速比。
2. 算子融合的两个加速来源分别是什么?在 prefill(大张量)和 decode(batch=1 小张量)场景下,哪个来源更主导?
3. 为什么"算力受限的大 GEMM"融合一个 bias 收益不大,但 cuBLAS 还是要做 epilogue fusion?
4. `torch.compile` 已能自动融合逐元素算子,那为什么 FlashAttention 还需要人手写(而不是指望编译器自动融出来)?

<details>
<summary>参考答案</summary>

1. 未融合:读 x、写 tmp、读 tmp、写 out = 4 次;融合:读 x、写 out = 2 次(bias 那一点读忽略不计,因为它小且可缓存)。理论加速 ≈ 4/2 = **2 倍**,与实测吻合。
2. 来源一:减少 global memory 往返(中间结果不落地);来源二:减少 kernel 启动开销。prefill 张量大,访存往返是大头,**来源一主导**;decode batch=1 时每个 kernel 很小,启动开销占比高,**来源二主导**(也是为什么 decode 还要叠加 CUDA Graph)。
3. 大 GEMM 是算力受限,它的总时间由 FLOP 决定,bias 那点访存占比极小,单独看收益小。但 epilogue fusion 仍有意义:GEMM 算完的结果**本来就在寄存器/shared 里**,顺手做完 bias+激活再写回,省掉了"GEMM 写回 + 激活 kernel 再读"这一整趟往返,且省一次 kernel 启动,几乎零成本白赚。
4. 自动融合擅长**逐元素算子链**(无复杂数据依赖)。FlashAttention 涉及 softmax 的**跨块归约依赖**(需要 online softmax 重写算法)、tiling、以及避免实例化 O(N²) 中间矩阵——这是算法层面的重构,不是简单的算子链拼接,通用编译器目前难以自动推导出来,需要人把算法改写好。

</details>

---

## 小结

- 算子融合加速有**两个独立来源**:① 减少 global memory 往返(中间结果留片上)② 减少 kernel 启动开销。
- 主要救**带宽受限**算子(逐元素链、归约链);算力受限算子融合收益小(但 epilogue fusion 仍划算)。
- 手写 Triton fused bias+GELU 可得约 2 倍加速,与"访存减半"的理论一致;`torch.compile` 能自动做简单融合。
- 推理框架的 fusion pass、以及 **FlashAttention** 都是这一思想的工程化与极致化。

## 自测验收
- [ ] 能说清融合的两个加速来源及各自主导的场景。
- [ ] 能用 HBM 访问次数算出 bias+gelu 融合的理论加速比。
- [ ] `fused_bias_gelu.py` 跑通,看到融合版接近 2 倍加速。
- [ ] 能解释为什么 FlashAttention 不能靠编译器自动融出来。

下一课:**Lesson 6 — Softmax 与 LayerNorm 的高效实现**,深入数值稳定性与 online/两遍策略,为 FlashAttention 铺最后一块砖。
