# M3 · Lesson 2:自回归生成——prefill vs decode

> 上一课我们把 Transformer 拆成了子模块,知道了「谁计算密集、谁访存密集」。本课把镜头拉到**生成过程**:大模型不是一次吐出整段回答,而是**一个 token 一个 token 地挤出来**。这个过程天然分成两个特性截然不同的阶段——**prefill** 和 **decode**。看懂它们的差异,你就理解了大模型推理优化的「主战场」。
> 预计用时:2 小时(阅读 + 实测两阶段耗时)。
> 前置:Lesson 1(子模块计算/访存特性、roofline 归类)。
> 环境:本课实测代码建议有 GPU(Colab T4 / AutoDL 4090 均可);纯 CPU 也能跑通,只是数值不同。

## 学习目标

学完本课你应该能回答:
1. 「自回归生成」具体是怎么循环的?为什么必须串行?
2. prefill 和 decode 在输入 shape、计算特性上有什么本质区别?
3. 为什么 prefill 是**算力受限**、decode 是**带宽受限**?
4. 「首 token 延迟(TTFT)」和「每 token 延迟(TPOT)」分别由哪个阶段决定?
5. 为什么说「decode 慢」是大模型推理的核心矛盾?

---

## 1. 自回归:语言模型怎么「写字」

大模型生成文本的方式,本质是一个**条件概率的连续采样**。给定已有 token 序列 \(x_1, \dots, x_t\),模型预测下一个 token 的分布:

\[
P(x_{t+1} \mid x_1, \dots, x_t)
\]

从这个分布里采一个 token(怎么采是 Lesson 4 的事),**拼回输入末尾**,再预测下一个。如此循环,直到生成结束符或达到长度上限。

```
prompt: "中国的首都是"
  step 0: 输入 "中国的首都是"        → 预测 "北"      → 拼接
  step 1: 输入 "中国的首都是北"      → 预测 "京"      → 拼接
  step 2: 输入 "中国的首都是北京"    → 预测 "。"      → 拼接
  step 3: 输入 "中国的首都是北京。"  → 预测 <eos>     → 停止
```

**关键认知**:第 `t+1` 步的输入依赖第 `t` 步的输出。这是一条**强串行依赖链**——你没法并行生成第 100 个和第 101 个 token,因为算第 101 个时必须先知道第 100 个是什么。

> 回忆 M1 L1 「什么不适合 GPU」那一节:**强串行依赖**正是 GPU 的天敌。自回归 decode 恰恰是强串行的,这是大模型推理慢的**结构性根源**,不是工程没做好。

---

## 2. 两个阶段:prefill 与 decode

仔细看上面的循环,会发现它其实分两段:

### 阶段一:Prefill(预填充)—— 处理 prompt

第一步,模型要消化整个 prompt(比如 50 个 token)。这 50 个 token 是**一次性、全部已知**的,可以**并行**喂进模型,一次前向就把它们全部算完,并顺便得到第一个新 token。

```
Prefill:  输入 [S_prompt 个 token]  ──一次前向──►  logits + 第 1 个新 token
          ↑ 所有 prompt token 并行处理,塞满 GPU
```

### 阶段二:Decode(解码)—— 逐 token 生成

之后每一步,输入**只有 1 个**刚生成的 token(配合缓存的历史,见 Lesson 3),前向一次只产出 1 个新 token。要生成 200 个 token,就要串行跑 200 次。

```
Decode:  输入 [1 个 token]  ──前向──►  1 个新 token   (重复 N 次,串行)
         输入 [1 个 token]  ──前向──►  1 个新 token
         ...
```

把两阶段画在一起:

```
时间 ──────────────────────────────────────────────►
│ Prefill │ D │ D │ D │ D │ D │ D │ D │ ...  │ D │
│ (一次)  │   逐 token,每个一次前向,串行       │
   ▲           ▲
   │           └─ 每个 D 产出 1 token,决定 TPOT(每 token 延迟)
   └─ 处理整个 prompt,决定 TTFT(首 token 延迟)
```

| 维度 | Prefill | Decode |
|---|---|---|
| 输入序列长度 | `S_prompt`(几十~几千) | `1`(每步) |
| 一次前向产出 | 整个 prompt 的表示 + 第 1 个 token | 1 个 token |
| 执行次数 | 1 次 | `N_gen` 次(串行) |
| 主要算子形态 | 大矩阵 × 大矩阵(GEMM) | 瘦向量 × 大矩阵(GEMV-like) |
| 决定的指标 | **TTFT**(首 token 延迟) | **TPOT**(每 token 延迟) |

---

## 3. 为什么 prefill 算力受限、decode 带宽受限

这是本课最重要的一节,把 Lesson 1 的 roofline 结论落到两阶段上。回忆 roofline:**计算强度(FLOP/Byte)** 决定你卡在算力还是带宽。

### Prefill:大 GEMM,喂得饱算力

prefill 时输入是 `[S, d]`,`S` 很大。以 FFN 的矩阵乘为例,是 `[S, d] × [d, d_ff]`——一个**又高又宽**的矩阵乘。

- 读权重 `[d, d_ff]`:一份字节。
- 但这份权重被 `S` 行输入**复用了 S 次**(每行都要乘一遍)。
- **计算强度 ∝ S**:S 越大,每读 1 字节权重就做越多次乘加,计算强度高 → 落在 roofline 屋顶下 → **算力受限**。

> 直觉:prefill 像「批处理」,一大批数据复用同一份权重,GPU 的几千个核心都有活干,Tensor Core 喂得饱。

### Decode:GEMV,带宽白白浪费

decode 时输入是 `[1, d]`(只有 1 个 token)。同样的 FFN 矩阵乘变成 `[1, d] × [d, d_ff]`——一个**向量 × 矩阵**(GEMV)。

- 读权重 `[d, d_ff]`:还是一份字节(权重大小不变!)。
- 但这份权重只被 **1 行**输入用了 **1 次**。
- **计算强度 ≈ 常数(很低)**:读了一大堆权重,只做了一点点计算 → 落在 roofline 斜坡上 → **带宽受限**。

```
Prefill GEMM:                      Decode GEMV:
 [S,d] x [d,d_ff]                   [1,d] x [d,d_ff]
 权重复用 S 次,算力打满            权重只用 1 次,大部分时间在等显存读权重
 compute-bound ✅                   memory-bound ❌(算力闲置)
```

**核心结论(背下来)**:
- decode 每生成 1 个 token,都要把**整个模型的权重**(7B 模型 fp16 ≈ 14GB)从显存完整读一遍,而只做了 1 个 token 的计算。
- 所以 **decode 的速度上限 ≈ 显存带宽 ÷ 模型权重字节数**,与算力几乎无关。

举个硬数字:一张 A100 显存带宽约 2TB/s,跑 fp16 的 13GB 权重,理论 decode 上限 ≈ 2000/13 ≈ **150 token/s**(单请求、不计其他开销)。**这就是为什么单条请求 decode 那么慢——不是 GPU 不行,是带宽天花板。**

> 这条结论直接推出后续整个课程的两大优化方向:
> 1. **减少要读的字节数** → 量化(M5):权重从 fp16 变 int4,带宽需求降 4×,decode 直接快几倍。
> 2. **让一次读权重服务更多 token** → batching(M4):多条请求拼一起 decode,权重读一次给 B 个 token 用,提高计算强度。

---

## 4. 一个常被忽略的细节:decode 也要读 KV Cache

decode 慢,除了读权重,还有一个隐藏开销:**读 KV Cache**(下一课主角)。每生成一个 token,attention 要回看**全部历史 token 的 K、V**,这些缓存随序列变长而变大,也要从显存读出来。

所以 decode 的带宽开销 = **读权重(固定)+ 读 KV Cache(随上下文增长)**。短上下文时权重是大头;长上下文(几万 token)时 KV Cache 甚至会超过权重——这是 Lesson 3 和 PagedAttention(M4)的动机。先记住这个伏笔。

---

## 5. 动手实验:亲手测出两阶段差异

代码见 `code/prefill_vs_decode.py`。我们加载一个真实小模型(默认 GPT-2,也可换 Qwen2-0.5B),分别测量:
1. **prefill**:喂一个长度为 `S` 的 prompt,记录耗时与输出 shape。
2. **decode**:逐 token 生成 `N` 个,记录每步 shape 与平均每 token 耗时。
3. 对比两者的「每 token 摊销耗时」,直观看到 decode 的串行代价。

核心逻辑(完整见文件,这里展示两阶段 shape 对比):

```python
# Prefill:一次性吃下整个 prompt
logits = model(input_ids).logits        # input_ids: [1, S_prompt]
next_token = logits[:, -1, :].argmax(-1) # 只取最后一个位置预测下一个

# Decode:每次只喂 1 个 token(此处未用 KV Cache,Lesson 3 再加)
for _ in range(n_gen):
    logits = model(seq).logits           # seq 每步增长 1
    next_token = logits[:, -1, :].argmax(-1)
    seq = torch.cat([seq, next_token[:, None]], dim=-1)
```

运行:

```bash
cd code
python prefill_vs_decode.py --model gpt2 --prompt-len 64 --gen-len 64
```

你会看到类似(数值随硬件不同):

```
[prefill] 输入 shape [1, 64] -> 耗时 18.3 ms,产出 1 个 token
[decode ] 生成 64 个 token,总耗时 612 ms,平均每 token 9.6 ms
[对比   ] prefill 64 token 用 18.3ms(0.29ms/token);decode 单 token 9.6ms
          → decode 每 token 比 prefill 摊销慢约 33×
```

**观察重点**:prefill 把 64 个 token 并行处理只用了十几毫秒,而 decode 生成同样 64 个却用了几百毫秒。**同样的算力,prefill 喂得饱、decode 喂不饱**——这就是 roofline 两个区间的真实体现。

### 实验任务
- **必做**:跑通脚本,改变 `--prompt-len`(32/128/512),观察 prefill 耗时如何变化、decode 单 token 耗时是否基本不变。
- **必做**:补全文件中 `TODO` 的「计算强度估算」函数,粗略算出 prefill 与 decode 的 FLOP/Byte 比值。
- **选做**:用 `torch.cuda.synchronize()` 确保 GPU 计时正确(回忆 M1 L5:不同步会测到假的快)。脚本已留位置,体会异步计时的坑。

---

## 练习题

1. 一个请求 prompt 长 1000 token,要生成 500 token。哪个阶段做的「前向次数」多?哪个阶段做的「总浮点运算」可能更多?
2. 用「读权重」的视角解释:为什么把 batch size 从 1 提到 32,decode 的**总吞吐**几乎线性涨,但**单请求延迟**几乎不变?
3. TTFT(首 token 延迟)主要由谁决定?如果用户抱怨「点了发送后等很久才出第一个字」,你先怀疑哪个阶段、哪类瓶颈?
4. 为什么说「量化」对 decode 的加速效果,通常比对 prefill 更明显?

<details>
<summary>参考答案(想完再看)</summary>

1. **前向次数**:prefill 1 次,decode 500 次,decode 多得多。**总浮点运算**:prefill 一次处理 1000 个 token 的大 GEMM,单次 FLOP 巨大;decode 每次只算 1 token 但跑 500 次。具体谁多取决于 prompt 与生成长度比例,长 prompt + 短生成时 prefill 总 FLOP 反而可能更大。但**耗时**上 decode 通常是大头,因为它带宽受限且串行。
2. decode 读一次权重本来只服务 1 个 token(浪费带宽);batch=32 时,读一次权重同时给 32 个 token 算(权重复用 32 次),计算强度提高 32×,带宽利用率上去了,总吞吐近似线性涨。但每个请求仍要等这一整步算完,单请求延迟不变(甚至略增)。这就是 continuous batching 的核心收益(M4)。
3. TTFT 主要由 **prefill** 决定(还含排队、调度等)。长 prompt 时 prefill 是大 GEMM,算力受限;若抱怨首 token 慢,先查 prompt 长度与 prefill 阶段(以及是否在排队)。
4. decode 是带宽受限,瓶颈是「读权重的字节数」;量化把权重字节数降 4×(int4),直接把带宽瓶颈缓解 4×。prefill 是算力受限,减少字节数对它帮助小(它本来就喂得饱算力),所以量化对 decode 收益更显著。

</details>

---

## 小结

- 自回归生成是**强串行**过程:第 `t+1` 个 token 依赖第 `t` 个,无法并行 —— 这是 decode 慢的结构性根源。
- 生成分两阶段:**prefill**(一次并行处理整个 prompt,定 TTFT)与 **decode**(逐 token 串行生成,定 TPOT)。
- **prefill = 大 GEMM = 算力受限**;**decode = GEMV = 带宽受限**。decode 每个 token 都要把整个模型权重读一遍。
- decode 速度上限 ≈ 显存带宽 ÷ 权重字节数,与算力几乎无关 → 引出两大优化:**量化**(降字节)与 **batching**(提复用)。
- decode 还要读随上下文增长的 **KV Cache**,这是下一课的主角。

## 自测验收(过了再进 Lesson 3)
- [ ] 能画出 prefill→decode 的时间线,并标出 TTFT / TPOT 各由谁决定。
- [ ] 能用 roofline 解释为什么 prefill 算力受限、decode 带宽受限。
- [ ] 能估算给定带宽和模型大小下,单请求 decode 的理论 token/s 上限。
- [ ] `prefill_vs_decode.py` 跑通,亲眼看到 decode 单 token 远慢于 prefill 摊销。
- [ ] 能说清 batching 和量化分别从哪个角度加速 decode。

下一课:**Lesson 3 — KV Cache**,我们解决「decode 每步都重算历史」的浪费,推导显存占用公式,并给迷你推理加上缓存看提速。
