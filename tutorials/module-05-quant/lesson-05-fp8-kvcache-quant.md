# M5 · Lesson 5:FP8 与 KV Cache 量化

> 前四课都是**整数**量化(INT8/INT4)。本课换一种低精度:**FP8**——8 bit 的浮点。它不是整数,而是把 FP16 的指数和尾数都砍短。为什么大模型偏爱浮点而非整数低精度?Hopper(H100)又为它做了什么硬件支持?最后我们攻克长上下文推理的真正杀手——**KV Cache 量化**:当上下文长到几万 token,KV Cache 会比模型权重还吃显存和带宽,量化它是长文本推理的命脉。动手在 vLLM 里一行开关打开 fp8 kv cache,亲眼看显存怎么降。
> 预计用时:2.5 小时(理论 + vLLM 实战需 N 卡)。
> 前置:L1(量化误差、roofline、decode 带宽)、Module 3(KV Cache 原理与显存计算)。

## 学习目标

学完本课你应该能回答:
1. FP8 是什么?E4M3 和 E5M2 两种格式差在哪、各用在哪?
2. 为什么 FP8 比 INT8 更适合表示有 outlier 的张量?
3. Hopper 的 FP8 硬件支持带来什么(对比 INT8/FP16)?
4. KV Cache 为什么是长上下文的瓶颈?量化它能省什么?
5. 实战:在 vLLM 里开启 FP8 KV Cache,量化显存/吞吐变化。

---

## 1. FP8:把浮点砍到 8 bit

回忆浮点数的结构:`符号位(sign) + 指数位(exponent) + 尾数位(mantissa)`。FP16 是 1+5+10。FP8 把它压到 8 位,有两种标准切分(OCP/NVIDIA 规范):

```
FP16:  S EEEEE MMMMMMMMMM     (1 + 5 + 10 = 16 bit)

FP8 E4M3: S EEEE MMM          (1 + 4 + 3 = 8 bit)  尾数多 → 精度高,范围小
FP8 E5M2: S EEEEE MM          (1 + 5 + 2 = 8 bit)  指数多 → 范围大,精度低
```

| 格式 | 指数位 | 尾数位 | 动态范围 | 精度 | 典型用途 |
|---|---|---|---|---|---|
| **E4M3** | 4 | 3 | 较小(±448) | 较高 | **前向**:权重、激活、KV Cache |
| **E5M2** | 5 | 2 | 较大(±57344) | 较低 | **反向**:梯度(范围大,精度要求低) |

记忆法:**E4M3 精度优先(推理常用),E5M2 范围优先(训练梯度常用)。** 本课讲推理,主角是 **E4M3**。

> **指数 vs 尾数的取舍**:指数位决定能表示多大/多小的数(动态范围),尾数位决定相邻可表示数之间有多密(精度)。8 个 bit 就这么多,给指数多了就给尾数少了,这是 E4M3/E5M2 的本质区别。

---

## 2. 为什么 FP8 比 INT8 更"扛 outlier"

这是 FP8 在 LLM 里受宠的关键。回忆 L1:INT8 是**均匀量化**——所有整数格子等间距,scale 固定。FP8 是**浮点量化**——格子是**非均匀**的,越靠近 0 越密,越往大值越稀疏。

```
INT8(均匀):  | | | | | | | | | | | | | | | |   格子等距
FP8 (浮点):   ||||| | |  |   |    |     |      格子近 0 密、远 0 疏
                 ↑小值精度高     ↑大值也能表示(范围大)
```

神经网络的张量分布特点是:**大量小值集中在 0 附近 + 少数大幅 outlier**。FP8 的非均匀分布天生契合:
- 0 附近格子密 → 占多数的小值精度高;
- 大值处格子稀但能覆盖 → outlier 不会被 clip 掉,只是精度低一点(但 outlier 本来数量就少)。

> **一句话**:INT8 用均匀格子,被 outlier 撑大 scale 后小值精度崩;FP8 用浮点的非均匀格子,小值密、大值疏,**同时兼顾了精度和动态范围**,因此对 LLM 这种"小值多 + 有 outlier"的分布更友好,常常 PTQ 直接量化就能保住精度,无需 GPTQ/AWQ 那样的复杂处理。

代价:FP8 量化/反量化比 INT8 略复杂(浮点编解码),且需要硬件原生支持才划算——这就引出 Hopper。

---

## 3. Hopper 的 FP8 硬件支持

FP8 不是软件模拟才有意义,关键在硬件。**NVIDIA Hopper 架构(H100/H800)的第四代 Tensor Core 原生支持 FP8 矩阵乘**(Ada/L40S 也支持;Ampere/A100 **不**原生支持 FP8)。

```
Tensor Core 算力(粗略量级,越低精度越快):
FP16 ████████              1×
FP8  ████████████████      2×   ← Hopper 原生,吞吐约 FP16 两倍
INT8 ████████████████      2×   (整数,但 outlier 难处理)
```

Hopper FP8 的价值:
- **吞吐**:FP8 矩阵乘吞吐约为 FP16 的 2 倍(prefill/计算受限场景直接受益)。
- **显存/带宽**:权重、激活、KV Cache 用 FP8 存,字节数减半 → decode 带宽减半。
- **精度**:相比 INT8,FP8 保精度更容易(上一节的非均匀优势),很多场景 PTQ 即可。

> **对照你的认知**:INT8 走的是"整数算力 + 复杂 outlier 处理"路线(SmoothQuant 那套);FP8 走的是"原生浮点低精度 + 硬件支持"路线,工程上更省心,但**绑定 Hopper 及以后**的卡。没有 H 卡(比如只有 A100/4090)时,FP8 收益有限甚至不支持,这时仍用 INT4/INT8 方案。这是选型时必须确认的硬件前提。

---

## 4. KV Cache:长上下文的真正瓶颈

前面的量化都在管**权重**。但长上下文场景下,有个东西会膨胀到比权重还大——**KV Cache**。

回忆 Module 3:自回归生成时,为避免重复计算,每个 token 在每一层的 Key 和 Value 都被缓存下来。KV Cache 的大小:

\[
\text{KV Cache 字节} = 2 \times L \times H \times d \times S \times B \times \text{dtype\_bytes}
\]

其中 2(K 和 V)、\(L\) 层数、\(H\) 注意力头数、\(d\) 头维度、\(S\) 序列长度、\(B\) batch、dtype 字节数。**注意它随序列长度 \(S\) 和 batch \(B\) 线性增长。**

```
短上下文(S=512):    权重 ████████████  KV ██           权重是大头
长上下文(S=32768):  权重 ████████████  KV ████████████████████  KV 反超!
                                          ↑随上下文线性膨胀
```

举例(粗算):一个 7B 模型在 32K 上下文、batch=8 时,KV Cache 可达几十 GB,**超过模型权重本身**。后果:
- **显存爆**:KV Cache 占满显存,放不下更大 batch / 更长上下文。
- **带宽爆**:decode 时每生成一个 token,attention 要读取**全部历史 KV**,KV 越大,decode 越慢(又是带宽受限!)。

> **结论**:长上下文 + 大并发场景,**KV Cache 量化和权重量化同等重要,甚至更重要**。把 KV Cache 从 FP16 量化到 FP8(或 INT8),显存和 KV 读取带宽直接减半——这是支撑长文本、高并发推理的关键技术。

### 为什么 KV Cache 常用 FP8 而非 INT4?

KV Cache 是**激活**性质的张量(动态、有 outlier),且 attention 对 KV 的精度较敏感。所以:
- INT4 太激进,KV 用 INT4 易掉点;
- **FP8(E4M3)** 是当前 KV Cache 量化的甜点:非均匀格子扛 outlier、半字节省一半、Hopper 原生支持。
- INT8 KV Cache 也常见(需要 per-token/per-channel 校准来对付 outlier)。

---

## 5. 动手实验

### 实验 A:FP8 vs INT8 在 outlier 数据上的对比(必做,纯 CPU)

运行 `code/fp8_vs_int8.py`:用一个"小值多 + 有 outlier"的张量,分别模拟 INT8 均匀量化和 FP8(E4M3)非均匀量化,对比量化误差,直观验证第 2 节的结论。(FP8 用软件模拟编解码,不需要 H 卡。)`fake_fp8_e4m3` 的尾数舍入留了 TODO。

### 实验 B:KV Cache 显存计算器(必做,纯 CPU)

运行 `code/kvcache_calc.py`:输入模型规格(层数/头数/头维/上下文/batch),算出 FP16 与 FP8 下的 KV Cache 显存,看长上下文时它如何反超权重。

### 实验 C:vLLM 开启 FP8 KV Cache(必做,需 N 卡)

⚠️ 需要 N 卡(FP8 KV Cache 的最佳硬件是 Hopper;部分版本在 Ampere/Ada 上也能跑 FP8 KV,以软件转换实现)。

`code/vllm_fp8_kv.py` 演示:同一模型分别用默认(FP16 KV)和 `kv_cache_dtype="fp8"` 启动 vLLM,对比:
- KV Cache 能容纳的最大 token 数(吞吐/并发上限);
- 长 prompt 下的显存占用;
- 输出质量(抽几条对比是否明显退化)。

```bash
pip install vllm
python vllm_fp8_kv.py
```

关键就一个开关:

```python
# FP16 KV(默认)
llm = LLM(model=MODEL)
# FP8 KV:KV Cache 显存减半,可容纳约 2 倍 token
llm = LLM(model=MODEL, kv_cache_dtype="fp8")
```

---

## 练习题

1. E4M3 和 E5M2 的位分配各是多少?推理(前向)更常用哪个,为什么?
2. 为什么说 FP8 比 INT8 更适合表示带 outlier 的神经网络张量?
3. A100(Ampere)能像 H100 一样原生加速 FP8 矩阵乘吗?这对你的量化选型有什么影响?
4. 一个模型上下文从 2K 拉到 64K,KV Cache 显存怎么变?为什么这时 KV Cache 量化变得关键?
5. 为什么 KV Cache 量化常选 FP8/INT8 而很少用 INT4?

<details>
<summary>参考答案(想完再看)</summary>

1. E4M3 = 1 符号 + 4 指数 + 3 尾数;E5M2 = 1 符号 + 5 指数 + 2 尾数。推理前向常用 **E4M3**,因为它尾数多、精度高、动态范围对前向激活/权重/KV 足够;E5M2 范围大精度低,更适合训练时数值范围跨度大的梯度。

2. INT8 是均匀量化,格子等距,被 outlier 撑大 scale 后,占多数的小值被挤进极少格子精度崩;FP8 是浮点非均匀量化,0 附近格子密(小值精度高)、大值处格子稀但能覆盖(outlier 不被 clip)。神经网络张量正是"小值多 + 少量 outlier",FP8 同时兼顾精度和范围,更契合。

3. **不能**。FP8 原生 Tensor Core 加速从 Hopper(H100/H800)开始(Ada/L40S 也支持),Ampere/A100 不原生支持 FP8。影响:只有 A100/4090 时,FP8 收益有限或不可用,应优先 INT4(AWQ/GPTQ,省带宽)或 INT8(SmoothQuant)。选型前必须确认硬件代际。

4. KV Cache 随序列长度线性增长,2K→64K 约增大 32 倍,长上下文下 KV Cache 会超过模型权重成为显存和 decode 带宽的最大头。此时量化 KV Cache(FP16→FP8 减半)能直接放大可用上下文/并发,并加速 decode(attention 要读全部历史 KV),所以变得关键。

5. KV Cache 是激活性质的张量(动态、有 outlier),且 attention 对其精度较敏感,INT4 太激进易明显掉点;FP8(E4M3)非均匀格子扛 outlier、半字节省显存、Hopper 原生支持,是甜点;INT8 也可但需 per-token/channel 校准处理 outlier。

</details>

---

## 小结

- **FP8** 是 8 bit 浮点,两种格式:**E4M3**(精度优先,推理前向)、**E5M2**(范围优先,训练梯度)。
- FP8 是**非均匀**量化(近 0 密、远 0 疏),天生契合"小值多 + 有 outlier"的 LLM 张量,**比 INT8 更扛 outlier**,常 PTQ 即可保精度。
- **Hopper(H100)原生支持 FP8 Tensor Core**(A100 不支持):吞吐约 FP16 两倍,字节减半省带宽;选型须确认硬件代际。
- **KV Cache** 随上下文/batch 线性膨胀,长上下文下会反超权重成为显存与 decode 带宽瓶颈;**量化 KV Cache 是长文本/高并发推理的命脉**,常用 **FP8/INT8**(不用 INT4)。
- vLLM 一个 `kv_cache_dtype="fp8"` 开关即可让 KV 显存减半、容纳约 2 倍 token。

## 自测验收(完成本模块)
- [ ] 能说清 E4M3/E5M2 的位分配与各自用途。
- [ ] 能解释 FP8 为何比 INT8 更扛 outlier(非均匀量化)。
- [ ] 能说清 Hopper FP8 支持的意义及硬件前提。
- [ ] 能用公式解释 KV Cache 为何随上下文膨胀、为何要量化它。
- [ ] `fp8_vs_int8.py`、`kvcache_calc.py` 跑通,补全 FP8 模拟 TODO。
- [ ](有卡)`vllm_fp8_kv.py` 跑通,观察 FP8 KV 的显存/容量变化。

---

## Module 5 收官

到这里,你已经把量化从"点开关的黑盒"拆成了能讲清原理、能手写、能选型的完整知识体系:

- **L1 数学基础** → 仿射量化、scale/zero-point、粒度、误差、roofline 动机
- **L2 PTQ vs QAT** → 校准、STE、权重易/激活难
- **L3 GPTQ** → 二阶 Hessian 误差补偿(W4A16)
- **L4 AWQ / SmoothQuant** → 缩放保护显著权重 / 难度迁移(W4A16 vs W8A8)
- **L5 FP8 / KV Cache** → 非均匀浮点低精度、长上下文 KV 量化

一条选型主线串起全模块:**看你的负载在 roofline 哪一侧、用什么硬件、上下文多长**——decode/带宽受限走 W4A16(AWQ/GPTQ),prefill/算力受限走 W8A8(SmoothQuant)或 Hopper FP8,长上下文务必量化 KV Cache。

下一模块:**Module 6 — 推理服务化与工程**,我们把量化好的模型真正搬上线,讲服务架构、TensorRT-LLM 部署、多卡张量并行与压测容量规划。
