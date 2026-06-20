# M4 · Lesson 6:部署 vLLM 并做 benchmark——把机制跑成数字

> 本模块收官课。前五课讲的 continuous batching、PagedAttention、投机解码,都是"原理"和"模拟"。这一课把它们在**真实 vLLM** 上跑起来:装好、起一个 OpenAI 兼容 server、用脚本测出**吞吐、TTFT、TPOT、P99**,亲手验证"调大并发,吞吐怎么涨、延迟怎么变"。
> 你做过大数据压测和容量规划,这一课会让你把那套"压测—看指标—定位瓶颈"的肌肉记忆迁移到推理服务上。学完你就能独立评估"一张卡能扛多少 QPS"。
> 本课需要 **N 卡环境**(本地或 AutoDL 租一张 24G+ 显存的卡,如 4090/A10/A100)。没有卡可先读懂指标定义和脚本,租到卡再跑。
> 预计用时:3 小时(含装环境和跑测)。
> 前置:Lesson 1-5;有命令行和基本服务部署经验。

## 学习目标

1. 在 N 卡环境装好 vLLM,起一个 **OpenAI 兼容 server**。
2. 精确说出四个核心指标的定义:**吞吐、TTFT、TPOT、P99**,以及它们分别衡量什么、谁关心。
3. 会用压测脚本测这些指标,并读懂不同**并发 / batch** 下的变化趋势。
4. 把测到的数字和前五课的机制对应起来(吞吐随并发上升 = continuous batching 在起作用)。
5. 建立"吞吐 vs 延迟"的权衡直觉,会回答"这张卡能扛多大负载"。

---

## 1. 环境准备:装 vLLM、起 server

**环境**:Linux + N 卡 + 合适的 CUDA 驱动。AutoDL 选个带 CUDA 12.x 的镜像最省事。建议显卡显存 ≥ 24G(跑 7B 模型),没有就换更小的模型(如 Qwen2.5-1.5B)。

安装(建议用独立虚拟环境):

```bash
# 强烈建议新建环境,vLLM 对 torch/CUDA 版本敏感
conda create -n vllm python=3.10 -y && conda activate vllm
pip install vllm        # 会一并带上匹配的 torch、flash-attn 等
python -c "import vllm; print(vllm.__version__)"   # 验证装好
```

**起 OpenAI 兼容 server**(这是 vLLM 最常用的部署方式,接口和 OpenAI API 一模一样,客户端零改造):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen7b \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --port 8000
# 首次会从 HuggingFace / ModelScope 下载权重,耐心等。国内可设 HF_ENDPOINT 镜像加速。
```

几个关键参数,正是前几课机制的"旋钮":

| 参数 | 含义 | 对应前面哪一课 |
|---|---|---|
| `--gpu-memory-utilization` | 用多少比例显存做 KV Cache(默认 0.9) | Lesson 3:KV Cache 越大,batch 上限越高 |
| `--max-num-seqs` | running batch 最多容纳几个请求 | Lesson 2:continuous batching 的并发上限 |
| `--max-num-batched-tokens` | 一步最多处理多少 token | Lesson 2/4:chunked prefill 的预算 |
| `--enable-prefix-caching` | 开启前缀缓存 | Lesson 3/4:prefix sharing |
| `--max-model-len` | 单请求最大长度 | 影响单请求 KV 上限 |

server 起来后,确认它活着(应返回模型列表):

```bash
curl http://localhost:8000/v1/models
```

发一条流式请求试水(`stream: true` 是后面测 TTFT 的关键):

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen7b",
  "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
  "max_tokens": 64, "stream": true
}'
```

---

## 2. 四个核心指标:定义清楚,别混

压测之前,必须把指标定义钉死。推理服务的性能,从来不是一个数字,而是一组——因为**吞吐和延迟是一对矛盾**(你在大数据里早就体会过)。

一张图先建立时间轴:

```
请求发出
   │
   │◀──── TTFT ────▶│ 第1个token   第2个  第3个 ...  最后一个token
   │                 ●─────────────●──────●──────────●
   │                 │◀── TPOT ──▶│                  │
   │◀──────────────── 端到端延迟 (latency) ──────────▶│
```

逐个讲清楚:

- **TTFT(Time To First Token,首 token 延迟)**:从发出请求到收到**第一个** token 的时间。它主要由 **prefill** 决定(要先把整个 prompt 算完才能吐第一个字)。**谁关心**:用户体验——聊天界面"转圈"多久才开始出字。prompt 越长、排队越久,TTFT 越大。

- **TPOT(Time Per Output Token,每 token 时间)**:进入生成阶段后,平均每个 token 的间隔(也叫 ITL,inter-token latency)。它由 **decode** 决定。**谁关心**:用户感知的"打字速度"流不流畅。`1/TPOT` 就是单请求的 token 生成速率。

- **吞吐(Throughput)**:整个系统每秒生成的 token 总数(所有请求合计,tok/s)。**谁关心**:运维/成本——一张卡能服务多少量、单位 token 成本多少。**这是 continuous batching + PagedAttention 主攻的指标。**

- **P99 延迟**:端到端延迟(从发请求到收完整回复)的 99 分位数。**谁关心**:SLA / 尾延迟——99% 的用户体验不差于这个值。**这正是你在分布式系统里盯的尾延迟(tail latency)**,均值好看没用,长尾才是体验杀手。

> **关键结论:别用单一数字评价推理服务。** 吞吐高但 P99 爆炸,用户照样骂;TTFT 小但吞吐低,机器烧钱。正确的姿势是**在满足延迟 SLA(如 TTFT<500ms、P99<5s)的前提下,把吞吐做到最大**——这就是容量规划的本质,和你做大数据集群容量评估一模一样。

---

## 3. 动手:用脚本压测

`code/bench_vllm.py` 是一个异步压测脚本,用**流式接口**(才能精确测 TTFT),在不同并发档位下闭环压测,汇总四个指标。

依赖(在能访问 server 的机器上,可以就是 server 同一台):

```bash
pip install aiohttp numpy
```

跑一组并发对比(并发 1 → 8 → 32 → 64):

```bash
cd code
python bench_vllm.py --model qwen7b --concurrency 1 8 32 64 --duration 20
```

脚本核心逻辑(看懂它就懂了怎么测 TTFT):

```python
async def one_request(session, url, model, max_tokens):
    start = time.perf_counter()
    first_token_time = None
    async with session.post(url, json=payload) as resp:   # payload 里 stream=True
        async for raw in resp.content:                     # 逐块读 SSE 流
            ...
            if first_token_time is None:
                first_token_time = time.perf_counter()     # ★第一块到达 = TTFT 终点
            metric.output_tokens += 1
    metric.ttft = first_token_time - start                 # 首 token 延迟
    metric.latency = end - start                           # 端到端延迟
    # TPOT = (端到端 - TTFT) / (token 数 - 1)
```

`worker` 在固定时间窗内持续发请求(闭环压测),`concurrency` 个 worker 并发,模拟 N 个用户同时压。

一组**示意性**结果(真实数字随卡、模型、版本变化,重在看**趋势**):

```
并发   请求数   吞吐(tok/s)   TTFT均值(ms)  TTFT_P99(ms)  TPOT均值(ms)  延迟_P99(ms)
1      28       820.0         55.0          70.0          18.0          5200.0
8      190      4600.0        90.0          150.0         22.0          5600.0
32     560      9800.0        210.0         480.0         31.0          6900.0
64     680      11200.0       520.0         1400.0        55.0          9800.0
```

**这张表怎么读(本课精华)**:

- **吞吐随并发上升**(820 → 11200 tok/s):这就是 **continuous batching + PagedAttention 在起作用**——并发越高,batch 越大,decode 阶段权重搬运被摊到越多请求上,吞吐越高。这是 Lesson 1-3 原理的实测兑现。
- **但吞吐会饱和**:从 32→64,吞吐只从 9800 涨到 11200(增幅变小),因为 GPU 算力/显存逼近上限,batch 再大也塞不下或算不动了。**这个拐点就是这张卡的吞吐天花板。**
- **延迟随并发恶化**(TPOT 18→55ms,P99 5.2s→9.8s):并发越高,每个请求在 batch 里被"摊薄"的算力越少,吐字变慢、排队变长,尾延迟显著上升。
- **TTFT 上升尤其快**(55→520ms):高并发下新请求要排队等 prefill 槽位,首 token 延迟被拉长。

> **关键结论:吞吐和延迟是跷跷板。** 调大并发/batch → 吞吐涨、延迟恶化;调小 → 延迟好、吞吐低。**没有最优解,只有在你的延迟 SLA 约束下的最大吞吐点。** 找这个点,就是推理服务容量规划的核心动作。

**动手任务(必做)**:
1. 起 server,跑通压测,得到你自己卡上的那张表。
2. **找拐点**:把并发档位加密(1,2,4,8,16,32,64,128),画出"并发-吞吐"和"并发-P99"两条曲线,找到"吞吐基本饱和、P99 开始陡升"的并发数——那就是你这张卡的**最佳工作点**。
3. **验证 KV Cache 是瓶颈**:把 `--gpu-memory-utilization` 从 0.9 降到 0.5 重启 server,再压测。观察最大可承受并发和吞吐怎么下降——这直接印证 Lesson 3"KV Cache 显存决定 batch 上限"。
4. **prompt 长度的影响**:把脚本里 `PROMPT` 换成一个很长的(几百字),观察 TTFT 怎么变化,理解"TTFT 由 prefill 主导"。

**进阶任务(选做)**:
- 开 `--enable-prefix-caching`,用一批"共享同一长 system prompt"的请求压测,对比 TTFT/吞吐变化(验证 Lesson 3/4 的 prefix caching)。
- 起两个 server:一个开投机解码(`--speculative-config`,具体参数查 vLLM 版本文档),一个不开,在**低并发**下对比 TPOT(验证 Lesson 5"投机解码低并发降延迟")。

---

## 4. 把数字和机制对上号(模块总复盘)

这一课的最大价值,是让你把整个 Module 4 的机制和可测量的指标连起来。一张总表收束本模块:

| 你观察到的现象 | 背后的机制 | 哪一课 |
|---|---|---|
| 并发上升,吞吐显著上升 | continuous batching 把更多请求拼进同一步,摊销权重搬运 | L1、L2 |
| 同样显存能容纳更多并发请求 | PagedAttention 消除 KV Cache 碎片/预留浪费 | L3 |
| 降低 `gpu-memory-utilization` 后并发上限骤降 | KV Cache 显存是 batch 上限的硬约束 | L3 |
| 相同 system prompt 的请求 TTFT 变低 | prefix caching 跳过重复前缀的 prefill | L3、L4 |
| 低并发下开投机解码,TPOT 下降 | 用闲置算力一次验证多 token | L5 |
| 高并发下投机解码反而拖累 | batch 已打满算力,没有闲置给验证 | L5 |
| 高并发下 P99/TTFT 恶化 | 吞吐-延迟权衡:batch 越大单请求被摊越薄 | L1、L6 |

> **整个 Module 4 的主线一句话:推理引擎就是在"显存约束 + 延迟 SLA"下,把 batch 尽可能塞大塞满,以最大化吞吐、压低单位 token 成本。** continuous batching 负责"塞满"(调度),PagedAttention 负责"能塞下"(显存),投机解码负责"低负载时也快"(单请求延迟),benchmark 负责"用数字证明它有效"。

---

## 练习题

1. 用一句话分别说清 TTFT、TPOT、吞吐、P99 各衡量什么、谁(用户/运维)最关心。

2. 为什么测 TTFT 必须用流式(stream)接口?用非流式接口测到的"延迟"是什么?

3. 压测中"并发从 32 加到 64,吞吐只涨一点点但 P99 大涨",说明系统进入了什么状态?这个拐点对容量规划意味着什么?

4. 把 `--gpu-memory-utilization` 调低后,你预期最大并发和吞吐会怎么变?为什么?(用 Lesson 3 的知识答。)

5. 你的 SLA 是"TTFT < 300ms 且 P99 < 4s"。给定第 3 节那张示意表,你会把这张卡的并发设到多少?为什么不设更高?

6.(开放)老板说"我们要同时把吞吐和 P99 都做到最好"。请用本课知识解释为什么这通常做不到,以及实践中该怎么取舍。

<details>
<summary>参考答案(想完再看)</summary>

1. TTFT:发请求到首 token 的时间,衡量"开始响应快不快",用户(交互体验)关心;TPOT:生成阶段每 token 间隔,衡量"吐字流畅度",用户关心;吞吐:每秒总生成 token 数,衡量"机器服务能力/成本",运维关心;P99:端到端延迟 99 分位,衡量"尾延迟/SLA",运维和重体验的用户都关心。

2. 因为 TTFT 是"首 token 到达"的时刻,只有流式接口才会在第一个 token 生成时就推回来、让你测到那个时间点;非流式接口要等**整个回复生成完**才一次性返回,测到的是端到端延迟(latency),拿不到首 token 时刻。

3. 说明系统接近**饱和/拥塞**:GPU 算力或显存逼近上限,再加并发已无法转化成更多吞吐,只会让请求排队、互相摊薄,导致尾延迟陡升。这个拐点就是该卡在当前模型下的**实际容量上限**,容量规划应把工作点设在拐点之前(吞吐接近饱和但延迟还可接受处)。

4. 最大并发和吞吐都会下降。因为 `gpu-memory-utilization` 决定了留给 KV Cache 的显存,KV Cache 越小,能同时容纳的请求(batch)上限越低(Lesson 3:KV Cache 随并发×长度线性增长),batch 做不大,decode 摊销变差,吞吐随之下降。

5. 看表:并发 32 时 TTFT 均值 210ms、P99 480ms(<300ms 的是均值,P99 480 已超 TTFT 但题目 SLA 是端到端 P99<4s——注意区分)。端到端 P99:并发 1=5.2s、8=5.6s 已超 4s。严格按"P99<4s"这张示意表其实都不满足(因为 max_tokens 大导致单请求本就长)。合理答法:应选择**满足两个 SLA 的最大并发**——若都不满足,需降低 `max_tokens`/优化或扩容;在能满足的范围内取并发最大者。不设更高是因为再高会突破延迟 SLA,违背"延迟约束下最大化吞吐"的原则。(本题重在让你学会"以 SLA 为约束反推并发",而非记数字。)

6. 因为吞吐和延迟是结构性矛盾:提吞吐要靠做大 batch(更多请求拼一步),而 batch 越大,单个请求分到的算力越少、排队越久,延迟(尤其 P99)必然恶化;反之亦然。实践中只能取舍:**先定延迟 SLA 作为硬约束,在满足 SLA 的前提下最大化吞吐**;若两者都要更好,只能换更强的硬件、上量化/投机等优化、或加机器扩容,而不是在单卡上奢望同时拉满。

</details>

---

## 小结

- vLLM 最常用部署方式是 **OpenAI 兼容 server**(`vllm serve`),客户端零改造;关键参数(`gpu-memory-utilization`、`max-num-seqs` 等)正是前几课机制的旋钮。
- 四个核心指标:**TTFT**(首 token,prefill 主导)、**TPOT**(每 token,decode 主导)、**吞吐**(每秒总 token,运维关心)、**P99**(尾延迟,SLA)。别用单一数字评价服务。
- 压测要用**流式接口**测 TTFT;闭环并发压测看趋势:**并发↑ → 吞吐↑(会饱和)、延迟↑**。
- 找到"吞吐饱和、P99 陡升"的**拐点 = 这张卡的容量上限**;容量规划 = **延迟 SLA 约束下最大化吞吐**。
- 测到的趋势能逐一对应回 continuous batching、PagedAttention、prefix caching、投机解码——本模块机制的实测兑现。

## 自测验收(完成即通关 Module 4)
- [ ] 能独立装好 vLLM、起 OpenAI 兼容 server 并发出流式请求。
- [ ] 能准确定义 TTFT/TPOT/吞吐/P99,并说清各自由 prefill 还是 decode 主导、谁关心。
- [ ] 跑通压测脚本,得到自己卡上的并发-指标表,并能解读吞吐上升与延迟恶化的趋势。
- [ ] 能找到吞吐拐点,理解它就是容量上限,并能在给定 SLA 下反推合理并发。
- [ ] 能把测到的每个现象对应回本模块某一课的机制。

---

## Module 4 结业语

你从"迷你推理循环"出发,一路打通了现代推理引擎的核心机制:

1. **看清问题**(L1 静态 batching 三宗罪)→
2. **调度解药**(L2 continuous batching)→
3. **显存解药**(L3 PagedAttention)→
4. **真实代码**(L4 nano-vLLM 源码)→
5. **单请求提速**(L5 投机解码)→
6. **跑成数字**(L6 部署与 benchmark)。

现在你不仅会用 vLLM,更**懂它为什么快、瓶颈在哪、怎么调**——这正是"打开黑盒"的推理 Infra 工程师该有的能力。

下一模块:**Module 5 — 量化与模型压缩**。我们去解决另一个核心约束:模型太大、显存太贵。还记得 Lesson 1 埋的伏笔吗——decode 是带宽受限,而量化(减少要读的字节数)能直接加速 decode。我们将从量化的数学基础讲到 GPTQ/AWQ/FP8 的原理与实战。
