# M6 · Lesson 1:推理服务架构——从一次 HTTP 请求到一个 token

> 前五个模块,你已经能让一个模型在单机上跑得又快又省。但「能跑」和「能上线」之间,隔着一整套服务工程。本课不写 kernel,而是把镜头拉远:一个用户的请求,是怎么穿过排队、调度、批处理,最后把 token 一个一个吐回浏览器的。
> 预计用时:2 小时(阅读 + 跑通一个迷你 serving demo)。
> 前置:Module 4(continuous batching、PagedAttention),会用 vLLM 起一个服务;有微服务 / 后端经验最好(本课大量用它类比)。

## 学习目标

学完本课你应该能回答:
1. 一个推理服务从请求进来到响应返回,中间有哪些环节?为什么不能像普通 Web 服务那样「一个请求一个线程跑到底」?
2. 为什么要把 **API server** 和 **inference engine** 拆成两层?它们各自的瓶颈是什么?
3. 什么是流式输出(streaming)?SSE 和普通 HTTP 响应有什么区别?为什么 LLM 服务几乎都要流式?
4. 衡量一个推理服务好坏的指标有哪些?**QPS、并发、TTFT、TPOT、P99** 分别是什么,优化时它们怎么互相打架?

---

## 1. 先对齐认知:LLM 服务和普通微服务有什么不一样

你写过微服务,熟悉这个模型:**请求进来 → 线程池里取一个 worker → 同步处理(查 DB、算逻辑)→ 返回 → 释放 worker**。每个请求几十毫秒,worker 处理完立刻还回池子,靠「线程池大小」控制并发。

LLM 推理把这个模型彻底打破了,原因有三个,你必须先在脑子里建立这三个「反直觉」:

| 维度 | 普通微服务 | LLM 推理服务 | 后果 |
|---|---|---|---|
| **单请求耗时** | 几十 ms | 几百 ms ~ 几十 s(取决于生成长度) | 不能让连接「干等到底」,要流式 |
| **请求是否等长** | 基本均匀 | 生成长度差异极大(10 token vs 2000 token) | 静态批处理会被最长的拖死 |
| **资源瓶颈** | CPU / DB 连接 | **GPU 显存 + 显存带宽** | 并发上限由显存(KV Cache)决定,不是线程数 |

> 一句话:**LLM 服务的并发不是「能开多少线程」,而是「显存里能同时塞下多少条序列的 KV Cache」。** 这是从微服务思维切换到推理服务思维的第一关。

第二个反直觉:**请求不是「各跑各的」,而是被攒成一个 batch 一起在 GPU 上算**。Module 4 讲的 continuous batching 就是干这个——多个用户的请求在同一次 forward 里并肩前进。所以服务层的核心工作,其实是「攒批 + 调度」,这和你熟悉的「一个请求一个 goroutine」是完全不同的范式。

---

## 2. 全景图:一次请求的完整旅程

先上架构图,后面逐段拆解。把它和你画过的微服务时序图对照着看:

```
                      ┌──────────────────────────────────────────────────┐
   用户/客户端         │                  推理服务进程                       │
   (浏览器/SDK)        │                                                    │
      │               │   ┌─────────────┐         ┌──────────────────┐    │
      │  HTTP POST     │   │  API Server  │         │  Inference Engine │    │
      │  /v1/chat ───────▶ │ (FastAPI/    │         │   (vLLM/TRT-LLM)  │    │
      │                │   │  Uvicorn)    │         │                   │    │
      │                │   │  · 鉴权/限流 │  入队    │  ┌─────────────┐  │    │
      │                │   │  · 参数校验  │ ──────▶ │  │ 等待队列     │  │    │
      │                │   │  · 模板拼接  │         │  │ (waiting)   │  │    │
      │                │   │  · 放入队列  │         │  └──────┬──────┘  │    │
      │                │   │             │         │         ▼          │    │
      │                │   │             │         │  ┌─────────────┐  │    │
      │                │   │             │         │  │  调度器       │  │    │
      │                │   │             │         │  │ (scheduler)  │  │    │
      │  SSE 流        │   │             │         │  │ 选 batch     │  │    │
      │  data:{token}  │   │  异步取回   │         │  └──────┬──────┘  │    │
      │  data:{token}  │ ◀──────────────────────── │         ▼          │    │
      │  ...           │   │ (streaming) │  逐 token │  ┌─────────────┐  │    │
      │  data:[DONE]   │   │             │ ◀─────── │  │ GPU forward  │  │    │
      │                │   └─────────────┘         │  │ (batched)    │  │    │
      │                │                           │  └─────────────┘  │    │
      │                │                           │   running batch    │    │
      │                │                           └──────────────────┘    │
      │               │                                                    │
                      └──────────────────────────────────────────────────┘
```

六个环节,逐一拆:

1. **入口(API Server)**:接 HTTP、鉴权、限流、校验参数、套对话模板(chat template),把「人话请求」翻译成 engine 能吃的 `(token_ids, sampling_params)`。
2. **排队(waiting queue)**:请求不会立刻执行,先进等待队列。这一步和你在消息队列(Kafka/RabbitMQ)里做削峰填谷一模一样——把突发流量缓冲住。
3. **调度(scheduler)**:每一步(iteration)从等待队列和正在运行的请求里,挑出一批能塞进显存的序列组成 batch。这是整个系统的大脑,Module 4 的 continuous batching 就发生在这里。
4. **批处理 forward(GPU)**:选出的 batch 在 GPU 上做一次前向,prefill 阶段算 prompt、decode 阶段每次吐 1 个 token。
5. **流式返回(streaming)**:每生成一个 token,就通过 SSE 推回客户端,而不是等整段生成完。
6. **回收(free)**:请求生成完(遇到 EOS 或达到 max_tokens),释放它占的 KV Cache 显存,让等待队列里的新请求能进来。

> 把它对到你的微服务经验:**API Server = 网关/BFF 层(无状态、IO 密集),Inference Engine = 有状态的计算后端(GPU 密集),中间的队列 = 削峰的 MQ。** 整个系统是一个「生产者-消费者 + 批处理」模型。

---

## 3. 为什么必须把 API Server 和 Engine 拆开

这是本课最重要的工程决策,值得单独讲。它们的拆分理由,和你做微服务时「IO 密集服务」与「计算密集服务」要分开部署,是同一个道理。

**两层的负载特征完全相反:**

```
API Server 层                        Engine 层
──────────────                       ──────────
· IO 密集(等网络、等队列)            · 计算密集(GPU 满负荷)
· 高并发连接(成千上万长连接)         · 低「并发实体」(batch 内几十条序列)
· 要 async,不能阻塞                  · 要吃满 GPU,不能空转
· 扩容容易(无状态,加副本)           · 扩容贵(每副本要一张/多张卡)
· CPU 上跑                            · GPU 上跑
```

如果把它们糅在一起会怎样?设想 API server 用同步阻塞模型:一个请求要生成 2000 个 token、耗时 30 秒,这个 worker 线程就被占 30 秒。1000 个并发长连接就要 1000 个线程,CPU 调度直接爆炸——而这 30 秒里 GPU 其实只为这个请求算了极小一部分时间,大量时间它在和别的请求拼 batch。**连接的生命周期和 GPU 的计算节奏根本不在一个时间尺度上**,必须解耦。

解耦后:

- **API Server 用异步(async)**:一个事件循环就能 hold 住上万个长连接,每个连接只是在 `await` 一个队列。这就是你熟悉的 Reactor 模型 / 协程,Python 里是 `asyncio` + Uvicorn。
- **Engine 独立循环**:engine 跑一个永不停歇的 `step()` 循环,每一步选 batch、forward、分发结果,只关心怎么把 GPU 喂饱,不关心 HTTP 协议。

> 生产里这个拆分常常做到**进程级甚至机器级**:API server 是一组无状态副本(可以随便水平扩展、做灰度),engine 是带 GPU 的 worker 池,中间用队列(本地 asyncio.Queue,或跨机的 ZMQ / Redis / gRPC)连接。vLLM 的 `AsyncLLMEngine`、TGI 的 router+shard 架构都是这个形态。

---

## 4. 异步与流式:为什么 LLM 服务离不开它们

### 4.1 异步:别让 worker 干等

普通同步代码 `result = engine.generate(prompt)` 会**阻塞**到整段生成完。在 LLM 场景这是灾难。异步的写法是:把请求丢进队列后立刻 `await`,让出控制权,等 engine 那边有 token 产出了再被唤醒。

伪代码对比(感受范式差异):

```python
# 同步:worker 被占满整个生成周期(30s),并发上不去
def handle(request):
    output = engine.generate(request)   # 阻塞 30 秒
    return output

# 异步:让出控制权,一个事件循环服务上万连接
async def handle(request):
    async for token in engine.generate_stream(request):  # 不阻塞,逐 token 唤醒
        yield token
```

### 4.2 流式输出与 SSE

**为什么要流式?** 一个 2000 token 的回答可能要 20 秒。如果等全部算完再返回,用户盯着空白屏幕 20 秒,体感极差。流式让 token 像打字机一样实时蹦出来,**首字延迟(TTFT)只有几百毫秒**,体验天差地别。这就是为什么 ChatGPT 的字是一个个冒出来的。

**怎么实现?** 主流方案是 **SSE(Server-Sent Events)**,一种基于 HTTP 的单向流式协议。和你熟悉的协议对比:

| 方案 | 方向 | 连接 | 适用 |
|---|---|---|---|
| 普通 HTTP | 一问一答 | 短连接 | 非流式,等全部结果 |
| **SSE** | 服务端 → 客户端(单向) | 一个长连接持续推 | **LLM 流式输出的事实标准** |
| WebSocket | 全双工 | 长连接 | 需要双向实时(语音、协同) |

SSE 的报文长这样,本质是「一个 HTTP 响应,body 不结束,服务端不断往里 `write` 数据块」:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream      ← 关键头:告诉客户端这是事件流

data: {"choices":[{"delta":{"content":"你"}}]}

data: {"choices":[{"delta":{"content":"好"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: [DONE]                          ← OpenAI 风格的结束标记
```

每个事件以 `data: ` 开头、`\n\n` 结尾。客户端收到一块就渲染一块。OpenAI 兼容 API(vLLM、TGI、TRT-LLM 都实现了)就是这个格式,所以前端可以无缝切换后端。

> 工程提醒:流式时要注意**反压(backpressure)**。如果客户端网络慢、消费不过来,token 会在缓冲区堆积。生产中要设超时和队列上限,否则慢客户端会拖垮内存——这和你在流处理(Flink)里处理反压是同一类问题。

---

## 5. 指标体系:你怎么知道服务「好不好」

做大数据和微服务,你早就知道「没有度量就没有优化」。LLM 服务有一套**专属指标**,必须先建立直觉。先分两类:**延迟类**(单个用户体感)和**吞吐类**(系统整体效率),它们天生互相拉扯。

### 5.1 延迟类:用户体感

```
请求到达                                                 生成结束
   │                                                        │
   │◀────── TTFT ──────▶│◀─ TPOT ─▶│◀─ TPOT ─▶│◀─ TPOT ─▶│
   │                    │          │          │           │
   │   排队 + prefill    │  token1  │  token2  │  token3   │
   │                    ▲          ▲          ▲           ▲
   │                  首 token    第2个      第3个        末token
```

- **TTFT(Time To First Token,首 token 延迟)**:从请求发出到收到第一个 token 的时间。它 = 排队等待时间 + prefill 计算时间。**直接决定「人话开始蹦出来有多快」**,聊天场景最关键的体感指标。
- **TPOT(Time Per Output Token,每 token 延迟)**:生成阶段平均每个 token 的间隔。也叫 ITL(Inter-Token Latency)。它决定「打字速度」。TPOT 100ms 就是每秒 10 个字,够快;TPOT 500ms 就卡顿。
- **端到端延迟(E2E latency)** = TTFT + TPOT × 输出 token 数。这是用户拿到完整回答的总时间。

> 记住这个分解:**端到端延迟 = TTFT + (输出长度 - 1) × TPOT**。优化 TTFT 和优化 TPOT 是两件不同的事,因为 prefill 和 decode 是两种不同的计算特征(prefill 算力受限、decode 带宽受限,这是 Module 1/4 的结论)。

### 5.2 吞吐类:系统效率

- **QPS / RPS(每秒请求数)**:经典指标,但在 LLM 里有点失真,因为请求长度差异巨大。
- **Token throughput(每秒生成 token 数,tokens/s)**:**LLM 服务更核心的吞吐指标**,通常分开看 input tokens/s(prefill 吞吐)和 output tokens/s(decode 吞吐)。
- **并发数(concurrency)**:同时在 engine 里被处理的请求数。受 KV Cache 显存上限约束。

### 5.3 统计口径:为什么看 P99 不看平均

这点你在微服务里很熟,LLM 里更要命。延迟分布是**长尾**的:大部分请求很快,少数请求(超长输出、刚好赶上大 batch)很慢。

- **平均值(mean)会骗人**:1 个请求 10 秒 + 99 个请求 0.1 秒,平均 0.2 秒,看着很美,但那个 10 秒的用户已经走了。
- **P99(99 分位)**:把所有请求延迟排序,第 99% 那个值。**意思是「99% 的用户体验不差于这个数」**,这才是 SLA 该承诺的。
- 常用还有 P50(中位数)、P95、P99。**SLA 通常写成「P99 TTFT < 2s」这种形式**。

> 一句话总结指标的核心矛盾:**延迟和吞吐是一对冤家。** 把 batch 攒大 → GPU 利用率高、吞吐大,但每个请求等更久、TTFT/TPOT 变差;把 batch 缩小 → 延迟好,但 GPU 闲、吞吐低。整个容量规划(Lesson 4)就是在这条曲线上找一个满足 SLA 的最优点。

---

## 6. 动手实验:200 行写一个迷你 serving 框架

光看图不够。我们用 FastAPI 写一个**极简但结构完整**的推理服务,把上面所有概念落到代码:API server / engine 分离、异步队列、continuous batching、SSE 流式。为了不依赖 GPU,engine 用一个「假模型」(每步给每条序列吐一个字符)替代真实 forward——**架构是真的,只有算 token 那一步是假的**。你以后把那一步换成 vLLM 的 `step()` 就是生产骨架。

代码在 `code/mini_serving.py`,核心结构:

```python
# 见 code/mini_serving.py,这里只摘最能说明架构的三段

# ① engine 的核心循环:永不停歇地选 batch、forward、分发
async def engine_loop(self) -> None:
    while True:
        self._admit_new_requests()          # 从 waiting 队列招新请求进 running
        if not self.running:
            await asyncio.sleep(0.005)       # 没活干,让出 CPU
            continue
        finished = self._step()              # 对整个 running batch 走一步(假 forward)
        for req in finished:
            req.queue.put_nowait(None)       # 用 None 作为「生成结束」哨兵

# ② API 层:把请求丢进队列后立刻 await,绝不阻塞事件循环
async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
    req = Request(prompt=prompt, queue=asyncio.Queue())
    self.waiting.append(req)                 # 入队,等 engine 调度
    while True:
        token = await req.queue.get()        # 不阻塞:有 token 才被唤醒
        if token is None:
            break
        yield token

# ③ SSE 端点:FastAPI 用 StreamingResponse 把 token 逐个推出去
@app.post("/generate")
async def generate(body: dict):
    async def event_source():
        async for token in engine.generate_stream(body["prompt"]):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_source(), media_type="text/event-stream")
```

完整文件里还实现了:`max_batch_size` 限制(模拟显存上限对并发的约束)、每条请求独立的 `max_tokens`(模拟不等长生成)、简单的指标统计(TTFT、吞吐)。

### 实验 A:跑通并观察流式(必做)

1. 装依赖、起服务:

```bash
cd code
pip install fastapi uvicorn httpx
python mini_serving.py
```

2. 另开一个终端,用 curl 看 SSE 流式效果(注意 `-N` 关闭缓冲,才能看到逐字蹦出):

```bash
curl -N -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "hello", "max_tokens": 20}'
```

你会看到 `data: {...}` 一行行地、有节奏地冒出来,最后是 `data: [DONE]`。**这就是 SSE 流式,和 ChatGPT 打字机效果同源。**

### 实验 B:观察 continuous batching 与并发(必做)

运行压测脚本 `code/load_client.py`,它会**同时发起 N 个请求**,然后打印每个请求的 TTFT 和整体吞吐:

```bash
python load_client.py --concurrency 1
python load_client.py --concurrency 8
python load_client.py --concurrency 32
```

观察并记录:并发从 1 → 8 → 32 时,**总吞吐(tokens/s)怎么变,TTFT 怎么变**。你会看到吞吐先涨后平、TTFT 持续变差——这就是 §5.3 说的「延迟 vs 吞吐」的拉扯,在你自己写的迷你系统上复现了。

### 实验 C:改造为带「显存上限」的拒绝策略(选做,留 TODO)

打开 `mini_serving.py`,找到 `_admit_new_requests` 里的 TODO:当 `running` 已满且 `waiting` 持续堆积超过阈值时,实现一个**限流 / 快速失败**(返回 429),而不是让队列无限堆积。想想这和你在微服务里做的「过载保护」是不是一回事。

---

## 练习题

1. 为什么 LLM 服务的「最大并发数」不是由 CPU 线程数决定,而是由显存决定?具体是显存里的什么东西在限制它?
2. 一个请求:排队 0.3s,prefill 0.2s,然后生成 100 个 token、每个间隔 50ms。它的 TTFT 和端到端延迟分别是多少?
3. 把 batch 攒得更大,对 QPS / token 吞吐 / TTFT / TPOT 分别是变好还是变坏?为什么?
4. SSE 和 WebSocket 都能做流式,为什么 LLM 推理普遍选 SSE 而不是 WebSocket?
5. (开放)你们公司微服务的「网关 + 后端服务 + MQ」架构,和本课的「API server + engine + 队列」如何一一对应?有哪些经验可以直接迁移,哪些不能?

<details>
<summary>参考答案</summary>

1. 因为每条正在生成的序列都要在显存里维护它的 **KV Cache**,长度随生成增长。显存装得下多少条序列的 KV Cache,就是并发上限。CPU 线程在异步模型下几乎不是瓶颈(一个事件循环 hold 上万连接),真正的硬约束是 **GPU 显存(主要被 KV Cache 吃掉)**。
2. TTFT = 排队 0.3s + prefill 0.2s = **0.5s**。端到端 = TTFT + (100-1)×0.05s = 0.5 + 4.95 = **5.45s**。(第一个 token 在 prefill 后产出,之后 99 个 token 每个隔 50ms。)
3. QPS / token 吞吐:**变好**(GPU 一次算更多,利用率高)。TTFT:**变差**(请求要等攒够一批、排队更久,且大 batch 的 prefill 更慢)。TPOT:**通常变差**(batch 越大,每步 decode 算得越久),但在带宽受限区,增大 batch 摊薄了权重读取成本,单位 token 效率反而更高——所以适度增大 batch 对 TPOT 影响有限,这正是 continuous batching 划算的原因。
4. ① LLM 输出是**单向**的(服务端推、客户端只读),SSE 单向正好够用,WebSocket 的全双工是浪费;② SSE 就是普通 HTTP,天然兼容现有网关、负载均衡、鉴权、CDN,WebSocket 要额外支持;③ SSE 自带断线重连语义,实现简单。所以 OpenAI 及所有兼容实现都用 SSE。
5. 对应关系:网关 ≈ API server(鉴权/限流/路由),后端服务 ≈ engine,MQ ≈ 请求队列。可迁移:限流、熔断、超时、P99 监控、灰度发布、无状态水平扩展(针对 API server)。不能直接迁移:engine 是**有状态且 GPU 强约束**的,不能像无状态服务那样随意加副本(每副本要卡、要加载几十 GB 权重、扩容以分钟计),并发模型也从「一请求一线程」变成「攒批调度」。

</details>

---

## 小结

- LLM 服务和普通微服务有三个根本不同:**单请求长、长度不等、瓶颈在显存**,所以并发上限由 **KV Cache 显存**决定,不是线程数。
- 一次请求的旅程:**入口 → 排队 → 调度选 batch → GPU 批处理 forward → 流式返回 → 回收显存**。
- **API server(IO 密集、异步、易扩展)和 engine(GPU 密集、有状态、扩容贵)必须分层解耦**,中间用队列连接——本质是生产者-消费者 + 批处理模型。
- 流式输出靠 **SSE**(`text/event-stream`,`data: ...\n\n`,以 `[DONE]` 收尾),把 TTFT 降到几百毫秒,是 LLM 服务的事实标准。
- 指标分两类:延迟类(**TTFT、TPOT**、E2E)和吞吐类(**token throughput**、并发数),用 **P99** 而非平均值写 SLA。**延迟和吞吐是一对冤家,容量规划就是在这条曲线上找最优点。**

## 自测验收(过了再进 Lesson 2)
- [ ] 能画出一次请求的完整架构图,并说清每个环节在干什么。
- [ ] 能解释为什么 API server 和 engine 要分层,各自的负载特征是什么。
- [ ] 能手写一段 SSE 报文,说清它和普通 HTTP、WebSocket 的区别。
- [ ] 能准确定义 TTFT、TPOT、P99,并解释「延迟 vs 吞吐」的矛盾。
- [ ] `mini_serving.py` 跑通,实验 B 里观察到了并发增大时吞吐与 TTFT 的反向变化。

下一课:**Lesson 2 — TensorRT-LLM 部署**,我们从「自己搭框架」转向「用工业级引擎榨干 GPU 峰值性能」,并搞清它和 vLLM 该怎么取舍。
