# M4 · Lesson 4:精读 nano-vLLM 源码——看机制如何变成代码

> 前三课我们讲透了 continuous batching 和 PagedAttention 的**原理**。但原理和工程之间永远隔着一条河。这一课带你过河:打开一个**约 1200 行**的真实极简推理引擎 **nano-vLLM**,看 waiting/running 队列、block table、引用计数、prefix caching、采样这些概念在代码里**长什么样、怎么串起来**。
> nano-vLLM 是 vLLM 核心贡献者写的"教学版 vLLM",麻雀虽小五脏俱全——它保留了 vLLM 的核心架构,却砍掉了让你读不下去的工程枝节。读懂它,你再去读 vLLM 主仓(几十万行)就有了地图。
> 本课是**源码阅读课**:动手任务是 clone、读、trace,不是写新代码。
> 预计用时:4 小时(读代码是慢功夫,别赶)。
> 前置:Lesson 1-3 全部;PyTorch 基础;读过 Module 3 的迷你推理循环。

## 学习目标

1. 建立 nano-vLLM 的**全局地图**:有哪些模块、各管什么、调用关系如何。
2. 在真实代码里认出前三课的概念:`Scheduler` 的 waiting/running、`BlockManager` 的 block table 与 ref_count、prefix caching。
3. 读懂引擎主循环 `step()`:schedule → run → postprocess 三段式。
4. 理解 nano-vLLM 与教科书描述的差异(block_size=256、靠 flash-attn 实现 paged kernel、preemption 抢占)。
5. 完成核心任务:**trace 一个请求从 `add_request` 到吐出 EOS 的完整生命周期**。

---

## 1. 先把仓库拉下来,跑个 example

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
# 看一眼整体规模(约 1200 行核心代码)
find nanovllm -name "*.py" | xargs wc -l | sort -n | tail -1
```

它依赖 `torch`、`transformers`、`flash-attn`、`triton`、`xxhash`。真正跑起来需要 N 卡(flash-attn 要 GPU),没有卡就**只读代码 + 看本课讲解**,一样能学到精髓。有卡的话(AutoDL 租一张),按 README 下个 Qwen3-0.6B 小模型,跑 `example.py` 感受一下。

> 学习策略:**先读代码、建立心智模型,再上机验证。** 读代码这一步即使没有 GPU 也能完整做完,这正是本课的重点。

---

## 2. 全局地图:7 个核心文件,各司其职

nano-vLLM 的目录结构(只列核心):

```
nanovllm/
├── llm.py                      # 对外入口:class LLM(LLMEngine),就一行 pass
├── config.py                   # 全局配置(max_num_seqs、block_size、显存占比...)
├── sampling_params.py          # 采样参数(temperature、max_tokens...)
├── engine/                     # ★引擎核心,本课主战场
│   ├── llm_engine.py           #   总指挥:add_request / step / generate 主循环
│   ├── scheduler.py            #   调度器:continuous batching 的大脑(waiting/running)
│   ├── block_manager.py        #   显存块管理:PagedAttention 的 block table + ref_count
│   ├── sequence.py             #   一个请求的状态机(token_ids、block_table、status)
│   └── model_runner.py         #   模型执行器:准备张量、跑前向、CUDA graph、张量并行
├── layers/                     # 模型的各层实现
│   ├── attention.py            #   ★PagedAttention 落地:store_kvcache + flash-attn paged
│   ├── sampler.py              #   采样(gumbel-max 技巧)
│   ├── linear.py / ...         #   线性层、norm、rotary、激活(支持张量并行)
└── models/qwen3.py             # 具体模型结构(Qwen3),拼装上面的 layers
```

调用关系一张图理清(从你调用 `LLM.generate` 开始,自上而下):

```
你的代码
  │ llm.generate(prompts, sampling_params)
  ▼
LLMEngine.generate ──循环调用──▶ LLMEngine.step()   ← 引擎心跳,一次一个 batch
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  Scheduler.schedule()       ModelRunner.run()         Scheduler.postprocess()
  (选出这步跑哪些 seq)        (真正上 GPU 算前向+采样)    (写回 token、判 EOS、回收块)
        │                           │
        ▼                           ▼
  BlockManager               layers/attention.py
  (分配/释放 KV 块)           (store_kvcache + flash-attn paged attention)
```

> **关键认知:把 Lesson 2 的"调度"和 Lesson 3 的"显存管理"对号入座——`Scheduler` = continuous batching 的大脑,`BlockManager` = PagedAttention 的页表。** 整个引擎就是这两者 + 模型前向的协奏。

**建议的阅读顺序**(由数据结构到流程,避免一上来陷进张量细节):
`sampling_params.py` → `sequence.py` → `block_manager.py` → `scheduler.py` → `llm_engine.py` → `layers/attention.py` → `model_runner.py` → `models/qwen3.py`。

---

## 3. `Sequence`:一个请求的状态机

`engine/sequence.py`。先认识"被调度的最小单位"。一个请求就是一个 `Sequence` 对象,它是个状态机:

```
WAITING ──被调度─▶ RUNNING ──吐EOS/到max─▶ FINISHED
   ▲                  │
   └──── 被抢占 preempt ┘   (显存不够时退回等待,Lesson 3 的换出)
```

关键字段(对照前三课):

```python
class Sequence:
    block_size = 256            # ← 注意!nano-vLLM 默认 256,不是 vLLM 的 16
    def __init__(self, token_ids, sampling_params):
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)   # prompt + 已生成的 token,会不断 append
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0          # 已经算好 KV 并缓存的 token 数(prefix cache 用)
        self.is_prefill = True              # 还在 prefill 阶段?
        self.block_table = []              # ★逻辑块号 → 物理块号 的映射(就是 Lesson 3 那张表)
```

几个要读懂的属性:

- `num_blocks`:当前需要几个块 = `⌈num_tokens / block_size⌉`。
- `last_block_num_tokens`:最后一块装了几个 token(算 slot 位置用)。
- `block(i)`:取第 i 个逻辑块对应的 token_ids(算 hash 做 prefix cache 用)。
- `append_token(id)`:生成一个新 token 就追加进 `token_ids`,`num_tokens += 1`。

> 留意 `block_size = 256` 这个差异:Lesson 3 为了画图用了 16,vLLM 默认也是 16,但 nano-vLLM 用 256。block 越大,块表越短、管理越简单,但内部碎片(最后一块的浪费)越大——这正是 Lesson 3 练习题 2 让你体会的权衡,nano-vLLM 选了"简单优先"。

---

## 4. `BlockManager`:PagedAttention 的页表落地

`engine/block_manager.py`,**本课最该精读的文件之一**。它就是 Lesson 3 那个 `BlockAllocator` 的工业版,还多了 prefix caching。

数据结构:

```python
class Block:
    def __init__(self, block_id):
        self.ref_count = 0       # ★引用计数(Lesson 3 的共享/CoW 基础)
        self.hash = -1           # 块内容的哈希(prefix cache 用,只对"满块"算)
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.blocks = [Block(i) for i in range(num_blocks)]   # 物理块池
        self.hash_to_block_id = dict()    # 内容哈希 → 物理块号(命中即可复用 = prefix cache)
        self.free_block_ids = deque(...)  # 空闲块队列
        self.used_block_ids = set()
```

把它的方法和 Lesson 3 概念一一对应:

| 方法 | 干什么 | 对应 Lesson 3 |
|---|---|---|
| `can_allocate(seq)` | 检查显存够不够,**顺便查 prefix cache 命中几块** | 按需分配前的检查 |
| `allocate(seq, n)` | 给 seq 分配块,前 n 块复用缓存(ref_count+1),其余新分配 | 块表填充 + 前缀共享 |
| `deallocate(seq)` | 释放 seq 的所有块,**ref_count 减到 0 才真回收** | 引用计数安全释放 |
| `can_append(seq)` | decode 时要不要新块?(当前块满了才要) | 按需分配(demand paging) |
| `may_append(seq)` | 真要新块时分配一块 | 生成到边界追加新块 |
| `hash_blocks(seq)` | 给写满的块算哈希、登记到 `hash_to_block_id` | 让后来者能命中 prefix cache |

**重点看 prefix caching 怎么实现的**(这是 nano-vLLM 比 Lesson 3 多出来的精华):

```python
def can_allocate(self, seq):
    h = -1
    num_cached_blocks = 0
    for i in range(seq.num_blocks - 1):     # 遍历每个"满块"
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) # 链式哈希:把前缀也混进去,保证"同前缀同内容"才命中
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                            # 没命中或哈希碰撞,停止复用
        num_cached_blocks += 1               # 命中一块,这块的 KV 可以直接复用,不用重算!
    ...
```

> **关键认知:prefix caching = 用"块内容的哈希"做去重。** 两个请求只要前缀 token 完全相同,它们对应的满块哈希就相同,第二个请求直接复用第一个的物理块(ref_count+1),**这段前缀的 prefill 完全跳过**。这就是 Lesson 3 "前缀共享"红利在真实代码里的样子——靠 `compute_hash` + `hash_to_block_id` 字典实现。链式哈希(把前缀 hash 作为 `prefix` 传入)保证了"前缀不同则哈希不同",避免误命中。

---

## 5. `Scheduler`:continuous batching 的大脑

`engine/scheduler.py`,**另一个必须精读的文件**。Lesson 2 的 waiting/running 双队列,这里是工业版,还实现了 **chunked prefill** 和 **preemption(抢占)**。

```python
class Scheduler:
    def __init__(self, config):
        self.block_manager = BlockManager(...)
        self.waiting = deque()    # ← Lesson 2 的等待队列
        self.running = deque()    # ← Lesson 2 的运行批次
```

核心是 `schedule()`,它返回"这一步要跑哪些 seq、是 prefill 还是 decode"。注意它的**两段式策略:prefill 优先,没有 prefill 才做 decode**:

```python
def schedule(self):
    scheduled_seqs = []
    num_batched_tokens = 0
    # === 第一段:尽量调度 waiting 里的请求做 prefill ===
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]
        remaining = self.max_num_batched_tokens - num_batched_tokens
        ...
        num_cached_blocks = self.block_manager.can_allocate(seq)  # 查显存+prefix cache
        if num_cached_blocks == -1:        # 显存不够,这一步先不收它
            break
        # 关键:chunked prefill —— 只有 batch 里"第一个"请求允许被切块,
        # 避免一个超长 prompt 把整步撑爆(对应 Lesson 2 的 chunked prefill)
        if remaining < num_tokens and scheduled_seqs:
            break
        self.block_manager.allocate(seq, num_cached_blocks)
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        ...
    if scheduled_seqs:
        return scheduled_seqs, True        # 这一步是 prefill batch

    # === 第二段:没有 prefill 可做,就推进 running 里的 decode ===
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):   # decode 要新块但显存不够
            if self.running:
                self.preempt(self.running.pop())  # ★抢占:把别的请求换出去腾显存
            else:
                self.preempt(seq)                 # 实在不行把自己换出
                break
        else:
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)   # 按需追加新块
            scheduled_seqs.append(seq)
    return scheduled_seqs, False                 # 这一步是 decode batch
```

两个工程细节,面试爱问:

- **prefill 优先**:nano-vLLM 一步要么全做 prefill、要么全做 decode(不混)。这比 vLLM 的"prefill/decode 混批"简单。chunked prefill 在这里体现为"只允许批里第一个长请求被切块"。
- **preemption(抢占)**:这是 Lesson 3 埋的伏笔——**显存真的不够时怎么办?** 答案:把某个 running 请求**换出**(`preempt`),释放它的 KV 块给更紧急的请求,被换出的请求退回 waiting 队首、状态置回 prefill,**以后重新算**(recompute)。这就是 OS 里"内存不足换页"在推理引擎里的对应物。

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True               # 退回去重新当 prefill 处理
    self.block_manager.deallocate(seq)  # 释放它占的 KV 块
    self.waiting.appendleft(seq)        # 插回等待队首,优先恢复
```

> **关键认知:scheduler 是"显存预算下的连续批处理调度器"。** 它每一步都在回答:在不超显存(`can_allocate`/`can_append`)、不超 token 预算(`max_num_batched_tokens`)、不超并发数(`max_num_seqs`)的约束下,这步该跑谁。这就是 Lesson 2 那场"把 batch 塞满"的战争的真实代码。

---

## 6. `LLMEngine.step()`:把一切串起来的心跳

`engine/llm_engine.py`。引擎的主循环极其简洁,把前面所有模块串成三段式:

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()          # ① 调度:这步跑谁
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 执行:上 GPU 算 + 采样
    self.scheduler.postprocess(seqs, token_ids, is_prefill)      # ③ 善后:写回/判结束/回收
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

`postprocess` 里能看到 Lesson 2 的"退出"和 Lesson 3 的"回收"合体:

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)     # 给新写满的块登记哈希(供后来者命中 prefix cache)
        ...
        seq.append_token(token_id)              # 把新生成的 token 追加进序列
        if (token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # ★吐EOS,立刻释放 KV 块(Lesson 2 的"退出")
            self.running.remove(seq)            #    从 running 移除,腾出名额
```

而 `generate()` 就是反复敲心跳直到全部结束,顺便统计 prefill/decode 吞吐:

```python
while not self.is_finished():
    output, num_tokens = self.step()
    # num_tokens > 0 是 prefill 步,< 0 是 decode 步,用来分别算两种吞吐
    ...
```

> 注意 `model_runner.call("run", ...)`:nano-vLLM 用多进程做张量并行(`tensor_parallel_size > 1` 时 spawn 多个 `ModelRunner`),rank 0 主进程调度、其余进程跟着算同一批。单卡时就一个进程。这块细节可略读,知道它是 TP(张量并行)入口即可,Module 6 会专门讲 TP。

---

## 7. `attention.py`:PagedAttention 在 GPU 上怎么落地

`layers/attention.py`。这里揭晓 Lesson 3 末尾那个问题——**paged attention kernel 到底怎么实现的?** nano-vLLM 的答案很务实:**不自己手写,借 flash-attn 的 paged 支持**。

```python
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():
        # ① 先把这步新算出来的 k,v 写进 KV Cache 的对应 slot(slot_mapping 由 model_runner 算好)
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        # ② prefill:变长 flash-attn,block_table 传进去 → 它按块表去分散的物理块读 KV
        o = flash_attn_varlen_func(q, k, v, ..., block_table=context.block_tables)
    else:
        # ③ decode:flash-attn 的 with_kvcache,同样靠 block_table 间接寻址读分散的 KV
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    block_table=context.block_tables, ...)
    return o
```

两个关键点:

- **`store_kvcache`** 是一个手写的 **Triton kernel**(回忆 Module 2 的 Triton),作用是把这步新算的 K、V 按 `slot_mapping` 散写进 KV Cache 的物理槽位。`slot_mapping` 把"逻辑位置"翻译成"物理块内的绝对槽号",正是 block table 的另一种表达。
- **`block_table=...`** 传给 flash-attn:这就是 Lesson 3 说的"kernel 按块表间接寻址读分散的 KV"。nano-vLLM 没重造轮子,而是利用了 flash-attn 已经内置的 paged KV cache 支持。**这告诉你一个工程真相:真实引擎大量站在 flash-attn / FlashInfer 这些高性能算子库的肩膀上。**

`slot_mapping`、`block_tables` 这些张量是在 `model_runner.py` 的 `prepare_prefill` / `prepare_decode` 里,根据每个 seq 的 `block_table` 拼出来的——想深入可以读这两个函数,它们是"Python 端的块表"翻译成"GPU kernel 能用的扁平张量"的地方。

---

## 8. 动手任务:trace 一个请求的完整生命周期(核心)

这是本课的**核心作业**,不写代码,而是**当人肉 debugger**。拿 `example.py` 里那个 prompt `"introduce yourself"`,在源码里追踪它从进引擎到吐完 EOS 的全过程,写一份 trace 笔记。

按这个清单逐步追(在代码里找到对应行,记下函数名和它对这个请求做了什么):

1. **诞生**:`LLM.generate` → `add_request`。prompt 字符串怎么变成 token_ids?`Sequence` 对象初始状态是什么(status、block_table、is_prefill)?它进了哪个队列?

2. **第一次被调度(prefill)**:`step` → `scheduler.schedule` 第一段。
   - `can_allocate` 检查了什么?它给这个全新请求返回的 `num_cached_blocks` 大概是几(提示:第一个请求没有任何缓存可命中)?
   - `block_manager.allocate` 给它分了几个块(用 prompt 长度和 block_size=256 算)?`block_table` 现在长什么样?
   - 它的 status 变成了什么?进了哪个队列?

3. **执行 prefill**:`model_runner.run` → `prepare_prefill` → 模型前向 → `attention.forward` 走哪个分支?→ `sampler` 采出第一个生成 token。

4. **善后**:`postprocess`。新 token 怎么 append?`hash_blocks` 给它的满块登记了哈希(这一步让**下一个相同前缀的请求**能命中 prefix cache)。它结束了吗(EOS?)?

5. **进入 decode 循环**:接下来若干个 `step` 走 `schedule` 第二段。
   - 每步 `can_append` 在判断什么?什么时候 `may_append` 会真的分一个新块(提示:`num_tokens % 256 == 1` 时,即刚跨过块边界)?
   - decode 的 `attention.forward` 走哪个分支?

6. **死亡**:某一步采到 EOS(或到 `max_tokens=256`)。`postprocess` 里:status → FINISHED,`deallocate` 释放了它的几个块(这些块回到 `free_block_ids`,立刻能给别人用),从 running 移除。`generate` 把它的 token decode 回字符串返回。

**交付物**:一份按上面 6 步组织的笔记,每步写清"调用了哪个函数 + 对这个请求的状态(status / block_table / num_tokens)做了什么改变"。能讲清这一条生命线,你就真正打通了前四课。

**进阶任务(选做)**:
- 追踪 example.py 里**第二个** prompt,如果它和第一个有相同前缀(比如都用了同一段 chat template 前缀),观察它在 `can_allocate` 时 `num_cached_blocks` 是否 > 0(prefix cache 命中)。
- 构造一个会触发 **preemption** 的场景:把 `num_kvcache_blocks` 调到很小、塞很多长请求,在 `preempt` 处打断点(或加日志),观察请求被换出又换回的过程。

---

## 练习题

1. nano-vLLM 的 `Scheduler` 用哪两个队列实现 continuous batching?`schedule()` 为什么是"prefill 优先"?这和 vLLM 的"prefill/decode 混批"相比简化在哪?

2. `Block.ref_count` 是干嘛的?在什么操作时 +1、什么时候 -1、什么时候块才真正被回收?这对应 Lesson 3 的什么概念?

3. prefix caching 在 `BlockManager` 里靠哪个字典实现?两个请求要满足什么条件才能复用同一个物理块?`compute_hash` 为什么要把前缀 hash 也混进去?

4. `can_append` 里 `len(seq) % self.block_size == 1` 这个判断是什么意思?为什么"模等于 1"就代表需要新块?

5. preemption(抢占)发生在什么情况下?被抢占的请求会怎样?为什么把它 `is_prefill = True` 退回去?这对应 OS 的什么机制?

6. nano-vLLM 的 paged attention 是自己写 CUDA kernel 实现的吗?它实际依赖了什么?`store_kvcache` 这个 Triton kernel 的职责是什么?

<details>
<summary>参考答案(想完再看)</summary>

1. `waiting` 和 `running` 两个 deque。prefill 优先是因为:只要有等待中的新请求能塞下,就先把它们 prefill 进来填满 batch(提升并发),没有可 prefill 的才推进 decode。vLLM 会把 prefill 和 decode 请求**混在同一个 batch**(配合 chunked prefill 拉平单步耗时),nano-vLLM 简化成"一步要么全 prefill 要么全 decode",只对批里第一个长请求允许 chunk。

2. `ref_count` 是物理块的引用计数,记录有多少个请求/逻辑块在用它。`allocate` 复用缓存块时 +1、新分配置 1;`deallocate` 时 -1;**减到 0 才真正回收**(放回 `free_block_ids`)。对应 Lesson 3 的引用计数 / 共享块安全释放 / copy-on-write 基础。

3. 靠 `hash_to_block_id` 字典(块内容哈希 → 物理块号)。两个请求要复用同一物理块,需要这一块的 **token_ids 完全相同**(且哈希命中、`token_ids` 比对一致防碰撞)。`compute_hash` 把前缀 hash 混进去做**链式哈希**,保证"只有从头到这一块的整段前缀都相同"才命中,避免"中间某块内容偶然相同"导致误复用。

4. `len(seq)` 是当前总 token 数。decode 每步只 +1 个 token,当 `总数 % 256 == 1` 时,说明刚刚新增的这个 token 是**某个新块的第一个 token**(前面的块刚好填满),所以需要追加一个新块。`may_append` 就在这个时刻分配新块。

5. 当 decode 时某请求要新块(`can_append` 为假)但显存没有空闲块了,就触发抢占:挑一个 running 请求 `preempt` —— 释放它全部 KV 块、状态退回 WAITING、`is_prefill=True`、插回 waiting 队首。退回当 prefill 是因为它的 KV 已被丢弃,恢复时要**重新计算(recompute)**整段 KV。对应 OS 的"内存不足时换页 / 牺牲页"。

6. 不是自己手写完整 paged attention kernel。它依赖 **flash-attn**(`flash_attn_varlen_func` 做 prefill、`flash_attn_with_kvcache` 做 decode),把 `block_table` 传进去由 flash-attn 完成按块表间接寻址。`store_kvcache` 是个手写 **Triton kernel**,职责是把这步新算出的 K、V 按 `slot_mapping` 散写进 KV Cache 的物理槽位。

</details>

---

## 小结

- nano-vLLM ≈ **教学版 vLLM**,约 1200 行,核心是 `engine/` 下的 5 个文件 + `layers/attention.py`。
- **`Scheduler` = continuous batching 的大脑**(waiting/running 双队列、prefill 优先、chunked prefill、preemption)。
- **`BlockManager` = PagedAttention 的页表**(block 池、block_table、ref_count、prefix caching 靠内容哈希字典)。
- **`LLMEngine.step()` = 三段式心跳**:schedule → run → postprocess,把调度、执行、善后串起来。
- **`attention.py`** 揭示工程真相:paged attention 靠 **flash-attn + Triton kernel(store_kvcache)** 落地,不重造轮子。
- 与教科书的差异:block_size=256、prefill/decode 不混批、显存不足靠 **preemption** 换出重算。
- 核心收获来自**亲手 trace 一个请求的生命周期**,把前三课的原理钉进真实代码。

## 自测验收(过了再进 Lesson 5)
- [ ] 能画出 nano-vLLM 的模块调用图,说清 Scheduler / BlockManager / ModelRunner 各管什么。
- [ ] 能在代码里指出 Lesson 2 的 waiting/running 和 Lesson 3 的 block_table/ref_count。
- [ ] 能复述 `step()` 的三段式和 `postprocess` 里"吐 EOS 即回收块"的逻辑。
- [ ] 能解释 prefix caching 的哈希字典实现,以及 preemption 何时触发、被抢占请求的命运。
- [ ] **完成 trace 一个请求完整生命周期的笔记(6 步)。**
- [ ] 知道 paged attention 实际靠 flash-attn + Triton 落地。

下一课:**Lesson 5 — 投机解码(Speculative Decoding)**。前面我们用调度和显存把 batch 做大、把吞吐做高;下一课换个维度,从"减少大模型的前向次数"入手给**单个请求**的 decode 提速,看小模型当"草稿员"、大模型当"审稿员"的精妙配合。
