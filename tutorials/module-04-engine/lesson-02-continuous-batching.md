# M4 · Lesson 2:Continuous Batching——迭代级调度

> 上一课我们把静态 batching 的"三宗罪"钉在了墙上。这一课治前两宗:**padding 浪费** 和 **同生共死 / HOL 阻塞**。武器叫 **Continuous Batching(连续批处理)**,也叫 **iteration-level scheduling(迭代级调度)**。
> 这是现代推理引擎(vLLM、TGI、TensorRT-LLM)吞吐的命根子。理解它,你就理解了"为什么 vLLM 比朴素 HuggingFace `generate` 快一个数量级"的一半原因(另一半是下一课的 PagedAttention)。
> 本课理论 + 一段**可运行的纯 Python 调度模拟**(不需要 GPU),你会亲手跑出吞吐和利用率的对比数字。
> 预计用时:2 小时。
> 前置:Lesson 1(静态 batching 三宗罪)。

## 学习目标

1. 说清 continuous batching 的核心思想:把调度粒度从"请求级"降到"迭代级(step 级)"。
2. 理解请求如何"随时加入、随时退出",以及这如何同时干掉 padding 和 HOL 阻塞。
3. 看懂并能复述简化版调度循环(waiting / running 两个队列怎么流转)。
4. 理解 prefill 和 decode 混在一起调度的难点(chunked prefill 的由来)。
5. 知道业界 "23× throughput" 这个数字的出处和适用条件,不把它当万能咒语。

---

## 1. 一句话抓住核心:把调度粒度从"请求"降到"step"

静态 batching 的调度单位是**整个请求的一生**:一进 batch,就锁定到死,中途不能加人、不能踢人。

continuous batching 的洞察极其简单,却威力巨大:

> **自回归生成本来就是一步一步(step by step)走的。那为什么不在每一步结束后,都重新决定一次"下一步这个 batch 里该有谁"?**

这就是 **iteration-level scheduling(迭代级调度)**:调度发生在**每个 decode step 的边界**,而不是请求的边界。

类比你最熟的领域:

- 静态 batching ≈ **批处理作业(batch job)**:一个 job 提交后整块运行到结束,中途不可变。
- continuous batching ≈ **流处理 / 抢占式分时调度**:以极细的时间片(一个 step)为单位,持续地把就绪任务塞进资源、把完成的任务回收。

你在 OS 课学的"时间片轮转"、在 Flink 里见的"连续算子",思想是一脉相承的:**用细粒度调度换高利用率**。

---

## 2. 请求随时进出:两个病根一起治

continuous batching 维护两个队列(后面 nano-vLLM、vLLM 的 scheduler 都是这个骨架):

```
        ┌──────────────┐  调度(有空槽就拉人)   ┌──────────────┐
新请求 → │ waiting 队列  │ ───────────────────→ │ running 批次  │ → 每 step 各生成 1 token
        │ (排队等资源) │                       │ (正在 GPU 上)│
        └──────────────┘ ←─────────────────── └──────────────┘
                          被抢占/换出(显存不够时,Lesson 3 讲)
                                                     │ 某请求吐 EOS
                                                     ▼
                                                  立刻退出、腾槽位
```

每个 step 干三件事:

1. **退出**:上一步吐了 EOS 的请求,立刻离开 running,**马上释放它的槽位和 KV Cache**。
2. **加入**:只要 running 还有空槽,就从 waiting 队首拉新请求进来(它先做 prefill,再汇入 decode)。
3. **推进**:对 running 里的每个请求各算一个新 token。

看这套机制怎么把上一课的病根逐个干掉:

| 静态 batching 的病 | continuous batching 的解法 |
|---|---|
| **同生共死**:短请求陪着话痨空跑到死 | 短请求一吐 EOS **立刻退出**,槽位马上给新请求 → 木桶效应消失 |
| **HOL 队头阻塞**:话痨卡住整批,新请求干等 | 槽位一空就拉新请求进来,**不必等整批结束** → 队头不再阻塞 |
| **padding 浪费**:全 batch 按最长对齐 | 每 step 重组 batch,decode 阶段每个请求都只算"当前这 1 个 token",**根本不需要跨请求 padding** |

> **关键结论:continuous batching 让 GPU 始终"满载有效请求"。** 一个槽位空出来的那一刻就被新请求填上,而不是空跑到整批结束。利用率从"被最长请求绑架"变成"持续接近 100%"。

那 padding 为什么就没了?因为在 decode 阶段,**每个请求每步都只产出 1 个 token**——大家的"当前步工作量"天然一样大(都是 1),拼成 batch 不需要补 pad。各请求的历史长度不同?没关系,那是 KV Cache 的事(每个请求各读各的 KV,长度不同也无妨),不影响这一步的矩阵乘形状。这一点等 Lesson 3 PagedAttention 会彻底讲透:KV Cache 不再要求"对齐成一个规整张量"。

---

## 3. 难点:prefill 和 decode 混在一起怎么办

理想很美好,但有个现实摩擦点,也是面试高频题:

**prefill 和 decode 的计算形状完全不同。**

- decode step:每个请求只算 1 个新 token(矩阵-向量乘,轻)。
- prefill:一个新请求进来,要一次性算它**整个 prompt**(几百上千 token,矩阵-矩阵乘,重)。

如果某一步刚好有个长 prompt 的新请求要 prefill,这一步就会**特别慢**——而这一步里那些只做 decode 的老请求,只能干等它算完。结果:**正在 decode 的请求被一次大 prefill 卡出明显的延迟尖刺(TTFT/TPOT 抖动)。**

业界有两套主流解法,你知道名字和思路即可:

- **chunked prefill(分块 prefill)**:把长 prompt 的 prefill 切成若干小块,每个 step 只做一块,和 decode 请求混在同一个 batch 里推进。这样单步耗时被拉平,decode 不再被长 prefill 卡住。vLLM 默认开启。
- **prefill-decode disaggregation(PD 分离)**:干脆把 prefill 和 decode 放到**不同的 GPU / 实例**上跑,各自优化、互不干扰。这是更激进的架构,DeepSeek、Mooncake 等用得多。

> 记住这条权衡:**continuous batching 把"请求级"的 HOL 阻塞消掉了,但在"step 内"引入了"prefill 拖慢 decode"的新摩擦。chunked prefill 就是来填这个坑的。** 调度系统里没有银弹,只有把粗粒度问题换成更可控的细粒度问题。

---

## 4. 动手:用纯 Python 模拟跑出吞吐对比

光说不练假把式。`code/continuous_batching_sim.py` 是一个**不需要 GPU、不依赖任何框架**的离散事件模拟器,把"真实计算"抽象成"token 数 = 工作量",专注对比两种**调度策略**。

它造了 200 个请求,长度方差很大(90% 短请求、10% 话痨,正是真实 LLM 流量的样子),分别用静态和连续两种策略跑,统计总耗时、GPU 槽位利用率、端到端延迟。

核心是两个调度函数。先看静态(同生共死):

```python
def run_static(requests):
    pending = list(requests)
    step = 0
    while pending:
        batch = pending[:MAX_BATCH_SIZE]      # 攒一批
        pending = pending[MAX_BATCH_SIZE:]
        batch_steps = max(r.output_len for r in batch)  # 木桶:跑到最长的结束
        for _ in range(batch_steps):
            for r in batch:
                if not r.done:
                    r.generated += 1          # done 的请求仍占着槽位空跑
            step += 1
    ...
```

再看连续(随时进出),注意 `waiting` / `running` 两个队列的流转,这就是 vLLM scheduler 的骨架:

```python
def run_continuous(requests):
    waiting, running, step = list(requests), [], 0
    while waiting or running:
        # ① 加入:有空槽就从 waiting 拉新请求
        while len(running) < MAX_BATCH_SIZE and waiting:
            running.append(waiting.pop(0))
        # ② 推进:每个 running 请求各生成 1 token
        for r in running:
            r.generated += 1
        step += 1
        # ③ 退出:done 的立刻离开,腾出的槽位下一轮就能给别人
        running = [r for r in running if not r.done]
    ...
```

运行它:

```bash
cd code
python continuous_batching_sim.py
```

在参考实现上(`MAX_BATCH_SIZE=8`,200 个请求,seed=42),典型输出:

```
策略=static
  总 step 数          : 4152
  完成请求数          : 200
  GPU 槽位利用率      : 24.7%
  平均端到端延迟(step): 2090.3

策略=continuous
  总 step 数          : 1139
  完成请求数          : 200
  GPU 槽位利用率      : 90.0%
  平均端到端延迟(step): 499.6

吞吐提升(总 step 之比): 3.6x
平均端到端延迟: 静态 2090.3 -> 连续 499.6 step
```

读懂这组数字:

- **吞吐 3.6×**:同样 200 个请求,连续批处理只用了约 1/3.6 的 step 数就全部跑完。
- **利用率 24.7% → 90%**:静态批处理里四分之三的 GPU"槽位"在给已结束的请求空跑;连续批处理几乎槽槽有活。这正是 Lesson 1 那张"空跑图"的量化版。
- **延迟 2090 → 500**:静态批处理里排在后面批次的请求要苦等前面所有批跑完(HOL),端到端延迟被严重拉长;连续批处理让请求持续流动,延迟降到约 1/4。

> **同时改善吞吐和延迟,这在系统优化里很罕见——通常二者要权衡。** continuous batching 之所以能两头都赚,是因为静态 batching 实在浪费得太离谱了,它消的是**纯粹的空转**,不是从延迟里偷吞吐。

**动手任务(必做)**:
1. 跑通脚本,确认你能复现上面的数字。
2. 把话痨比例从 `0.10` 调到 `0.30`(改 `make_workload`),**先预测**两种策略的差距会变大还是变小,再验证。
3. 把 `MAX_BATCH_SIZE` 从 8 调到 32,观察吞吐和利用率怎么变,想想为什么 batch 越大连续批处理的优势越明显。
4. (进阶 TODO,脚本里已留位置)给请求加上**随时到达**(`arrival` 不全为 0,按泊松过程到达),让模拟更接近真实在线服务。看看静态 batching 在"请求陆续到达"时会糟糕到什么程度。

---

## 5. 那个著名的 "23× throughput" 到底怎么来的

你在博客、论文、面经里一定见过这个数字:**continuous batching 带来高达 23× 的吞吐提升**。要会用、更要会泼冷水。

出处:Anyscale 2023 年那篇广为流传的博客《How continuous batching enables 23x throughput in LLM inference while reducing p50 latency》。它对比的是:

- 基线:朴素的静态 batching(HuggingFace `generate` 那种)。
- 对照:continuous batching(+ 当时一并引入的优化)。

**为什么能到 23× 而不是我们模拟的 3.6×?** 关键在于真实场景的几个放大因素,理解它们比记住数字重要:

- **输出长度方差极大**:真实流量里话痨请求更极端,静态 batching 的木桶效应被放大到离谱。我们的模拟比较温和。
- **更大的 batch 上限**:真实引擎(配合 PagedAttention)能开很大的 batch,连续批处理把空槽利用得更彻底,优势随 batch 上限放大。
- **同时叠加了显存优化**:那篇博客里 continuous batching 常和高效显存管理一起出现,二者协同。
- **基线实现很朴素**:对照组是没怎么优化的静态批处理,差距自然大。

> **关键结论:"23×" 是特定负载、特定基线下的上限数字,不是你随手就能拿到的保证。** 真实加速比取决于**输出长度方差**和**能开多大 batch**。方差越大、batch 上限越高,连续批处理赢得越多;反之(比如所有请求长度都一样)收益会缩小。面试时能讲清楚"它依赖什么条件"才是真懂,张口就来"23 倍"反而露怯。

这也顺势引出了下一课:**能开多大 batch,卡在显存上**。而 KV Cache 是显存大户,它的管理方式直接决定了 batch 上限。怎么把 KV Cache 管得又省又灵活?这就是 PagedAttention。

---

## 练习题

1. 用你自己的话解释"迭代级调度(iteration-level scheduling)"和"请求级调度"的区别,以及为什么前者能同时改善吞吐和延迟。

2. continuous batching 在 decode 阶段为什么"不需要 padding"?(提示:每个请求每步的工作量是多少?各请求历史长度不同会不会破坏这一点?)

3. 一个长 prompt(2000 token)的新请求进来做 prefill,会对同 batch 里正在 decode 的请求造成什么影响?业界用什么办法缓解?

4. 把模拟脚本里话痨比例从 0.10 提到 0.30 后,你观察到静态 vs 连续的差距变大了还是变小了?从"木桶效应"角度解释。

5. 有人说"我们的线上请求输出长度都差不多(都是固定模板),所以上不上 continuous batching 无所谓"。这话对吗?在什么前提下他说得有几分道理?

<details>
<summary>参考答案(想完再看)</summary>

1. 请求级调度在请求进入 batch 后就锁死到它结束;迭代级调度在**每个 step 边界**重新决定 batch 成员,完成的立刻退出、新来的随时加入。它能两头赚是因为静态批处理浪费的是**纯空转**(已结束请求占着槽位空跑),消除空转既缩短了后续请求的等待(降延迟)又让单位时间服务更多请求(提吞吐),不是从延迟里换吞吐。

2. 因为 decode 阶段每个请求每步**只产出 1 个 token**,大家"当前步的工作量"都是 1,拼 batch 时形状天然一致,无需补 pad。各请求历史长度不同只影响各自要读的 KV Cache(各读各的),不改变"这一步算 1 个 token"的矩阵乘形状,所以不破坏。

3. 这一步会变得特别慢(要一次算 2000 个 token 的 prefill),同 batch 里只做 decode 的请求被迫干等,造成延迟尖刺(TTFT/TPOT 抖动)。缓解办法:**chunked prefill**(把长 prefill 切块,每步只做一块,和 decode 混跑,拉平单步耗时)或 **PD 分离**(prefill 和 decode 放不同实例)。

4. 差距变大。话痨越多,静态批处理里"一个话痨绑架整批短请求空跑"的木桶效应越严重,有效利用率越低;而连续批处理里短请求照样及时退出、不受话痨连累,所以话痨越多,二者差距越大。

5. 不全对。如果输出长度**真的高度一致**,木桶效应确实很弱,continuous batching 在"消除空跑"上的收益会缩小——这是他有道理的前提。但即使长度一致,continuous batching 在**请求陆续到达**(不是一次性全到)时仍有价值:它能让新到请求立刻填补空槽,而不必等下一整批攒齐,依然降低排队延迟、提升利用率。所以"无所谓"过于绝对。

</details>

---

## 小结

- continuous batching = **迭代级调度**:在每个 decode step 边界重组 batch,而非锁定整个请求生命周期。
- 用 **waiting / running 两队列** 实现"随时加入、随时退出":完成的立刻退出释放槽位,新请求立刻填空。
- 它同时治了静态 batching 的两宗罪:**木桶效应**(短请求及时退出)和 **HOL 阻塞**(槽位空了就拉新人),decode 阶段还**天然免 padding**。
- 新摩擦:**prefill 拖慢同批 decode**,用 **chunked prefill** 或 **PD 分离** 缓解。
- "**23× 吞吐**"是特定负载/基线的上限,真实加速比取决于**输出长度方差**和**batch 上限**;后者卡在显存,引出下一课。

## 自测验收(过了再进 Lesson 3)
- [ ] 能向别人讲清"请求级 vs 迭代级"调度的区别,以及为什么能两头赚。
- [ ] 能默写出连续批处理的 waiting/running 调度循环骨架(加入→推进→退出)。
- [ ] 跑通模拟脚本,复现吞吐/利用率/延迟三组数字,并完成话痨比例和 batch 大小两个对照实验。
- [ ] 能解释 chunked prefill 解决的是什么问题。
- [ ] 能说清 "23×" 依赖哪些条件,不会把它当万能数字。

下一课:**Lesson 3 — PagedAttention**。我们去解决"能开多大 batch"背后的显存难题:KV Cache 怎么管才不浪费、不碎片化。答案藏在你操作系统课学过的**虚拟内存分页**里。
