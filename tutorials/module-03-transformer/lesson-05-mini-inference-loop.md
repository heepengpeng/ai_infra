# M3 · Lesson 5:从零搭一个迷你推理循环

> 这是 Module 3 的收官课,也是「打开黑盒」的高潮。前四课的零件——**Transformer 前向、prefill/decode 两阶段、KV Cache、采样策略**——现在我们把它们**亲手焊成一台能跑的迷你推理引擎**,加载真实小模型生成文本,并 benchmark 它的吞吐。跑通它,你就从「会用 `model.generate`」升级到「知道 generate 内部每一行在干什么」。最后我们留一组数据,作为 Module 4(真正的推理引擎)的对照基线。
> 预计用时:3 小时(阅读 + 补全推理循环 + benchmark + 调参)。
> 前置:Lesson 1~4 全部。
> 环境:建议 GPU(Colab T4 / AutoDL 4090)。模型用 Qwen2-0.5B 或 GPT-2,显存需求很小,CPU 也能跑通(慢)。

## 学习目标

学完本课你应该能回答:
1. 一个最小推理引擎的主循环长什么样?prefill 和 decode 在代码里如何分工?
2. 怎么把 KV Cache 和采样正确接进循环?边界(EOS、max_len)怎么处理?
3. 如何正确 benchmark 吞吐(token/s)、TTFT、TPOT?需要避开哪些计时坑?
4. 我这台「单请求、串行」迷你引擎,瓶颈在哪?Module 4 的引擎要解决它的什么短板?

---

## 1. 推理循环全景:把四课焊在一起

先把整个生成过程用伪代码写清楚——这就是所有推理引擎(包括 vLLM)最内核的骨架,只是它们在外面套了调度、batching、分页:

```
加载模型与 tokenizer
prompt 文本 → tokenizer → input_ids [1, S]

# ===== Prefill 阶段(一次)=====
out = model(input_ids, use_cache=True)      # 并行处理整个 prompt
past = out.past_key_values                  # ← KV Cache 建好(Lesson 3)
logits = out.logits[:, -1, :]               # 只要最后一个位置的 logits

# ===== Decode 阶段(循环 N 次)=====
generated = []
for step in range(max_new_tokens):
    next_id = sample(logits, ...)           # 采样(Lesson 4)
    if next_id == eos: break                # 边界:遇到结束符停
    generated.append(next_id)
    # 只喂这 1 个新 token + 历史 cache(Lesson 2 的 decode 形态)
    out = model([[next_id]], past_key_values=past, use_cache=True)
    past = out.past_key_values              # cache 追加更新
    logits = out.logits[:, -1, :]

文本 = tokenizer.decode(generated)
```

**关键认知**:这 20 行就是大模型推理的「心脏」。`model.generate()` 帮你封装了它,但封装的代价是你看不见 prefill/decode 边界、看不见 cache 怎么流转。**自己写一遍,黑盒就透明了。**

```
       前四课的零件 → 本课的整机
  ┌─────────────────────────────────────────────┐
  │  L1 Transformer 前向 ──┐                      │
  │  L2 prefill/decode ────┼──► 推理主循环 ──► 文本│
  │  L3 KV Cache ──────────┤                      │
  │  L4 采样 ──────────────┘                      │
  └─────────────────────────────────────────────┘
```

---

## 2. 三个工程细节,决定循环对不对

光有骨架不够,以下三点是「跑得对」和「跑得快」的分水岭:

### 2.1 prefill 只取最后一个位置的 logits

prefill 一次前向产出 `[1, S, V]` 的 logits(每个位置都有预测),但生成时**只需要最后一个位置**(它预测的是 prompt 之后的第一个新 token)。前面位置的 logits 在生成里用不上(它们在训练算 loss 时才用)。取 `logits[:, -1, :]`,别浪费。

### 2.2 decode 每步只喂 1 个 token

这是 KV Cache 的全部意义(Lesson 3)。如果你不小心又把整个序列喂进去,cache 就白建了,退化成 Lesson 2 的朴素 \(O(N^2)\) 慢法。**喂进 `model` 的 `input_ids` 在 decode 阶段长度必须是 1。**

### 2.3 计时要同步、要分阶段

回忆 M1 L5:GPU kernel 异步下发,`time.perf_counter()` 前后必须 `torch.cuda.synchronize()`,否则测到的是「下发时间」不是「执行时间」。而且要**分别**计:
- **TTFT(Time To First Token)** = prefill 耗时(决定用户等多久看到第一个字)。
- **TPOT(Time Per Output Token)** = decode 总耗时 / 生成 token 数(决定后续吐字速度)。
- **吞吐(throughput)** = 生成 token 数 / 总耗时(token/s)。

> 业界报告推理性能,几乎都是这三个数。能把它们算对、解释清,你就具备了推理 benchmark 的基本功——Module 4、6 全程都在优化它们。

---

## 3. 这台迷你引擎的「短板」(Module 4 伏笔)

我们这台引擎能跑,但它有两个致命短板,正是真实引擎要解决的:

```
短板 1:单请求,串行
  ┌──────────────────────────────────────┐
  │ 请求 A: prefill → decode decode ... 完 │  ← 必须等 A 全跑完
  │ 请求 B:                          等待... │     B 才能开始
  └──────────────────────────────────────┘
  GPU 在 decode 时算力大量闲置(Lesson 2:带宽受限),却没拿去服务别的请求。
  → Module 4 的 Continuous Batching:把多个请求的 decode 步拼在一起,
    一次读权重服务 B 个 token,吞吐成倍提升。

短板 2:KV Cache 静态预分配,显存浪费
  每个请求按 max_len 预留一整块连续显存,实际只用了一小段 → 大量碎片。
  → Module 4 的 PagedAttention:像 OS 管虚拟内存一样分页管理 KV Cache,
    显存利用率从 ~40% 提到 ~90%,并发翻倍。
```

> 所以本课结尾,我们会把这台单请求引擎的 **吞吐 / TTFT / TPOT 记录下来**,作为 Module 4 的「优化前基线」。等你学完 continuous batching 和 PagedAttention,回头对比这组数字,会非常有成就感——那才是「优化 X×」的真实体感。

---

## 4. 动手实验:补全并运行迷你引擎

代码见 `code/mini_inference.py`。它实现了:
- 加载模型(默认 `Qwen/Qwen2-0.5B`,可 `--model gpt2` 退化到更小);
- `generate()` 主循环:prefill + 带 KV Cache 的 decode + 采样(复用 Lesson 4 的采样思想,内置简化版);
- `benchmark()`:计算 TTFT、TPOT、吞吐,正确同步计时。

核心循环(完整见文件):

```python
@torch.no_grad()
def generate(self, prompt, max_new_tokens, **sample_kwargs):
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

    # ---- prefill ----
    out = self.model(input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    # ---- decode ----
    generated = []
    for _ in range(max_new_tokens):
        next_id = self.sample(logits[0], **sample_kwargs)
        if next_id == self.tokenizer.eos_token_id:
            break
        generated.append(next_id)
        cur = torch.tensor([[next_id]], device=self.device)
        out = self.model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return self.tokenizer.decode(generated)
```

运行:

```bash
cd code
pip install torch transformers accelerate
python mini_inference.py --model Qwen/Qwen2-0.5B --prompt "用一句话解释什么是KV Cache:" --max-new-tokens 128
```

预期(数值随硬件):

```
[生成结果] KV Cache 是把历史 token 的 Key 和 Value 缓存起来,避免每步重复计算……
=== benchmark(单请求基线) ===
TTFT(首 token / prefill): 41.2 ms
TPOT(每输出 token)      : 7.8 ms/token
吞吐(decode)            : 128.2 token/s
prompt 长度 12, 生成 128 token
→ 记下这组数字,作为 Module 4 的优化前基线
```

### 实验任务
- **必做**:补全 `mini_inference.py` 中标注 `TODO` 的 decode 循环关键步骤(喂单 token + 更新 cache),跑通生成。
- **必做**:用 greedy 和 `temperature=0.8, top_p=0.9` 各生成一次,对比结果的确定性与多样性(呼应 Lesson 4)。
- **必做**:记录你机器上的 TTFT / TPOT / 吞吐,写进 `PROGRESS.md` 或笔记,留作 Module 4 基线。
- **选做**:把 `max_new_tokens` 从 64 加到 512,观察 TPOT 是否随上下文变长而略升(提示:Lesson 3,KV Cache 变长 → 读 cache 带宽增加)。
- **选做**:故意关掉 KV Cache(每步喂整段),对比吞吐暴跌,亲手复现 Lesson 3 的结论。

---

## 练习题

1. 在主循环里,如果忘了写 `past_key_values=past`(每步重建空 cache),会发生什么?吞吐会怎样变化?为什么?
2. TTFT 和 TPOT 哪个对「长 prompt、短回答」的场景更关键?哪个对「短 prompt、长回答」更关键?
3. 你测出吞吐 120 token/s。要把它翻倍,基于本模块所学,列出至少 3 条不同方向的优化(分别属于哪一课/哪个模块)?
4. 为什么单请求串行时,GPU 利用率在 decode 阶段往往很低?这和 Module 4 的 continuous batching 有什么关系?

<details>
<summary>参考答案(想完再看)</summary>

1. 退化成 Lesson 2 的朴素法,但更糟:若每步只喂 1 个 token 又没给 cache,模型看不到历史,生成会错乱;若每步喂整段又不用 cache,则重复计算,吞吐随生成长度呈 \(O(N^2)\) 暴跌。正确写法必须「喂单 token + 传 past」。
2. **长 prompt、短回答**:prefill 重(长 prompt),TTFT 是大头,优化 prefill/TTFT 更关键。**短 prompt、长回答**:decode 步数多,TPOT/吞吐主导,优化 decode 更关键。
3. 例:(a) **量化权重 int4**(M5)——decode 带宽减 4×;(b) **continuous batching**(M4)——多请求拼批提高权重复用;(c) **KV Cache 量化 / GQA**(L3、M5)——减少 KV Cache 读写;(d) **FlashAttention**(M2 L7)——长序列 attention 提速。属于不同课/模块。
4. decode 带宽受限(Lesson 2),单请求时一次只算 1 个 token,几千个 CUDA 核心大量闲置,算力浪费。continuous batching 把多个请求的 decode 步合并,一次读权重同时服务多个 token,把闲置算力利用起来,大幅提升整体吞吐。

</details>

---

## 小结

- 推理引擎的内核就是 **prefill + 带 KV Cache 的 decode 循环 + 采样**,约 20 行;`model.generate` 只是它的封装。
- 三个对错关键:prefill **只取最后位置 logits**、decode **每步只喂 1 token**、计时要**同步 + 分阶段(TTFT/TPOT/吞吐)**。
- 你这台引擎是**单请求、串行**的,短板是 decode 时 GPU 闲置、KV Cache 静态预分配浪费。
- 这两个短板正是 Module 4 的 **Continuous Batching** 与 **PagedAttention** 要解决的;请记下你的基线数字,以便日后对比。
- 至此你已**亲手打开了大模型推理的黑盒**:从结构、两阶段、缓存、采样到完整循环,全程不靠调 API 而是理解原理。

## 自测验收(Module 3 通关标准)
- [ ] 能默写推理主循环骨架,讲清 prefill 与 decode 各做什么。
- [ ] `mini_inference.py` 补全跑通,能用 greedy 与采样两种方式生成并解释差异。
- [ ] 能正确测出并解释 TTFT / TPOT / 吞吐,知道计时同步的坑。
- [ ] 能说清这台迷你引擎的两个短板,以及 Module 4 如何解决。
- [ ] 回头能用本模块知识,完整回答「7B 模型在某卡上能跑多少并发、decode 为什么慢、怎么优化」。

---

## Module 3 收官 & 下一步

你已经走完「打开黑盒」这一程:**结构(L1)→ 两阶段(L2)→ KV Cache(L3)→ 采样(L4)→ 整机(L5)**。从这里开始,推理对你不再是一个 `generate()` 黑盒,而是一条你能逐段拆解、定位瓶颈、动手优化的流水线。

**下一站:Module 4 — 推理引擎核心机制**。我们从你这台迷你引擎的短板出发,学静态 batching 的问题、Continuous Batching、PagedAttention,最后精读 nano-vLLM 源码,把「迷你玩具」升级成「准工业引擎」。带上你今天记下的基线数字,我们去把它优化 X×。
