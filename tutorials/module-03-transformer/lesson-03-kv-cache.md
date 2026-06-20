# M3 · Lesson 3:KV Cache——原理与显存计算

> 上一课结尾我们留了个坑:朴素 decode 每生成一个 token,都把不断变长的整个序列重新前向一遍——这里面有巨大的**重复计算**。本课的 KV Cache 就是干掉这份浪费的关键技术,它是所有现代推理引擎的基石。你不仅要懂原理,还要能**手推显存占用公式**、估算并发容量——这是推理 Infra 工程师的必备硬技能。
> 预计用时:2.5 小时(阅读 + 推导 + 给迷你推理加缓存测速)。
> 前置:Lesson 1(attention 计算)、Lesson 2(prefill/decode、带宽受限)。
> 环境:实测建议 GPU;显存公式手算部分纸笔即可。

## 学习目标

学完本课你应该能回答:
1. 朴素 decode 到底在重复算什么?为什么 K、V 可以缓存而 Q 不能?
2. KV Cache 的原理是什么?它把 decode 的单步计算量从 \(O(S^2)\) 降到了多少?
3. 能**默写并推导** KV Cache 显存占用公式,并手算 7B 模型在某并发下的显存。
4. 为什么 KV Cache 是「用显存换计算」?它带来了哪些新的工程问题(伏笔 PagedAttention)?

---

## 1. 朴素 decode 的浪费:每步都重算历史

回忆 attention:位置 `i` 的输出,需要它的 Query 去和**所有历史位置**的 Key、Value 做加权。decode 第 `t` 步,我们要算的是**最新一个 token**(位置 `t`)的输出:

\[
\text{out}_t = \text{softmax}\!\left(\frac{q_t \cdot [k_1, \dots, k_t]^\top}{\sqrt{d_k}}\right) [v_1, \dots, v_t]
\]

注意:这里只需要 \(q_t\)(最新 token 的 Query),但需要**全部历史的 \(k_1..k_t\) 和 \(v_1..v_t\)**。

朴素做法(Lesson 2 的代码)是把整个序列 `[1..t]` 重新前向,于是 \(k_1..k_{t-1}\)、\(v_1..v_{t-1}\) 被**反复重算了无数遍**:

```
朴素 decode(重复计算):
  step t=1: 算 k1,v1
  step t=2: 算 k1,v1,k2,v2        ← k1,v1 又算了一遍
  step t=3: 算 k1,v1,k2,v2,k3,v3  ← k1,v1,k2,v2 又算了一遍
  ...
  累计计算量 ~ O(N^2),N 为生成长度
```

**关键洞察**:对一个已经处理过的历史 token,它的 K 和 V **永远不变**(因为 K、V 只依赖该 token 自己和它之前的内容,后面新增 token 不影响它)。既然不变,**为什么要重算?缓存起来就好。**

> 为什么 Q 不缓存?因为每一步我们只关心**最新 token 的 Q**(去查历史),历史 token 的 Q 在它那一步用完就再也不需要了。**只有 K、V 需要被未来反复查询,所以只缓存 K、V。** 这就是名字「KV Cache」的由来。

---

## 2. KV Cache 原理:存历史的 K、V

做法极简单:开两块缓存 `K_cache`、`V_cache`,每生成一个 token 就把它新算的 \(k_t, v_t\) **追加(append)**进去。decode 每步:

```
带 KV Cache 的 decode:
  输入:仅最新 1 个 token  →  算它的 q_t, k_t, v_t
  把 k_t, v_t 追加到 cache:  K_cache += k_t,  V_cache += v_t
  attention:q_t 与 K_cache 全部、V_cache 全部 做注意力
  产出 1 个 token
```

对比 prefill 与 decode 的张量形态:

```
Prefill(处理 prompt,S 个 token 一起):
  Q,K,V: [S, d]      →  attention 分数 [S, S]      →  填满 KV Cache 前 S 项

Decode(每步 1 个 token,带 cache):
  q_t:   [1, d]
  K_cache: [t, d]    →  attention 分数 [1, t]      →  只算 1 行!
  追加新的 k_t,v_t,cache 长度 +1
```

**收益**:decode 单步的 attention 从「算整个 `[t,t]` 矩阵」降到「算一行 `[1,t]`」。

- 单步计算量:\(O(t^2) \to O(t)\)。
- 生成全程累计:\(O(N^2) \to O(N \cdot \bar{t})\)(仍随上下文线性,但消除了重复)。
- QKV 投影:从「重算 `t` 个 token 的投影」降到「只算 1 个 token」。

> 这正是 Lesson 2 里 decode 那张图能成立的前提:**因为有 KV Cache,decode 每步输入才能只喂 1 个 token。** 前面我们假装它存在,现在补上了原理。

**代价**:你得把所有历史的 K、V 一直存在显存里。这是一笔**显存换计算**的交易——而这笔账,正是推理部署的核心约束。

---

## 3. 显存占用公式:必须会手推

这是本课的硬核。我们一步步推导 KV Cache 占多少显存。

**单个 token、单层、单个 KV 头**,缓存一个 K 向量需要 `head_dim` 个数,V 同理,所以是 `2 × head_dim` 个数。

往上累乘各维度:

\[
\boxed{\text{KVCache 字节数} = \underbrace{L}_{\text{层数}} \times \underbrace{2}_{K,V} \times \underbrace{S}_{\text{序列长}} \times \underbrace{n_{kv}}_{\text{KV头数}} \times \underbrace{d_h}_{\text{head\_dim}} \times \underbrace{P}_{\text{每个数字节}} \times \underbrace{B}_{\text{batch}}}
\]

各因子含义:

| 符号 | 含义 | 7B 典型值(LLaMA-2-7B) |
|---|---|---|
| `L` | 层数 | 32 |
| `2` | K 和 V 各一份 | 2 |
| `S` | 序列长度(prompt+生成) | 视场景,例 2048 |
| `n_kv` | KV 头数(GQA 下 < n_heads) | 32(7B 无 GQA);GQA 模型更小 |
| `d_h` | head_dim = d / n_heads | 128 |
| `P` | 每个数的字节(fp16=2,fp8=1) | 2 |
| `B` | batch / 并发请求数 | 视并发 |

> 注意 \(n_{kv} \times d_h\) 这一项:对**没有 GQA** 的模型,`n_kv = n_heads`,于是 \(n_{kv}\times d_h = d\)(隐藏维)。所以常见简写形式是 `L × 2 × S × d × P × B`。**有了 GQA,就要老老实实用 \(n_{kv}\times d_h\),不能用 d。** 这是面试和实战的高频坑。

### 3.1 手算:LLaMA-2-7B,单请求,S=2048

\[
32 \times 2 \times 2048 \times (32 \times 128) \times 2 \text{ byte} \times 1
\]

逐步算:
- \(n_{kv}\times d_h = 32 \times 128 = 4096\)(等于 d,因为 7B 无 GQA)。
- \(32 \times 2 \times 2048 = 131072\)。
- \(131072 \times 4096 = 5.37\times 10^8\) 个数。
- \(\times 2\) 字节 \(= 1.07\times 10^9\) 字节 ≈ **1.0 GB**。

**单条 2048 长度的请求,KV Cache 就要 1GB。** 模型权重 fp16 是 14GB,假设卡是 24GB(4090),那留给 KV Cache 的大约 10GB:

### 3.2 算并发容量

\[
\text{最大并发} \approx \frac{\text{显存} - \text{权重}}{\text{每请求 KVCache}} = \frac{24 - 14}{1.0} \approx 10 \text{ 条}
\]

**结论:一张 24GB 卡跑 fp16 的 7B,2048 上下文,大约只能塞 ~10 条并发。** 这个数字直接决定了你的服务吞吐上限。想提高,要么:
- **量化权重**(M5):14GB→4GB(int4),腾出更多显存给 KV Cache。
- **量化 KV Cache**(M5):fp16→fp8,`P` 从 2→1,KV Cache 减半。
- **GQA**:`n_kv` 从 32→8,KV Cache 直接降 4×(这就是 LLaMA-2-70B 用 GQA 的原因)。
- **PagedAttention**(M4):解决「预分配 max_len 造成的碎片浪费」,让显存利用率从 ~40% 提到 ~90%。

> 把这套手算练到张口就来。面试官问「你这卡能跑多少并发」,这就是标准答题路径:**权重多大 → 单请求 KV Cache 多大 → (显存−权重)/单请求 = 并发**。

---

## 4. KV Cache 带来的新问题(承上启下)

KV Cache 解决了重复计算,但引入了三个新麻烦,正是后续模块的主题:

1. **显存暴涨且随上下文增长**:长上下文(32K、128K)时 KV Cache 可能超过权重本身 → 量化 KV Cache、GQA。
2. **变长 + 碎片**:每个请求长度不同、还在动态增长,提前按 max_len 预分配会浪费大量显存 → **PagedAttention**(M4)用「分页」像操作系统管理虚拟内存一样管理 KV Cache。
3. **decode 带宽里多了一项**:回忆 Lesson 2,decode 每步要读权重 + 读 KV Cache,长上下文时读 KV Cache 成为新的带宽瓶颈 → FlashAttention 的 decode 变体、KV Cache 量化。

> 一句话:**KV Cache 把「计算瓶颈」转化成了「显存与显存带宽瓶颈」。** 理解这个转化,你就理解了 M4/M5 大半内容的动机。

---

## 5. 动手实验:给迷你推理加 KV Cache 测速

代码见 `code/kv_cache_demo.py`,包含两部分:

**Part A — 显存计算器**:输入模型超参(L、n_kv、head_dim、dtype)与并发、序列长度,打印单请求 KV Cache 大小与最大并发数。用它复算上面的 7B 例子。

**Part B — 实测提速**:用真实小模型(GPT-2),对比「不用 cache」与「用 cache(`use_cache=True` / `past_key_values`)」生成相同长度的耗时。HuggingFace 模型原生支持 KV Cache,我们手动驱动这个循环来看清机制:

```python
# 用 KV Cache 的 decode:每步只喂 1 个 token + 历史 past_key_values
past = None
cur = input_ids                      # 第一次喂整个 prompt(prefill)
for _ in range(n_gen):
    out = model(cur, past_key_values=past, use_cache=True)
    past = out.past_key_values        # 缓存被追加更新
    next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
    cur = next_token                  # 之后每步只喂 1 个新 token!
```

运行:

```bash
cd code
python kv_cache_demo.py --model gpt2 --prompt-len 64 --gen-len 128
```

预期(数值随硬件):

```
=== 显存计算器:LLaMA-2-7B, S=2048, B=1 ===
单请求 KV Cache: 1.00 GB
24GB 卡(权重 14GB)最大并发 ≈ 10 条

=== 实测:GPT-2 生成 128 token ===
[无 cache] 总耗时 1820 ms  (每步重算整段)
[有 cache] 总耗时  240 ms  (每步只喂 1 token)
加速比 ≈ 7.6×
```

**观察重点**:加速比随生成长度增大而增大(因为朴素法是 \(O(N^2)\),cache 法是 \(O(N)\))。把 `--gen-len` 改大,看加速比怎么涨。

### 实验任务
- **必做**:跑通,验证「有 cache」显著快于「无 cache」,且加速比随 gen-len 增大。
- **必做**:补全 Part A 中 `TODO` 的 `kv_cache_bytes` 函数,用它复算 7B 的 1GB 结论,误差应在 5% 内。
- **选做**:把计算器参数改成一个 GQA 模型(如 Qwen2-7B:`n_kv=4`),对比 KV Cache 缩小了多少倍。

---

## 练习题

1. 默写 KV Cache 显存公式,并解释为什么是「×2」、为什么 GQA 模型不能用 `d` 而要用 `n_kv × head_dim`。
2. LLaMA-2-7B(L=32, n_heads=32, head_dim=128, fp16),要支持 batch=16、每条 S=4096,KV Cache 共需多少显存?80GB 的 A100(权重 14GB)能放下吗?
3. 同样模型改用 fp8 存 KV Cache,第 2 题的显存变成多少?这是 KV Cache 量化的直接收益。
4. 为什么 KV Cache 是「用显存换计算」?在什么场景下这笔交易可能不划算(提示:极短生成)?

<details>
<summary>参考答案(想完再看)</summary>

1. 公式:`L × 2 × S × n_kv × head_dim × P × B`。「×2」是 K 和 V 各存一份;GQA 下 KV 头数 `n_kv < n_heads`,缓存的是 KV 头而非 Q 头,所以维度是 `n_kv × head_dim`(它 < d),用 `d` 会高估。
2. \(32\times2\times4096\times(32\times128)\times2\times16\)。先算单请求 S=4096:是 3.1 节 S=2048 的 2 倍 = 2GB;×16 = **32GB**。加上权重 14GB 共 46GB,80GB A100 放得下(还需留激活等开销,实际更紧)。
3. fp8 下 `P=1`,KV Cache 减半 → **16GB**。加权重 14GB 共 30GB,余量充足,可进一步加大并发或上下文。
4. 因为它把「重复计算 K、V」换成了「常驻显存存 K、V」。当生成极短(比如只生成 1~2 个 token)时,重复计算本来就没几次,缓存省下的计算有限,却仍占用显存与分配/读取开销,可能不划算;但绝大多数生成场景(几十到上千 token)它都是巨大净收益。

</details>

---

## 小结

- 朴素 decode 反复重算历史 K、V(\(O(N^2)\) 浪费);**历史 token 的 K、V 不变,故可缓存**,Q 用完即弃故不缓存。
- KV Cache 把 decode 单步 attention 从 `[t,t]` 降到 `[1,t]`,投影只算 1 个 token,这是 decode「每步只喂 1 token」的前提。
- **显存公式**:`L × 2 × S × n_kv × head_dim × P × B`,务必会手推;GQA 模型用 `n_kv×head_dim` 而非 `d`。
- 标准容量估算:**(显存 − 权重)/ 单请求 KV Cache = 最大并发**;7B/24GB/2048 ≈ 10 条。
- KV Cache 把计算瓶颈转成了**显存与显存带宽瓶颈**,直接引出量化 KV Cache、GQA、PagedAttention。

## 自测验收(过了再进 Lesson 4)
- [ ] 能用一句话说清「为什么缓存 K、V 而不缓存 Q」。
- [ ] 能不看资料默写显存公式,并手算 7B 在给定并发/长度下的 KV Cache。
- [ ] 能由「权重大小 + 单请求 KV Cache + 显存」推出最大并发数。
- [ ] `kv_cache_demo.py` 跑通,实测加速比随 gen-len 增大,计算器复算 7B 误差 < 5%。
- [ ] 能说出 KV Cache 引出的三个后续工程问题(量化/GQA/PagedAttention)。

下一课:**Lesson 4 — 采样策略**,有了 logits,如何把它变成下一个 token?greedy / temperature / top-k / top-p / repetition penalty 的原理与手写实现。
