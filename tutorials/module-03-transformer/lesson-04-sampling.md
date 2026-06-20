# M3 · Lesson 4:采样策略——从 logits 到下一个 token

> 前三课我们把模型「前向出 logits」这条链路打通了。但 logits 不是 token——它只是词表上每个候选 token 的「未归一化打分」。**怎么从这堆打分里挑出下一个 token,直接决定了生成结果是死板复读还是天马行空。** 这一步叫采样(sampling),是推理循环里看似简单、实则讲究的一环,也是工程上参数最多、最容易被问的部分。
> 预计用时:2 小时(阅读 + 手写各采样函数 + 观察多样性)。
> 前置:Lesson 1~3;概率分布、softmax 的基本概念。
> 环境:纯 CPU 即可(只在 logits 向量上操作,不依赖大模型)。

## 学习目标

学完本课你应该能回答:
1. logits、概率、softmax 三者什么关系?temperature 在数学上动了什么?
2. greedy、top-k、top-p(nucleus)、repetition penalty 各自的原理与取舍?
3. 这些策略如何组合成工业级的采样管线(顺序很重要)?
4. 为什么采样几乎不影响推理性能(相对 decode 的带宽瓶颈而言)?

---

## 1. logits → 概率:softmax 与 temperature

模型最后一层 LM Head 输出 `logits ∈ [V]`(V 是词表大小),每个分量是对应 token 的「打分」,范围任意(可正可负)。要变成概率分布,过 **softmax**:

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

`p_i` 即「下一个 token 是第 `i` 个词」的概率,所有 `p_i` 非负且和为 1。

**Temperature(温度 \(T\))** 在 softmax 之前把 logits 除以 \(T\):

\[
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
\]

直觉上 `T` 控制分布的「陡峭/平缓」:

```
原始 logits: [3.0, 1.0, 0.5]

T = 0.5(更陡,更确定):     T = 1.0(原样):           T = 2.0(更平,更随机):
 高分被进一步放大            标准分布                   差距被压缩
 [0.88, 0.10, 0.02]         [0.71, 0.21, 0.08]         [0.50, 0.30, 0.20]
   ▁▁█                        ▁██                        ███
```

- **\(T < 1\)**:放大差距,分布更尖,高概率 token 更容易被选 → **更确定、更保守**(适合代码、事实问答)。
- **\(T > 1\)**:压缩差距,分布更平,低概率 token 也有机会 → **更随机、更有创意**(适合写作、头脑风暴)。
- **\(T \to 0\)**:退化成 greedy(永远选最大的)。

> 关键结论:**temperature 不改变 token 的相对排序,只改变它们之间的概率"差距"。** 它是控制「确定性 ↔ 多样性」的总旋钮。

---

## 2. Greedy:永远选最大的

最简单的策略:直接取 logits 最大的那个 token(等价于 \(\arg\max\),不需要 softmax)。

\[
x_{t+1} = \arg\max_i z_i
\]

- **优点**:确定性(同样输入永远同样输出)、最简单、对事实性任务往往最稳。
- **缺点**:容易陷入**重复循环**(「我觉得我觉得我觉得……」),缺乏多样性。因为它每步都贪心选局部最优,没有全局视野。

> 你做模型迁移做对齐时,验证「迁移后输出和原模型一致」,几乎都用 greedy + 固定输入——因为它**可复现**,是调试的黄金标准。这是 greedy 在工程里最重要的用途。

---

## 3. Top-k:只在前 k 个里采

greedy 太死,纯按概率随机采(multinomial)又可能采到长尾的离谱 token。**Top-k** 折中:**只保留概率最高的 `k` 个 token,其余置零,在这 k 个里按概率重新归一化后采样。**

```
logits 排序后取前 k=3:
  [the:5.0, a:4.0, cat:3.0, | dog:1.0, ... 其余全部丢弃]
                              ↑ 截断线
  在 {the, a, cat} 上重新归一化概率,采样
```

- **优点**:挡住长尾垃圾 token,同时保留多样性。
- **缺点**:`k` 是**固定**的,不适应分布形状。当模型很确定时(某个 token 0.99),k=50 会引入 49 个本不该考虑的词;当模型很不确定时(分布很平),k=50 又可能砍掉合理候选。

---

## 4. Top-p(nucleus):按累积概率动态截断

**Top-p(核采样)** 解决 top-k 的「固定 k」问题:**按概率从高到低累加,直到累积概率 ≥ p,只保留这批 token(称为 nucleus 核),在其中归一化采样。** 候选数量是**动态**的。

```
按概率降序累加,p = 0.9:
  the: 0.5   累积 0.5
  a:   0.3   累积 0.8
  cat: 0.15  累积 0.95 ≥ 0.9  ← 到此为止,保留 {the, a, cat}
  dog: 0.03  丢弃
  ...

模型很确定时(the:0.95):只保留 1 个,近似 greedy(自适应收紧)
模型不确定时(分布很平):保留很多个(自适应放开)
```

- **优点**:**自适应**——分布尖时候选少、分布平时候选多,比 top-k 更合理。这是目前最主流的策略。
- **典型组合**:`temperature=0.7~1.0` + `top_p=0.9` + 有时叠加一个较大的 `top_k`(如 50)兜底。

> top-k 和 top-p 可以叠加:先 top-k 砍掉绝对长尾,再 top-p 动态收。工业实现(vLLM/HF)都支持同时配置,**顺序是先 k 后 p**。

---

## 5. Repetition penalty:压制复读

即使有 top-p,模型仍可能重复。**Repetition penalty** 对**已经出现过**的 token 的 logits 施加惩罚,降低它们被再次选中的概率:

\[
z_i' = \begin{cases} z_i / \rho & z_i > 0 \\ z_i \cdot \rho & z_i \le 0 \end{cases} \quad (\rho > 1,\ i \in \text{已出现 token})
\]

注意正负 logits 的处理不同:对正 logit 除以 \(\rho\)(变小),对负 logit 乘以 \(\rho\)(更负),两种都是「往下压」。\(\rho=1.0\) 即不惩罚,常用 `1.1~1.3`。

- **变体**:`frequency_penalty`(按出现次数线性加罚)、`presence_penalty`(出现过就罚固定值),GPT API 用的是后两者。
- **坑**:罚太重会让模型不敢用正常的高频词(如「的」「the」),语句变得别扭。要适度。

> 这一步是**在 softmax 之前**改 logits,和 temperature 一样属于「logits 处理器」。记住整个管线的顺序很关键(下一节)。

---

## 6. 完整采样管线:顺序很重要

工业级采样把上述步骤串成一条流水线,**顺序约定如下**(vLLM、HF generate 基本一致):

```
原始 logits [V]
   │ ① repetition / frequency penalty  (改 logits,压制已出现 token)
   ▼
   │ ② temperature 缩放                 (logits / T)
   ▼
   │ ③ top-k 过滤                       (保留前 k,其余设 -inf)
   ▼
   │ ④ top-p 过滤                       (保留累积概率核,其余设 -inf)
   ▼
   │ ⑤ softmax → 概率分布
   ▼
   │ ⑥ multinomial 采样(或 T=0 时 argmax)
   ▼
下一个 token id
```

**为什么这个顺序**:惩罚和温度是「调分」,要在筛选前做完;top-k/top-p 是「筛选候选集」;最后才归一化采样。如果顺序乱了(比如先 softmax 再加惩罚),数学含义就错了。

> 性能视角:回忆 Lesson 2,decode 是**带宽受限**,瓶颈在读权重。采样只是在一个 `[V]` 向量上做排序/筛选,计算量相比整个前向**微不足道**(V 才几万,前向是几十亿 FLOP)。所以**采样策略基本不影响推理速度**,可以放心用复杂策略。唯一例外:超大词表 + 极致优化时,topk 的排序也会被搬上 GPU 做 kernel 优化。

---

## 7. 动手实验:手写采样函数并观察多样性

代码见 `code/sampling.py`,从零实现 `softmax_with_temperature`、`top_k_filter`、`top_p_filter`、`apply_repetition_penalty`,并组装成一个 `sample` 管线。然后用一组固定 logits 反复采样,统计不同策略下的**输出分布/多样性**。

核心管线(完整见文件):

```python
def sample(logits, temperature, top_k, top_p, penalty, prev_tokens):
    logits = apply_repetition_penalty(logits, prev_tokens, penalty)  # ①
    if temperature == 0.0:
        return int(logits.argmax())                                  # greedy
    logits = logits / temperature                                    # ②
    logits = top_k_filter(logits, top_k)                             # ③
    logits = top_p_filter(logits, top_p)                             # ④
    probs = torch.softmax(logits, dim=-1)                            # ⑤
    return int(torch.multinomial(probs, num_samples=1))              # ⑥
```

运行:

```bash
cd code
python sampling.py
```

预期输出(展示同一组 logits 在不同策略下采样 1000 次的去重数/熵):

```
[greedy      ] 1000 次采样 -> 唯一 token 数 1   (完全确定)
[T=0.7       ] 1000 次采样 -> 唯一 token 数 6   熵 1.42
[T=1.0       ] 1000 次采样 -> 唯一 token 数 9   熵 2.01
[top_k=3     ] 1000 次采样 -> 唯一 token 数 3   熵 1.05
[top_p=0.9   ] 1000 次采样 -> 唯一 token 数 5   熵 1.38
[T=1.3+top_p ] 1000 次采样 -> 唯一 token 数 8   熵 1.95
```

**观察重点**:greedy 永远同一个;温度越高、唯一 token 越多、熵越大(越随机);top-k/top-p 在保证不跑偏的前提下保留适度多样性。

### 实验任务
- **必做**:补全 `top_p_filter` 中标注 `TODO` 的「累积概率截断」逻辑,使其通过文件内的单元断言。
- **必做**:跑通主程序,改变 `temperature`(0.3 / 1.0 / 1.8),观察唯一 token 数与熵的单调变化。
- **选做**:接上一个真实模型(GPT-2),用你的 `sample` 替换 `model.generate`,对同一 prompt 用不同策略各生成 3 句,直观感受文风差异。

---

## 练习题

1. temperature 从 1.0 调到 0.1,生成会变得更确定还是更随机?调到 2.0 呢?用 softmax 公式解释。
2. top-k=1 和 greedy 等价吗?top-p 取多少时近似 greedy?
3. 给出一个场景说明「top-p 比 top-k 更合理」,再给一个「固定 top-k 反而更可控」的场景。
4. repetition penalty 对正、负 logit 的处理为什么不同?如果统一「减去一个常数」会有什么问题?
5. 为什么说采样策略几乎不影响 decode 的吞吐?哪种极端情况例外?

<details>
<summary>参考答案(想完再看)</summary>

1. 调到 0.1:logits 被除以 0.1 即放大 10 倍,分布极陡 → 更确定(近 greedy)。调到 2.0:logits 减半,分布更平 → 更随机。本质是 \(e^{z/T}\) 里 T 改变指数差距。
2. top-k=1 即只保留最大 logit 的那个 token,采样必然选它,**与 greedy 等价**。top-p → 0(很小)时只保留概率最高的一个,也近似 greedy。
3. top-p 更合理:当下一个词高度确定(如固定搭配)时,top-p 自适应只留 1~2 个,而 top-k=50 会引入 48 个无关词;top-k 更可控:需要严格限定候选数量(如受控生成、对齐评测要求可复现的候选集)时,固定 k 更可预测。
4. 因为目标是「降低被选概率」。正 logit 除以 \(\rho>1\) 会变小(下压);负 logit 乘以 \(\rho\) 会更负(也下压)。若统一「减常数」,对极大正 logit 影响微弱、对接近 0 的可能直接翻负号,惩罚不均衡且可能扭曲分布。
5. 采样只在 `[V]` 向量上做排序/归一化,计算量相比整个前向(几十亿 FLOP)可忽略,而 decode 瓶颈是读权重的带宽,所以采样不影响吞吐。例外:超大词表 + 极致优化场景下,top-k 排序本身也可能成为可优化的 GPU kernel。

</details>

---

## 小结

- **softmax** 把 logits 变概率;**temperature** 调分布陡缓(确定性↔多样性总旋钮),不改相对排序。
- **greedy**:确定、可复现(调试/迁移对齐黄金标准),但易复读。
- **top-k**:保留前 k 个,挡长尾但 k 固定不自适应;**top-p(nucleus)**:按累积概率动态截断,主流选择。
- **repetition penalty**:在 softmax 前压制已出现 token,注意正负 logit 区别处理。
- 工业管线顺序:**penalty → temperature → top-k → top-p → softmax → 采样**,顺序错则语义错。
- 采样计算量相比前向可忽略,**基本不影响 decode 吞吐**。

## 自测验收(过了再进 Lesson 5)
- [ ] 能用 softmax 公式解释 temperature 的作用,并说明它不改排序。
- [ ] 能讲清 top-k 与 top-p 的区别与各自适用场景。
- [ ] 能默写完整采样管线的步骤顺序,并说明为何不能乱序。
- [ ] `sampling.py` 跑通,补全的 `top_p_filter` 通过断言,观察到熵随温度单调变化。
- [ ] 能解释为什么采样不影响推理速度。

下一课:**Lesson 5 — 从零搭一个迷你推理循环**,把 Transformer + KV Cache + 采样 全部串起来,加载真实小模型,做完整生成与吞吐 benchmark,为 Module 4 引擎篇埋伏笔。
