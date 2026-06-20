# M7 · Lesson 1:Capstone 项目——把一个 7B 模型推理优化 X×

> 这是整个系列的"毕业设计"。前面 M0–M6 你把零件一个个造好了:CUDA kernel、FlashAttention、KV Cache、Continuous Batching、PagedAttention、量化、服务化。本课把它们**拼成一件能拿去面试、能写进简历、能发成博客的作品**。
> 一句话目标:**选一个 7B 模型,把它的推理性能优化 X×(≥2× 吞吐 或 ≥30% 延迟),全程有数据、有图、有结论。**
> 预计用时:10–20 小时(2 周内分摊完成),这是一个项目而不是一节课。
> 前置:M4(Continuous Batching/PagedAttention)、M5(量化)、M6(服务化与压测)。没学完也能做,但理解会浅。

## 学习目标

学完(做完)本课你应该能:
1. 独立设计并执行一个完整的推理优化项目:**选型 → baseline 测量 → 多手段优化 → 对比 → 复盘成文**。
2. 用**科学的方法**做性能优化:先量化现状、再改一个变量、再测量、再归因,而不是"凭感觉调参"。
3. 产出一套**可被陌生人复现**的成果物:GitHub repo + README + benchmark 图 + 一篇技术博客。
4. 把这个项目讲清楚(为 Lesson 3 的面试做准备)。

---

## 1. 为什么必须做这个项目

先说一个扎心的事实:**面试官不信"我学过"的清单,只信"我做过"的证据。**

你的简历已经有很硬的料(SDXL 推理优化 10×、Tachyons 引擎内存/CPU 双降 50%、九问编排引擎),但这些都是**别的方向**(扩散模型、大数据、应用框架)。招推理 Infra 的面试官会问一个尖锐的问题:

> "你做过这么多优化,但你**对 LLM 推理本身**做过什么?KV Cache、PagedAttention、连续批处理你**自己动手**搞过吗?"

这个 Capstone 就是为了回答这一问。它的价值不在于优化了多少倍(虽然倍数好看更好),而在于**它证明你能独立打开 LLM 推理的黑盒、定位瓶颈、用对的技术解决、并量化收益**——这正是推理 Infra 工程师的核心动作。

> **记住:Capstone 的产出物不是"代码",是"一个可信的故事 + 支撑它的数据"。** 面试时你讲的是故事,GitHub 是它的证据链。

---

## 2. 项目选型:选什么模型、优化什么、在什么卡上

选型阶段最容易犯的错是**贪大求全**。原则:**小而完整 > 大而烂尾。**

### 2.1 选模型(选一个 7B 级别的)

| 候选 | 优点 | 适合理由 |
|---|---|---|
| **Qwen2.5-7B-Instruct** | 中文友好、社区活跃、vLLM 一等支持 | 首选,中文岗位面试加分 |
| Llama-3.1-8B-Instruct | 英文生态最全、对标 benchmark 多 | 想对齐国际论文数据时用 |
| Mistral-7B-Instruct | 结构经典、GQA、资料多 | 想顺带讲 GQA 时用 |

> 建议直接用 **Qwen2.5-7B-Instruct**:中文岗位面试时,面试官自己也常用它,有共同语言。

### 2.2 选卡(按预算)

| 卡 | 显存 | 来源 | 备注 |
|---|---|---|---|
| RTX 4090 | 24GB | AutoDL 按小时租(约 ¥2/h) | **首选**,FP16 7B 刚好放下,跑 batching 够用 |
| A100 40G/80G | 40/80GB | AutoDL/AutoPanel | 想测大 batch、长序列时租 |
| L20 / L40S | 48GB | 云厂商 | FP8 友好(后面量化能用上) |

> 一台 4090 + 几十块钱,足够把整个 Capstone 跑完。**不要因为没有 8 卡就不开始。** 单卡优化的故事一样完整。

### 2.3 定优化目标(写下来,贴在 README 顶部)

目标必须是**可测量的数字**,二选一(能两个都达成最好):
- **吞吐目标**:整体吞吐(tokens/s,所有并发请求加总)**提升 ≥ 2×**。
- **延迟目标**:固定并发下,**P99 端到端延迟降低 ≥ 30%**,或 **TTFT(首 token 时间)降低 ≥ 30%**。

> 关键纪律:**先定目标,再做优化**。否则你会陷入"调到哪算哪"的无底洞,也写不出有说服力的结论。

---

## 3. 推理性能的"考试科目":你要测哪些指标

优化前先认清"分数怎么算"。LLM 推理服务的核心指标(M6 学过,这里作为项目语言统一一下):

| 指标 | 含义 | 谁关心 |
|---|---|---|
| **Throughput(吞吐)** | 每秒生成的总 token 数(含所有并发) | 平台/成本(每 token 成本) |
| **TTFT** | Time To First Token,首 token 延迟 | 用户体感(等待感) |
| **TPOT / ITL** | 每个输出 token 的平均时间(token 间延迟) | 用户体感(流式顺滑度) |
| **P50/P99 Latency** | 端到端延迟中位数/尾延迟 | SLA |
| **显存占用** | 峰值显存、KV Cache 可容纳的并发数 | 部署密度/成本 |

> 一个核心权衡你必须在报告里讲清:**吞吐和延迟通常是此消彼长的**。加大 batch → 吞吐上升,但单请求延迟上升。优化的艺术是**在满足延迟 SLA 的前提下,把吞吐推到最高**。这正是你 Tachyons 引擎做"内存/CPU 双降"时熟悉的多目标权衡思路,可迁移。

---

## 4. 项目五阶段法(主线流程)

整个 Capstone 就是这条流水线,**每一步都要留下数据**:

```
①  选型 & 环境       ②  Baseline 测量        ③  逐项优化(改一个测一个)
  定模型/卡/目标   →   HF 朴素推理 跑分    →   量化 → batching → 引擎 → kernel
                                                      ↓
⑤  复盘成文        ④  优化后对比测量
  README+博客   ←   同一套压测脚本重测   ←   (每项单独记一次增量收益)
```

**铁律:控制变量。** 每次只改一个东西,重测一次,记录这一项带来的增量收益。否则你最后只知道"快了 2 倍",却说不清"哪一项贡献了多少"——而面试官恰恰爱问这个。

---

## 5. 阶段②:Baseline——先把"慢"测准

Baseline 是整个项目的标尺。**baseline 不准,后面所有倍数都是假的。**

### 5.1 用 HuggingFace 朴素推理做 baseline

故意用最朴素、最没优化的方式(transformers `generate`,逐请求串行),它代表"没学过 Infra 的人会怎么跑模型"——这正是你要超越的对象。

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

prompt = "请用三句话解释什么是 KV Cache。"
inputs = tok(prompt, return_tensors="pt").to("cuda")

# 预热(第一次有 CUDA 初始化/编译开销,不计入)
_ = model.generate(**inputs, max_new_tokens=8)
torch.cuda.synchronize()

t0 = time.perf_counter()
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

n_new = out.shape[1] - inputs["input_ids"].shape[1]
print(f"生成 {n_new} tokens 用时 {dt:.2f}s,吞吐 {n_new/dt:.1f} tok/s")
```

> 测量纪律(M1 Lesson 5 的 benchmark 规范在这里复用):**必须预热、必须 `torch.cuda.synchronize()`、必须多次取中位数。** 不 sync 你测的是"发指令的时间"不是"算完的时间",这是新手最常见的假数据来源。

### 5.2 baseline 也要测"并发吞吐"

单请求只能体现延迟。要体现吞吐,得并发打。先用一个简单的并发脚本(或直接用下文的 vLLM `benchmark_serving`),记录:并发 1/8/32/64 下的总吞吐与 P99 延迟。

把 baseline 数据存成一张表,这是你的"考前成绩":

| 并发 | 吞吐 tok/s | TTFT(ms) | P99 延迟(s) | 峰值显存 |
|---|---|---|---|---|
| 1 | … | … | … | … |
| 32 | … | … | … | … |

---

## 6. 阶段③:优化——四个递进的武器(逐个上,逐个测)

按"性价比 + 学习价值"排序,**建议全做,但每做一项就重测一次并记录增量**。

### 武器一:换引擎(vLLM)——一步拿到 Continuous Batching + PagedAttention

这是收益最大、最快见效的一步。把朴素 HF 换成 vLLM,你几乎免费获得了 M4 学的两大杀器:**连续批处理**和 **PagedAttention**。

```bash
pip install vllm
# 起服务(OpenAI 兼容接口)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

压测(vLLM 自带脚本,数据可信、可复现):

```bash
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset-name random --num-prompts 200 \
    --request-rate 16
```

> **预期**:相比 HF 朴素串行,并发吞吐通常能直接涨 **5–20×**(主要来自 continuous batching 把 GPU 喂饱)。这一步你已经大概率达成 ≥2× 目标——但**别停**,后面的优化才是体现你深度的地方。

### 武器二:量化(AWQ / GPTQ / FP8)——降显存、提吞吐

M5 学的量化在这里变现。量化把权重从 FP16 压到 INT4/FP8,带来两个收益:
- **显存↓**:7B 从 ~14GB(FP16)降到 ~4–5GB(INT4),省出的显存全给 KV Cache → **能塞更多并发请求** → 吞吐↑。
- **decode 加速**:decode 是带宽受限(M1 埋的伏笔),读更少字节 = 更快。

```bash
# 用现成的 AWQ 量化权重(社区有 Qwen2.5-7B-Instruct-AWQ)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq --dtype float16 \
    --gpu-memory-utilization 0.9
```

> 报告里要讲清**权衡**:量化几乎必然带来微小精度损失。所以你要**同时测一个精度指标**(下文 6.5),证明"我提速的同时,质量没崩"。"只报速度不报精度"是外行,"速度+精度一起报"才是 Infra 工程师。

### 武器三:调度与并行参数调优——把卡榨干

不写代码,纯调参,但要懂每个参数背后的原理(面试爱问):

| 参数 | 作用 | 调优方向 |
|---|---|---|
| `--max-num-seqs` | 最大并发序列数 | 调大 → 吞吐↑ 但延迟↑,找 SLA 内最大值 |
| `--gpu-memory-utilization` | 显存用于 KV Cache 的比例 | 调高(0.9~0.95)→ 更多 KV 块 → 更多并发 |
| `--max-num-batched-tokens` | 单次迭代最大 token 数 | 影响 prefill/decode 平衡 |
| `--enable-chunked-prefill` | 长 prompt 分块,避免阻塞 decode | 长输入场景开,降 TTFT 抖动 |
| `--tensor-parallel-size` | 张量并行(多卡时) | 单请求延迟↓,需多卡 |

> 做一组**参数扫描实验**:固定其他、单独扫 `--max-num-seqs`(如 8/32/64/128),画一条"吞吐 vs 延迟"曲线,找到拐点。这张图是博客里最专业的一张。

### 武器四(进阶,加分项):自定义 kernel / 投机解码 / KV Cache 量化

如果还有时间和体力,挑一个做深,这是和别的候选人**拉开差距**的部分:
- **投机解码(speculative decoding)**:vLLM 支持 `--speculative-model`,用小模型起草,大模型验证,decode 提速。
- **KV Cache 量化**:`--kv-cache-dtype fp8`,长上下文场景省显存。
- **Triton kernel**:如果你 M2 的 Triton 学扎实了,可以替换某个算子(如 RMSNorm/采样)并 profile 对比——哪怕只优化一个小算子,"我手写过 kernel 并 profile 验证收益"这句话在面试里价值极高。

### 6.5 别忘了测精度(质量护栏)

速度优化不能以质量崩塌为代价。挑一个轻量评测当护栏:
- **困惑度(PPL)**:在一小段中文文本上算,量化前后对比,涨幅 < 1% 一般可接受。
- **任务正确率**:跑一个小子集(如 100 道 CEval/GSM8K 题),对比量化前后准确率。

把"速度 vs 质量"做成一行结论:**"AWQ 量化后吞吐 +40%,显存 -60%,CEval 子集准确率从 72% 微降到 71%(可接受)。"** ——这一句顶一百句口号。

---

## 7. 阶段④⑤:对比成图 + 复盘成文

### 7.1 必出的几张图(博客/README 的灵魂)

用同一套压测脚本、同一组 prompt 重测,画图(matplotlib 即可):

1. **吞吐对比柱状图**:HF baseline → +vLLM → +量化 → +调参,一根根往上长,直观看到"X×"。
2. **吞吐-延迟权衡曲线**:横轴并发/吞吐,纵轴 P99 延迟,标出 SLA 线。
3. **显存对比**:量化前后峰值显存 + 可容纳并发数。
4. **(可选)精度对比**:量化前后 PPL/准确率。

> 这几张图就是你简历项目栏里那行字的**证据**。面试官扫一眼图,信任度立刻不一样。

### 7.2 优化收益归因表(最有说服力的一张表)

| 优化项 | 吞吐(tok/s) | 相对 baseline | 本项增量 | 代价 |
|---|---|---|---|---|
| HF baseline | 100 | 1.0× | — | — |
| + vLLM(CB+PagedAttn) | 850 | 8.5× | +750 | 无 |
| + AWQ 量化 | 1180 | 11.8× | +330 | 精度 -1% |
| + 调度调参 | 1400 | 14.0× | +220 | 延迟 +15% |

> (数字是示意,以你实测为准。)这张表回答了面试官最爱的追问:**"每一项分别贡献了多少?代价是什么?"** 能答出来,说明你是真做过、真理解。

---

## 8. 动手任务

### 任务 A:跑通 baseline(必做)
租一台 4090,用第 5 节脚本跑出 HF 朴素推理的单请求吞吐 + 并发吞吐表。**截图保存,这是你的起点。**

### 任务 B:四件武器逐个上(必做)
按第 6 节顺序:vLLM → 量化 → 调参,**每做一项重测一次**,填进第 7.2 节的归因表。至少达成 ≥2× 吞吐目标。

### 任务 C:成图 + 写博客(必做,最重要)
按下面模板产出**全部成果物**。记住:**没写成文章的优化,等于没做。**

---

## 9. 产出物清单与模板(逐项交付)

### 9.1 产出物 Checklist

- [ ] **GitHub 公开 repo**:含全部脚本、配置、压测结果(原始数据)。
- [ ] **README.md**:项目目标 + 结论(X× 摆在最上面)+ 复现步骤 + 图。
- [ ] **benchmark 图**:至少"吞吐对比柱状图"+"吞吐-延迟曲线"两张。
- [ ] **优化归因表**:第 7.2 节那张。
- [ ] **一篇技术博客**:发知乎/掘金/个人站/Medium,讲清"为什么慢、怎么优化、收益与权衡"。
- [ ] **(加分)精度护栏数据**:量化前后质量对比。

### 9.2 GitHub repo 结构模板

```
qwen2.5-7b-inference-optimization/
├── README.md              # 结论先行:X× + 一张主图 + 复现步骤
├── requirements.txt
├── benchmarks/
│   ├── bench_hf.py        # baseline
│   ├── bench_vllm.sh      # vLLM 压测命令
│   └── results/           # 原始 json/csv(可复现的证据)
├── analysis/
│   ├── plot.py            # 出图脚本
│   └── figures/           # 生成的 png
├── eval/
│   └── eval_quality.py    # 精度护栏
└── docs/
    └── blog.md            # 博客原文
```

### 9.3 博客结构模板(直接套用)

```
标题:把 Qwen2.5-7B 推理吞吐优化 14×:从 HuggingFace 到 vLLM 的实战复盘

1. 背景与目标:为什么要优化、目标是什么(X×)、在什么卡上
2. 怎么测才算数:指标定义 + benchmark 方法(预热/sync/中位数)
3. Baseline:HF 朴素推理有多慢,瓶颈在哪(GPU 利用率截图)
4. 优化一:换 vLLM —— Continuous Batching 与 PagedAttention 原理 + 收益
5. 优化二:AWQ 量化 —— 原理(为什么 decode 是带宽受限)+ 速度/精度权衡
6. 优化三:调度调参 —— 吞吐-延迟曲线,如何在 SLA 内找最优
7. 收益归因表 + 主图
8. 踩过的坑 & 没做完的(诚实加分:speculative decoding 留作后续)
9. 总结:学到了什么
```

> 博客写作的黄金法则:**结论先行,数据说话,讲清权衡,诚实坦白局限。** 一篇这样的博客,比简历上十行"精通 XXX"都管用。

---

## 练习题

1. 你测出 vLLM 比 HF 快了 8×,但单请求延迟反而变高了一点。这正常吗?怎么解释?
2. 量化后吞吐只涨了 5%,远低于预期。可能的原因有哪些?怎么排查?
3. 面试官问:"你这个 14× 里,如果只准你保留一项优化,你留哪个?为什么?"

<details>
<summary>参考答案(做完项目再看)</summary>

1. **正常**。8× 主要来自 continuous batching——它靠"把多个请求攒成大 batch 喂满 GPU"提升整体吞吐,但单个请求要和别人共享算力、还可能排队,所以单请求延迟可能略升。这正是"吞吐 vs 延迟"权衡的体现。报告里要把这点讲透。

2. 可能原因:① 你的瓶颈不在显存/带宽,而在别处(如并发不够,GPU 没喂满,量化省的带宽没转化成吞吐);② batch 还很小,decode 带宽优势没放大;③ 量化 kernel 实现没走到优化路径(dtype/后端配置不对);④ 测量没控制变量,被其他波动淹没。排查:看 GPU 利用率、加大并发再测、确认量化 kernel 真生效。

3. **留 vLLM(continuous batching + PagedAttention)**。因为它贡献了绝大部分倍数(8.5×),且零精度代价、零额外硬件成本。这个回答展示了你懂"性价比"和"工程权衡",而不是只迷恋花哨技术。

</details>

---

## 小结

- Capstone 的本质是**用一个可复现的项目,证明你能独立优化 LLM 推理**——这是简历清单替代不了的硬证据。
- 方法论是**五阶段法 + 控制变量**:选型 → baseline → 逐项优化(每项单测)→ 对比 → 成文。
- 四件武器按性价比排序:**换 vLLM(最大收益)→ 量化 → 调度调参 → 自定义 kernel/投机解码(加分)**。
- **速度和精度一起报、吞吐和延迟讲权衡、每项收益做归因**——这三点区分"外行调参"和"Infra 工程师"。
- 产出物=**GitHub repo + README + benchmark 图 + 博客**,缺一不可。**没写成文的优化等于没做。**

## 自测验收(达成才算完成 Capstone)
- [ ] 有一份可信的 baseline 数据(预热、sync、多次中位数)。
- [ ] 至少达成 ≥2× 吞吐 或 ≥30% 延迟 目标。
- [ ] 有一张"优化收益归因表",能说清每一项贡献多少、代价是什么。
- [ ] 量化项附带了精度护栏数据(速度+质量一起报)。
- [ ] GitHub repo 别人能照 README 复现。
- [ ] 写出并发布了一篇技术博客。
- [ ] 能用 3 分钟把这个项目对人讲清楚(为 Lesson 3 面试做准备)。

---

下一课:**Lesson 2 — 开源贡献指南**。Capstone 证明你"能独立做事",而向 vLLM/SGLang 提一个被合入的 PR,则证明你"能在世界级工程团队的标准下做事"——这是另一种、甚至更强的背书。我们下一课就讲怎么两周内提交第一个 PR。
