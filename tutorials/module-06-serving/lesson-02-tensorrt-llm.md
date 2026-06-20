# M6 · Lesson 2:TensorRT-LLM 部署——把 GPU 的峰值性能榨干

> 上一课你用 vLLM 思路搭了服务骨架。vLLM 的哲学是「灵活、易用、够快」。但当业务进入「同样的卡要服务更多用户、延迟卡得更死」的阶段,就该认识另一个选手:**TensorRT-LLM**——NVIDIA 官方的推理编译器,用「提前编译 + 深度算子融合」把自家 GPU 榨到极致。本课讲清它的原理、和 vLLM 的取舍,并带你完整编译、部署一个模型。
> 预计用时:3 小时(含一次完整的 engine 编译,GPU 上跑)。
> 前置:Module 5(FP8/INT8 量化原理)、Lesson 1(serving 架构、in-flight batching 即 continuous batching);需要一张 NVIDIA GPU(建议 AutoDL 上 4090 / L20 / A100)。

## 学习目标

学完本课你应该能回答:
1. TensorRT-LLM 和 vLLM 到底差在哪?为什么说一个是「编译器」、一个是「运行时」?
2. 「编译一个 engine」这个流程里到底发生了什么?为什么编译要花几分钟、还绑定具体 GPU 型号?
3. 什么时候该选 TensorRT-LLM,什么时候 vLLM 更香?(工程取舍)
4. FP8 / INT8 在 TRT-LLM 里怎么落地?in-flight batching 是什么?
5. 能独立把一个 Llama / Qwen 模型编成 engine 并起 OpenAI 兼容服务。

---

## 1. 一个类比:解释执行 vs 提前编译

你是搞系统结构的,这个类比你秒懂:

- **vLLM ≈ 解释器 / JIT(PyTorch eager)**:模型权重加载进来,推理时**每一步动态地**调度 kernel。灵活——换模型、改采样、调 batch 都是改配置;但每次都要走一遍框架的调度开销,算子之间的边界没法跨过去优化。
- **TensorRT-LLM ≈ AOT 编译器(像 GCC -O3)**:部署前先把整个模型「编译」成一个针对**特定 GPU、特定精度、特定 shape 范围**高度优化的二进制 **engine**。编译期它做了大量重活:算子融合、kernel 自动选优(autotuning)、内存复用规划、精度降级。运行时几乎没有框架开销,直接执行编译好的计划。

> 核心权衡一句话:**TensorRT-LLM 用「编译期的不灵活 + 一次性的编译成本」,换来「运行期的极致峰值性能」。vLLM 用「一点点运行时开销」,换来「随手就能跑、随手就能换」的灵活。** 这和「C++ 编译型 vs Python 解释型」是同一种取舍哲学。

为什么编译能更快?因为编译器掌握了**全局信息**:它知道整个计算图长什么样,于是能把「LayerNorm → QKV 投影 → RoPE → Attention」这一长串本来要启动很多次 kernel、反复读写显存的操作,**融合成少数几个大 kernel**(回忆 Module 2 的算子融合),还能为你这张具体的卡(SM 数量、显存带宽、是否有 FP8 Tensor Core)挑选实测最快的实现。vLLM 在运行时拿不到这么完整的优化窗口。

---

## 2. TensorRT-LLM 的工作流:三步走

它的使用流程和 vLLM「一行 `LLM(model=...)`」很不一样,是经典的**编译型工具链**:

```
  HuggingFace 权重 (.safetensors)
        │
        │  ① convert / quantize  —— 转成 TRT-LLM 的 checkpoint 格式,可顺带量化(FP8/INT8/INT4)
        ▼
  TRT-LLM checkpoint (统一中间格式 + config.json)
        │
        │  ② trtllm-build  —— 编译!算子融合 + autotuning + 选 kernel + 规划内存
        ▼                     (绑定: GPU 型号 + 精度 + max_batch_size + max_seq_len)
  TensorRT engine (.engine, 二进制 plan)
        │
        │  ③ 部署  —— trtllm-serve / Triton + tensorrtllm_backend,起 OpenAI 兼容服务
        ▼
  线上服务 (HTTP /v1/chat/completions, 支持 in-flight batching + 流式)
```

每一步的关键认知:

- **① 转换**:HF 权重的张量布局、命名和 TRT-LLM 内部不一样,要先转成统一 checkpoint。量化通常在这一步做(把 FP16 权重压成 FP8/INT8,并标定 scale)。
- **② 编译(`trtllm-build`)**:整个工具链的核心,也是最耗时的一步(几分钟到几十分钟)。这里要传一堆「形状契约」参数:`max_batch_size`、`max_input_len`、`max_seq_len`、`max_num_tokens`。**编译器需要预先知道输入规模的上界,才能规划内存、选 kernel。** 这就是它「不灵活」的代价。
- **③ 部署**:engine 本身只是个计算计划,要套一层 runtime(`trtllm-serve` 或 Triton Inference Server + `tensorrtllm_backend`)才能接 HTTP、做批处理调度。

> **重要:engine 是「一次性、强绑定」的产物。** 它绑定了:GPU 型号(在 A100 上编的不能拿到 H100 跑)、TRT-LLM/TensorRT 版本、精度、并行度(TP/PP size)、形状上界。换任何一个,基本都要**重新编译**。这是从 vLLM 切过来时最容易踩的认知坑——你不能再「随便换张卡跑跑」。

---

## 3. in-flight batching:TRT-LLM 版的 continuous batching

Lesson 1 和 Module 4 讲的 continuous batching(迭代级、请求随时进出 batch),在 TensorRT-LLM 的语境里叫 **in-flight batching(IFB)**,有时也叫 inflight fused batching。**它俩是同一个东西的不同叫法**,解决的是同一个问题:别让短请求等着长请求一起结束(静态 batching 的致命伤)。

```
静态 batching(被最长的拖死):
  req A (生成3个就完): ███_____________   ← 完成了还占着 batch 槽位空等
  req B (生成15个):     ███████████████
  时间轴 ──────────────────────────────▶   GPU 后段大量空转

in-flight batching(完成即退、空位即补):
  req A: ███            ← 第3步就退出,释放槽位
  req C:    ██████████  ← 立刻补进来填上 A 的空位
  req B: ███████████████
  时间轴 ──────────────────────────────▶   GPU 持续满载
```

TRT-LLM 的 IFB 配合它自己实现的 **paged KV cache**(对应 vLLM 的 PagedAttention)一起工作:KV Cache 分页管理,序列进出时按页分配/回收,避免碎片。在部署时这通常是默认开启的,你需要关心的是配 `max_num_tokens`(一步里所有序列的 token 总预算)和 KV cache 显存占比。

> 认知对齐:**vLLM 的「continuous batching + PagedAttention」 ≈ TRT-LLM 的「in-flight batching + paged KV cache」。** 两家殊途同归,差别在 TRT-LLM 把这套机制和编译后的融合 kernel 绑得更紧,峰值更高。

---

## 4. FP8 / INT8 部署:把 Module 5 的量化用起来

这是 TRT-LLM 最能打的场景之一,因为它对 NVIDIA 新硬件的低精度支持是第一梯队。回忆 Module 5 的结论:量化减少要搬的字节数,直接加速带宽受限的 decode,还省显存。在 TRT-LLM 里量化是编译流程的一环。

几种常见量化方案的取舍(选给你的卡和精度需求):

| 方案 | 精度损失 | 加速来源 | 适用硬件 | 备注 |
|---|---|---|---|---|
| **FP8** (E4M3) | 很小 | 权重+激活都 8bit,用 FP8 Tensor Core | Hopper(H100)、Ada(L20/4090 部分) | 新硬件首选,精度/速度平衡最好 |
| **INT8 SmoothQuant** | 小 | 权重+激活 INT8 | Ampere(A100)及以上 | Hopper 前的主力方案 |
| **INT4 AWQ / GPTQ** | 中 | 仅权重 4bit(W4A16) | 广泛 | 显存极省,适合大模型挤小卡;decode 快但 prefill 收益小 |
| **FP8 KV cache** | 很小 | KV Cache 用 FP8 存 | Hopper/Ada | 显存翻倍利用,长上下文/高并发关键 |

实操上,量化通过 NVIDIA 的 **ModelOpt(TensorRT Model Optimizer)** 工具在「转换」步完成,产出带 scale 的 checkpoint,再 `trtllm-build`。FP8 几乎不用校准数据(per-tensor scale 简单),INT8 SmoothQuant / INT4 AWQ 需要一小批校准数据集做激活标定——这些你在 Module 5 已经理解了原理,这里只是换了工具落地。

> 实践建议:**有 Hopper/Ada 卡就直接上 FP8**(精度损失通常可忽略,速度和显存双赢);只有 Ampere(A100)就用 INT8 SmoothQuant;**显存实在不够**(比如 4090 24G 跑大模型)再考虑 INT4 AWQ,但要接受 prefill 加速有限、精度损失略大。

---

## 5. 工程取舍:到底选 vLLM 还是 TensorRT-LLM

这是面试高频题,也是真实决策。别背结论,理解维度:

| 维度 | vLLM | TensorRT-LLM |
|---|---|---|
| **上手 / 迭代速度** | 极快,一行起服务,换模型改配置 | 慢,每次要 convert + build,绑定多 |
| **峰值性能(同卡同精度)** | 很好 | **通常更高**(尤其 NVIDIA 新卡 + 低精度) |
| **灵活性** | 高,易换模型/采样/LoRA 热插拔 | 低,改东西常要重编译 |
| **硬件** | NVIDIA / AMD / 其他后端 | **仅 NVIDIA**(自家深度优化) |
| **新模型支持** | 社区快,出得早 | 官方支持,稍滞后但质量高 |
| **运维复杂度** | 低 | 高(engine 管理、版本/卡型绑定) |
| **典型场景** | 快速上线、多模型、研究、中小规模 | 单一主力模型、大规模、延迟/成本极致敏感、全 NVIDIA |

决策心法:

> - **业务还在快速变、模型经常换、团队小、要快速验证** → **vLLM**。灵活性的价值远大于那点峰值差距。
> - **模型已经稳定、要在固定 NVIDIA 卡上把每一分钱榨出来、规模大、SLA 严** → **TensorRT-LLM**。多出来的峰值性能能直接换成「少买卡」。
> - 很多成熟团队是**两者并用**:研发/灰度用 vLLM 快速迭代,主力流量稳定后切 TRT-LLM 压成本。这和你在大数据里「探索用 Spark SQL、稳定后固化成优化作业」的思路一致。

注意:vLLM 也在持续逼近 TRT-LLM 的峰值(它内部也有 CUDA graph、各种融合 kernel),差距在缩小。所以**别把「TRT-LLM 一定更快」当教条**,具体场景一定要用 Lesson 4 的方法**实测**(同卡、同精度、同数据集压一遍看 P99 和吞吐),用数据决策。

---

## 6. 动手实验:把一个模型编成 engine 并起服务

> 强烈建议在 AutoDL 上租一张 Ada/Hopper 卡(L20 / 4090 / A100)。TensorRT-LLM 对环境(CUDA / driver / TensorRT 版本)敏感,**最省心的是用 NVIDIA 官方 NGC 容器**,别在裸机硬装。

下面以 Qwen2.5 或 Llama 系列的一个小模型(如 1.5B / 7B)为例,代码与脚本在 `code/`。流程对所有 decoder-only 模型基本一致。

### 6.0 环境准备(用官方容器最稳)

```bash
# 方式一(推荐):pip 安装(需匹配的 CUDA 环境)
pip install tensorrt-llm -U
trtllm-build --help   # 验证安装

# 方式二:NGC 容器(环境最干净,AutoDL 若支持 docker 时用)
# docker run --gpus all -it nvcr.io/nvidia/tensorrt-llm/release:<tag>
```

下载模型权重(用国内镜像加速):

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B
```

### 6.1 路线 A:高阶 API(最简单,先跑通)

新版 TensorRT-LLM 提供了 `LLM` 高阶 API,把 convert/build 都藏在背后,接口和 vLLM 几乎一样——**先用它跑通,建立信心**。见 `code/trtllm_quickstart.py`:

```python
# 摘自 code/trtllm_quickstart.py
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="./Qwen2.5-1.5B")  # 首次会自动构建 engine 并缓存
params = SamplingParams(temperature=0.7, max_tokens=128)
for out in llm.generate(["用一句话解释什么是张量并行。"], params):
    logger.info("生成: %s", out.outputs[0].text)
```

```bash
cd code
python trtllm_quickstart.py
```

### 6.2 路线 B:显式三步编译(理解全流程,生产用)

这才是你要真正掌握的。脚本 `code/build_engine.sh` 把三步串起来,**注意观察每一步的产物和耗时**:

```bash
# ① 转换 HF 权重 → TRT-LLM checkpoint(此处用 FP16;量化见下)
python convert_checkpoint.py \
    --model_dir ./Qwen2.5-1.5B \
    --output_dir ./ckpt_fp16 \
    --dtype float16

# ② 编译 engine —— 最耗时的一步,留意它在做 autotuning
trtllm-build \
    --checkpoint_dir ./ckpt_fp16 \
    --output_dir ./engine_fp16 \
    --gemm_plugin float16 \
    --max_batch_size 16 \
    --max_input_len 2048 \
    --max_seq_len 4096      # 形状上界:编译期必须知道,运行期不能超

# ③ 起 OpenAI 兼容服务
trtllm-serve ./engine_fp16 --tokenizer ./Qwen2.5-1.5B --port 8000
```

> `convert_checkpoint.py` 是 TRT-LLM 官方按模型族提供的脚本(在其 `examples/<model>/` 目录下),不同模型族脚本不同。本课 `code/` 里给的是**调用封装与参数说明**,真正的 convert 脚本请从你安装的 TRT-LLM examples 里取,文件头注释有指引。

### 6.3 FP8 量化编译(有 Hopper/Ada 卡时做)

把转换步换成 ModelOpt 量化,体会显存和吞吐的变化:

```bash
# 用 ModelOpt 做 FP8 量化转换(产出带 scale 的 checkpoint)
python quantize.py \
    --model_dir ./Qwen2.5-1.5B \
    --output_dir ./ckpt_fp8 \
    --qformat fp8 \
    --kv_cache_dtype fp8       # 顺便把 KV cache 也压成 FP8

trtllm-build --checkpoint_dir ./ckpt_fp8 --output_dir ./engine_fp8 \
    --gemm_plugin auto --max_batch_size 16 --max_seq_len 4096
```

### 6.4 验证服务(与 OpenAI SDK 兼容)

服务起来后,用标准 OpenAI 客户端访问(`code/test_client.py`),**验证 TRT-LLM 和 vLLM 在 API 层是无缝替换的**:

```bash
python test_client.py --port 8000 --stream
```

### 实验任务(必做)

1. 跑通路线 A,确认能生成。
2. 跑通路线 B 的三步,**记录第 ② 步编译耗时**,并打开产物目录看 `.engine` 文件多大。
3. 故意把请求的 `max_tokens` 设到超过编译时的 `max_seq_len`,观察报错——亲身体会「编译期形状契约」的硬约束。
4. (有合适卡时)编一个 FP8 engine,用 Lesson 4 的压测方法对比 FP16 vs FP8 的吞吐和显存。

### 实验任务(选做,留 TODO)

`code/test_client.py` 里留了一个 TODO:实现一个**简易 A/B 对比**——同一批 prompt 分别打到 vLLM 服务(你 Module 4 部署的)和 TRT-LLM 服务,统计两者的 TTFT p99 和 token 吞吐,输出一张对比表。这就是 §5 说的「用数据决策」。

---

## 练习题

1. 为什么 TensorRT-LLM 的 engine 换张卡(A100 编的拿到 H100)就基本要重编译?从「编译器优化绑定硬件」的角度解释。
2. `trtllm-build` 为什么必须传 `max_batch_size` / `max_seq_len`?如果运行时请求超过这个上界会怎样?
3. in-flight batching 和 Module 4 学的 continuous batching 是什么关系?
4. 你的业务:一个 7B 模型,已稳定半年,全 H100 集群,日请求量大、老板要砍 30% 推理成本。你会选 vLLM 还是 TRT-LLM?为什么?还会配哪种量化?
5. 有人说「TRT-LLM 永远比 vLLM 快」,这句话错在哪?你会怎么验证某个具体场景下谁更快?

<details>
<summary>参考答案</summary>

1. 因为编译期做了**绑定硬件的优化**:autotuning 时它针对该 GPU 的 SM 数量、显存带宽、Tensor Core 类型(是否支持 FP8)实测挑选了最快的 kernel 和内存布局。换架构后这些最优选择不再成立,kernel 甚至可能用到目标卡没有的指令,所以要重编译重新 autotune。
2. 编译器要**预先规划显存**(KV cache、activation buffer)和**选择适配该规模的 kernel**,这些都需要 shape 上界。运行时超过上界,请求会被拒绝/报错(输入超 `max_input_len` 或总长超 `max_seq_len`),因为没有为更大的 shape 编译过对应计划。
3. **同一个东西的不同命名**。都是迭代级调度:请求随时进出 batch、完成即退、空位即补,解决静态 batching 被最长请求拖死的问题。TRT-LLM 叫 in-flight batching,vLLM 叫 continuous batching,都配套分页 KV cache。
4. 选 **TensorRT-LLM**:模型已稳定(不灵活的代价不痛)、全 NVIDIA H100(TRT-LLM 主场)、规模大且要砍成本(峰值性能直接换成少买卡)。量化用 **FP8**(H100 有 FP8 Tensor Core,精度损失极小,速度+显存双赢),并开 **FP8 KV cache** 提高并发。这正是 TRT-LLM 最能打的场景。
5. 错在「永远」「绝对」。vLLM 也在用 CUDA graph、融合 kernel 持续逼近,差距随版本缩小;而且结果高度依赖模型、精度、序列长度分布、batch 配置。**正确做法**:在同一张卡、同精度、同一份真实流量分布(prompt/输出长度分布)下,用 Lesson 4 的压测脚本各压一遍,对比相同 SLA(如 P99 TTFT<2s)下谁的吞吐更高 / 谁需要的卡更少,用数据决策。

</details>

---

## 小结

- TensorRT-LLM 是 NVIDIA 的**推理编译器**(AOT),vLLM 是**灵活的运行时**;前者用编译期的不灵活换运行期的极致峰值,后者用一点运行时开销换灵活。
- 工作流三步:**convert(可量化)→ trtllm-build(编译,最耗时)→ 部署(trtllm-serve / Triton)**。engine 强绑定 **GPU 型号 + 精度 + 形状上界 + 版本**,换任一项要重编译。
- **in-flight batching = continuous batching**,配 paged KV cache,机制同源、TRT-LLM 集成更紧。
- 量化:**Hopper/Ada 上 FP8 首选**,Ampere 用 INT8 SmoothQuant,显存紧张用 INT4 AWQ;还可开 FP8 KV cache 提并发。
- 选型不是教条:**模型稳定+全 NVIDIA+极致成本 → TRT-LLM;快速迭代+多模型 → vLLM;成熟团队常两者并用**。永远用 Lesson 4 的实测数据决策,别迷信「谁一定更快」。

## 自测验收(过了再进 Lesson 3)
- [ ] 能用「编译型 vs 解释型」向人解释 TRT-LLM 和 vLLM 的本质区别。
- [ ] 能默写出 convert → build → serve 三步,说清每步的产物和 build 为什么慢。
- [ ] 能说清 engine 为什么绑定硬件和形状,以及超出形状上界会怎样。
- [ ] 能在 5 个维度上对比 vLLM / TRT-LLM,并给出一个真实场景的选型理由。
- [ ] 在 GPU 上跑通了路线 A,或完整跑通了路线 B 的三步编译+部署。

下一课:**Lesson 3 — 多卡与张量并行入门**,当一个模型一张卡放不下,我们就要把它「切开」放到多张卡上协同推理,理解张量并行、NCCL AllReduce 和通信开销。
