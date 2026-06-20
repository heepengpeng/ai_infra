# AI Infra 转型计划(推理优化方向 · 6 个月)

> 目标:6 个月内具备「大模型推理 Infra 工程师」能力并成功跳槽
> 方向:Inference Infra(推理优化)
> 起点:CUDA 零基础 / C++ 待强化 / 云 GPU
> 已有强项:计算机系统结构(研究生)、SDXL 推理优化 10×、Tachyons 性能调优、大数据系统、九问/Agent 应用框架

---

## 0. 转型逻辑(给自己讲清楚的故事)

你不是转行,而是从 **「AI 应用 + 大数据系统」** 平移到 **「AI 计算底层」**。
面试时的主线叙事:

> "我有计算机系统结构科班 + 大数据极致性能调优(内存/CPU 双降 50%)+ 已经在 MindOne 做过 SDXL 推理 10× 优化(算子融合/量化/内存复用)的经验,现在系统化补齐 GPU/CUDA 与主流推理引擎(vLLM/TensorRT-LLM),把性能优化能力迁移到大模型推理场景。"

最大短板 = **GPU/CUDA 硬功夫**,也是 6 个月里投入产出比最高的地方。

---

## 1. 云 GPU 使用策略(省钱是关键)

- **按需租用,用完即关**,不要包月。学习阶段大部分时间在看资料/写 CPU 可调的代码。
- 平台建议:AutoDL(国内便宜,按小时)、阿里云/腾讯云竞价实例、Lambda/Runpod/Vast.ai(海外)。
- 卡型选择:
  - 入门 CUDA / 写 kernel:**RTX 4090 / 3090**(便宜够用,约 ¥1-3/小时)。
  - 跑 7B 推理 / vLLM:**单张 A100 40G 或 4090 24G**(7B 量化后可跑)。
  - benchmark / 量化:A100 / L20 / L40S 视手头预算。
- 省钱技巧:本地写代码 + 同步到云端跑;一次性把要跑的实验排队批量跑完;用小模型(GPT-2 / Qwen-0.5B / Llama-1B)验证逻辑,大模型只做最终验证。

---

## 2. 六个月总览

| 月 | 主题 | 核心产出(简历项目) |
|---|---|---|
| M1 | 地基:C++ 强化 + CUDA 入门 + GPU 架构 + PyTorch 内部 | 手写 CUDA kernel 集(vec_add/reduce/softmax) |
| M2 | CUDA 进阶 + Triton + 手写高性能算子 | 手写 GEMM / Fused Softmax / LayerNorm,逼近 cuBLAS |
| M3 | Transformer 推理原理 + 从零写迷你推理引擎 | mini-infer:支持 KV Cache + 采样的 GPT/Llama 推理 |
| M4 | 啃透 vLLM(PagedAttention / Continuous Batching) | vLLM 源码笔记 + 部署 benchmark 报告 |
| M5 | 量化压缩 + TensorRT-LLM + 推理服务化 | 量化对比实验(AWQ/GPTQ/FP8)+ TRT-LLM 部署 |
| M6 | 综合优化 capstone + 开源贡献 + 简历/面试冲刺 | 1 个吞吐提升 X× 的优化项目 + 技术博客 + offer |

---

## 3. 分周详细计划(26 周)

### M1 — 地基(W1-W4)

- **W1 环境 + C++ 强化**
  - 打通云 GPU 工作流(SSH、VSCode Remote、conda、CUDA Toolkit、nvidia-smi)。
  - C++ 复习:指针/引用、内存模型、模板、RAII、move 语义(够用即可)。
  - 实战:`01_cuda/00_setup` 跑通第一个 `hello_cuda`。
- **W2 CUDA 编程模型**
  - 线程层级(grid/block/thread/warp)、内存层级(global/shared/register/L2)。
  - 实战:`vec_add`、`matrix_transpose`、理解 coalesced access。
- **W3 GPU 架构**
  - SM 结构、Tensor Core、显存带宽 vs 算力(roofline 模型)、occupancy。
  - 工具:`nsight-compute` / `nsight-systems` 初步 profiling。
  - 实战:`reduce`(从 naive 到 warp shuffle 多版本对比)。
- **W4 PyTorch 内部机制**
  - Tensor 存储、Autograd 计算图、`nn.Module`、CUDA stream、`torch.compile` 入门。
  - 自定义 CUDA 扩展(`torch.utils.cpp_extension`)。
  - 实战:`softmax` CUDA kernel 并封装成 PyTorch 算子,与官方对比精度/速度。

### M2 — 算子优化(W5-W8)

- **W5 共享内存与 Tiling** — tiled `matrix_mul`,理解访存优化。
- **W6 手写 GEMM** — 逐步优化 GEMM,benchmark 对比 cuBLAS,理解差距来源。
- **W7 Triton 入门** — 用 Triton 写 fused softmax / layernorm / dropout,对比手写 CUDA。
- **W8 FlashAttention** — 读懂论文与思想(online softmax、IO 感知),用 Triton 实现简版 flash attention。
- 产出:`02_kernels/` 一套算子 + 性能对比报告(图表)。

### M3 — 推理引擎原理(W9-W12)

- **W9 Transformer 推理全流程** — prefill vs decode、autoregressive、采样(greedy/top-k/top-p/温度)。
- **W10 KV Cache** — 原理、显存占用计算、为什么是推理瓶颈。
- **W11-W12 从零写 mini-infer** — 加载 GPT-2 / Qwen-0.5B 权重,实现带 KV Cache 的推理 + 采样 + 简单 batching。
- 产出:`03_mini_infer/` 可运行的迷你推理引擎 + README 讲清设计。

### M4 — vLLM 深入(W13-W16)

- **W13 vLLM 部署与 benchmark** — OpenAI 兼容 server、吞吐/延迟/首 token 延迟(TTFT)指标。
- **W14 PagedAttention** — 显存分页管理思想,读对应源码模块。
- **W15 Continuous Batching + 调度器** — scheduler、prefill/decode 调度、抢占。
- **W16 源码精读 + 笔记** — 画出 vLLM 请求生命周期图,整理成博客。
- 产出:`04_vllm/` benchmark 脚本 + 源码笔记 + 一篇技术博客。

### M5 — 量化与服务化(W17-W20)

- **W17 量化基础** — INT8/FP16/BF16/FP8、对称/非对称、per-tensor/per-channel、PTQ vs QAT。
- **W18 LLM 量化实战** — GPTQ / AWQ / SmoothQuant,跑通并对比精度-速度-显存。
- **W19 TensorRT-LLM** — 编译 engine、FP8/INT8 部署,与 vLLM 对比。
- **W20 服务化** — Triton Inference Server / 推理服务架构、动态 batching、多副本。
- 产出:`05_quant_serving/` 量化对比表 + 部署方案文档。

### M6 — 冲刺与求职(W21-W26)

- **W21-W22 Capstone 项目** — 选一个模型(如 Qwen-7B),综合运用所学把吞吐/延迟优化 X×,写成完整项目 + 博客。
- **W23 开源贡献** — 给 vLLM / SGLang / TensorRT-LLM 提 PR(文档→bugfix→feature 循序渐进),建立背书。
- **W24 简历重写** — 见第 4 节,把项目对接推理 Infra 叙事。
- **W25 面试八股** — CUDA/GPU、推理引擎、量化、分布式基础(见第 5 节)。
- **W26 模拟面试 + 投递** — 海投 + 内推,复盘迭代。

---

## 4. 简历重写要点(转型后)

把现有经历"翻译"成 Infra 语言:

- **MindOne / SDXL** → 提到 **推理链路优化、算子融合、量化压缩、内存复用、精度对齐**,量化 10× 是亮点,补充吞吐/延迟/显存指标。
- **Tachyons / Spark** → 强调 **内存管理、任务调度、Native 高性能引擎**,体现系统级性能优化方法论(profiling → 定位瓶颈 → 优化 → 验证)。
- **计算机系统结构** → 前置,体现硬件/体系结构底子。
- **新增项目** → mini-infer、kernel 优化集、vLLM benchmark、量化对比、Capstone 都写上,并附 GitHub。
- 关键词覆盖:CUDA / Triton / FlashAttention / KV Cache / PagedAttention / Continuous Batching / GPTQ / AWQ / FP8 / vLLM / TensorRT-LLM / NCCL。

---

## 5. 面试八股清单(M6 重点背)

- **CUDA/GPU**:线程层级、内存层级、warp divergence、bank conflict、coalesced access、occupancy、roofline。
- **推理优化**:KV Cache 显存计算、PagedAttention 解决什么、Continuous Batching 原理、prefill/decode 差异、TTFT/TPOT 指标。
- **量化**:各精度区别、PTQ/QAT、GPTQ vs AWQ 原理、为什么能加速。
- **算子**:FlashAttention 思想、算子融合收益来源、GEMM 优化层次。
- **分布式(了解即可)**:DP/TP/PP、AllReduce、ZeRO、NCCL。
- **系统**:显存/带宽/算力瓶颈判断、CUDA stream、异步与 overlap。

---

## 6. 学习资源

- 课程:CUDA 官方 C++ Programming Guide、《Programming Massively Parallel Processors》(PMPP 圣经)、GPU MODE(原 CUDA MODE)系列。
- 推理:vLLM 官方文档与源码、FlashAttention 论文、PagedAttention 论文、Lilian Weng 博客。
- 实战:Triton 官方 tutorials、TensorRT-LLM examples、llm.c(Karpathy)。
- 社区:GPU MODE Discord、相关公众号/知乎专栏。

---

## 7. 项目目录结构(实战代码)

```
ai_infra/
├── ROADMAP.md              # 本计划
├── 01_cuda/                # M1 CUDA 入门 kernel
├── 02_kernels/             # M2 高性能算子(GEMM/Triton/FlashAttn)
├── 03_mini_infer/          # M3 迷你推理引擎
├── 04_vllm/                # M4 vLLM benchmark 与源码笔记
├── 05_quant_serving/       # M5 量化与服务化
└── 06_capstone/            # M6 综合优化项目
```

---

## 8. 进度追踪

| 周 | 计划 | 状态 | 产出链接/备注 |
|---|---|---|---|
| W1 | 环境 + C++ + hello_cuda | ⬜ | |
| W2 | CUDA 编程模型 + vec_add | ⬜ | |
| ... | ... | ⬜ | |

> 每周末花 30 分钟复盘:本周产出了什么可写进简历的东西?卡在哪?下周调整。
