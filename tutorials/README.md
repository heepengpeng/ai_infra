# AI Infra 推理方向 · 系列教程

这是一套**由浅入深、理论 + 动手**的自学教程,目标:把你从「AI 应用 / 大数据工程师」带到「大模型推理 Infra 工程师」。

> 与 `../ROADMAP.md` 的关系:`ROADMAP.md` 是时间规划(什么时候学),本目录 `tutorials/` 是**教程内容本身**(具体学什么、怎么学)。建议以本目录为主线学习,用 ROADMAP 控节奏,用 `PROGRESS.md` 打勾追踪。

---

## 一、你现在缺什么基础(基于简历的诊断)

你的简历显示:计算机系统结构(研究生)、Python 熟练、大数据系统(Spark/Flink)、AI 应用框架(九问/Agent)、模型迁移与黑盒推理优化(MindOne / MindSpore Lite)。底子很好,但**做推理 Infra 需要"打开黑盒"**,以下是从「会用」到「会造」的关键缺口:

| 缺口 | 你现在的状态 | 推理 Infra 要求 | 对应模块 |
|---|---|---|---|
| **GPU 编程(CUDA)** | ❌ 没写过 | 能手写、调试、优化 CUDA kernel | Module 1、2 |
| **GPU 硬件架构** | △ 有 CPU 体系结构基础,GPU 特性需补 | 懂 SM/warp/显存层级/Tensor Core | Module 1 |
| **现代 C++** | △ 待强化 | 写算子和读引擎源码够用 | Module 0 |
| **DL 底层数学/算子** | △ 会调框架,没手写过 | 能手写 softmax/attention/量化 | Module 2、3 |
| **推理引擎内部原理** | △ 黑盒用过 MindSpore Lite | 懂 KV Cache/PagedAttention/调度 | Module 4、5 |
| **量化原理** | △ 用过,不懂数学 | 懂 GPTQ/AWQ/FP8 原理与实现 | Module 6(量化) |
| **性能分析** | ✅ 有大数据调优经验,可迁移 | Nsight、roofline、瓶颈定位 | 贯穿全程 |

一句话:**核心缺口是 GPU/CUDA 这条主线**,其余围绕它展开。所以教程从 CUDA 起步(C++ 仅做够用铺垫)。

---

## 二、教程总大纲(系列目录)

每一课都遵循统一结构:**学习目标 → 理论讲解(配图解)→ 动手实验(可运行代码)→ 练习题(含答案)→ 小结 → 自测验收**。

### Module 0:现代 C++ 够用基础(`module-00-cpp/`)
> 不求精通,只为能写算子、读源码。4 课。
- L1 指针、引用、内存模型
- L2 模板与函数重载
- L3 RAII、智能指针、move 语义
- L4 编译链接、CMake 入门

### Module 1:CUDA 编程基础(`module-01-cuda-basics/`)
> 重头戏,从"GPU 为什么快"到能写正确的 kernel。8 课。
- L1 GPU 为什么快:CPU vs GPU 架构(纯理论)
- L2 第一个 CUDA 程序:kernel 与编译运行
- L3 线程层级:grid / block / thread / warp 与 2D 索引
- L4 内存模型:host/device、global memory、数据传输
- L5 错误处理与计时:CUDA event、正确 benchmark
- L6 共享内存与线程协作:`__syncthreads`、bank conflict
- L7 经典模式一:并行归约(reduce)
- L8 经典模式二:矩阵转置与访存合并(coalescing)

### Module 2:GPU 性能优化与核心算子(`module-02-kernels/`)
> 写出"快"的 kernel,并理解为什么快。7 课。
- L1 性能模型:roofline
- L2 手写 GEMM(一):tiling 分块
- L3 手写 GEMM(二):寄存器分块、向量化
- L4 Triton 入门:用 Python 写高性能 kernel
- L5 算子融合
- L6 Softmax 与 LayerNorm 的高效实现
- L7 FlashAttention:online softmax 与 IO 感知

### Module 3:Transformer 推理原理(`module-03-transformer/`)
> 打开大模型推理的黑盒。5 课。
- L1 Transformer 结构(推理视角)
- L2 自回归生成:prefill vs decode
- L3 KV Cache:原理、显存计算
- L4 采样策略:greedy/top-k/top-p/温度
- L5 从零搭一个迷你推理循环

### Module 4:推理引擎核心机制(`module-04-engine/`)
> 从迷你引擎到 vLLM 级别。6 课。
- L1 静态 batching 的问题
- L2 Continuous Batching:迭代级调度
- L3 PagedAttention
- L4 精读 nano-vLLM 源码
- L5 投机解码(speculative decoding)
- L6 部署 vLLM 并做 benchmark

### Module 5:量化与模型压缩(`module-05-quant/`)
> 降本核心技术。5 课。
- L1 量化数学基础
- L2 PTQ vs QAT
- L3 GPTQ 原理与实战
- L4 AWQ 与 SmoothQuant
- L5 FP8 与 KV Cache 量化

### Module 6:推理服务化与工程(`module-06-serving/`)
> 从模型到线上服务。4 课。
- L1 推理服务架构
- L2 TensorRT-LLM 部署
- L3 多卡 / 张量并行入门
- L4 监控、压测与容量规划

### Module 7:综合项目与求职(`module-07-capstone/`)
- L1 Capstone:把一个 7B 模型推理优化 X×
- L2 开源贡献指南
- L3 简历重写 + 面试八股精讲

---

## 三、怎么用这套教程

1. **按模块顺序学**,每课先读理论,再动手敲代码(别复制粘贴),最后做练习。
2. **每课都有验收标准**,过了再进下一课,不要囫囵吞枣。
3. 动手代码统一放在各模块的 `code/` 子目录。
4. 没有本地卡时:CUDA 课用 Colab / LeetGPU;大模型课用 AutoDL 按小时租。
5. 用 `PROGRESS.md` 边学边打勾。
