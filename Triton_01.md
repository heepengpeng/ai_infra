# Triton 入门：用 Python 写 GPU Kernel

前面几篇把 CUDA 的硬件模型、内存体系、Roofline 诊断、Occupancy、原生 C++ 编程都讲透了。这篇讲 **Triton**——OpenAI 开源的 GPU kernel 开发框架，现在是 vLLM、FlashAttention-2/3、torch.compile 等主流 LLM 推理框架背后的重要工具之一。

核心问题是：**Triton 到底帮你省了什么，又没有省什么**——这决定了你什么时候该用它、什么时候还得手写 CUDA。

---

## 1. Triton 解决的是什么问题

原生 CUDA 编程里，前面几篇讲过的这些事情都要**手动管理**：
- 手动算 `threadIdx`/`blockIdx` 索引
- 手动决定要不要用 shared memory、什么时候搬数据、什么时候 `__syncthreads()`
- 手动处理 Memory Coalescing（数据怎么排布才能让访问连续）
- 手动处理 Bank Conflict

Triton 的定位：**block 级别的编程模型，把"一个 block 内怎么把线程排布成 warp、怎么用 shared memory、怎么保证访问合并"这些活交给编译器自动做**，程序员只需要用类似 NumPy 的语法描述"这一个 block 要处理哪块数据、做什么运算"。

**关键区别一句话概括**：
- CUDA：你在写"每个线程做什么"（thread 级别）
- Triton：你在写"每个 block 做什么"（block 级别），线程内部怎么分工、内存怎么访问，编译器帮你优化

---

## 2. 一个例子：向量加法，CUDA vs Triton

### CUDA 版（前一篇写过的）

```cuda
__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

### Triton 版

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # 对应 CUDA 里的 blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 这个 block 要处理的一整段索引
    mask = offsets < n  # 对应 CUDA 里的 if (idx < n) 边界检查

    a = tl.load(a_ptr + offsets, mask=mask)  # 一次性加载一整块数据
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# 调用
grid = (triton.cdiv(n, BLOCK_SIZE),)
add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
```

**对照关系一目了然**：
- `tl.program_id(0)` ↔ `blockIdx.x`
- 没有 `threadIdx` 这个概念——Triton 里你操作的是一整块数据（`offsets` 是一个向量），而不是单个线程处理单个元素
- `mask` ↔ CUDA 里手写的边界判断，但 Triton 里 `mask` 直接作为 `load`/`store` 的参数，语义更清晰
- **没有 block 内怎么分线程、要不要用 shared memory这些事**——`tl.load` 这一整块数据怎么从 global memory 搬进来、要不要过一道 shared memory，编译器自动决定并优化

---

## 3. Triton 帮你自动做了前面几篇讲的什么

对照前面讲过的理论，Triton 编译器在背后自动处理的部分：

| 前面文章讲的手动优化 | Triton 里怎么处理 |
|---|---|
| Memory Coalescing（访问要连续对齐） | `tl.load`/`tl.store` 操作的是一个连续的 block，编译器自动生成合并访问的指令 |
| Shared Memory 手动搬运 + `__syncthreads()` | 编译器根据数据复用模式自动决定要不要用 shared memory，何时同步 |
| Bank Conflict | 编译器自动处理 shared memory 内部的访问排布 |
| Warp 内线程怎么分工 | 完全由编译器决定，你写的代码里看不到 warp/thread 这一层 |

**这不代表 Triton 就没有性能可调的空间**——`BLOCK_SIZE` 这类参数你依然要手动设置和调优（对应前面讲的 Occupancy 权衡：block 太大寄存器/shared memory 不够，太小并行度不够），Triton 生态里常见的做法是写一个 **autotune**（自动网格搜索最优 `BLOCK_SIZE` 等超参数）：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n'],
)
@triton.jit
def add_kernel(...):
    ...
```

这一步做的事情，本质上就是前面 Occupancy 那篇讲的"手算三个限制因素找最优 block size"，只是从人工计算变成了自动搜索。

---

## 4. Triton 没有帮你省的部分

**算法层面的设计，Triton 不会替你想**。比如 FlashAttention 的核心创新——用 tiling + 在线 softmax 把 memory-bound 的 attention 变成 compute-bound——这个"怎么分块、什么时候该复用数据、怎么组织计算顺序"的算法设计，依然需要你自己想清楚。Triton 只是让你**实现**这个算法时不用再操心线程级别的琐碎细节，写出来的代码更接近伪代码、更容易维护和迭代。

**Roofline 诊断思维依然是你自己的活**。Triton 写出来的 kernel 一样可以用 Nsight Compute 分析，一样要看 Compute/Memory Throughput 判断瓶颈在哪——前面几篇讲的诊断框架完全不变，变的只是"确定了要往哪个方向优化之后，具体怎么改代码"这一步变简单了。

**Tensor Core 的使用，需要显式调用矩阵乘 API**：
```python
acc = tl.dot(a, b)  # 显式调用矩阵乘，编译器会尽量映射到 Tensor Core
```
精度选择（FP16/BF16/INT8）依然是你要做的决策，Triton 不会替你决定该不该量化。

---

## 5. 什么时候用 Triton，什么时候手写 CUDA

**适合 Triton 的场景**：
- 自定义算子融合（比如把某几个逐元素操作 + reduction 融合成一个 kernel，减少 memory-bound 场景下的显存往返）——这正是 LLM 推理里最常见的优化诉求
- 需要快速迭代实验不同的算法变体（比如 attention 的各种变种），Triton 代码量小、调试快
- 团队里没有大量 CUDA 专家，但需要写自定义高性能算子（Triton 学习曲线远低于原生 CUDA）

**可能还是需要原生 CUDA（或 CUTLASS）的场景**：
- 需要对 shared memory 布局、bank conflict 做极致的手动控制，榨最后几个百分点的性能
- 需要用到 Triton 还没很好支持的硬件特性（比如某些新架构的特殊 Tensor Core 模式）
- 生产级、长期维护、需要跟硬件厂商深度联调的核心 kernel（比如 cuDNN/cuBLAS 这个级别的库本身）

**现实情况**：目前主流做法是**混合使用**——vLLM、FlashAttention-2/3 这些项目里，性能不那么敏感或者需要快速迭代的部分用 Triton 写，最核心、最反复被调用的少数几个 kernel 用手写 CUDA/CUTLASS 做到极致。

---

## 6. 小结

Triton 不是"CUDA 的替代品"，而是**换了一个抽象层级**：把你从 thread 级别的手动管理中解放出来，让你在 block 级别思考问题，同时保留了对关键性能参数（block size、是否用 Tensor Core、数据精度）的控制。

理解 Triton 的前提，恰恰是前面几篇讲的那套 CUDA 硬件模型——**不懂 Occupancy、Memory Coalescing、Roofline 诊断，你依然写不出高性能的 Triton kernel，只是不用再手写 `__syncthreads()` 了**。这也是为什么把 CUDA 基础打透，即使最终工作中主要写 Triton，也是划算的投入。

---

下一步方向建议：
1. 拿一个具体算子（比如 Softmax 或简单的矩阵乘）用 Triton 实际写一遍，对照本文的向量加法例子练手
2. 深入 FlashAttention 的 Triton 官方实现源码，把"tiling + 在线 softmax"这个算法设计和 Triton 代码对应起来读
3. 转向 LLM 推理优化的系统层面：KV Cache 管理、PagedAttention、Continuous Batching——这些是算子之外、推理服务层面的优化技术
