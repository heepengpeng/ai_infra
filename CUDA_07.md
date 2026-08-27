# CUDA 编程核心（基于官方 CUDA Programming Guide 整理）

这篇按官方 CUDA Programming Guide（v13.3）的 Programming Model 一章重新梳理核心概念，在之前几篇的基础上，补充官方文档里几个之前没讲透、或者已经更新的关键点——尤其是 **GPC**、**Thread Block Cluster**、以及一个全新的 **Tile Programming 模型**。

---

## 1. 异构系统：Host 和 Device 的分工

CUDA 编程模型基于**异构计算系统**：一台机器里同时有 CPU（host）和 GPU（device），各自连着自己的内存（host memory / device memory）。

- 应用程序**永远从 CPU 开始执行**，host 代码负责用 CUDA API 在 host memory 和 device memory 之间拷贝数据、启动 GPU 上的代码执行、等待数据拷贝或 GPU 计算完成
- CPU 和 GPU 可以**同时执行代码**，性能最优的做法通常是让两边都保持忙碌，而不是 CPU 等 GPU 或反过来
- GPU 上执行的代码叫 **device code**，其中被调用来在 GPU 上执行的函数叫 **kernel**（历史遗留命名）。"启动"一个 kernel 意味着在 GPU 上并行启动大量线程去执行这份 kernel 代码

---

## 2. GPU 硬件模型：SM 之上还有 GPC

之前几篇讲到"GPU 是若干 SM 的集合"，官方文档补了一层——**SM 会被组织成更大的组，叫 GPC（Graphics Processing Cluster）**。

```
GPU
 └── GPC（Graphics Processing Cluster）
      └── SM（Streaming Multiprocessor）
           └── 寄存器文件 + Unified Data Cache（承载 Shared Memory 和 L1 Cache）+ 若干功能单元
```

- 每个 SM 内部有一个本地寄存器文件、一个 unified data cache，以及一批负责实际计算的功能单元
- **unified data cache 同时承载了 Shared Memory 和 L1 Cache 这两种角色**——它们共享同一块物理硬件资源，具体怎么分配（多少给 shared memory、多少给 L1）可以在运行时配置。这个细节之前的文章没讲清楚：Shared Memory 不是独立于 L1 Cache 之外的一块单独硬件，而是同一块物理存储的两种不同用法配置
- 不同架构里，这些存储的大小、功能单元的数量都会变化——这也是为什么 CUDA 编程模型特意设计成**跟具体硬件布局解耦**：同一份代码，不需要关心某一代显卡到底怎么物理实现，正确性不受影响

---

## 3. Thread Block 与 Grid：调度关系的精确表述

**核心事实（之前讲过，这里更精确一次）**：一个 kernel 启动时会带着大量线程（经常是百万级），这些线程组织成 **thread block**，多个 thread block 组织成 **grid**。Grid 里所有的 thread block 大小和维度都相同。Thread block 和 grid 都可以是 1 维、2 维或 3 维，方便把线程索引映射到具体的数据结构上。

**调度关系的关键规则**：
- **一个 thread block 的所有线程，一定运行在同一个 SM 上**，这使得 block 内线程可以高效地通信和同步（通过 on-chip shared memory）
- 一个 grid 可能有几百万个 thread block，而执行这个 grid 的 GPU 可能只有几十到几百个 SM——所以**大部分 block 是在排队等待被调度**，不是同时执行
- **block 之间的调度顺序没有任何保证**。这意味着 CUDA 编程模型要求：**不同 thread block 之间不能有数据依赖**——一个线程不应该依赖另一个 block 里线程的结果，也不能跨 block 同步。正因为有这条规则，CUDA 程序才能在只有 1 个 SM 的小 GPU 和有几千个 SM 的大 GPU 上，**用同一份代码正确运行**——block 之间可以并行执行，也可以完全串行执行，程序的正确性不能依赖于具体的调度方式

这条规则是理解"为什么 CUDA 里没有 grid 级别的全局同步原语"的根源——`__syncthreads()` 只能同步一个 block 内的线程，就是因为跨 block 同步这件事在编程模型层面是被禁止假设的。

### Thread Block Cluster（compute capability 9.0+ 的新概念）

这是之前几篇完全没提到的一层，Hopper 架构（compute capability 9.0）开始引入：

- **Cluster 是若干相邻 thread block 的分组**，同样可以是 1/2/3 维
- 指定 cluster 不会改变 grid 的维度或者 block 在 grid 中的索引，只是给相邻的 block **额外加了一层分组关系**
- **一个 cluster 里所有的 thread block，会被调度到同一个 GPC 内**（对应前面提到的"SM 之上还有 GPC"这一层硬件结构）
- 因为 cluster 内的 block 是**同时调度、且在同一个 GPC 内**，所以它们之间可以用 Cooperative Groups 提供的软件接口做**通信和同步**——这是之前"block 之间不能同步"这条规则的一个受限例外
- Cluster 内的线程还可以访问 cluster 内所有 block 的 shared memory，这被称为**distributed shared memory**——相当于把 shared memory 的可见范围从"一个 block"扩大到了"一个 cluster"

**这层概念的意义**：之前"block 之间完全无法通信"的限制，在新架构上通过 cluster 被部分打开了一个口子，为需要跨 block 协作的算法（比如更大规模的矩阵分块）提供了新的硬件支持路径。

---

## 4. Warp 与 SIMT：细节补充

核心机制之前讲过：一个 thread block 内的线程被组织成 32 个一组的 **warp**，以 SIMT（Single-Instruction Multiple-Threads）方式执行。这里补充几个官方文档强调、但之前几篇没讲精确的点：

**Warp lane 编号**：warp 内的每个线程会被分配一个 **warp lane**，编号 0 到 31。Thread block 里的线程按可预测的规则被分配到具体的 warp 和 lane 上。

**Warp divergence 的精确机制**：SIMT 不等于 SIMD。**同一个 warp 里所有线程执行的是同一份代码，但每个线程可以走不同的分支路径**——这是 SIMT 和传统 SIMD 最大的区别（SIMD 只有单一控制流，没有"每个通道各走各的分支"这个概念）。当 warp 内线程分叉走不同分支时，没有走某条分支的线程会被**屏蔽（masked off）**，而走这条分支的线程正常执行；反过来执行另一条分支时，屏蔽状态互换。整个 warp 的吞吐因此受影响——这就是为什么让同一 warp 内线程尽量走相同控制流路径能最大化 GPU 利用率。

**一个重要提醒**：官方文档特别强调，**不应该依赖具体硬件如何调度被屏蔽的 warp lane 这种底层细节去写代码**。SIMT 编程模型的保证只是"warp 内所有线程步调一致地执行代码"，硬件在满足这个模型的前提下，具体怎么优化被屏蔽的 lane 是它自己的事——如果程序违反了这个编程模型的假设（比如依赖某种具体的底层执行顺序），会导致**未定义行为**，且在不同 GPU 硬件上表现可能不同。

**Block size 应该是 32 的倍数**——这条规则是合法但非强制的：block 里线程总数不是 32 的倍数依然能跑，但最后一个 warp 会有一些 lane 全程闲置，导致功能单元利用率和内存访问效率都不理想。

---

## 5. Tile Programming：CUDA 原生的"block 级别编程模型"（重要更新）

这是官方文档里一个之前完全没讲到、但和你之前问的 Triton 高度相关的内容——**CUDA 现在原生支持一种 Tile Programming 模型**，和 SIMT 模型并存。

### 核心思路

在 SIMT 模型里，程序员写的是"每个线程做什么"。**Tile Programming 里，程序员写的是"整个 thread block 做什么"**——描述对多维数据集合（称为 **tile**）的操作，由**编译器自动把这些操作映射到 block 内的具体线程上**。

这和前面讲 Triton 时的思路几乎一模一样：Triton 也是 block 级别编程、编译器负责线程级细节。区别在于，**Tile Programming 现在是 CUDA 语言本身的一部分**，不需要额外的框架。

### 两个核心数据类型：Array 和 Tile

- **Array（数组）**：存储在 device memory 里的多维数据容器，可读可写，有明确的 shape 和数据类型
- **Tile（瓦片）**：只存在于 tile 代码内部、属于单个 block 的多维数据集合。**Tile 是不可变的**——每个操作都产生一个新 tile，而不是修改已有的。Tile 不一定在内存里有实际存储形式，具体存放在寄存器还是 shared memory，由编译器决定。**Tile 的每个维度必须是 2 的幂，且必须在编译期确定**，Tile 不能作为 kernel 参数传递，只能在 tile 代码内部创建和消费

### Tile Space：数据怎么在 Array 和 Tile 之间搬运

把一个 array 按固定的 tile 形状**概念上切分成网格**，就得到了 **tile space**。比如一个形状 (M, N) 的二维数组，如果 load 操作指定 tile 形状是 (tm, tn)，这个数组就被概念上划分成 ⌈M/tm⌉ 行 × ⌈N/tn⌉ 列个 tile。用 tile space 里的索引 (i, j) 去 load，就能取出对应位置的一整块 tile；store 则是反过来，把一个 tile 写回 array 里对应的位置。**如果 tile 超出了数组边界（比如数组大小不是 tile 大小的整数倍），load 可以指定越界部分怎么处理（比如填零）**，store 时越界的写入会被静默丢弃。

### Tile 上的操作

Tile Programming 提供一套内置的 tile 级操作：**逐元素运算、矩阵乘法、沿一个或多个轴的规约（比如求和、求最大值）、形状变换（reshape、transpose）、类型转换**。当两个形状不同的 tile 参与同一个运算时，较小的 tile 会自动广播扩展到较大 tile 的形状。

### 和 SIMT 的关系

**Tile Programming 不是要取代 SIMT**，两者在 CUDA 里并存，一个应用可以同时包含 SIMT kernel 和 tile kernel，两者可以操作同一块 device memory 上的数据，选用哪种模型是**逐 kernel 决定**的：

- SIMT：对单个线程有精细控制，某些算法和优化技巧仍然需要这种细粒度控制
- Tile Programming：更高层的抽象，简化 kernel 开发；因为线程级别的具体决策交给了编译器，**同一份 tile kernel 代码可以在不同 GPU 架构上运行，不需要为每一代硬件重新调参**——这一点和 Triton 的价值主张几乎完全一致

两种模型底层用的是同一套硬件（SM、thread block、grid），也用同一套 device memory 空间——只是编程时的抽象层级不同。

**这个发现对你的学习路径的意义**：之前我们讲 Triton 时说它是"第三方框架，帮你省掉 SIMT 层面的细节"，现在看，CUDA 语言本身已经原生提供了几乎一样的抽象（Tile Programming）。如果你后续要在 CUDA 和 Triton 之间做选型，值得关注官方 Tile Programming 这条路径的成熟度和生态支持情况。

---

## 6. GPU 内存：官方视角的精确层次

### Global Memory：GPU 上的 DRAM

GPU 和 CPU 都各自挂了 DRAM 芯片。从 device 代码的视角看，GPU 挂载的这块 DRAM 叫 **global memory**——之所以叫"global"，是因为 GPU 上所有 SM 都能访问它（不代表系统里其他地方也能访问）。CPU 挂载的 DRAM 则叫 system memory 或 host memory。

**统一虚拟地址空间**：现在所有受支持的系统里，CPU 和 GPU 共用同一个虚拟内存地址空间——每个 GPU 的虚拟地址范围和 CPU、以及系统里其他 GPU 的地址范围都是唯一且不重叠的。给定一个虚拟地址，系统能判断它到底落在 GPU 显存还是系统内存里，多 GPU 系统还能判断具体落在哪块 GPU 显存上。

### On-Chip Memory：寄存器与 Shared Memory 的分配规则

每个 SM 有自己的寄存器文件和 shared memory，这些是 SM 内部的资源，可以被 SM 内正在执行的线程极快地访问。

- **寄存器文件**存的是线程的局部变量，通常由编译器自动分配
- **Shared memory** 可以被同一个 thread block（或 cluster，见前面）内的所有线程访问，用于线程间数据交换
- **调度约束（这条是之前 Occupancy 那篇手算逻辑的官方依据）**：要把一个 thread block 调度到某个 SM 上，"每个线程所需寄存器数 × block 内线程数"必须小于等于这个 SM 可用的寄存器总量。**如果一个 block 所需的寄存器超过了 SM 寄存器文件的大小，这个 kernel 根本无法启动**，必须减少 block 内的线程数才能让它可调度
- **Shared memory 的分配是以整个 thread block 为单位的**，不像寄存器是按线程分配——这也是为什么之前 Occupancy 计算里，shared memory 用量是按"每个 block 用多少"而不是"每个线程用多少"来算的

### 缓存体系

除了程序员可编程的寄存器和 shared memory，GPU 还有硬件自动管理的缓存：
- **每个 SM 有自己的 L1 Cache**，是 unified data cache 的一部分（也就是前面提到的、和 shared memory 共享物理资源的那块存储）
- **L2 Cache 更大，由 GPU 上所有 SM 共享**
- **每个 SM 还有一块独立的 constant cache**，专门用来缓存那些在 kernel 生命周期内声明为常量、存放在 global memory 里的值。编译器也可能把 kernel 参数放进 constant memory——这样 kernel 参数就能被缓存在和 L1 data cache 分开的独立空间里，从而提升性能

### Unified Memory：让 CUDA 自动管理数据放在哪

正常情况下，显式在 GPU 或 CPU 上分配的内存，只能被对应那一侧的代码访问——CPU 内存只能被 CPU 代码访问，GPU 内存只能被 GPU 上跑的 kernel 访问，两者之间需要显式调用 CUDA API 做拷贝。

**Unified Memory** 允许分配一块可以同时被 CPU 或 GPU 访问的内存，CUDA runtime 或底层硬件会在需要的时候自动完成访问授权或数据搬运，不需要手写拷贝逻辑。**但即便用了 unified memory，最优性能依然来自于尽量减少数据迁移、尽量让数据被"直接挂载在其上的那个处理器"访问**——unified memory 解决的是编程便利性问题，不是让你可以完全不关心数据locality。

---

## 小结：这篇更新了什么

对照之前几篇内容，这次基于官方文档整理，主要补充和修正了这几点：

1. **SM 之上还有 GPC 这一层**，Thread Block Cluster（新架构）就是在 GPC 层面调度的
2. **Shared Memory 和 L1 Cache 共享同一块物理硬件（unified data cache）**，不是两块独立存储
3. **Block 之间不能同步不是工程限制，而是编程模型的根本约束**——是为了让同一份代码能在任意规模的 GPU 上正确运行；Cluster 是这条规则在新硬件上的一个受限例外
4. **CUDA 现在原生支持 Tile Programming**，一个和 Triton 思路几乎一致的 block 级别编程模型，官方文档甚至把它和 SIMT 并列为两大编程范式
5. **Occupancy 计算里"寄存器按线程算、shared memory 按 block 算"**这条规则，在官方文档里能找到明确依据

建议下一步：如果你对 Tile Programming 感兴趣，可以直接读官方文档 2.4 节"Writing Tile Kernels"，看它的实际语法和一个具体例子（比如矩阵乘法在 Tile Programming 下怎么写），这样能直接对比它和 Triton、以及手写 SIMT CUDA 三者在同一个任务上的代码形态差异。
