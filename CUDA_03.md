# CUDA 内存（Memory）体系详解

内存这条线要回答的核心问题是：**数据怎么从"很远很慢"的地方，一路搬到"很近很快"的地方，让计算单元不用干等**。

这篇文档按数据实际流动的顺序展开：从磁盘开始，经过主机内存、PCIe，进入 GPU 显存，再到 SM 内部的缓存体系，最后抵达寄存器和计算单元。

---

## 0. 完整数据搬运链路总览

```
磁盘（Disk/SSD）
   ↓ 文件系统读取
主机内存（Host RAM / Pageable Memory）
   ↓ PCIe / NVLink 传输
GPU 显存（Device Global Memory）
   ↓ Memory Coalescing 访问
L2 Cache
   ↓
Shared Memory / L1 Cache
   ↓
Register
   ↓
计算单元（CUDA Core / Tensor Core）
```

前半段（磁盘 → 主机内存 → GPU 显存）是"跨设备搬运"，后半段（GPU 显存 → … → 计算单元）是"设备内搬运"。这两段的瓶颈性质完全不同，分别对应不同的优化手段，下面逐段展开。

---

## 1. 磁盘 → 主机内存（Host RAM）

数据首先从磁盘（HDD / SSD / NVMe）读入 CPU 侧的内存，这一步是操作系统层面的常规文件 I/O，不涉及 CUDA，但速度差异很大：

| 存储介质 | 大致带宽 |
|---|---|
| 机械硬盘（HDD） | 上百 MB/s |
| SATA SSD | 几百 MB/s |
| NVMe SSD | 几 GB/s |

如果训练或推理需要反复从磁盘读取大数据集，这一步经常是被忽视的瓶颈——尤其是数据加载（dataloader）没有做好**预取（prefetch）**和**多进程并行读取**的时候，GPU 常常在"等数据"而不是在计算。

---

## 2. 主机内存 → GPU 显存：PCIe 这一跳是关键卡点

这是"设备间"传输，也是整条链路里最容易被忽视、但影响很大的一环。

### Pageable Memory vs Pinned Memory（锁页内存）

普通的主机内存（`malloc` 分配的）是 **pageable memory**，操作系统可能随时把它换出（swap）到磁盘上。CUDA 要把这种内存里的数据传给 GPU 时，不能直接用 DMA（直接内存访问）搬运，必须先把数据**拷贝到一块临时的 pinned memory（锁页内存，保证不会被换出）**，再从 pinned memory 用 DMA 传给 GPU——多了一次中间拷贝。

如果显式用 `cudaMallocHost()` 分配 **pinned memory**，数据可以直接 DMA 传输，省掉中间那次拷贝，传输带宽通常能提升数倍。这也是为什么很多高性能数据加载 pipeline（比如 PyTorch DataLoader 的 `pin_memory=True`）会专门做这个优化。

### PCIe 带宽是硬限制

主机到设备（H2D）的传输走 PCIe（或更高端的 NVLink），带宽远低于 GPU 内部的显存带宽：

| 链路 | 大致带宽 |
|---|---|
| PCIe 4.0 x16 | ~32 GB/s |
| PCIe 5.0 x16 | ~64 GB/s |
| NVLink（卡间互联） | 几百 GB/s |
| GPU 显存带宽（HBM） | 几千 GB/s |

PCIe 带宽比 GPU 显存内部带宽低一到两个数量级。**如果一个任务需要频繁在 CPU 和 GPU 之间搬运数据（比如数据加载速度跟不上训练/推理速度），PCIe 这一跳很可能才是真正的瓶颈，而不是 GPU 内部的计算或显存访问。**

### 异步传输与计算重叠

CUDA 提供 `cudaMemcpyAsync` 配合 **CUDA Stream**，可以让"数据从主机传到设备"和"GPU 正在计算上一批数据"这两件事同时进行，而不是串行等待。这是数据加载 pipeline 优化的核心手段——用计算时间"掩盖"传输时间，前提是用了 pinned memory 和异步 API，否则同步拷贝会直接阻塞计算流水线。

### 进阶：GPUDirect Storage

对于超大数据集（大模型训练常见场景），现在有 **GPUDirect Storage** 技术，可以让 NVMe SSD 的数据**绕过 CPU 主机内存，直接 DMA 到 GPU 显存**，省掉"磁盘 → 主机内存 → GPU 显存"里主机内存中转的那一步。在数据 I/O 是瓶颈的大规模训练场景里，这是比较前沿的优化手段。

---

## 3. GPU 内部的内存层次结构

数据进入 Global Memory 之后，才进入 GPU 内部的存储层次。从快到慢、从小到大排列：

```
Register（寄存器）
   ↓
Shared Memory / L1 Cache（SM 内部）
   ↓
L2 Cache（所有 SM 共享）
   ↓
Global Memory（显存，HBM/GDDR）
```

**每往下一层，延迟大约增加一个数量级，但容量也大一个数量级**，这是所有存储系统的经典权衡，GPU 也不例外。

| 层级 | 大致延迟 | 大致容量 | 归属范围 |
|---|---|---|---|
| Register | ~1 周期 | 每线程几十到上百个 | 单个线程私有 |
| Shared Memory | ~几十周期 | 每 SM 几十~上百 KB | block 内共享 |
| L2 Cache | ~几百周期 | 几十 MB | 所有 SM 共享 |
| Global Memory | ~几百到上千周期 | 几十 GB | 整个 GPU |

**关键认知**：Global Memory（即显存）虽然容量最大，但延迟是 Register 的几百倍。一个 kernel 如果频繁直接读写 global memory 而不加以优化，大部分时间都在"等数据"，这就是 memory-bound 的根源。

---

## 4. Global Memory 与显存带宽

**显存带宽（Memory Bandwidth）**，单位 GB/s，决定了单位时间内能从 global memory 搬运多少数据。这是 memory-bound kernel 的天花板，类似计算那边的"峰值 FLOPS"。

现代 GPU 用 HBM（High Bandwidth Memory）堆叠内存来提高带宽，但即便如此，带宽相对算力依然是稀缺资源——这也是为什么"计算强度"（FLOPs / 访存字节数）这个指标如此重要：同样多的计算，如果能少读一点显存，就能更接近算力上限，而不是被带宽拖累。

---

## 5. Memory Coalescing（内存合并访问）——最容易被忽视的性能杀手

这是 GPU 内部内存优化里**最基础也最容易踩坑**的点。

**原理**：一个 warp 里的 32 个线程如果访问的是**连续对齐**的显存地址，硬件可以把这 32 次访问**合并成一次（或很少几次）内存事务**完成。反之，如果每个线程访问的地址零散、跨步很大，硬件就要拆成多次事务，实际带宽利用率可能骤降到理论值的几分之一甚至更低。

**一个直观例子**：

```cuda
// 合并访问（coalesced）——线程 i 访问 array[i]，地址连续
data[threadIdx.x] = value;

// 非合并访问（strided）——线程 i 访问 array[i * stride]，地址跨步很大
data[threadIdx.x * stride] = value;
```

第一种写法，32 个线程的访问地址是连续的一整块，硬件一次搬运就够了。第二种写法，地址之间隔着 `stride` 个元素，硬件可能要发起 32 次独立的内存事务——同样搬运 32 个数据，实际耗时可能差几倍到几十倍。

这也是为什么数据在显存里的排布方式（row-major vs column-major、是否做了 transpose）会直接影响 kernel 性能——很多"优化"工作其实就是在调整数据布局，让 warp 的访问模式尽量连续。

---

## 6. Shared Memory——程序员手动管理的"可编程缓存"

Shared Memory 是 SM 内部的一块高速存储，由程序员用 `__shared__` 显式声明和管理，同一个 block 内的所有线程都能访问。

**核心用途：作为 Global Memory 和计算单元之间的缓冲区**，典型使用模式：

1. block 内的线程合作把一块数据从 global memory 搬到 shared memory（一次性、尽量合并访问）
2. `__syncthreads()` 同步，确保数据搬运完成
3. 线程反复复用 shared memory 里的数据做计算，不再重复访问 global memory
4. 把结果写回 global memory

**为什么这样能提速**：如果同一块数据会被多个线程重复读取（比如矩阵乘法里，同一行/列的数据会被多次用到），先搬进 shared memory 再复用，相当于用一次慢速访问换来多次快速访问，大幅降低了对 global memory 的总访问次数。这正是矩阵乘法分块（tiling）优化的核心思路，也是 FlashAttention 的核心手段。

**Shared Memory 的代价**：它是稀缺资源，每个 SM 容量有限（几十到上百 KB），一个 block 用得越多，能同时驻留在这个 SM 上的 block 数量就越少——这直接影响 **Occupancy**。所以 shared memory 的使用是一个权衡：用多了换来复用率提升，但可能压低 occupancy；用少了 occupancy 高，但复用率不够，还是要频繁访问 global memory。

---

## 7. Bank Conflict——Shared Memory 里的"内部拥堵"

Shared Memory 内部被划分成若干个 **bank**（通常 32 个，对应一个 warp 的 32 个线程），不同 bank 可以并行访问，但**同一个 bank 在同一时刻只能服务一个访问请求**。

如果一个 warp 里多个线程恰好访问了同一个 bank 的不同地址，就会发生 **bank conflict**，这些访问会被拆成多次串行完成，shared memory 本该有的高速优势就打了折扣。

典型场景：按某个 stride 访问 shared memory 数组，如果 stride 和 bank 数量（32）有公约数关系，很容易造成多个线程落在同一个 bank 上。常见解决办法是 **padding**（故意在数组里多加一个无用元素，错开访问步长，避免规律性的 bank 冲突）。

---

## 8. 内存延迟隐藏——和计算那条线同一套哲学

当一个 warp 发起 global memory 读取请求后，不会傻等，SM 的 warp 调度器会切换去执行另一个已经就绪的 warp。**只要同时驻留的 warp 数量足够多，GPU 就能用"warp 之间的切换"把内存访问的几百周期延迟藏起来**，而不需要像 CPU 那样依赖大缓存和复杂的预取策略。

这再次说明：**Occupancy 同时是计算效率和内存延迟隐藏能力的共同瓶颈**，是连接计算和内存两条线的关键资源。

---

## 小结：完整链路的瓶颈全景图

```
磁盘I/O（百MB/s ~ 几GB/s）
   ↓ 瓶颈点1：磁盘带宽/文件系统开销 → 靠预取、并行读取缓解
主机内存（pageable → pinned 拷贝，如果没用 pinned memory）
   ↓ 瓶颈点2：PCIe带宽（~32-64 GB/s）→ 靠 pinned memory + 异步传输 + 计算重叠缓解
GPU显存（几千 GB/s 内部带宽）
   ↓ 瓶颈点3：Memory Coalescing → 靠访问模式优化缓解
Shared Memory（更快，但容量小）
   ↓ 瓶颈点4：Bank Conflict / Occupancy → 靠合理分配和 padding 缓解
Register → 计算单元
```

**一个直观的认知**：越往链路前端（磁盘、PCIe），带宽越低但数据量通常越大（整个数据集）；越往链路后端（shared memory、register），带宽越高但容量越小。所以整体优化思路是"能少搬就少搬、能在前端就把该搬的一次搬够、后面尽量复用不再回头找前面要数据"——这和 Shared Memory 复用 Global Memory 数据的逻辑，其实是同一套哲学在不同层级的重复应用。

---

*下一篇：Roofline Model——用计算强度这个指标，把"计算"和"内存"两条线正式连接起来，回答"CUDA 优化的核心是什么"这个问题。*
