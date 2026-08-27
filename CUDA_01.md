# CUDA 基本概念：Grid / Block / Thread / Warp / SM

## 1. 软件层面的层次结构

CUDA 编程模型把并行计算组织成三层网格状结构：

```
Grid（网格）
  └── Block（线程块）
        └── Thread（线程）
```

### Thread（线程）
- 最小的执行单元，每个线程执行同一份 kernel 代码，但处理不同的数据（SIMT 模型：Single Instruction, Multiple Threads）
- 每个线程有自己的寄存器、程序计数器，以及唯一的索引（`threadIdx`）

### Block（线程块）
- 一组线程的集合，同一个 block 内的线程：
  - 可以通过 `__shared__` 内存高效通信
  - 可以用 `__syncthreads()` 同步
  - 会被调度到**同一个 SM** 上执行（不会跨 SM）
- 每个 block 有唯一索引 `blockIdx`，block 内线程也有 `threadIdx`（可以是 1D/2D/3D）
- 一个 block 最多 1024 个线程（具体上限取决于硬件架构）

### Grid（网格）
- 一次 kernel launch 启动的所有 block 的集合
- Grid 内的 block 之间**没有同步保证**，也不共享内存（除了全局内存 / L2 cache）
- 一个 kernel 调用对应一个 grid

调用形式：
```cuda
kernel<<<gridDim, blockDim>>>(...);
```

---

## 2. 硬件层面

### SM（Streaming Multiprocessor，流多处理器）
- GPU 的核心计算单元，一块 GPU 芯片上有若干个 SM（比如几十到上百个）
- 每个 SM 包含：CUDA Core（算术单元）、Tensor Core（矩阵计算，较新架构）、寄存器文件、共享内存（Shared Memory / L1 Cache）、Warp 调度器等
- **一个 block 会被完整分配到一个 SM 上执行**，一个 SM 可以同时驻留多个 block（资源允许的情况下）

### Warp（线程束）
- 硬件调度和执行的**真正基本单位**，固定为 **32 个线程**
- 一个 block 内的线程会按顺序被划分成若干个 warp（比如 block 有 256 个线程，就是 8 个 warp）
- 同一个 warp 内的 32 个线程在 SM 上**以 SIMD 方式同步执行同一条指令**（这就是 SIMT 的本质：warp 级别的 SIMD）
- **Warp Divergence（线程束分化）**：如果同一 warp 内的线程因为 `if/else` 走了不同分支，硬件会串行执行各个分支路径，造成性能损失——这是 CUDA 优化中非常重要的点

---

## 3. 关系图解

```
                     GPU
                      │
        ┌─────────────┼─────────────┐
       SM 0          SM 1    ...   SM N        ← 硬件计算单元
        │
   ┌────┼────┐
 Block0 Block1 ...                              ← 软件调度到 SM 上（一个 block 固定在一个 SM）
   │
 ┌─┼─┬─────┐
Warp0 Warp1 ...                                 ← block 内线程按 32 个一组划成 warp（硬件执行单位）
   │
 Thread0..31                                    ← 每个 warp 内 32 个线程 SIMT 执行
```

### 对应关系总结

| 层级 | 归属 | 说明 |
|---|---|---|
| Grid | 一次 kernel 调用 | 包含多个 block |
| Block | 逻辑分组，由程序员定义 | 固定调度到一个 SM，可用共享内存 + `__syncthreads()` 同步 |
| Warp | 硬件自动划分（每 32 线程一个） | SM 实际执行调度的基本单位 |
| Thread | 最小执行单元 | SIMT 方式在 warp 内同步执行指令 |
| SM | 物理硬件单元 | 可同时容纳多个 block/warp，通过 warp 调度器切换隐藏延迟 |

---

## 4. 关键要点

**为什么 block size 通常设成 32 的倍数？**
如果 block size 不是 32 的倍数，最后一个 warp 会有线程"空闲"（inactive），浪费计算资源。常见设置如 128、256。

**SM 如何隐藏内存延迟？**
SM 通过在多个 warp 之间快速切换来隐藏内存访问延迟——当一个 warp 在等待内存读取时，SM 调度器会切换执行另一个已经就绪的 warp。这是 GPU 高吞吐量设计的核心思想（区别于 CPU 靠大缓存和分支预测降低延迟）。
