# Occupancy 计算方法与 Nsight Compute 实战

前面几篇讲的都是"为什么"——为什么 Occupancy 重要、为什么它连接计算和内存两条线。这篇讲"怎么算""怎么测"，把理论落地成能在真实 kernel 上用的工具。

---

## 1. Occupancy 的定义

$$
\text{Occupancy} = \frac{\text{每个 SM 实际驻留的 warp 数}}{\text{每个 SM 硬件支持的最大 warp 数}}
$$

分母是硬件规格里的固定值（比如某代架构一个 SM 最多驻留 64 个 warp，即 2048 个线程）。分子由你的 kernel 实际用了多少资源决定——**三个资源中最紧张的那个，决定了实际能驻留多少 block/warp**。

---

## 2. 三个限制因素，怎么算

### 限制因素一：线程数 / Block Size

每个 SM 支持的最大线程数是固定的（硬件规格），你的 block size 决定了一个 SM 能塞下几个 block：

$$
\text{受线程数限制的 block 数} = \left\lfloor \frac{\text{每 SM 最大线程数}}{\text{block size}} \right\rfloor
$$

**举例**：某 GPU 每 SM 最大 2048 线程，block size 设成 256，理论上一个 SM 能塞 `2048/256 = 8` 个 block（还要看下面两个限制会不会更早卡住）。

**这里有个坑**：block size 如果不是 32 的倍数，会有"零头"线程浪费——比如 block size 定成 100，实际会占用 4 个 warp 的位置（128 线程的资源），但只用了 100 个，白白浪费 28 个线程的槽位。

### 限制因素二：寄存器数量

每个 SM 的寄存器总量是固定的（比如 65536 个 32 位寄存器），每个线程用的寄存器数量由编译器决定（可以通过 `nvcc` 的 `--maxrregcount` 或 kernel 里的 `__launch_bounds__` 限制）：

$$
\text{受寄存器限制的 block 数} = \left\lfloor \frac{\text{每 SM 寄存器总数}}{\text{每线程寄存器数} \times \text{block size}} \right\rfloor
$$

**举例**：每 SM 65536 个寄存器，每个线程用 64 个寄存器，block size 256：
$$
\left\lfloor \frac{65536}{64 \times 256} \right\rfloor = \left\lfloor \frac{65536}{16384} \right\rfloor = 4 \text{ 个 block}
$$

**关键认知**：kernel 写得越复杂（局部变量多、循环展开多），编译器分配的寄存器就越多，Occupancy 天花板就越低。这是个真实的权衡——有时候手动限制寄存器数量反而能提升整体性能（更高 Occupancy 换更好的延迟隐藏），但也可能导致寄存器溢出到 local memory（其实是 global memory 的一部分，速度骤降），需要实测。

### 限制因素三：Shared Memory 用量

每个 SM 的 shared memory 总量固定（比如 100KB 出头，具体因架构而异，且部分容量可能要跟 L1 Cache 分享，可配置比例）：

$$
\text{受 shared memory 限制的 block 数} = \left\lfloor \frac{\text{每 SM shared memory 总量}}{\text{每 block 使用的 shared memory}} \right\rfloor
$$

**举例**：每 SM 96KB shared memory，每个 block 用 32KB：
$$
\left\lfloor \frac{96}{32} \right\rfloor = 3 \text{ 个 block}
$$

### 最终 Occupancy：取三者最小值

$$
\text{实际驻留 block 数} = \min(\text{线程数限制}, \text{寄存器限制}, \text{shared memory 限制})
$$

再换算成 warp 数除以硬件最大 warp 数，就是最终 Occupancy 百分比。

**这也是为什么"提高 Occupancy"经常是一个三方博弈**：加大 block size 能提升线程数维度的利用率，但可能因为寄存器或 shared memory 更早触顶而不起作用；减少每线程寄存器用量能提升寄存器维度的上限，但可能逼着数据溢出到 local memory，反而更慢。**没有单一的"调大某个参数就变好"的规律，必须实测。**

---

## 3. 不用手算：CUDA Occupancy Calculator

手算三个限制因素比较繁琐，实际工作中很少手算，有两种更实用的方式：

**方式一：Occupancy API（写在代码里，运行时查询）**
```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
// blockSize 就是理论上能给出最高 occupancy 的 block size
```

**方式二：Nsight Compute（跑完 kernel 之后，直接看实测数据）**——这是更常用、更准确的方式，因为它反映的是真实运行情况，不只是理论计算。下面重点讲这个。

---

## 4. Nsight Compute 实战

Nsight Compute（简称 NCU）是 NVIDIA 官方的 kernel 级性能分析工具，专门用来深挖单个 kernel 的执行细节，跟 Nsight Systems（看整体 timeline、多 kernel 之间的调度关系）是互补关系。

### 基本用法

```bash
# 最简单的用法：跑一遍程序，抓取所有 kernel 的性能数据
ncu ./my_program

# 只关注某个特定 kernel（用正则匹配名字）
ncu -k my_kernel_name ./my_program

# 生成报告文件，之后用 GUI 打开细看
ncu -o report ./my_program
```

### 需要重点看的几个指标

**1. Occupancy 相关**
- `Achieved Occupancy`：实测的实际 occupancy，和理论值经常有差距（理论值假设 block 均匀分布、warp 一直有活干，实际运行中会有 tail effect 等因素拉低）
- `Theoretical Occupancy`：根据 kernel 的资源用量（寄存器/shared memory/block size）算出的理论上限，对应前面手算的过程
- 如果两者差距很大，说明问题不在资源限制，而在**调度不均衡**（比如某些 SM 分到的 block 少，warp divergence 让部分线程提前退出等）

**2. 判断 compute-bound 还是 memory-bound（对应 Roofline）**
- `Compute (SM) Throughput`：计算单元的利用率百分比
- `Memory Throughput`：显存带宽的利用率百分比
- 哪个数值更接近 100%，kernel 就更偏向那一类瓶颈——这是 NCU 直接帮你把 Roofline 诊断做了的部分，甚至它自带 **Roofline 图表**，会把你的 kernel 实测点直接画在算力屋顶和带宽屋顶的图上

**3. Warp 相关的细节指标**
- `Warp Execution Efficiency`：反映 warp divergence 的严重程度，越接近 100% 说明分支分化越少
- `Stall Reasons`（阻塞原因分类）：NCU 会告诉你 warp 在等待什么——是在等内存（`Stall Long Scoreboard`，通常对应等 global memory）、等同步（`Stall Barrier`，对应 `__syncthreads()`）、还是指令依赖（`Stall Short Scoreboard`）。**这是最直接定位问题的地方**——与其猜"是不是内存卡了"，不如直接看 stall reason 排名第一的是什么

**4. Memory Coalescing 相关**
- `L2 Cache Hit Rate`、`Global Memory Load/Store Efficiency`：如果这个效率远低于 100%，基本可以断定访问模式没有合并（对应前面讲的 Memory Coalescing 问题），需要检查数据布局或访问方式

### 典型排查流程

```
1. 先看 Compute Throughput vs Memory Throughput
   → 哪个高，先怀疑哪个是瓶颈（对应 Roofline 诊断）

2. 如果 Memory Throughput 高但 kernel 还是慢
   → 看 Global Memory Load/Store Efficiency 是否偏低
   → 偏低说明访问没合并，去查数据布局/访问 pattern

3. 如果 Compute Throughput 和 Memory Throughput 都不高
   → 大概率是 Occupancy 不够，SM 大量空闲
   → 看 Achieved Occupancy vs Theoretical Occupancy
   → 理论值就低 → 调整 block size / 减少寄存器或 shared memory 用量
   → 理论值不低但实际值低 → 看调度不均衡或 warp divergence（Warp Execution Efficiency）

4. 具体等在哪 → 看 Stall Reasons 排名，直接定位到根因
```

这套流程本质上就是把之前讲的"计算 vs 内存 vs Occupancy"理论框架，套上 NCU 给出的具体数字，从"猜测"变成"读数据下结论"。

---

## 5. 小结

Occupancy 的计算逻辑是"三个资源哪个先触顶，哪个就是天花板"（线程数 / 寄存器 / shared memory），手算能建立直觉，但实际工作中更依赖 Nsight Compute 给出的**实测数据**——尤其是 `Achieved Occupancy`、`Compute/Memory Throughput`、`Stall Reasons` 这几项，能直接把"这个 kernel 到底卡在哪"从理论推测变成有数据支撑的结论。

这也是把 Roofline 诊断思路真正落地到日常调优工作流的方式：**先用 NCU 的 Throughput 数据做 Roofline 式诊断，再用 Occupancy 和 Stall Reasons 找到具体的资源瓶颈，最后对症下药**（调 block size、减少寄存器/shared memory 用量、改善访问模式、减少 divergence）。
