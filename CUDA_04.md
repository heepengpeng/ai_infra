# Roofline Model：连接计算与内存的桥梁

前两篇分别讲透了计算这条线（CUDA Core → Tensor Core → 精度 → Occupancy）和内存这条线（磁盘 → PCIe → Global Memory → Shared Memory → Bank Conflict）。这篇要回答的问题是：**面对一个具体的 kernel，怎么知道该往哪条线使劲？**

Roofline Model 就是回答这个问题的工具。

---

## 1. 核心指标：计算强度（Arithmetic Intensity）

$$
\text{计算强度（AI）} = \frac{\text{浮点运算次数（FLOPs）}}{\text{访存字节数（Bytes）}}
$$

单位是 FLOPs/Byte，意思是"每从显存搬运 1 字节数据，能配套做多少次浮点运算"。

- **AI 越高** → 数据搬一次，能反复算很多次 → 越可能是 compute-bound
- **AI 越低** → 数据搬一次，只够算一两次 → 越可能是 memory-bound

**举例直觉**：
- 大矩阵乘法（GEMM）：读进来的每个数据会被复用很多次（矩阵乘法的经典性质），AI 高
- 逐元素操作（比如 ReLU、加法）：读一个数据只做一次运算就完事，AI 极低
- Attention 中的 softmax、LayerNorm：同样是访存量大、计算量相对小，AI 偏低

---

## 2. Roofline 图怎么画

横轴是计算强度（AI，对数坐标），纵轴是实际达到的性能（FLOPS，对数坐标）。图上有两条"房顶线"：

```
性能(FLOPS)
   ↑
   │           ┌──────────────────  ← 算力屋顶（硬件峰值 FLOPS，水平线）
   │          ╱
   │         ╱
   │        ╱    ← 带宽屋顶（斜线，斜率 = 显存带宽）
   │       ╱
   │      ╱
   │     ╱
   └────╱─────────────────────────→ 计算强度(FLOPs/Byte)
        ↑
     两条线的交点 = 硬件的"平衡点"
```

- **斜线部分（左侧）**：性能被显存带宽卡住，此时 `性能 = 计算强度 × 带宽`，AI 越高性能越高，这个区域是 **memory-bound**
- **水平线部分（右侧）**：性能被算力卡住，无论 AI 再怎么涨，性能也涨不上去，封顶在硬件峰值 FLOPS，这个区域是 **compute-bound**
- **两条线的交点**：这个 AI 值就是这块硬件的"平衡点"——低于它，你天生是 memory-bound；高于它，你天生是 compute-bound，这是硬件本身的属性（峰值算力 / 显存带宽算出来的），和你写的 kernel 无关

**把自己的 kernel 实际测出来的 (AI, 性能) 这个点画到图上**，落在哪个区域，立刻知道该往哪个方向优化。

---

## 3. 这就是"CUDA 优化核心"问题的完整闭环

回到之前那个问题——**CUDA 优化的核心是什么**，现在可以给出完整版答案：

**第一步：算出你的 kernel 的计算强度，判断落在 Roofline 图的哪一侧**

- 落在斜线区（memory-bound）→ 榨计算没用，SM 大部分时间在等数据。该做的是内存那条线的手段：Kernel Fusion 减少中间结果读写、Memory Coalescing、Shared Memory 复用、减少 kernel 数量
- 落在水平区（compute-bound）→ 数据搬运不是瓶颈，SM 一直有活干但算得不够快。该做的是计算那条线的手段：用好 Tensor Core、降精度（量化）、提高 Occupancy、减少 warp divergence

**第二步：更高级的优化——把点在图上"挪位置"**

单纯在原地优化是有限的（斜线区你最多把带宽利用率做到 100%，水平区最多把 MFU 做到接近 100%）。真正厉害的优化是通过**算法设计改变计算强度本身**，把一个天生 memory-bound 的问题，变成计算强度更高、更接近甚至跨过平衡点的问题。

**FlashAttention 就是这个思路的经典案例**：

标准 Attention 实现要算 `Q@K^T`、softmax、再乘 `V`，中间结果（尤其是 `N×N` 的 attention score 矩阵）要写回 global memory 再读出来，访存量随序列长度平方增长，AI 很低，是典型的 memory-bound。

FlashAttention 用 **tiling**：把 Q、K、V 分块搬进 shared memory，在 shared memory 里完成分块的矩阵乘、softmax 累积（用在线 softmax 算法避免存储完整的 `N×N` 矩阵），最后才把结果写回 global memory。**计算量（FLOPs）基本没变，但访存量大幅下降**——这直接把这个 kernel 在 Roofline 图上的位置往右推（AI 变高），让它离算力屋顶更近，从而跑得更快。这不是"算得更快"，而是"用更少的数据搬运换到了同样多的计算"。

---

## 4. 小结：三条线的关系

```
计算这条线（Tensor Core / 精度 / Occupancy）
       ╲
        ╲── 都受制于 Occupancy（同时驻留 warp/block 数）
       ╱
内存这条线（Coalescing / Shared Memory / Bank Conflict）

          ↓
   两条线的相对强弱，由计算强度（AI）决定谁是瓶颈
          ↓
   Roofline Model：先诊断你在哪个区域，再决定往哪条线使劲
          ↓
   终极优化：不是在原地榨到极限，而是用算法设计（如 tiling、kernel fusion）
            改变计算强度本身，把 kernel 从 memory-bound 推向 compute-bound
```

至此，"CUDA 优化的核心是什么"这个问题有了完整闭环的答案：

> **核心不是"让计算更快"或"让内存更省"这两者之一，而是先用计算强度诊断瓶颈所在，再用对应资源（Tensor Core/精度/Occupancy，或 Coalescing/Shared Memory/Fusion）对症下药；更高阶的优化则是通过算法设计改变计算强度本身，从根本上把问题从一个区域推向另一个区域。**

这套框架（Roofline + 诊断优先）就是理解后续所有 LLM 推理优化技术（KV Cache、PagedAttention、量化、算子融合、FlashAttention 系列）为什么有效的通用钥匙——它们本质上都是在这张图上，想办法把自己的 kernel 挪到一个更有利的位置。
