# M2 · Lesson 2:手写 GEMM(一)——从 naive 到 shared memory 分块

> GEMM(通用矩阵乘 General Matrix Multiply)是深度学习的"主食":Linear 层、Attention 的 QKV 投影、FFN,全是 GEMM。把 GEMM 写快,你就掌握了推理性能的命脉。
> 上一课我们算出 GEMM 的理论算术强度是 N/6(很高,应该算力受限)。但你马上会看到:**naive 写法把这个高 AI 全浪费了**,实测只能跑到峰值算力的几个百分点。本课用 shared memory 分块(tiling)把它救回来。
> 预计用时:3 小时(理论 + 敲两版 kernel + benchmark)。
> 前置:M1 L6(shared memory、`__syncthreads`、bank conflict)、L8(coalescing);M2 L1(roofline、算术强度)。

## 学习目标

1. 写出正确的 naive GEMM,并用 roofline 解释它为什么慢(访存爆炸)。
2. 理解 tiling 的核心思想:**把数据搬进 shared memory 复用,缩小"有效访存量"**。
3. 写出 shared memory 分块版 GEMM,benchmark 对比 naive,看到数倍加速。
4. 算清楚分块带来的复用率提升(从理论上理解加速的来源)。

---

## 1. 问题定义与 naive 实现

计算 `C = A · B`,其中 A 是 M×K,B 是 K×N,C 是 M×N。元素公式:

```
C[row][col] = Σ_{k=0}^{K-1} A[row][k] * B[k][col]
```

最直接的并行化:**一个线程负责一个输出元素 `C[row][col]`**,自己跑完那条求和:

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];   // 每次循环 2 次全局访存
        }
        C[row * N + col] = acc;
    }
}
```

逻辑完全正确,但性能很差。用 roofline 拆解一下。

### naive 的致命伤:访存爆炸

看那条内层循环:每次迭代要从**全局显存**读 `A[row][k]` 和 `B[k][col]` 各一次。整个 kernel 的全局访存量:

```
每个输出元素读 2K 个 float,共 M×N 个输出
→ 总访存 = 2·M·N·K × 4 字节
```

而计算量是 `2·M·N·K` FLOP。于是 naive 的**有效算术强度**:

```
AI_naive = 2·M·N·K / (8·M·N·K) = 0.25 FLOP/Byte
```

**注意:这是个常数,跟 N 完全无关!** 对比 L1 算的理论值 N/6(N=1024 时 ≈170),naive 把算术强度从 170 砸到了 0.25。回到 roofline 图上,它被死死按在带宽受限区的最左端——**一个本该算力受限的运算,被 naive 写法搞成了带宽受限。**

> 根因(呼应 L1 练习题 3):相邻线程之间毫无数据复用。`C[0][0]` 和 `C[0][1]` 都要读整行 `A[0][:]`,但它们各读各的,同一份 A 数据被从显存反复搬运。**数据复用率几乎为 0,带宽全浪费在重复搬运上。**

这和你在大数据里见过的"每个 task 都去重新 scan 一遍同一张维表"如出一辙——解法也一样:**把热数据缓存到近端、共享复用**。GPU 上的"近端高速缓存"就是 shared memory。

---

## 2. Tiling:把数据搬进 shared memory 复用

核心思想:让**一个 block 协作计算 C 的一个小方块(tile)**,计算这个 tile 需要的 A、B 数据,先由 block 内所有线程**合作搬进 shared memory**,然后大家都从 shared memory 读——一份数据搬一次,被整个 tile 的线程复用很多次。

设 tile 大小为 `T×T`(比如 32×32)。算 C 的一个 `T×T` 块,需要 A 的 `T×K` 一条横带、B 的 `K×T` 一条竖带。K 可能很大,放不进 shared memory,所以沿 K 方向**也切成 T 段,分阶段(phase)处理**:

```
       B 沿 K 切成若干 T 宽的竖段
        ┌──┬──┬──┬──┐
        │B0│B1│B2│..│   每个 phase 取一段 As(TxT)、一段 Bs(TxT)
   A    ├──┼──┼──┼──┤   搬进 shared memory,在片上做 TxT 的小矩阵乘累加
 ┌────┐ │  │  │  │  │
 │A0A1│ └──┴──┴──┴──┘
 │....│
 └────┘
   A 沿 K 切成若干 T 宽的横段

每个 phase:
  1) block 内线程合作把 As、Bs 从 global -> shared
  2) __syncthreads()  确保都搬完
  3) 在 shared memory 里做 T 次乘加累加到寄存器 acc
  4) __syncthreads()  确保都算完才能覆盖下一段
最后把 acc 写回 C。
```

代码骨架(完整版见 `code/gemm_tiled.cu`):

```cpp
#define TILE 32

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;   // 本线程负责的 C 行
    int col = blockIdx.x * TILE + tx;   // 本线程负责的 C 列

    float acc = 0.0f;
    // 沿 K 方向分阶段;每段宽 TILE。
    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {
        // 协作搬运:每个线程搬 As、Bs 各一个元素(注意边界与合并访问)。
        int a_col = ph * TILE + tx;
        int b_row = ph * TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();                 // 等所有线程搬完

        // 在片上做这一段的乘加,数据来自快的 shared memory。
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();                 // 等所有线程算完再覆盖共享内存
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

两处 `__syncthreads()` 缺一不可(M1 L6 强调过):第一处保证"数据搬完才开算",第二处保证"算完才覆盖下一段",否则会读到脏数据。

> **As 的访问 `As[ty][k]`、Bs 的访问 `Bs[k][tx]`**:Bs 按列读会不会 bank conflict?这里 `Bs[k][tx]`,同一 warp 内 tx 连续变化、k 固定,访问的是同一行的连续元素,落在不同 bank,**无冲突**。这正是 M1 L6 bank conflict 知识的应用——tile 的布局要让 warp 内访问错开 bank。

---

## 3. 为什么 tiling 快:复用率的定量分析

关键在分母——**搬进 shared memory 的每个元素被复用了多少次**。

在一个 phase 里,block 搬进了 `TILE×TILE` 个 A 元素和 `TILE×TILE` 个 B 元素。这些数据用来更新 `TILE×TILE` 个 acc,每个 acc 在这一 phase 做了 `TILE` 次乘加。也就是:

- 搬进的 A 数据每个被用了 `TILE` 次(被同一行的 TILE 个输出列复用)。
- 搬进的 B 数据每个被用了 `TILE` 次(被同一列的 TILE 个输出行复用)。

于是全局访存量从 naive 的 `2·M·N·K` 降到约 `2·M·N·K / TILE`。算术强度:

```
AI_tiled ≈ 0.25 × TILE
```

TILE=32 时 AI ≈ 8 FLOP/Byte——比 naive 提升 32 倍!回到 roofline:naive 在 0.25 处贴着带宽斜坡爬,tiled 在 8 处(已接近很多卡的拐点),离算力屋顶近多了。

> **一句话:tiling 把"每个数据从显存搬一次只用一次"变成"搬一次用 TILE 次",有效算术强度翻 TILE 倍。** 这就是几乎所有高性能 GEMM 的第一性原理。下一课的寄存器分块,本质是在 shared memory 之上再加一层复用(shared → 寄存器),把 AI 推得更高。

---

## 4. 动手实验:benchmark naive vs tiled

`code/gemm_tiled.cu` 包含两个 kernel、一个 CPU 参考实现(校验正确性)、CUDA event 计时,并计算每个 kernel 达到的 GFLOP/s。

编译运行:

```bash
cd code
nvcc -O3 gemm_tiled.cu -o gemm_tiled
./gemm_tiled 1024        # 参数是方阵边长 N(M=N=K)
```

参考输出(数值随卡不同):

```
[INFO] GEMM M=N=K=1024
[INFO] naive : 6.42 ms, 334.2 GFLOP/s  (correct)
[INFO] tiled : 1.18 ms, 1820.5 GFLOP/s (correct)
[INFO] speedup tiled/naive = 5.4x
```

**读数要点**:
- tiled 比 naive 快 4~8 倍是常见结果(取决于卡和 N)。
- 即便 tiled 也只到峰值算力的 10%~20%——别灰心,这正是 Lesson 3 要继续榨的空间(寄存器分块、向量化能再翻几倍,逼近 cuBLAS)。
- 一定要看到 `(correct)`:GEMM 极易写错索引,**先正确再快**。校验用 CPU 算一个小规模对拍。

### 留给你的 TODO

`gemm_tiled.cu` 里有两个 `TODO`:
1. 把 `TILE` 改成 16 和 64(注意 64×64 的 shared memory 和线程数限制),benchmark 对比,验证 AI≈0.25×TILE 的趋势。
2. 处理非整除边界:当 N 不是 TILE 整数倍(如 N=1000),确认结果仍 `correct`(代码已留了边界判断,你来验证)。

---

## 5. 接回推理

你现在写的这个 tiled GEMM,就是 cuBLAS / CUTLASS 内核的"骨架雏形"。推理框架里:
- 每个 Linear 层 `y = xW` 是一个 GEMM;
- prefill 阶段(处理长 prompt)是大 GEMM,算力受限,tiling 这类优化直接决定吞吐;
- 真实生产当然用 cuBLAS(它做到了寄存器分块 + 向量化 + Tensor Core + 自动调参),但**理解 tiling 让你能看懂 profile、判断一个 GEMM 是否已接近硬件极限**,而不是把库当黑盒。

---

## 练习题

1. naive GEMM 的算术强度为什么和 N 无关,而理论值是 N/6?差距来自哪里?
2. tile 取 32×32 时,一个 block 用了多少 shared memory(字节)?如果想用 64×64,会遇到什么限制?
3. 为什么两次 `__syncthreads()` 都不能省?分别说清省掉后会发生什么错误。
4. `Bs[k][tx]` 这种访问为什么没有 bank conflict?如果把 Bs 声明成 `[TILE][TILE+1]`(padding)是为了解决什么场景的冲突?

<details>
<summary>参考答案</summary>

1. naive 里每个输出元素都独立地从全局显存重读整行 A、整列 B,数据零复用,访存量 = 2MNK,与计算量同阶,AI 退化成常数 0.25。理论值 N/6 假设每个输入只从显存读一次(完美复用)。差距 = **复用没做出来**,tiling 就是来补这个复用的。
2. 两个 `float[32][32]` = 2 × 32 × 32 × 4 = 8192 字节 = 8 KB/block。64×64 需要 2×64×64×4 = 32 KB(很多卡单 block shared memory 上限附近),且 64×64=4096 线程 > 单 block 最多 1024 线程的限制,所以不能简单地让一个线程算一个元素——必须让每线程算多个元素(正是 Lesson 3 的寄存器分块)。
3. 第一个 `__syncthreads()`(搬运后):若省掉,有的线程还没把 As/Bs 写完,别的线程就开始读,读到旧/未初始化数据 → 结果错。第二个(计算后):若省掉,快的线程进入下一 phase 把 As/Bs 覆盖了,慢的线程还在用上一段数据 → 结果错。
4. 同一 warp 内 `tx` 连续(0..31)、`k` 固定,`Bs[k][tx]` 访问连续地址,落在 32 个不同 bank,无冲突。padding 成 `[TILE][TILE+1]` 是为了解决**按列访问**(如 `Bs[tx][k]` 那种同列不同行)时,步长正好是 32 的倍数导致所有线程撞同一 bank 的冲突——错开一列就把地址打散到不同 bank。

</details>

---

## 小结

- naive GEMM 因**零数据复用**,有效算术强度只有 0.25(与 N 无关),被困在带宽受限区。
- **tiling**:一个 block 把 A、B 的小块搬进 **shared memory** 复用,把全局访存量降为 `1/TILE`,算术强度提到 `≈0.25×TILE`。
- 两次 `__syncthreads()` 保证搬运/计算的正确同步;tile 布局要避开 bank conflict。
- 实测 tiled 比 naive 快数倍,但离峰值算力还远——下一课继续榨。

## 自测验收
- [ ] 能默写 naive GEMM 并解释它为何带宽受限。
- [ ] 能讲清 tiling 的数据流(分 phase、协作搬运、两次 sync)。
- [ ] `gemm_tiled.cu` 跑通且 `correct`,看到数倍加速。
- [ ] 能定量说明 tiling 把算术强度提了 TILE 倍。

下一课:**Lesson 3 — 手写 GEMM(二):寄存器分块、向量化访存、循环展开**,把 GEMM 推到逼近 cuBLAS。
