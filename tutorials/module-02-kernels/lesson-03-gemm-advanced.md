# M2 · Lesson 3:手写 GEMM(二)——寄存器分块、向量化、循环展开

> 上一课 tiling 把 GEMM 从带宽受限拉回了一些,但实测还只有峰值算力的 10%~20%。差在哪?差在**计算指令的密度不够、访存还没打满**。本课用三板斧——**寄存器分块、float4 向量化访存、循环展开**——把 GEMM 再翻几倍,逼近 cuBLAS。
> 这是手写 kernel 优化的"内功心法",之后看任何高性能算子(包括 FlashAttention)都会反复见到同样的套路。
> 预计用时:3.5 小时(理论 + 敲 kernel + 画性能曲线)。
> 前置:M2 L2(tiling、shared memory 复用、算术强度);M1 L8(coalescing)。

## 学习目标

1. 理解 tiling 之后新的瓶颈:**每条乘加指令都伴随 shared memory 访问,计算访存比不够**。
2. 掌握**寄存器分块**:让一个线程算 `TM×TN` 个输出,把复用从 shared 再下沉到寄存器。
3. 用 **float4 向量化**把访存事务数减少 4 倍,用 **`#pragma unroll`** 消除循环开销。
4. 跑出从 naive → tiled → 寄存器分块 的性能曲线,看到逐步逼近 cuBLAS。

---

## 1. tiling 之后,新瓶颈是什么?

回顾 L2 的 tiled kernel 内层:

```cpp
for (int k = 0; k < TILE; ++k) {
    acc += As[ty][k] * Bs[k][tx];   // 1 次 FMA,但要 2 次 shared memory 读
}
```

每做 1 次乘加(FMA),要从 shared memory 读 2 个数。shared memory 虽然比全局显存快得多,但**每条算术指令都夹着访存指令**,计算单元(FMA 单元)有大量时间在等数据,利用率上不去。

> 类比:tiling 把"去仓库(全局显存)取货"改成了"去货架(shared memory)取货",快多了;但工人(FMA 单元)还是**取一个零件装一下、再取一个再装一下**,手没停过却效率低。真正的高手是**一次把一筐零件搬到工作台(寄存器)上,然后连续组装**——这就是寄存器分块。

我们要提高的是**计算访存比**:让从 shared memory 读进寄存器的每个数,在寄存器里被用更多次。

---

## 2. 寄存器分块:一个线程算一小块输出

L2 是"一个线程算 1 个输出 `C[row][col]`"。现在改成 **一个线程算 `TM×TN` 个输出**(比如 8×8=64 个)。

关键洞察:算这 `TM×TN` 个输出,需要的是 A 的 `TM` 个值(一列片段)和 B 的 `TN` 个值(一行片段)。把它们读进**寄存器**后,可以做 `TM×TN` 次乘加:

```
寄存器里持有:  a_reg[TM]  (来自 As 的一列)
              b_reg[TN]  (来自 Bs 的一行)
然后:
for i in 0..TM:
  for j in 0..TN:
    acc[i][j] += a_reg[i] * b_reg[j];   // TM*TN 次 FMA
共读了 TM + TN 个数,做了 TM*TN 次 FMA
```

**计算访存比从 tiled 的 1:2(每 FMA 两次访存)跃升到 `TM*TN : (TM+TN)`**。TM=TN=8 时是 64:16 = 4:1——每读 1 个数做 4 次 FMA,FMA 单元终于忙起来了。

这是 L2 思想的递归应用:

```
全局显存 ──tiling──▶ shared memory ──寄存器分块──▶ 寄存器
  (慢)              (一个 block 复用)         (一个线程复用)
   复用 0             复用 TILE 倍              再复用 TM/TN 倍
```

每下沉一层存储,复用一次,算术强度再上一个台阶。这就是 GEMM 优化的全部秘密。

代码骨架(完整见 `code/gemm_register.cu`,采用经典 1D thread-tile 布局):

```cpp
// 每个 block 算 BM×BN 的输出块;每个线程算 TM×TN 个输出。
// BK 是沿 K 方向每个 phase 的步长。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[BK][BM];   // 注意 As 转置存放,便于按列向量化读
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {0.0f};    // 每线程的 TM*TN 个累加器,全在寄存器
    float a_reg[TM], b_reg[TN];

    for (int ph = 0; ph < K; ph += BK) {
        // 1) 协作把 A、B 的块搬进 shared(此处用 float4 向量化,详见 §3)
        load_tiles_to_shared(...);
        __syncthreads();

        // 2) 沿 BK 逐步:从 shared 读一列 a_reg、一行 b_reg 到寄存器
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[k][thread_row * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][thread_col * TN + j];
            // 3) 在寄存器里做 TM*TN 次外积累加
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }
    // 写回 acc 到 C 的对应 TM×TN 块
}
```

> `acc[TM][TN]` 一定要小到能放进寄存器(每线程寄存器有限,约 255 个)。TM=TN=8 → 64 个累加器 + a_reg/b_reg + 索引,刚好。开太大(如 16×16=256)会**寄存器溢出(register spilling)**到本地显存,性能反而暴跌。这是寄存器分块的调参核心:在复用率和寄存器压力之间取平衡。

---

## 3. float4 向量化访存:一次搬 4 个

M1 L8 讲过 coalescing(warp 内连续访问合并)。向量化更进一步:**让单个线程用一条指令读连续的 4 个 float(128 bit)**。CUDA 提供 `float4` 类型:

```cpp
// 标量:4 条 load 指令
float a0 = A[i], a1 = A[i+1], a2 = A[i+2], a3 = A[i+3];

// 向量化:1 条 128-bit load 指令,访存事务数减少 4 倍
float4 v = reinterpret_cast<const float4*>(A)[i / 4];
// v.x, v.y, v.z, v.w 即 4 个元素
```

向量化的收益:
- **减少指令数**:访存指令少 4 倍,指令发射压力小。
- **更高的内存事务效率**:128-bit 事务能更好利用显存总线。

把它用在"global → shared"的搬运阶段:每个线程一次搬 4 个连续元素进 shared memory。要求地址 16 字节对齐(`float4` = 16B),所以矩阵的行宽通常要求是 4 的倍数。

> 注意一个常见坑:向量化和 shared memory 的布局要配合。代码里把 `As` **转置存放**(`As[BK][BM]`),就是为了让从全局显存按 float4 读进来的连续 4 个 A 元素,在 shared 里能落到后续按列读取时不冲突的位置。这类布局技巧是高性能 GEMM 的"脏活",理解动机即可,细节看代码注释。

---

## 4. 循环展开:`#pragma unroll`

内层那些固定次数的小循环(`for k in BK`、`for i in TM`),编译器加 `#pragma unroll` 后会**完全展开成顺序指令**,消除:
- 循环计数器的加法和比较;
- 分支跳转(以及可能的分支预测开销);
- 让编译器能更好地做指令调度和寄存器分配(把独立的 FMA 排在一起、隐藏延迟)。

因为 TM/TN/BK 都是**编译期常量**(模板参数),编译器才能展开——这也是为什么我们用模板传 tile 尺寸,而不是运行时变量。

> 三板斧是配合使用的:寄存器分块**制造了**大量独立的 FMA;循环展开**把它们排开**让流水线填满;float4 向量化**喂得够快**不让 FMA 饿着。少了任何一个,另外两个的收益都打折。

---

## 5. 动手实验:画出性能曲线

`code/gemm_register.cu` 在 L2 两个 kernel 基础上加了寄存器分块版(BM=BN=64, BK=8, TM=TN=4 的稳妥配置),并可选地和 cuBLAS 对比。

编译运行:

```bash
cd code
# 不连 cuBLAS(只看三版手写对比):
nvcc -O3 gemm_register.cu -o gemm_register
# 连 cuBLAS 当标杆(强烈建议):
nvcc -O3 -DUSE_CUBLAS gemm_register.cu -o gemm_register -lcublas
./gemm_register 2048
```

参考输出(A100,数值随卡不同):

```
[INFO] GEMM M=N=K=2048
[INFO] naive    :  51.3 ms,  334 GFLOP/s
[INFO] tiled    :   9.1 ms, 1887 GFLOP/s
[INFO] register :   2.4 ms, 7160 GFLOP/s   (correct)
[INFO] cublas   :   1.6 ms, 10700 GFLOP/s
[INFO] register reaches 66.9% of cuBLAS
```

把不同 N 的 GFLOP/s 画成曲线(横轴 N,纵轴 GFLOP/s),你会看到:

```
GFLOP/s
  ▲
  │                       ●────●────●  cuBLAS(标杆)
  │                  ●────                register(手写,逼近)
  │             ●───────●────●────●     register
  │        ●──●
  │   ●────●────●────●────●────●        tiled
  │ ●──────────────────────────●        naive(基本贴地)
  └────────────────────────────────►  N
```

**读图要点**:
- 手写寄存器分块能到 cuBLAS 的 60%~85%,对一个几十行的 kernel 已经非常可观——剩下的差距是 cuBLAS 的 Tensor Core、double buffering(预取下一段)、针对每个尺寸的 autotune。
- N 越大,各版本越能发挥(算术强度越高,越靠近算力屋顶,呼应 L1 的 AI=N/6)。
- 永远以 cuBLAS 为标杆判断"还有多少空间",而不是凭感觉。

### 留给你的 TODO

代码里有 `TODO`:把 `TM,TN` 从 4×4 调到 8×8,benchmark 对比;再试 16×16 观察寄存器溢出导致的性能下跌(用 `nvcc --ptxas-options=-v` 看 register/spill 用量)。

---

## 6. 一句话带过:Tensor Core / WMMA

现代 N 卡(Volta 起)有专门的 **Tensor Core**,一条指令直接算一个小矩阵块的乘加(如 16×16×16),FP16/BF16/TF32 吞吐是普通 FP32 FMA 的若干倍。cuBLAS、CUTLASS 的极致性能主要来自它。

编程上可以用 `nvcuda::wmma` API(Warp Matrix Multiply-Accumulate)或更上层的 CUTLASS 模板。**本系列不深入手写 WMMA**(收益/复杂度对学习阶段不划算),你只需知道:
- 它存在,是 GEMM 逼近峰值的最后一块拼图;
- 它要求数据是低精度(FP16/BF16/...)、特定 tile 形状;
- 实践中通过 cuBLAS / Triton(下一课)/ CUTLASS 间接使用它,而不是裸写。

> 记住优先级:**先用对库(cuBLAS/Triton),看懂 profile;只在库覆盖不到的融合算子里才手写 kernel。** 手写 GEMM 的价值是"理解",不是"重造 cuBLAS"。

---

## 练习题

1. 寄存器分块把计算访存比从 1:2 提到 `TM*TN:(TM+TN)`。TM=TN=8 时具体是多少?说明它为什么能提高 FMA 单元利用率。
2. 为什么 `acc[TM][TN]` 必须是编译期常量大小、且不能开太大?寄存器溢出会发生什么?
3. float4 向量化为什么要求 16 字节对齐?如果矩阵列数不是 4 的倍数怎么办?
4. 为什么 `#pragma unroll` 只对循环次数是编译期常量的循环有效?它消除了哪些开销?

<details>
<summary>参考答案</summary>

1. 8×8 时是 64:16 = **4:1**,即每从 shared memory 读 1 个数做 4 次 FMA。tiled 是每读 2 个做 1 次(1:2)。比例越高,FMA 单元等待访存的时间占比越低,利用率越高,越接近算力屋顶。
2. 必须编译期常量,编译器才能把 `acc` 分配到寄存器并展开循环。寄存器总数有限(每线程约 255 个),`acc` 太大(如 16×16=256)放不下,会**溢出(spill)到本地内存**(实际在全局显存里),访存暴增,性能断崖下跌——这与寄存器分块的初衷(减少访存)背道而驰。
3. `float4` 是 128 bit = 16 字节,硬件的 128-bit load/store 指令要求地址按 16 字节对齐,否则非法或退化。列数非 4 的倍数时:对矩阵做 padding 补到 4 的倍数,或对边界部分用标量回退路径处理。
4. `unroll` 需要在编译期知道展开几份;次数是运行时变量时无法静态展开。它消除了循环计数器的自增/比较、分支跳转,并让编译器把展开后相互独立的 FMA 重排、调度以隐藏指令延迟、优化寄存器分配。

</details>

---

## 小结

- tiling 之后的瓶颈是**计算访存比太低**(每 FMA 夹着 shared 访存);**寄存器分块**让一个线程算 `TM×TN` 个输出,把复用从 shared 下沉到寄存器,比例提到 `TM*TN:(TM+TN)`。
- **float4 向量化**减少 4 倍访存事务;**`#pragma unroll`** 消除循环开销、利于指令调度;三者配合才有效。
- 手写可达 cuBLAS 的 60%~85%;剩余差距来自 **Tensor Core / WMMA**、double buffering、autotune——了解即可,实践用库。
- 存储层级的核心套路:**全局显存 → shared(block 复用)→ 寄存器(线程复用)**,每下沉一层就复用一次、抬高算术强度。

## 自测验收
- [ ] 能解释 tiled 之后为什么还慢(计算访存比),以及寄存器分块如何解决。
- [ ] 能说清 `acc[TM][TN]` 的寄存器约束与溢出后果。
- [ ] `gemm_register.cu` 跑通且 correct,并能和 cuBLAS 对比出百分比。
- [ ] 能复述"全局→shared→寄存器"三层复用的优化主线。

下一课:**Lesson 4 — Triton 入门**,我们换一种武器:用 Python 写出接近手写 CUDA 的高性能 kernel,从此摆脱大部分 C++ 苦力活。
