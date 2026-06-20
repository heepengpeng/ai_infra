# M1 · Lesson 7:并行归约(reduce 求和)

> "把一个数组求和"在 CPU 上是一行 `for` 循环,但在 GPU 上它是一道经典难题:求和有**依赖**(每一步都要用到上一步的结果),天然反并行。本课用归约这个例子,把你前几课学的 shared memory、`__syncthreads`、warp、bank conflict、divergence **全部串起来实战**。我们写五个版本,一版比一版快,每一版都用 Lesson 5 的 benchmark 量出来、并解释清楚"快在哪"。
> 预计用时:3 小时(本模块最硬核的一课,值得慢慢啃)。
> 前置:Lesson 1(warp/divergence/coalescing)、Lesson 5(cudaEvent + 带宽)、Lesson 6(shared memory + `__syncthreads`)。

## 学习目标

1. 理解"归约"为什么不能像逐元素操作那样平凡并行,以及"树形归约"的思想。
2. 能写出 shared memory 树形归约,并解释 `__syncthreads()` 在每一层的作用。
3. 看懂三个经典优化:**消除 warp divergence**、**消除 bank conflict**、**warp shuffle 免共享内存**。
4. 用 benchmark + 带宽指标,亲手验证每版优化的收益,把"为什么更快"落到具体机制。

---

## 1. 归约是什么,为什么难

**归约(reduction)**:把一个数组用某个**可结合(associative)**的二元操作"压"成一个值。求和、求最大值、求最小值、求积都是归约。本课以求和为例:`sum = a[0]+a[1]+...+a[n-1]`。

CPU 版本平凡:

```cpp
float sum = 0;
for (int i = 0; i < n; ++i) sum += a[i];   // 串行,每步依赖上一步的 sum
```

难点在于这个 `sum +=` 是**串行依赖链**——第 i 步要用第 i-1 步的结果。GPU 有几千个线程,直接让它们都往一个 `sum` 上加会**数据竞争**(多个线程同时读改写同一个地址)。

但求和**可结合**:`(a+b)+(c+d) == a+b+c+d`。这给了并行的突破口——**树形归约**:

```
初始:  3   1   7   0   4   1   6   3        8 个数
        └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
第1步:   4       7       5       9          两两相加,4 个数(4 个加法并行)
        └───┬───┘       └───┬───┘
第2步:     11              14               2 个数(2 个加法并行)
            └───────┬───────┘
第3步:            25                        1 个数,完成
```

`n` 个数,串行要 `n-1` 步;树形归约只要 **log₂(n) 层**,每层内部全并行。8 个数 3 层,100 万个数也只要 20 层。这就是把"串行依赖"转成"对数层数的并行"。

> 类比:这正是 Spark/MapReduce 里 `reduce` 算子的并行化思想——分区内局部聚合 + 树形合并。你早就用过这个模式,只是现在要亲手在 GPU 线程层面实现它。

由于不同 block 之间不能直接同步(Lesson 6),完整方案是**两级归约**:每个 block 先把自己负责的一段归约成 1 个**部分和**,写回全局显存;再对这些部分和做第二次归约(可以再启动一次 kernel,或部分和已经很少时拿回 CPU 收尾)。本课聚焦"block 内怎么归约得快"。

---

## 2. 版本 0:基线——交错寻址(有 warp divergence)

最直观的 block 内树形归约:先把本 block 的数据搬进共享内存,然后每一层让"步长 stride 的整数倍"的线程做加法,stride 从 1 翻倍到 blockDim。

```cpp
__global__ void reduce_v0(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    s[t] = (gid < n) ? in[gid] : 0.0f;     // 搬入,越界补 0(归约的单位元)
    __syncthreads();

    // 交错寻址:stride = 1,2,4,...
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (t % (2 * stride) == 0) {        // ← 问题就在这个判断
            s[t] += s[t + stride];
        }
        __syncthreads();                    // 每层之间必须同步
    }
    if (t == 0) out[blockIdx.x] = s[0];     // 0 号线程写出本 block 的部分和
}
```

它能算对,但有个性能病:`if (t % (2*stride) == 0)`。

> **问题:warp divergence(回顾 Lesson 1)。** 第一层只有偶数线程(t=0,2,4...)干活,奇数线程闲置——同一个 warp 内一半线程走 if、一半不走,两条路径被串行化。而且活跃线程**分散**在各 warp 里,几乎每个 warp 都发散。这是最慢的版本。

---

## 3. 版本 1:连续寻址——消除 divergence

关键改动:**让活跃的线程是连续的一段(0,1,2,...),而不是隔一个一个**。把 stride 反过来,从大到小(blockDim/2 → 1),让前一半线程加后一半:

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {                 // 活跃线程是连续的 [0, stride)
        s[t] += s[t + stride];
    }
    __syncthreads();
}
```

为什么这就消除了 divergence?设 BLOCK=256:第一层 stride=128,线程 0~127 活跃、128~255 闲置。**前 4 个 warp(0~127)整体活跃,后 4 个 warp 整体闲置**——warp 内部不再有"一半走一半不走",发散消失。随着 stride 缩小,闲置的 warp 整批退出,直到只剩 warp 0。

ASCII 对比同一层(stride=4,8 线程):

```
v0 交错:  活跃线程 = 0,_,2,_,4,_,6,_   ← 同 warp 内隔位活跃 → 发散
v1 连续:  活跃线程 = 0,1,2,3,_,_,_,_   ← 活跃线程连成一段 → 整 warp 同进退
```

> 收获:**同样的加法次数、同样的 log 层数,只是把"谁干活"重排成连续段,就靠消除 warp divergence 拿到明显加速。** 这是 Lesson 1 那个抽象概念第一次帮你赚到真金白银的性能。

顺带,这个连续寻址访问共享内存 `s[t]` 和 `s[t+stride]` 的模式也基本无 bank conflict(stride 是 2 的幂且活跃线程连续,32 个线程落在不同 bank)。

---

## 4. 版本 2:首次加载就减半——别浪费一半线程

注意到一个浪费:搬入时我们启动了 `n` 个线程,但归约**第一层就有一半线程立刻闲置**。不如让**每个线程在搬入阶段就先加两个数**,等于免费做掉了第一层,且把 grid 缩小一半:

```cpp
int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;   // 每 block 管 2*BLOCK 个数
float v = (gid < n) ? in[gid] : 0.0f;
if (gid + blockDim.x < n) v += in[gid + blockDim.x];      // 加载时顺手加一次
s[t] = v;
__syncthreads();
// ... 后面和 v1 一样的连续寻址循环 ...
```

这一步让每个线程"上来就有活干",并把全局内存读取的并行度用满(两次读都是 coalesced 的)。对**带宽受限**的归约(它就是典型的带宽受限,计算只有加法、数据量大),少启动一半 block、让每个线程多读一个,是实打实的收益。

---

## 5. 版本 3:warp shuffle——最后 32 个线程免共享内存、免同步

当归约进行到 stride ≤ 32,**只剩一个 warp(32 线程)还在干活**。此时还在用 `__syncthreads()` 和 shared memory,有两点浪费:

1. 一个 warp 内的 32 个线程本就是**锁步(lockstep)**执行的(SIMT),它们之间其实**不需要 `__syncthreads()`** 这种全 block 屏障。
2. 通过 shared memory 中转(写 s、读 s)还要走片上内存。

CUDA 提供 **warp 级原语 `__shfl_down_sync`**:让一个 warp 内的线程**直接读另一个线程寄存器里的值**,不经过共享内存:

```cpp
// 把"我"上方 offset 个 lane 的 val 取过来。在寄存器间直接传递,极快。
__device__ float warp_reduce(float val) {
    // offset = 16,8,4,2,1:warp 内树形归约,全程在寄存器、无需 __syncthreads
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;   // lane 0 拿到这 32 个数的和
}
```

`0xffffffff` 是参与的 lane 掩码(32 位全 1 表示整个 warp 都参与)。完整 kernel:用 shared memory 归约到每个 warp 一个值,再用 `warp_reduce` 收尾;或更激进地整段用 shuffle。

> 为什么更快:**省掉了 shared memory 的读写往返,也省掉了最后 5 层的 `__syncthreads()` 开销**;数据在寄存器间流动,最快。这是现代 CUDA 归约的标配写法,后面 Module 2 的 softmax/LayerNorm 归约也都用它。

(注:`_sync` 后缀的原语是较新 CUDA 的要求,显式传 mask,保证 warp 内线程同步参与,避免老式 `__shfl_down` 在分歧执行下的未定义行为。)

---

## 6. 五版性能对照(示意)

在中端卡上对 `n = 2^24`(约 1678 万 float)求和,典型趋势如下(具体数字看你的卡,重点看相对关系):

| 版本 | 关键手法 | 相对耗时 | 主要瓶颈被解决 |
|---|---|---|---|
| v0 交错寻址 | 基线 | 1.00× | —(满身 divergence) |
| v1 连续寻址 | 消除 warp divergence | ~0.5× | divergence |
| v2 加载时减半 | 首层免费 + 减半 grid | ~0.35× | 闲置线程 + 带宽 |
| v3 warp shuffle | 收尾免 shared/同步 | ~0.28× | 同步开销 + shared 往返 |
| (对照)CPU 单线程 | `for` 循环 | 慢几十倍 | — |

> 关键认知:归约是**带宽受限**任务。优化到后期,你应该用 Lesson 5 的公式算**有效带宽**(读了 `n*4` 字节),看它逼近显存峰值带宽到什么程度。当带宽接近峰值,就说明"该读的数据已经以最快速度读完了",再优化计算是徒劳——这就是 roofline 的实战判读。

---

## 7. 动手实验

### 实验 A:跑通五版并 benchmark(必做)
`code/reduce.cu` 内含 v0~v3 四个 device 版本 + CPU 参考,统一用 cudaEvent 计时、统一校验正确性。直接跑,观察每版耗时与有效带宽:

```bash
nvcc reduce.cu -o reduce && ./reduce
```

对照表格,确认 v1 相对 v0 的提升确实来自 divergence 的消除(可以把 v0 的 `if (t % (2*stride)==0)` 和 v1 的 `if (t < stride)` 对照着读)。

### 实验 B:补全 warp shuffle 版(必做)
`reduce.cu` 里的 `warp_reduce`(v3 用到)留了 TODO,补全那个 `__shfl_down_sync` 循环,跑通并确认它最快且结果正确。

### 实验 C(选做):第二级归约
现在每个 block 输出一个部分和,代码里是把这些部分和拷回 CPU 求和收尾。试着改成**再启动一次同样的 reduce kernel** 对部分和归约,体会"两级归约 / kernel 即全局同步点"。

---

## 练习题

1. 归约为什么不能像向量加法那样"一个线程一个元素"平凡并行?树形归约把 `n-1` 次串行加法变成了多少层?
2. v0 和 v1 做的加法次数完全一样,为什么 v1 明显更快?用 warp divergence 解释,并说明 v1 里"哪些 warp 整体活跃、哪些整体闲置"。
3. v2 为什么要"加载时就加一个"?它解决了 v0/v1 的什么浪费?
4. 为什么最后 32 个线程可以不用 `__syncthreads()`?`__shfl_down_sync` 比走 shared memory 快在哪?
5. 你把归约优化到有效带宽达到显存峰值的 ~85%,还值得继续优化吗?为什么?

<details>
<summary>参考答案</summary>

1. 因为求和是**串行依赖**(每步用到上一步结果),多个线程同时写一个 `sum` 会数据竞争。利用加法可结合性做树形归约,把 `n-1` 次串行加法变成 **⌈log₂(n)⌉ 层**,每层内部并行。
2. 加法次数一样,但 v0 的 `t % (2*stride)==0` 让活跃线程**隔位分布**,几乎每个 warp 内部都"一半走一半不走" → divergence,两条路径串行化。v1 的 `t < stride` 让活跃线程是**连续段**:BLOCK=256、stride=128 时,warp 0~3(线程 0~127)整体活跃,warp 4~7 整体闲置,warp 内部不发散,所以更快。
3. v0/v1 启动 `n` 个线程,但归约第一层立刻有一半线程闲置,浪费。v2 让每个线程在**搬入阶段就先加两个全局元素**,等于免费做掉第一层,同时把 block/grid 数减半、把全局读带宽用满,对带宽受限的归约是实打实收益。
4. 一个 warp 的 32 线程在 SIMT 下**本就锁步执行**,天然同步,无需 block 级屏障 `__syncthreads()`。`__shfl_down_sync` 让线程**直接读彼此寄存器**,省掉 shared memory 的写入+读取往返以及屏障开销,所以更快。
5. **基本不值得了。** 归约是带宽受限任务,达到峰值 85% 说明显存带宽已接近喂满,"该读的字节已用接近最快速度读完",剩余 15% 多是不可避免的开销;继续抠计算对带宽受限任务无意义(roofline 上你已贴着内存屋顶/斜坡顶端)。

</details>

---

## 小结

- 归约是**有串行依赖**的操作,靠**可结合性 + 树形归约**把 `n-1` 步串行变成 `log n` 层并行;跨 block 靠两级归约。
- **v0→v1:连续寻址消除 warp divergence**——同样的计算量,重排"谁干活"就提速。
- **v1→v2:加载时先加一次**,免费做掉第一层、减半 grid、用满全局读带宽。
- **v2→v3:warp shuffle(`__shfl_down_sync`)** 让收尾的 32 线程在寄存器间归约,省掉 shared 往返与 `__syncthreads`。
- 归约是**带宽受限**:用有效带宽 / 峰值带宽判断优化是否到顶(roofline 实战)。

## 自测验收(过了再进 Lesson 8)
- [ ] 能讲清归约为什么难、树形归约为什么是 log 层。
- [ ] 能对照代码解释 v0 与 v1 的差异**就是** divergence 的消除。
- [ ] 五个版本都跑通、结果正确,且耗时趋势符合预期。
- [ ] 补全并理解 `__shfl_down_sync` 收尾,知道它为何免同步、免 shared。
- [ ] 会用有效带宽判断归约是否已优化到位。

下一课:**Lesson 8 — 矩阵转置与访存合并**。本课的归约让你吃透了 divergence;下一课的转置则让你吃透 **coalescing(访存合并)** 和 bank conflict——它是 Module 2 手写高性能 GEMM 的直接前置。
