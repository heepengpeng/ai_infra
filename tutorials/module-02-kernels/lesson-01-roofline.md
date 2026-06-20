# M2 · Lesson 1:性能模型 Roofline——先学会"判断瓶颈在哪"

> 进入 Module 2,我们的目标从"写对 kernel"升级到"写快 kernel"。但优化前必须先回答一个问题:**这段代码到底是被算力卡住,还是被带宽卡住?** 答错了方向,优化全是白费力气。Roofline 就是回答这个问题的标准工具。
> 你在 Module 1 Lesson 1 已经见过 roofline 的雏形(那张"屋顶 + 斜坡"的图),本课把它做实:会算、会测、会用它给真实 kernel 定性。
> 预计用时:2 小时(阅读 + 跑两个测速程序)。
> 前置:M1 全部(尤其 L1 的 roofline 直觉、L5 的 CUDA event 计时、L8 的 coalescing)。

## 学习目标

学完本课你应该能:
1. 手算一个 kernel 的**算术强度(Arithmetic Intensity, AI)**,单位 FLOP/Byte。
2. 实测出你手上这张卡的**峰值算力**和**峰值显存带宽**,画出属于你的 roofline。
3. 给定一个 kernel,判断它落在**带宽受限区**还是**算力受限区**,并据此选对优化方向。
4. 理解为什么大模型推理里"几乎一切逐元素算子和 decode 都是带宽受限"。

---

## 1. 一个大数据人最熟悉的直觉:吞吐 = min(算得多快, 喂得多快)

你在 Spark 调优里一定遇到过这种场景:一个 Stage 慢,你先得判断它是 **CPU-bound**(算子太重)还是 **IO/Shuffle-bound**(数据搬不过来)。判断错了,加 executor 内存对 CPU-bound 任务毫无帮助,加 CPU 对 IO-bound 任务也是浪费。

GPU 上完全是同一回事,只是两个"管道"换了名字:

- **算力管道**:GPU 每秒能做多少次浮点运算(FLOP/s)。
- **带宽管道**:GPU 每秒能从全局显存搬多少字节进 SM(Byte/s)。

一个 kernel 的实际性能,被这两个管道里**更窄的那个**卡死。Roofline 就是把这句话画成一张图。

> **一句话定性:计算强度高 → 大概率算力受限;计算强度低 → 大概率带宽受限。** 计算强度就是"每搬 1 字节,你能榨出多少次浮点运算"。

---

## 2. 算术强度(AI):核心量纲

定义:

```
算术强度 AI = 总浮点运算次数 FLOP / 总访存字节数 Byte   (单位:FLOP/Byte)
```

注意分母是**真正从全局显存读写的字节数**,不是逻辑上的数据量(命中缓存/复用到的不算重复搬)。这正是优化的着力点:**复用越多,分母越小,AI 越高,越往算力受限区移动。**

### 例 1:向量加 `c = a + b`(逐元素)

对每个元素 `c[i] = a[i] + b[i]`,以 float32(4 字节)计:

- FLOP:1 次加法 → 1 FLOP
- 字节:读 `a[i]`、读 `b[i]`、写 `c[i]` = 3 × 4 = 12 Byte

```
AI = 1 / 12 ≈ 0.083 FLOP/Byte
```

极低。这意味着每搬 12 字节才算 1 次加法——**铁定带宽受限**。再强的算力也喂不饱,因为数据根本来不及搬进来。

### 例 2:SAXPY `y = a*x + y`

- FLOP:1 次乘 + 1 次加 = 2 FLOP
- 字节:读 `x`、读 `y`、写 `y` = 12 Byte

```
AI = 2 / 12 ≈ 0.167 FLOP/Byte
```

还是带宽受限,只是比向量加好一倍。

### 例 3:方阵乘 GEMM `C = A·B`,A、B、C 都是 N×N(理想复用)

- FLOP:每个输出元素要做 N 次乘加 = 2N FLOP,共 N² 个输出 → **2N³ FLOP**
- 字节(理想最优,每个输入只从显存读一次):读 A、B、写 C = 3N² × 4 Byte

```
AI = 2N³ / (12N²) = N/6 FLOP/Byte
```

**关键结论:GEMM 的算术强度随 N 线性增长!** N=1024 时 AI ≈ 170 FLOP/Byte。这就是为什么大矩阵乘是 GPU 的主场,也是为什么后两课我们要拼命做 tiling/寄存器分块——**目的就是把这个理论上的高 AI 真正实现出来**(naive 实现会因为反复重读 A、B 把分母撑大、AI 拉低)。

> 把例 1 和例 3 放一起看:同样是 GPU,向量加只能跑到带宽上限,GEMM 却能逼近算力上限。**差别不在硬件,在算术强度。** Module 2 后面所有优化,本质都是"提高有效算术强度"。

---

## 3. Roofline 图:把两个上限画成屋顶

横轴算术强度 AI(FLOP/Byte,对数轴),纵轴可达算力(FLOP/s,对数轴):

```
可达算力 (FLOP/s, log)
  ▲
  │                      ┌──────────────── 屋顶:峰值算力 π (算力受限区)
  │                     /│
  │                    / │
  │   斜坡 = 带宽 β  /   │
  │  (带宽受限区)  /     │
  │              /       │
  │            /         │
  │          /           │
  └────────┴─────────────┴──────────────►  算术强度 AI (FLOP/Byte, log)
           AI*=π/β    (拐点 ridge point)
```

模型本身就一条公式:

```
可达算力 = min( π ,  β × AI )
                ↑        ↑
            峰值算力   带宽 × 算术强度
```

- 斜坡段 `β × AI`:你受带宽 β 限制,AI 越大跑得越高。
- 平台段 `π`:你撞到算力天花板,再高的 AI 也上不去。
- **拐点 AI\* = π / β**:左边带宽受限,右边算力受限。这个拐点是这张卡的"性格"。

举例,一张 A100(后面会实测,这里先用标称值):FP32 峰值 π ≈ 19.5 TFLOP/s,HBM2e 带宽 β ≈ 1555 GB/s。

```
拐点 AI* = 19.5e12 / 1555e9 ≈ 12.5 FLOP/Byte
```

含义:**算术强度低于约 12.5 的 kernel,在 A100 上一定是带宽受限。** 回头看例 1 向量加(AI≈0.083)、例 2 SAXPY(0.167),都远在拐点左边——再优化也只能贴着斜坡跑,天花板是带宽不是算力。而 GEMM(N=1024,AI≈170)在拐点右边,有机会逼近 19.5 TFLOP/s。

> 拐点告诉你一个朴素但深刻的事实:**现代 GPU 算力远远过剩,大多数算子其实"饿着"。** 这也是为什么"算子融合""量化"这些**减少访存**的技术,往往比"优化计算"收益大得多——后面 Lesson 5 会专门讲。

---

## 4. 怎么拿到 π 和 β:标称值不可信,要实测

厂商标称的峰值算力是"理论极限"(所有 FMA 单元满载、频率拉满),真实 kernel 几乎摸不到。带宽同理,标称是显存控制器理论值,实测通常只有 80%~90%。做 roofline 分析要用**实测峰值**,否则你会误判一个其实已经很优的 kernel"还差得远"。

实测方法:
- **峰值带宽**:跑一个**纯访存、零计算**的 kernel(比如把显存里一大块数据拷到另一块),AI 趋近 0,它必然贴着斜坡顶——测出的就是带宽上限。这就是本课的动手程序。
- **峰值算力**:跑一个**纯计算、几乎零访存**的 kernel(在寄存器里疯狂做 FMA),AI 趋于无穷,贴着平台跑——测出算力上限。原理类似,本课作为练习留给你。

带宽测量的标准做法是 **STREAM / memcpy 风格**:测 `copy`(读+写)。计算公式:

```
有效带宽 (Byte/s) = 实际搬运的字节数 / 耗时
```

对 `out[i] = in[i]` 这种 copy,每个元素读 4 写 4 = 8 字节,N 个元素就是 `8N` 字节。

---

## 5. 动手实验:实测你这张卡的峰值带宽

代码见 `code/bandwidth_test.cu`。核心是一个 copy kernel + 用 CUDA event 计时(M1 L5 学过),多次取最优避免抖动。先看 kernel 主体:

```cpp
// 纯访存 kernel:读一个 float、写一个 float,几乎没有计算。
// grid-stride loop:用有限线程覆盖任意大的数组(M1 经典写法)。
__global__ void copy_kernel(const float* __restrict__ in,
                            float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        out[i] = in[i];   // 每次循环:读 4 字节 + 写 4 字节
    }
}
```

编译运行(把数组开到几百 MB,远大于 L2 cache,才能测到真实 HBM 带宽):

```bash
cd code
nvcc -O3 bandwidth_test.cu -o bandwidth_test
./bandwidth_test
```

参考输出(A100,你的数值会不同):

```
[bandwidth] N = 67108864 floats (256.0 MB)
[bandwidth] best time = 0.356 ms over 50 runs
[bandwidth] moved 512.0 MB (read+write)
[bandwidth] effective bandwidth = 1438.2 GB/s
[bandwidth] vs spec 1555 GB/s -> 92.5% of peak
```

**读图要点**:
- 测到标称的 85%~95% 是正常且健康的,说明你的 copy kernel 已经把带宽吃满。
- 如果只有 50%,检查:数组太小(命中 L2 了)、访存没合并(M1 L8 的 coalescing,这里因为是连续 copy 所以天然合并)、block 配置太离谱。
- 把这个实测带宽 β 记下来,它是你后面所有 roofline 分析的标尺。

> 为什么用 `float` 连续 copy 就能测满带宽?因为一个 warp 的 32 个线程访问连续的 `in[idx]`,硬件合并成大事务(coalescing 拉满),这正是 M1 L8 的结论在性能上的兑现。

---

## 6. 动手实验:给一个真实 kernel 定性

现在把 roofline 用起来。`code/roofline_classify.cu` 里实现了 SAXPY(`y = a*x + y`),程序会:
1. 测它的实际耗时和达到的 GB/s。
2. 算它的算术强度 AI。
3. 用 `min(π, β×AI)` 算理论上限,告诉你它是带宽受限还是算力受限,以及离上限多远。

正文不重复贴全,关注它打印的判定逻辑:

```cpp
double ai = total_flop / (double)total_bytes;        // 算术强度
double ridge = peak_flops / peak_bw;                 // 拐点
const char* regime = (ai < ridge) ? "MEMORY-BOUND" : "COMPUTE-BOUND";
double roof = std::min(peak_flops, peak_bw * ai);    // roofline 上限
LOG_INFO("AI=%.3f FLOP/B, ridge=%.2f -> %s", ai, ridge, regime);
```

运行:

```bash
nvcc -O3 roofline_classify.cu -o roofline_classify
# 把上一步测到的带宽和你卡的算力传进去(下例 A100)
./roofline_classify --peak_bw 1438 --peak_tflops 19.5
```

你会看到 SAXPY 被判定为 `MEMORY-BOUND`,且实际带宽已接近峰值——**结论:SAXPY 已经优无可优,因为它本质就是带宽受限,你能做的只有"少搬数据"(比如融合到上游算子里,不要单独跑)。** 这正是 Lesson 5 算子融合的动机。

---

## 7. 把它接回大模型推理

用 roofline 重新理解 M1 L1 埋下的伏笔:

| 场景 | 主要运算 | 算术强度 | 瓶颈 | 优化方向 |
|---|---|---|---|---|
| 训练 / prefill(长序列) | 大 GEMM | 高(N/6) | 算力受限 | Tensor Core、提高 MFU |
| **decode(逐 token 生成)** | 矩阵×向量(GEMV) | **极低** | **带宽受限** | **量化**(减少要读的权重字节) |
| LayerNorm / softmax / 激活 | 逐元素 | 极低 | 带宽受限 | **算子融合** |
| Attention 中间矩阵 | GEMM + softmax | 中等但访存爆炸 | 带宽/显存受限 | **FlashAttention(不落地大矩阵)** |

这张表基本就是 Module 2 剩下 6 课的路线图:**先把 GEMM 的算力榨干(L2/L3),再用融合和 IO 感知把带宽受限的算子救回来(L5/L6/L7)。** 而判断每一步该往哪优化,靠的就是今天这把 roofline 尺子。

---

## 练习题

1. 一个 kernel 对每个 float 元素做 8 次 FMA(共 16 FLOP),读 1 个 float、写 1 个 float。算它的 AI。在 A100(π=19.5T, β=1438G,拐点≈13.6)上它是带宽受限还是算力受限?
2. 同样的 kernel,如果你把数据类型从 float32 换成 float16(2 字节),AI 怎么变?瓶颈区间会移动吗?(提示:FLOP 不变,字节减半)
3. GEMM 的理论 AI 是 N/6。但 naive 实现(每个线程独立从全局显存重读整行整列)实际访存远大于 3N²。定性说明:为什么 naive GEMM 的"有效 AI"远低于 N/6?这会把它推向 roofline 的哪一侧?
4. 仿照 `bandwidth_test.cu`,写一个测**峰值算力**的程序(在寄存器里循环做 FMA,几乎不访存),思路写出来即可。

<details>
<summary>参考答案</summary>

1. AI = 16 FLOP / (2×4 Byte) = 2 FLOP/Byte。2 < 13.6,**带宽受限**。
2. FLOP=16 不变,字节 = 2×2 = 4 Byte,AI = 16/4 = 4 FLOP/Byte,翻倍。瓶颈点右移了,但 4 仍 < 拐点(FP16 拐点也会变,因为 FP16 算力更高),通常**仍是带宽受限**。这也说明:低精度不仅省显存,还能提高算术强度。
3. naive GEMM 里,计算 `C[i][j]` 时整行 `A[i][:]` 和整列 `B[:][j]` 都要从全局显存读;相邻线程没有复用彼此已读的数据,导致同一份 A/B 数据被反复从显存读入,**实际搬运字节数远超 3N²**,分母被撑大,有效 AI 暴跌,把 kernel 推向**带宽受限侧**。tiling 的本质就是让一个 block 把数据搬进 shared memory 后被多个线程复用,缩小分母、把 AI 拉回理论值附近。
4. 写一个 kernel,每个线程在寄存器里持有几个累加变量 `acc`,做一个长循环 `acc = acc * x + y;`(几千次,编译器无法优化掉因为有数据依赖且最后要写出),循环外只写一次结果到全局显存。这样字节≈0、FLOP 巨大、AI→∞,kernel 贴着算力平台跑,测得的 FLOP/s 即峰值算力。注意要让 occupancy 足够高、用 `-O3`、防止编译器把循环常量折叠掉。

</details>

---

## 小结

- Roofline 一条公式:**可达算力 = min(峰值算力 π, 带宽 β × 算术强度 AI)**。
- **算术强度 AI = FLOP / 从显存搬运的字节数**;低 AI → 带宽受限,高 AI → 算力受限,分界是**拐点 AI\* = π/β**。
- 标称峰值不可信,要**实测**:纯 copy kernel 测带宽,纯 FMA kernel 测算力。
- 逐元素算子、decode 都是带宽受限(AI 极低),GEMM 是算力受限(AI 随 N 增长);**优化前先用 roofline 定性,再选方向**。

## 自测验收(过了再进 Lesson 2)
- [ ] 能手算向量加、SAXPY、GEMM 的算术强度。
- [ ] 能解释 roofline 的斜坡、平台、拐点各代表什么。
- [ ] `bandwidth_test.cu` 跑通,测出自己卡的实测带宽并解释占标称的百分比。
- [ ] 能说清为什么 decode 和逐元素算子是带宽受限,以及对应的优化方向。

下一课:**Lesson 2 — 手写 GEMM(一):从 naive 到 shared memory 分块**,我们把今天算出的"GEMM 理论 AI = N/6"真正变成性能。
