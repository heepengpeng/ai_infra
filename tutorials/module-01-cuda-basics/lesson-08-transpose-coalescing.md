# M1 · Lesson 8:矩阵转置与访存合并

> 矩阵转置 `B[i][j] = A[j][i]` 不做任何算术,纯粹搬数据。正因为"零计算",它成了检验你**访存合并(coalescing)** 功力的最纯净的标尺:写得好能接近显存峰值带宽,写不好慢好几倍。本课用转置把 Lesson 1 的 coalescing、Lesson 6 的 shared memory tile 与 bank conflict **全部落地**,并教你用 `ncu` 亲眼看到差距。这是 Module 2 手写 GEMM 的直接前置。
> 预计用时:2.5 小时。
> 前置:Lesson 1(coalescing)、Lesson 3(2D 索引、行优先布局)、Lesson 5(带宽计算)、Lesson 6(shared tile、bank conflict、padding)。

## 学习目标

1. 理解为什么"朴素转置"必然有一端访存不合并,并能从行优先布局推出来。
2. 会用 **shared memory tile** 把转置拆成"合并读 → 块内转置 → 合并写"。
3. 理解 tile 转置为什么会引入 bank conflict,以及为什么 `[TILE][TILE+1]` 的 padding 能消除它。
4. 会用 cudaEvent 测带宽、用 `ncu` 观察 coalescing 与 bank conflict 指标,形成"看数据下结论"的习惯。

---

## 1. 转置的本质:行列互换 = 访存模式的错位

`A` 是 `width × height` 行优先存储,转置后 `B[col][row] = A[row][col]`。先看最直白的 kernel:

```cpp
__global__ void transpose_naive(const float* A, float* B, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // A 的列
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // A 的行
    if (col < width && row < height) {
        // 读 A[row][col],写到 B[col][row]
        B[col * height + row] = A[row * width + col];
    }
}
```

它结果完全正确,但慢。关键看一个 **warp(32 个 threadIdx.x 连续的线程)** 到底访问了哪些地址。回顾 Lesson 3:同 warp 内 `threadIdx.x` 连续 → `col` 连续、`row` 相同。

**读 A 端**:`A[row*width + col]`,col 连续 → 地址连续 → **完美合并读**。✓

**写 B 端**:`B[col*height + row]`,col 连续而它乘了 `height` → 相邻线程写的地址**相隔 height 个元素**(跨一整行)→ 完全**不合并(strided)**,32 个线程的写被拆成 32 次独立内存事务。✗

```
读 A(同一 warp,row 固定,col=0,1,2,...):
  地址: A[row*W+0], A[row*W+1], A[row*W+2], ...   连续! → 1 次合并事务

写 B(同一 warp,col=0,1,2,..., row 固定):
  地址: B[0*H+row], B[1*H+row], B[2*H+row], ...   相隔 H! → 32 次散乱事务
```

> 核心结论:**转置天然让"读"和"写"中至少一端的访问步长是矩阵宽度,无法两端都合并。** 朴素写法把不合并丢给了写端,带宽利用率因此可能只有合并时的几分之一。这正是 Lesson 1"访存合并"那段抽象文字的最直观代价。

---

## 2. 解法:用 shared memory tile 把"错位"挡在片上

既然全局内存的读、写不能同时合并,思路是:**让对全局内存的读和写都保持合并,把"行列互换"这件不规则的事放到高速的共享内存里做**。共享内存是片上 SRAM,随机访问的代价远小于全局显存的非合并访问。

三步走(tile 取 32×32):

```
① 合并读:一个 block 把 A 的一个 32×32 子块整齐读进 shared tile
          (warp 内 col 连续 → 读 A 合并 ✓)

② 块内转置:在 shared 里把 tile 转置(写 tile[ty][tx],读 tile[tx][ty])
          —— shared 随机访问便宜,这步不碰全局显存

③ 合并写:把转置后的 tile 整齐写回 B 的对应子块
          (重新计算输出坐标,让 warp 内连续线程写 B 的连续地址 → 写 B 合并 ✓)
```

代码骨架(完整见 `code/transpose.cu`):

```cpp
#define TILE 32
__global__ void transpose_tiled(const float* A, float* B, int width, int height) {
    __shared__ float tile[TILE][TILE + 1];      // +1 padding,见第 3 节

    // --- ① 合并读:从 A 读入,按 (行,列) 原样放进 tile ---
    int x = blockIdx.x * TILE + threadIdx.x;     // A 的列
    int y = blockIdx.y * TILE + threadIdx.y;     // A 的行
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];   // 读 A 合并

    __syncthreads();                              // tile 装满再用

    // --- ③ 合并写:输出坐标按 block 转置后重新算,保证写 B 也连续 ---
    int xt = blockIdx.y * TILE + threadIdx.x;     // 注意:用 blockIdx.y!
    int yt = blockIdx.x * TILE + threadIdx.y;
    if (xt < height && yt < width)
        B[yt * height + xt] = tile[threadIdx.x][threadIdx.y]; // 读 tile 时转置
}
```

要点:**输出端把 block 的 x、y 角色对调**(`blockIdx.y*TILE + threadIdx.x`),这样写 B 时同一 warp 内连续的 `threadIdx.x` 对应 B 里连续的地址 → 合并写。"转置"动作发生在 `tile[threadIdx.x][threadIdx.y]` 这个**读 shared 时的下标交换**上。

---

## 3. tile 转置的暗礁:bank conflict 与那个 +1

第 2 节代码里 shared 数组写成了 `tile[TILE][TILE + 1]` 而不是 `[TILE][TILE]`。这正是 Lesson 6 学的 bank conflict 解药,这里是它最经典的实战。

分析:步骤 ② 读 tile 时是 `tile[threadIdx.x][threadIdx.y]`——一个 warp 内 `threadIdx.x` 连续(它是读 shared 的"行"下标),即同一 warp 的 32 个线程访问 `tile[0][ty], tile[1][ty], ..., tile[31][ty]`,**按列访问**。

若行宽是 32:`tile[k][ty]` 的 float 下标 = `k*32 + ty`,32 个线程(k=0..31)全部落在**同一个 bank**(下标模 32 相同)→ **32 路 bank conflict**,这一步慢 32 倍,刚省下的合并收益被吃掉一大半。

加一列 padding,行宽变 33:下标 = `k*33 + ty`,因 33 与 32 互质,32 个线程落到 **32 个不同 bank** → 冲突归零。代价仅是每个 tile 多 32 个 float 的共享内存。

```
[32][32] 按列访问:  下标 k*32+ty,bank = (k*32+ty)%32 = ty%32  → 全同 bank,32 路冲突
[32][33] 按列访问:  下标 k*33+ty,bank = (k*33+ty)%32 = (k+ty)%32 → 32 个不同 bank,0 冲突
```

> 一个 `+1` 就能换来可观加速,这是 CUDA 里"知道原理 vs 不知道"差距最戏剧化的例子之一。务必能自己推导上面那行 `(k*33+ty)%32 = (k+ty)%32`。

---

## 4. 三版性能对照与判读

对一个较大的方阵(如 4096×4096 float)做转置,典型趋势:

| 版本 | 关键 | 相对带宽 | 瓶颈 |
|---|---|---|---|
| naive | 直接 `B[col*H+row]=A[...]` | 低 | 写端非合并 |
| tiled 无 padding | shared tile,`[32][32]` | 中 | bank conflict |
| tiled + padding | shared tile,`[32][33]` | 高(接近峰值) | 已接近带宽上限 |

转置是**纯带宽受限**(零算术),所以判读标准很干脆:用 Lesson 5 的公式算**有效带宽**。转置读 `n*4` + 写 `n*4` = `2*n*4` 字节,带宽越接近显存峰值越好。

> 经验:padding 版的 tiled 转置通常能到峰值带宽的 **70%~90%**。如果你的 naive 版只有峰值的 10%~20%,那中间这 4~5 倍就是"理解 coalescing"实打实换来的。

---

## 5. 用 ncu 看见"看不见的"访存行为

口说无凭,Nsight Compute(`ncu`)能直接量出 coalescing 和 bank conflict。常用命令:

```bash
# 整体性能 + 访存效率概览
ncu --set basic ./transpose

# 直接看全局访存合并效率(每次请求的扇区利用率,越低说明越不合并)
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio ./transpose

# 看 shared memory 的 bank conflict(load 端)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./transpose
```

你应当观察到:

- **naive 版**:写端的 sectors/request 很高(请求被放大,合并差)。
- **tiled 无 padding**:全局访存合并改善,但 shared bank conflict 指标非零。
- **tiled + padding**:全局合并好、bank conflict 归零,带宽最高。

> 养成习惯:**优化不靠猜,靠 profiler。** 大数据里你看 Spark UI 的 shuffle read/spill 来定位瓶颈;GPU 里就看 `ncu` 的访存指标。这套"测量—定位—验证"方法论会贯穿后面所有性能课(Module 2 GEMM、FlashAttention 都靠它)。

---

## 6. 动手实验

### 实验 A:三版转置 benchmark(必做)
`code/transpose.cu` 含 naive / tiled(无 padding)/ tiled+padding 三版,统一 cudaEvent 计时、统一校验正确性,并打印各自有效带宽。先跑通,记录三版带宽比值:

```bash
nvcc -O3 transpose.cu -o transpose && ./transpose
```

### 实验 B:补全 padding 版并验证收益(必做)
`transpose.cu` 里 tiled+padding 版的 shared 声明和输出索引留了 TODO,补全后确认它带宽最高、结果正确。把 padding 的 `+1` 去掉再跑,亲眼看带宽下降——这就是 bank conflict 的代价。

### 实验 C(选做):上 ncu
若环境有 `ncu`(本地或 AutoDL),按第 5 节命令分别 profile 三版,对照"合并效率"和"bank conflict"指标,把第 1、3 节的理论和真实硬件计数器对上号。

---

## 练习题

1. 朴素转置里,为什么"读 A"是合并的而"写 B"不是?从行优先布局和 warp 内 threadIdx.x 连续两点推导。
2. tiled 版为什么能让全局内存的读和写**都**合并?"行列互换"这件不合并的事被挪到哪里做了?
3. 为什么 `tile[32][32]` 在按列读时是 32 路 bank conflict?推导 `[32][33]` 为什么变成 0 冲突。
4. 转置的"理论最优带宽"应该接近什么?为什么说它是判断转置 kernel 好坏的硬标准?
5. 如果矩阵不是 32 的整数倍(比如 4000×3000),tiled 版还要注意什么?

<details>
<summary>参考答案</summary>

1. A 行优先存储,`A[row*W+col]`,同 warp 内 row 固定、col 连续(threadIdx.x 连续)→ 地址连续 → 合并读。写 `B[col*H+row]`,col 连续但乘了 H,相邻线程地址相隔 H → strided、不合并。
2. tiled 版**读 A**时让连续线程读 A 连续地址(合并),**写 B**时通过对调 block 的 x/y 角色让连续线程写 B 连续地址(合并);真正的"行列互换"放到**共享内存**里通过下标交换 `tile[tx][ty]` 完成,而 shared 的随机访问代价远低于全局非合并访问。
3. 按列读 `tile[k][ty]`(k=threadIdx.x=0..31),float 下标 `k*32+ty`,bank=`(k*32+ty)%32=ty`,32 个线程全落 bank ty → 32 路冲突。`[32][33]` 下标 `k*33+ty`,bank=`(k*33+ty)%32=(k+ty)%32`,k 从 0 到 31 取遍 32 个不同 bank → 0 冲突。
4. 接近**显存峰值带宽**。因为转置零算术、纯搬数据(读 `2*n*4` 字节量级),它必然是带宽受限,所以"实测带宽 / 峰值带宽"直接衡量 kernel 是否把内存喂满,是最干脆的好坏标准。
5. 边界处理:不满一个 tile 的边缘 block,读入和写出都要做边界检查(`if (x<width && y<height)`)避免越界;tile 内未被填充的位置不要参与写回。padding 仍照常用。

</details>

---

## 7. 模块收官

到这里 Module 1 的 8 课走完了,你已经把"GPU 为什么快"的直觉,变成了能写、能测、能优化的具体能力:

- **L1–L2**:架构直觉 + 第一个 kernel(线程模型、五步流程)。
- **L3–L5**:2D 索引、内存模型与传输、错误处理与正确计时。
- **L6–L8**:共享内存协作、归约、转置——两个经典模式吃透了 **divergence、coalescing、bank conflict、tiling** 四大性能武器。

这四件武器,正是下一模块手写高性能 GEMM 的全部基础。

## 小结

- 转置零算术、纯访存,是 **coalescing** 的最纯净标尺;朴素写法必有一端(写 B)非合并。
- **shared memory tile** 把转置拆成"合并读 → 片上转置 → 合并写",让全局读写都保持合并。
- tile 按列访问会触发 **bank conflict**,`[TILE][TILE+1]` 的 padding 消除它(`(k*33+ty)%32` 取遍 32 bank)。
- 转置是带宽受限,用**有效带宽 / 峰值**判读;优化靠 **`ncu`** 看真实访存指标,不靠猜。

## 自测验收(本模块收官)
- [ ] 能推导朴素转置为什么写端不合并。
- [ ] 能讲清 tiled 转置如何让读写都合并,以及转置动作发生在哪。
- [ ] 能推导 `[32][33]` padding 为何消除 bank conflict。
- [ ] 三版转置跑通,带宽递增,padding 版接近峰值。
- [ ] (选做)用 `ncu` 看到合并效率与 bank conflict 指标的变化。

下一模块:**Module 2 — GPU 性能优化与核心算子**。第一课先正式建立 **roofline 性能模型**(L1 埋的伏笔),然后手写 **GEMM(矩阵乘)**——你在本模块学的 tiling、shared memory、coalescing、bank conflict 将全部派上用场,把一个朴素矩阵乘一步步优化到接近 cuBLAS 的水平。
