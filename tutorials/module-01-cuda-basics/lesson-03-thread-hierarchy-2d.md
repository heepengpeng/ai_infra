# M1 · Lesson 3:线程层级深入与 2D 索引

> 上一课你用一维线程做了数组平方。但深度学习里最常见的数据是**矩阵**和**张量**——它们天然是二维、三维的。本课把线程层级讲透,并学会用 2D grid/block 直接对应矩阵的行和列。
> 预计用时:2 小时。
> 前置:Lesson 2(全局线程 id 公式、CUDA 五步流程、`<<<grid, block>>>` 启动语法)。

## 学习目标

1. 彻底理解 grid/block/thread 的**多维**组织,会用 `dim3` 配置二维启动参数。
2. 会推导二维全局索引:`row = blockIdx.y * blockDim.y + threadIdx.y`、`col = blockIdx.x * blockDim.x + threadIdx.x`。
3. 理解"逻辑二维 + 物理一维"的本质:为什么内存是一维的、`row * width + col` 这个映射怎么来。
4. 能判断一个问题该用 1D 还是 2D 索引,并解释 2D 只是**写法方便**,不改变 warp 的调度本质。

---

## 1. 为什么需要二维:别让索引算术埋了你

先看一个最朴素的需求:两个 `M×N` 的矩阵逐元素相加,`C = A + B`。

如果只用一维线程,你得在 kernel 里手动把一维 tid 拆回行列:

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int row = tid / N;   // 一维 id 反推行
int col = tid % N;   // 一维 id 反推列
```

能用,但有两个问题:一是 `/` 和 `%` 这种整数除法在 GPU 上不便宜;二是当数据真的是二维网格(图像、特征图、attention 分数矩阵)时,**人脑映射成本高、容易写错**。CUDA 直接提供了二维(乃至三维)的线程组织,让你写 `row`/`col` 就像写嵌套循环一样自然。

> 记住定位:**多维索引是给程序员的"语法糖",方便把线程网格对齐到数据网格;它不改变底层 warp 仍是 32 线程一组顺序打包这一事实。** 性能规律(coalescing、divergence)依旧由底层一维排布决定,这点本课第 4 节会点破。

---

## 2. dim3:线程层级其实一直是三维的

Lesson 2 里你写的 `<<<2, 4>>>`,其实是 `<<<dim3(2,1,1), dim3(4,1,1)>>>` 的简写。CUDA 的 grid 和 block **天生就是三维的**,只是没用到的维度默认为 1。

`dim3` 是 CUDA 内置的三元结构体:

```cpp
dim3 block(16, 16);      // x=16, y=16, z=1 → 每个 block 有 16*16 = 256 个线程
dim3 grid(4, 8);         // x=4,  y=8,  z=1 → grid 里有 4*8 = 32 个 block
my_kernel<<<grid, block>>>(...);
```

对应六个内置变量(都带 `.x/.y/.z`):

| 变量 | 含义 | 本例的值 |
|---|---|---|
| `gridDim` | grid 在各维上有多少 block | `(4, 8, 1)` |
| `blockIdx` | 当前 block 在 grid 中的坐标 | `x∈[0,4)`, `y∈[0,8)` |
| `blockDim` | 每个 block 在各维上有多少线程 | `(16, 16, 1)` |
| `threadIdx` | 当前线程在 block 内的坐标 | `x∈[0,16)`, `y∈[0,16)` |

一个约束要记住:**一个 block 内的线程总数(x×y×z)不能超过 1024**。所以 `16×16=256`、`32×32=1024` 都合法,`32×32×2` 就超了。这个上限和后面共享内存、寄存器分配都相关,现在先记住数字。

---

## 3. 二维索引公式:把线程网格盖在矩阵上

想象把一张"线程网格"像渔网一样盖在矩阵上,每个线程负责网下的一个元素。每一维都套用 Lesson 2 那条全局 id 公式,只是分别用 x 和 y:

```
col = blockIdx.x * blockDim.x + threadIdx.x;   // 列方向(横,x)
row = blockIdx.y * blockDim.y + threadIdx.y;   // 行方向(纵,y)
```

ASCII 图解(block 是 2×2,grid 是 2×2,处理一个 4×4 矩阵):

```
        col→  0      1   |   2      3
 row         ┌──────────┬──────────┐
  ↓     0    │(0,0)(0,1)│(0,2)(0,3)│   block(0,0) 管左上, block(1,0) 管右上
        1    │(1,0)(1,1)│(1,2)(1,3)│
             ├──────────┼──────────┤
        2    │(2,0)(2,1)│(2,2)(2,3)│   block(0,1) 管左下, block(1,1) 管右下
        3    │(3,0)(3,1)│(3,2)(3,3)│
             └──────────┴──────────┘

看 row=2,col=3 这个元素由谁负责?
  blockIdx.y=1, threadIdx.y=0 → row = 1*2 + 0 = 2  ✓
  blockIdx.x=1, threadIdx.x=1 → col = 1*2 + 1 = 3  ✓
```

> 注意约定:**x 对应列(矩阵的横向、宽度 width),y 对应行(矩阵的纵向、高度 height)**。初学最爱踩的坑就是把 x/y 和 row/col 对应反了。记一个口诀:**x 像横轴往右走,就是 col**。

---

## 4. 关键:逻辑二维,物理一维

GPU 显存(以及 C/C++ 里的数组)是**一维线性**的。一个 `height × width` 的矩阵在内存里按**行优先(row-major)**铺平:

```
矩阵 (2行3列):                内存里实际是一条:
  a00 a01 a02                 [a00, a01, a02, a10, a11, a12]
  a10 a11 a12                  ↑idx0          ↑idx3
```

所以二维坐标 `(row, col)` 访问元素时,必须**自己换算成一维下标**:

```cpp
int idx = row * width + col;   // 行优先:跳过前面 row 整行,再走 col 列
out[idx] = a[idx] + b[idx];
```

这一步至关重要——它把"逻辑上的二维"翻译成"物理上的一维"。这也解释了为什么前面说 2D 只是语法糖:数据本就是一维的,2D 索引帮你算坐标,但**真正决定访存效率的是相邻线程访问的一维地址连不连续**。

回顾 Lesson 1 的 coalescing:一个 warp 是 32 个 `threadIdx.x` 连续的线程(x 维度优先打包)。它们的 `col` 连续 → `idx = row*width + col` 也连续 → 访存合并,带宽拉满。

反过来,如果你手贱写成 `idx = col * height + row`(列优先),同一个 warp 内 x 连续的线程访问的地址就会**间隔 height** 个元素,访存严重不合并,带宽暴跌。

> 一句话:**让 `threadIdx.x` 这一维去扫描内存里连续的那一维(即 col)。** 这是 2D kernel 性能的第一原则,本质和 Lesson 8 的转置优化是同一回事。

---

## 5. 完整范例:二维矩阵逐元素加法

把上面拼起来,这是标准的 2D kernel 写法(完整代码见 `code/matadd2d.cu`):

```cpp
__global__ void mat_add(const float* A, const float* B, float* C,
                        int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // 二维边界检查:行、列都要拦,因为两个维度都向上取整了
    if (row < height && col < width) {
        int idx = row * width + col;   // 二维坐标 → 一维下标(行优先)
        C[idx] = A[idx] + B[idx];
    }
}
```

host 端启动配置——grid 在**两个维度上都要向上取整**:

```cpp
dim3 block(16, 16);
dim3 grid((width  + block.x - 1) / block.x,    // x 方向覆盖 width
          (height + block.y - 1) / block.y);   // y 方向覆盖 height
mat_add<<<grid, block>>>(d_A, d_B, d_C, width, height);
```

这里 `(width + block.x - 1) / block.x` 就是 Lesson 2 那个向上取整套路,只是现在要在 x、y 两个方向各做一次。

> 思考:逐元素加法其实**用 1D 也完全能做**(把矩阵当成 `width*height` 的长数组,一条 tid 走到底,而且更省一次乘法)。这里用 2D 纯粹是为了练习 2D 索引。**2D 真正不可替代的场景是 Lesson 8 的转置和 Module 2 的矩阵乘 tiling**——那里线程要按二维块协作,1D 写起来会非常别扭。

---

## 6. 到底什么时候用 1D,什么时候用 2D

| 场景 | 推荐 | 原因 |
|---|---|---|
| 逐元素一元/二元运算(加、激活、缩放) | **1D** | 数据可看作平铺长数组,1D 更简洁、少一次坐标换算 |
| 需要按"行"或"列"做规约(如 softmax 按行) | 1D 或 2D 均可 | 常让一个 block 负责一行,看实现风格 |
| 矩阵乘、转置、卷积、stencil(分块协作) | **2D** | 线程要按二维 tile 协作,2D 索引天然对齐数据布局 |
| 处理图像/特征图(H×W×C) | **2D/3D** | 维度直接对应空间结构,可读性最高 |

经验法则:**问题的数据访问模式是几维、线程是否需要按几维分块协作,就用几维。** 不要为了"显得高级"硬上 3D——维度越多,索引和边界越易错。

---

## 7. 动手实验

### 实验 A:跑通矩阵加法(必做)
1. 手敲 `code/matadd2d.cu`,用 `width=1024, height=768`,编译运行,程序内会自动校验 `C[i]==A[i]+B[i]`。

```bash
nvcc matadd2d.cu -o matadd2d && ./matadd2d
```

2. 在 kernel 里 `printf` 出 `row==0 && col==0` 这个线程的 `blockIdx`/`threadIdx`,验证你对索引的理解(注意加 `if` 否则几十万行刷屏)。

### 实验 B:补全坐标打印(必做)
补全 `code/index2d.cu` 的 TODO:启动一个 `dim3 block(4,4)`、`dim3 grid(2,2)` 的小网格,让**每个线程打印自己的 `(row, col)` 和换算出的一维 `idx`**(矩阵宽设为 8),手动核对几个值。这是把第 3、4 节吃透的最佳方式。

---

## 练习题

1. `dim3 block(32, 32)` 合法吗?`dim3 block(32, 32, 2)` 呢?为什么?
2. 矩阵 `width=1000, height=1000`,`block(16,16)`,grid 的 x、y 各是多少?一共启动多少线程?多出来多少个被边界检查拦掉?
3. 如果把 kernel 里的下标写成 `idx = col * height + row`(列优先),程序结果对吗?性能会怎样?用 Lesson 1 的 coalescing 解释。
4. 为什么逐元素加法用 1D 反而更好,但矩阵转置必须用 2D?

<details>
<summary>参考答案</summary>

1. `block(32,32)` 合法,共 1024 线程,正好等于上限。`block(32,32,2)` = 2048 > 1024,**非法**,kernel 启动会失败(配合 Lesson 5 的错误检查能看到 `invalid configuration argument`)。
2. `grid.x = (1000+15)/16 = 63`,`grid.y = 63`。共 `63*16 × 63*16 = 1008 × 1008 = 1,016,064` 个线程;有效元素 `1000×1000 = 1,000,000`;被拦掉 `16,064` 个(右边一条 + 下边一条边缘线程)。
3. **结果仍然正确**(只要读写用同一个 idx,加法对每个元素独立)。但**性能很差**:同一 warp 内 `threadIdx.x` 连续的线程,`col` 连续,而 `col*height+row` 让它们的地址相隔 `height`,访存完全不合并,带宽利用率可能只有合并时的几分之一。
4. 逐元素加法每个线程只读写自己的一个元素、无跨线程协作,平铺成一维长数组最简洁且省一次乘法。转置则要把一块数据"行列互换"地搬运,涉及二维块内的协作与 shared memory tile,用 2D 索引才能自然表达(Lesson 8 详解)。

</details>

---

## 小结

- grid/block **天生三维**,用 `dim3` 配置;一个 block 内线程数(x×y×z)**上限 1024**。
- 二维索引:`col = blockIdx.x*blockDim.x + threadIdx.x`,`row = blockIdx.y*blockDim.y + threadIdx.y`;**x 配 col,y 配 row**。
- **逻辑二维、物理一维**:必须用 `idx = row*width + col`(行优先)换算成一维下标。
- 多维只是**语法糖**,warp 仍按 `threadIdx.x` 连续打包;**让 x 维扫描内存连续维**才能 coalescing。
- 选维度看数据访问模式:逐元素用 1D,分块协作(转置/GEMM)用 2D。

## 自测验收(过了再进 Lesson 4)
- [ ] 能默写二维 `row`/`col` 公式,并说清 x/y 与 col/row 的对应。
- [ ] 能解释"逻辑二维物理一维"和 `row*width+col` 的由来。
- [ ] `matadd2d.cu` 跑通并通过自校验。
- [ ] 能用 coalescing 解释为什么不能写成 `col*height+row`。
- [ ] 能说出 1D 与 2D 各自的适用场景。

下一课:**Lesson 4 — CUDA 内存模型与数据传输**,搞清楚 `cudaMalloc`/`cudaMemcpy` 背后的内存层级,以及为什么 H2D/D2H 传输常是性能第一杀手。
