# M2 · Lesson 4:Triton 入门——用 Python 写高性能 kernel

> 前三课你用 CUDA C++ 手写了 GEMM,体会到了寄存器分块、向量化、bank conflict 这些"脏活"的威力,也体会到了它们的繁琐。本课介绍 **Triton**:一个让你**用 Python 写出接近手写 CUDA 性能**的 kernel 语言。FlashAttention、绝大多数现代融合算子的开源实现,都用 Triton 写。
> 关键转变:在 CUDA 里你管"每个线程",在 Triton 里你管"每个 block"——线程内的调度、向量化、shared memory 由编译器替你做。
> 预计用时:2.5 小时(装环境 + 跑两个 kernel + 对比 PyTorch)。
> 前置:M2 L1~L3(算术强度、tiling、向量化的直觉);会用 PyTorch 基本张量操作。

## 学习目标

1. 说清 Triton 解决了什么痛点,以及它和 CUDA 的抽象层级差异。
2. 掌握 Triton 的核心 API:`@triton.jit`、`program_id`、`tl.arange`、`tl.load/store`、`mask`。
3. 用 Triton 写出 **vector-add** 和 **fused-softmax**,与 PyTorch 对比正确性和性能。
4. 理解为什么 Triton 特别适合写**融合算子**(下一课主题)。

## 环境准备

```bash
pip install triton        # 需要 Linux + N 卡;Triton 不支持 Windows 原生、macOS
python -c "import triton; print(triton.__version__)"
```

> Triton 只在 Linux + NVIDIA GPU(以及部分 AMD)上工作。没有本地卡照旧用 Colab(免费 T4 自带 triton)或 AutoDL。

---

## 1. 为什么需要 Triton:CUDA 的两个痛点

你手写 GEMM 时痛在哪?
1. **样板代码多**:cudaMalloc/Memcpy、grid/block 计算、边界检查、`__syncthreads`……真正的算法逻辑被淹没。
2. **优化是体力活**:寄存器分块、float4 对齐、bank conflict 规避、循环展开——每一个都要手动且易错,换张卡(SM 数、shared memory 大小不同)还得重调。

Triton 的答案:**你只描述"一个 block 处理一块数据"的逻辑(用类 NumPy 的向量语法),编译器自动帮你做线程映射、访存合并、shared memory 分配、向量化、指令调度。**

类比你熟悉的世界:

| 层级 | 大数据类比 | 你操心什么 |
|---|---|---|
| CUDA C++ | 手写 MapReduce、自己管分区和 shuffle | 每个线程、每次访存 |
| **Triton** | **Spark RDD/DataFrame:你写 transform,引擎管执行** | **每个 block 的算法逻辑** |
| PyTorch 算子 | Spark SQL:一句话,啥都不用管 | 啥都不管(但融合不了、定制不了) |

Triton 的甜点区:**比 PyTorch 灵活(能写自定义融合算子),比 CUDA 省心(不用手抠线程)**,性能却能到 cuBLAS/手写 CUDA 的 80%~100%。

---

## 2. 核心抽象:program 而非 thread

这是从 CUDA 切到 Triton **最重要的认知转变**。

- **CUDA**:你写的是单个 **thread** 的代码,32 个 thread 组成 warp,你要操心 warp 内的分支、访存合并。
- **Triton**:你写的是单个 **program(等价于一个 CUDA block)** 的代码。一个 program 处理一整块数据(比如 1024 个元素),你用**向量操作**一次处理这一整块。block 内部怎么拆成线程、怎么向量化、怎么合并访存,**编译器决定**。

```
CUDA 视角:           Triton 视角:
grid 里有很多 block    一个 1D/2D 的 program 网格
每个 block 很多 thread  每个 program 处理一个 BLOCK_SIZE 的数据块
你写 1 个 thread 干啥   你写 1 个 program 干啥(向量化地处理整块)
```

`pid = tl.program_id(0)` 就相当于 CUDA 的 `blockIdx.x`。但**没有 `threadIdx`**——因为你不在线程粒度编程。

---

## 3. 第一个 Triton kernel:vector-add

完整代码见 `code/triton_vector_add.py`。核心 kernel:

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                 # 本 program 的编号 = CUDA blockIdx.x
    block_start = pid * BLOCK_SIZE              # 本 program 负责的数据起点
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 一个长度 BLOCK_SIZE 的向量索引
    mask = offsets < n_elements                # 边界掩码:防止越界(等价 if tid<n)
    x = tl.load(x_ptr + offsets, mask=mask)    # 一次性把整块读进来(向量)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

对照 CUDA 体会差异:
- 没有 `for` 循环遍历元素,`offsets` 是一个**向量**,`x + y` 是**向量加**——一条 Python 语句处理 BLOCK_SIZE 个元素。
- `mask` 替代了 `if (tid < n)`:被 mask 掉的位置不读/不写,既防越界又不污染结果。
- `BLOCK_SIZE: tl.constexpr` 是编译期常量(像 L3 的模板参数),编译器据此向量化、展开。

启动(launch)端:

```python
def vector_add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    # grid 是一个函数:给定 meta(含 BLOCK_SIZE),返回 program 数量。
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)  # 直接传 torch tensor!
    return out
```

注意:**不用 cudaMalloc、不用 memcpy**——直接把 PyTorch 的 CUDA tensor 传进去,Triton 拿它的显存指针。这就是省心的地方。

运行:

```bash
cd code
python triton_vector_add.py
```

它会校验和 `torch` 结果一致,并 benchmark。vector-add 是带宽受限算子(L1 算过 AI≈0.083),所以 Triton 和 PyTorch 都会贴着显存带宽跑,**速度接近**——这正说明 Triton 没有性能损失。

---

## 4. 重头戏:fused-softmax

softmax 才能体现 Triton 的价值。先看 softmax 定义(对每一行):

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

数值稳定版必须**先减去每行最大值**(否则 `exp` 溢出,L6 会深入):

```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

PyTorch 的 `torch.softmax` 怎么执行?它会拆成好几个 kernel:`max` → `subtract` → `exp` → `sum` → `divide`,**每一步都把整个矩阵从显存读出、写回**。对带宽受限的逐元素操作,这是巨大浪费(呼应 L1:这类算子瓶颈全在访存)。

Triton 的 fused-softmax:**一个 program 处理一整行,把这行加载进片上(SRAM)一次,在片上完成 max/exp/sum/divide 全过程,只读一次、写一次。** 这就是"融合"——把多次显存往返压成一次。

核心 kernel(完整见 `code/triton_softmax.py`):

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                       # 一个 program 负责一行
    col_offsets = tl.arange(0, BLOCK_SIZE)       # 假设 BLOCK_SIZE >= n_cols
    in_ptrs = in_ptr + row * in_row_stride + col_offsets
    mask = col_offsets < n_cols
    # 越界处补 -inf,使其 exp 后为 0,不影响 max 和 sum。
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                    # 数值稳定:减每行 max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    out_ptrs = out_ptr + row * out_row_stride + col_offsets
    tl.store(out_ptrs, y, mask=mask)
```

`tl.max`、`tl.sum` 是 Triton 内置的 **block 内归约**(底层用 shared memory + warp shuffle,正是 M1 L7 reduce 的内容,但你不用手写了!)。整行驻留在片上寄存器/SRAM,中间结果不落地显存。

运行:

```bash
python triton_softmax.py
```

参考输出(数值随卡):

```
[INFO] correctness vs torch.softmax: max abs err = 2.4e-07
[INFO] shape=(4096, 4096)
[INFO] torch.softmax : 0.62 ms, 432 GB/s
[INFO] triton fused  : 0.31 ms, 864 GB/s
[INFO] speedup = 2.0x
```

**为什么快 2 倍?** 不是因为算得快(算力都没吃满),而是因为**访存少了**:PyTorch 多 kernel 要把矩阵反复读写好几遍,Triton 融合后只读一遍写一遍。带宽受限算子的优化 = 减少访存次数,这是 Lesson 5 的核心,fused-softmax 就是第一个实例。

> 注意本例假设一行能装进一个 BLOCK_SIZE(`BLOCK_SIZE >= n_cols`)。当列数极大(如词表 softmax 几万列)装不下时,要用 online softmax 分块——这正是 L6、L7 的关键技术,这里先埋伏笔。

---

## 5. Triton vs CUDA:什么时候用谁

| 维度 | CUDA C++ | Triton |
|---|---|---|
| 编程粒度 | thread | program(block) |
| 访存合并/向量化 | 手动 | 编译器自动 |
| shared memory | 手动声明管理 | 编译器自动(`tl.max` 等隐含) |
| 调参(tile 大小) | 手改重编译 | `triton.autotune` 自动搜 |
| 上手与迭代速度 | 慢 | 快(纯 Python) |
| 极致性能 / 特殊指令 | 完全可控(可手写 WMMA) | 大部分够用,极限场景不如手写 |
| 适合 | 库级算子、需要榨干每一滴 | **自定义融合算子、研究迭代** |

实务结论:**推理 Infra 日常写融合算子,Triton 是首选**;只有当 Triton 表达不了或性能不够、且该算子极其关键时,才下沉到 CUDA C++。FlashAttention 的官方实现既有 CUDA 版也有 Triton 版,后者可读性强得多——L7 我们就用 Triton 写。

---

## 练习题

1. vector-add 里 `mask = offsets < n_elements` 的作用是什么?对应 CUDA 里的哪行代码?去掉会怎样?
2. fused-softmax 为什么比 `torch.softmax` 快?快的来源是"算力"还是"带宽"?用 L1 的 roofline 语言解释。
3. `tl.load(..., other=-float("inf"))` 里 `other` 参数为什么对 softmax 要设成 `-inf` 而不是 0?
4. Triton 里没有 `threadIdx`,那"32 个线程一个 warp、访存要合并"这些事去哪了?谁负责?

<details>
<summary>参考答案</summary>

1. 当 `n_elements` 不是 `BLOCK_SIZE` 整数倍时,最后一个 program 的部分 `offsets` 会越界;`mask` 让这些位置不读不写。对应 CUDA 的 `if (tid < n)`。去掉会越界读写,导致非法访问或结果错误。
2. 快在**带宽**:softmax 是带宽受限算子(AI 极低),PyTorch 拆成 max/sub/exp/sum/div 多个 kernel,每个都要把矩阵从 HBM 读出写回,多趟往返;Triton 融合成一个 kernel,数据只读一遍写一遍,访存量大减。roofline 上两者都在斜坡(带宽受限区),但 Triton 的"有效搬运字节"更少,所以同样带宽下更快。
3. softmax 要先求每行 max,再对 `exp(x-max)` 求和。越界位置若填 0,会被算进 max(0 可能比真实值大,扰乱 max),且 `exp(0-max)` 会被加进分母。填 `-inf`:它不会成为 max,且 `exp(-inf)=0` 不污染求和。
4. 全交给 Triton 编译器:它把一个 program 的向量操作 lower 成多线程(warp)、自动做访存合并、shared memory 分配和向量化。你只在 block 粒度描述算法,线程级的事编译器负责。

</details>

---

## 小结

- Triton 让你**用 Python 在 block(program)粒度写 kernel**,把线程映射、访存合并、shared memory、向量化交给编译器。
- 核心 API:`@triton.jit`、`tl.program_id`、`tl.arange`、`tl.load/store(mask=...)`、`tl.max/tl.sum`(block 内归约)。
- 直接传 PyTorch CUDA tensor,免去 malloc/memcpy 样板。
- **fused-softmax** 比 PyTorch 快,根源是**减少显存往返**(带宽受限算子的通用优化)——这就是下一课算子融合的开胃菜。

## 自测验收
- [ ] 能解释 program 与 thread 的抽象差异,以及 Triton 替你做了哪些优化。
- [ ] `triton_vector_add.py` 跑通,结果与 torch 一致。
- [ ] `triton_softmax.py` 跑通,理解它为何比 `torch.softmax` 快。
- [ ] 能说清 `mask` 和 `other` 参数的作用。

下一课:**Lesson 5 — 算子融合**,把"减少 kernel 启动 + 减少显存往返"系统化,手动融合 bias+gelu 并 benchmark。
