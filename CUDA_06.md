# CUDA 编程实战指南

前面几篇讲的是"CUDA 硬件和执行模型是怎么回事"（Grid/Block/Warp/SM、计算体系、内存体系、Roofline、Occupancy）。这篇换个角度，讲**怎么真正动手写一个 CUDA 程序**——语法、API、内存管理、一个完整的例子，以及从这些语法背后能看到的、和前面理论一一对应的关系。

---

## 1. 一个 CUDA 程序的基本骨架

CUDA 程序是 **Host（CPU）代码 + Device（GPU）代码** 混合在一起的，用 `.cu` 文件写，`nvcc` 编译器负责把两部分分别编译、拼接。

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

// __global__ 标记这是一个 kernel 函数：Host 调用，Device 执行
__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局线程索引
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 100万个元素
    size_t bytes = n * sizeof(float);

    // 1. Host 端分配内存并初始化数据
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    // ...填充 h_a, h_b...

    // 2. Device 端分配显存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. 数据从 Host 拷贝到 Device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel：<<<grid大小, block大小>>>
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;  // 向上取整，确保覆盖所有元素
    addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // 5. 结果从 Device 拷贝回 Host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. 释放内存
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
```

这个骨架（**分配 → 拷贝 H2D → 计算 → 拷贝 D2H → 释放**）是几乎所有 CUDA 程序的通用模式，后面讲的所有内容都是在这个骨架基础上做优化和扩展。

---

## 2. 函数修饰符：代码跑在哪里

CUDA 用三个修饰符区分函数的"归属"：

| 修饰符 | 调用方 | 执行方 | 用途 |
|---|---|---|---|
| `__global__` | Host（也可以是 Device，动态并行） | Device | kernel 函数入口，必须返回 `void` |
| `__device__` | Device | Device | 被 kernel 内部调用的辅助函数，会被内联或编译成 device 代码 |
| `__host__` | Host | Host | 普通 CPU 函数（可以省略，函数默认就是 host） |

一个函数可以同时标记 `__host__ __device__`，编译器会生成两份代码（CPU 一份、GPU 一份），常用于数学工具函数（比如一个 clamp 函数，CPU 和 GPU 代码里都要用）。

---

## 3. 索引计算：怎么让每个线程知道"我是谁、该算哪份数据"

这是 CUDA 编程里最基础、也是新手最容易出错的地方。核心内置变量：

- `threadIdx`：线程在其所属 block 内的局部索引（`.x`/`.y`/`.z`）
- `blockIdx`：block 在 grid 内的索引
- `blockDim`：一个 block 的尺寸（每维有多少线程）
- `gridDim`：grid 的尺寸（每维有多少 block）

**一维索引计算**（最常见）：
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
直觉理解：先定位到第几个 block（`blockIdx.x`），跳过前面所有 block 占的线程数（`× blockDim.x`），再加上自己在 block 内的偏移（`threadIdx.x`）。

**二维索引计算**（常见于图像/矩阵处理）：
```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;  // 转换成一维数组下标
```

**为什么要判断 `if (idx < n)`**：grid 大小通常是向上取整算出来的（保证覆盖所有数据），这会导致最后一个 block 里有些线程"多余"（超出数据范围），如果不加边界检查，这些线程会访问越界内存，必须显式过滤掉。

---

## 4. Kernel Launch 语法：`<<<>>>` 里到底在配置什么

```cuda
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

四个参数（后两个可省略）：
- **gridDim**：grid 里有多少个 block（可以是 `int` 或 `dim3` 三维）
- **blockDim**：每个 block 里有多少个线程（同样可以是 `int` 或 `dim3`）
- **sharedMemBytes**（可选）：动态分配的 shared memory 大小（字节数），配合 `extern __shared__` 使用
- **stream**（可选）：指定在哪个 CUDA Stream 上执行，用于实现异步和并发（前面内存篇提到的"传输和计算重叠"就是靠这个）

**`dim3` 类型**：当数据是二维/三维的（比如图像、矩阵），可以用 `dim3` 直接表达多维的 grid/block：
```cuda
dim3 blockDim(16, 16);  // 每个 block 是 16×16 = 256 个线程
dim3 gridDim((width + 15) / 16, (height + 15) / 16);
kernel<<<gridDim, blockDim>>>(...);
```

---

## 5. Shared Memory 的具体写法

对应前面内存篇讲的"程序员手动管理的可编程缓存"，实际写法有两种：

**静态分配**（编译期确定大小）：
```cuda
__global__ void kernel(float* data) {
    __shared__ float tile[256];  // 每个 block 独立拥有一份
    int tid = threadIdx.x;
    tile[tid] = data[blockIdx.x * blockDim.x + tid];  // 从 global memory 搬入
    __syncthreads();  // 确保所有线程都搬运完成，再开始使用
    // ...用 tile 里的数据做计算...
}
```

**动态分配**（运行时确定大小，通过 kernel launch 的第三个参数指定）：
```cuda
extern __shared__ float tile[];  // 大小在 launch 时指定，不在这里写死

kernel<<<grid, block, sharedBytes>>>(...);  // sharedBytes 就是 tile 的实际字节数
```

**`__syncthreads()` 的作用**：block 内所有线程执行到这一行才会继续往下走，用来确保"大家把数据都搬进 shared memory 了，再开始用"，防止有的线程还没搬完、别的线程就已经在读脏数据。这是 block 内同步的核心工具，但注意它**不能跨 block 同步**——grid 内不同 block 之间没有直接同步机制（这也是为什么 kernel 之间的依赖通常靠"结束一个 kernel、再启动下一个"来实现）。

---

## 6. 一个具体例子：矩阵乘法的朴素实现 vs Tiling 优化

这个例子能把"索引计算 + shared memory + 前面讲的内存复用"全部串起来。

### 朴素实现（每次都直接读 global memory）

```cuda
__global__ void matMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // 每次循环都读 global memory
        }
        C[row * N + col] = sum;
    }
}
```

问题：计算 `C[row][col]` 时，`A` 的一整行和 `B` 的一整列会被反复从 global memory 读取，而这些数据其实会被同一个 block 内的其他线程重复用到——这正是内存篇里讲的"计算强度低、memory-bound"的典型场景。

### Tiling 优化（先搬进 shared memory 复用）

```cuda
#define TILE_SIZE 16

__global__ void matMulTiled(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0;

    // 分块遍历：每次只搬一小块 A 和 B 进 shared memory
    for (int t = 0; t < N / TILE_SIZE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();  // 确保这一块数据搬运完成

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];  // 从 shared memory 读，快很多
        }
        __syncthreads();  // 确保这一块算完，才能进入下一轮覆盖 tile
    }
    C[row * N + col] = sum;
}
```

这里体现的正是前面理论篇讲的核心逻辑：**用一次 global memory 读取，换来 shared memory 里多次复用**，大幅减少了总的显存访问次数，把 kernel 从"内存卡脖子"往"算力卡脖子"方向推——这就是 Roofline 图上"把点往右挪"的具体代码实现。

---

## 7. CUDA Stream：实现异步和并发的关键

前面内存篇提到"用计算掩盖传输延迟"，具体靠 Stream 实现：

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 两个 stream 上的操作可以并发执行（互不阻塞）
cudaMemcpyAsync(d_a1, h_a1, bytes, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d_a1, ...);

cudaMemcpyAsync(d_a2, h_a2, bytes, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream2>>>(d_a2, ...);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**默认情况下（不指定 stream）所有操作都在同一个默认 stream 上顺序执行**，这也是为什么很多"看起来在并行"的代码实际是串行的——想要真正的并发（比如"传下一批数据"和"算这一批数据"同时进行），必须显式用多个 stream。

---

## 8. 错误检查：容易被忽略但很重要

CUDA API 调用出错不会像 CPU 代码那样直接崩溃报错，而是返回一个错误码，需要手动检查，否则 bug 可能悄无声息：

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 使用方式
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
```

**kernel launch 本身是异步的，出错也不会立刻反映**，需要用 `cudaGetLastError()`（检查 launch 配置是否有问题）和 `cudaDeviceSynchronize()`（等待 kernel 真正跑完，再检查执行期间是否出错）配合排查。

---

## 9. 小结：语法和理论的对应关系

写代码时的每一个语法点，背后都能对应回前面几篇讲的理论：

| 代码里的东西 | 对应的理论概念 |
|---|---|
| `<<<gridDim, blockDim>>>` | Grid/Block 的软件层次结构 |
| `threadIdx`/`blockIdx` 索引计算 | Thread 如何映射到数据 |
| `__shared__` + `__syncthreads()` | Shared Memory 复用，减少 memory-bound |
| block size 设成 32 的倍数 | Warp 的硬件执行单位是 32 线程 |
| `if (idx < n)` 边界检查 | Grid 向上取整导致的多余线程 |
| Tiling 矩阵乘法 | 把 kernel 从 memory-bound 推向 compute-bound（Roofline） |
| `cudaStream` 异步执行 | 内存篇讲的"传输和计算重叠" |
| 寄存器/shared memory 用量 | 直接决定 Occupancy 上限 |

这也是为什么前面花了大量篇幅讲硬件模型和 Roofline 理论——**CUDA 语法本身并不复杂，真正决定代码好坏的，是写代码时脑子里有没有这套硬件模型**。同样的矩阵乘法，朴素版和 tiling 版语法难度差不多，但性能可能差出几倍甚至十几倍，差距就来自于是否理解并利用了 Shared Memory、Memory Coalescing、Occupancy 这些前面讲过的机制。

下一步如果要继续深入，可以是：写一个完整可跑的例子并用 Nsight Compute 实测对比朴素版和 tiling 版的差异（把整套理论工具链跑通一遍），或者转向更贴近你工作方向的 Triton（比原生 CUDA 更易写，但底层逻辑完全一致）。
