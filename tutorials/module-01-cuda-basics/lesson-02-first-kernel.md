# M1 · Lesson 2:第一个 CUDA 程序

> 本课开始动手。你会写、编译、运行人生第一个 GPU kernel,并彻底搞懂"一份代码被几千个线程同时执行"是什么意思。
> 预计用时:2 小时。
> 环境:有本地 N 卡最好;没有就用 Google Colab(免费 T4)或 AutoDL 租 4090。Colab 用法见文末附录。

## 学习目标

1. 理解 CPU(host)与 GPU(device)的代码如何分工。
2. 会用 `__global__` 定义 kernel,会用 `<<<grid, block>>>` 启动它。
3. 彻底理解 `blockIdx`、`blockDim`、`threadIdx` 三件套和全局线程 id 公式。
4. 能独立编译运行 `.cu` 文件并解释输出。

---

## 1. Host 与 Device:两个世界

写 CUDA,你的代码运行在两个地方:

- **Host(主机)** = CPU + 内存。负责:准备数据、申请 GPU 内存、把数据拷给 GPU、启动 kernel、把结果拷回来。普通 C++ 代码。
- **Device(设备)** = GPU + 显存。负责:并行执行 kernel。

三个函数修饰符要分清:

| 修饰符 | 在哪运行 | 谁能调用 | 用途 |
|---|---|---|---|
| `__global__` | Device | Host 调用 | kernel 入口,启动并行 |
| `__device__` | Device | Device | kernel 内部调用的辅助函数 |
| `__host__`(默认) | Host | Host | 普通 CPU 函数 |

---

## 2. 最小可运行例子:hello kernel

先看代码(`code/hello.cu`),逐行讲解:

```cpp
#include <cstdio>

// __global__ : 这是一个 kernel —— 在 GPU 上跑,由 CPU 启动。返回类型必须是 void。
__global__ void hello_kernel() {
    // 每个线程都会执行这一整段代码。下面这行算出"我是第几个线程"。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d (block=%d, threadInBlock=%d)\n",
           tid, blockIdx.x, threadIdx.x);
}

int main() {
    // <<<2, 4>>> : 启动 2 个 block,每个 block 4 个线程,共 8 个线程并行跑 hello_kernel。
    hello_kernel<<<2, 4>>>();
    // kernel 启动是"异步"的:CPU 发完命令立刻往下走。必须同步等待 GPU 跑完。
    cudaDeviceSynchronize();
    return 0;
}
```

编译运行:

```bash
nvcc hello.cu -o hello
./hello
```

你会看到 8 行输出(顺序可能乱,因为是并行的!):

```
Hello from thread 0 (block=0, threadInBlock=0)
Hello from thread 5 (block=1, threadInBlock=1)
...
```

**关键认知**:你只写了一份 kernel 代码,但它被 8 个线程同时执行了 8 遍。线程之间靠 `blockIdx`/`threadIdx` 知道"我是谁、该处理哪份数据"。输出顺序乱,正说明它们是真并行,不是循环。

---

## 3. 线程是怎么组织的:grid → block → thread

CUDA 把线程组织成**两层**结构:

```
启动配置 <<<gridDim, blockDim>>> = <<<2, 4>>>

Grid(网格,整个 kernel 的所有线程)
├── Block 0 (blockIdx.x = 0)
│     ├── Thread 0 (threadIdx.x = 0)   全局 tid = 0*4 + 0 = 0
│     ├── Thread 1 (threadIdx.x = 1)   全局 tid = 0*4 + 1 = 1
│     ├── Thread 2 (threadIdx.x = 2)   全局 tid = 0*4 + 2 = 2
│     └── Thread 3 (threadIdx.x = 3)   全局 tid = 0*4 + 3 = 3
└── Block 1 (blockIdx.x = 1)
      ├── Thread 0 (threadIdx.x = 0)   全局 tid = 1*4 + 0 = 4
      ├── ...                          ...
      └── Thread 3 (threadIdx.x = 3)   全局 tid = 1*4 + 3 = 7
```

三个内置变量(在 kernel 里随时可用):

- `blockDim.x`:每个 block 有多少线程(这里 = 4)。
- `blockIdx.x`:当前线程所在 block 的编号(0 或 1)。
- `threadIdx.x`:当前线程在自己 block 内的编号(0~3)。

**全局线程 id 公式(必须背下来,后面每个 kernel 都用)**:

```
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

> 为什么要分两层(block + thread)而不是一长串线程?因为同一个 block 内的线程可以**共享高速的 shared memory** 并能**互相同步**,而不同 block 之间不能。这个设计是后面性能优化的基础(Lesson 6 详讲)。现在先接受这个结构。

---

## 4. 完整范例:数组平方(含 host 端完整流程)

这是你第一个"真正做事"的 kernel,展示了 **CUDA 程序的标准五步流程**,以后所有程序都是这个套路:

```cpp
#include <cstdio>

__global__ void square_kernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查:线程总数通常会略多于 n,多出来的线程不能越界访问!
    if (tid < n) {
        out[tid] = in[tid] * in[tid];
    }
}

int main() {
    const int n = 10;
    const size_t bytes = n * sizeof(float);

    // ① Host 准备数据
    float h_in[n], h_out[n];
    for (int i = 0; i < n; ++i) h_in[i] = (float)i;   // 0,1,2,...,9

    // ② 在 Device 上申请显存
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // ③ 把输入从 Host 拷到 Device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // ④ 启动 kernel:每 block 256 线程,block 数向上取整保证覆盖 n
    int block = 256;
    int grid = (n + block - 1) / block;
    square_kernel<<<grid, block>>>(d_in, d_out, n);

    // ⑤ 把结果从 Device 拷回 Host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) printf("%.0f^2 = %.0f\n", h_in[i], h_out[i]);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

**五步流程记牢**:① 备数据 → ② cudaMalloc 申请显存 → ③ cudaMemcpy H2D → ④ 启动 kernel → ⑤ cudaMemcpy D2H。

两个新手必踩的坑:
- **边界检查 `if (tid < n)`**:因为 block 数向上取整,启动的线程数往往多于 `n`,多余线程若不拦住会越界访问。
- **指针不能混用**:`d_in` 指向显存,在 host 端不能直接解引用;kernel 里只能用 `d_` 指针。

---

## 5. 动手实验

### 实验 A:跑通 hello(必做)
1. 把 `code/hello.cu` **手敲**一遍(别复制,肌肉记忆很重要),编译运行。
2. 把 `<<<2, 4>>>` 改成 `<<<3, 5>>>`,**先预测**会打印几行、tid 范围,再运行验证。

### 实验 B:数组平方(必做)
补全 `code/square.cu` 的 4 个 TODO,跑通并验证输出 `0,1,4,9,...,81`。

---

## 练习题

1. 改写 square,实现"每个元素 ×2 再 +1"。
2. 把 `n` 改成 1000000,`block=256`,算 `grid` 是多少?程序还正确吗?
3. 故意去掉 `if (tid < n)`,把 `n` 设成 250(非 256 整数倍),观察会发生什么,理解为什么要边界检查。

<details>
<summary>第 2 题答案</summary>

`grid = (1000000 + 255) / 256 = 3907`,共启动 3907 × 256 = 1,000,192 个线程,多出的 192 个被 `if (tid < n)` 拦住。程序正确。
</details>

---

## 小结

- CUDA 程序分 **Host(CPU)** 和 **Device(GPU)**,kernel 用 `__global__` 定义、`<<<grid, block>>>` 启动。
- 牢记**全局线程 id 公式** `blockIdx.x * blockDim.x + threadIdx.x`。
- 牢记 **CUDA 五步流程**:备数据 → cudaMalloc → cudaMemcpy(H2D) → kernel → cudaMemcpy(D2H)。
- 两个必踩坑:**边界检查** 和 **host/device 指针不能混用**。

## 自测验收
- [ ] 能不看资料默写出 hello kernel 并跑通。
- [ ] 能解释 `<<<3, 5>>>` 会启动多少线程、tid 范围。
- [ ] `square.cu` 跑通并通过结果校验。
- [ ] 能说清为什么需要 `if (tid < n)`。

---

## 附录:没有本地 GPU 怎么办

**Google Colab(免费)**:打开 https://colab.research.google.com ,新建 notebook,菜单 `代码执行程序 → 更改运行时类型 → T4 GPU`。在 cell 里:

```python
%%writefile hello.cu
// 把 .cu 内容粘进来
```
```python
!nvcc hello.cu -o hello && ./hello
```

**AutoDL(按小时)**:租 4090 实例,VSCode Remote-SSH 连上,正常用 `nvcc`。学完关机停止计费。

下一课:**Lesson 3 — 线程层级与 2D 索引**,我们处理矩阵,真正用上并行的威力。
