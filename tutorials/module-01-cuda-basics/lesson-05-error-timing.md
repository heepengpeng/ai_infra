# M1 · Lesson 5:错误处理与正确计时

> 上一课的实验里我们埋了个雷:用 CPU 端的秒表给 kernel 计时。这会测出**完全错误**的数字,因为 kernel 启动是**异步**的。本课先把"CUDA 出错了你却看不到"这个隐形杀手解决掉,再教你用 `cudaEvent` 测出 kernel 的真实耗时,最后用它算出有效带宽——把性能从"感觉快"变成"有数字"。
> 预计用时:2 小时。
> 前置:Lesson 2(kernel 启动异步、`cudaDeviceSynchronize`)、Lesson 4(带宽/数据量概念)。

## 学习目标

1. 知道几乎所有 CUDA API 都返回错误码,会写一个错误检查宏 `CUDA_CHECK` 并到处用。
2. 理解 kernel 启动**不返回错误码**,要用 `cudaGetLastError` + 同步来捕获 kernel 内/启动错误。
3. 讲清"异步性"如何让 CPU 计时失真,学会用 `cudaEvent` 正确测 kernel 时间(含 warmup、同步)。
4. 会用实测时间计算**有效带宽 GB/s**,并对照硬件峰值判断 kernel 跑得好不好。

---

## 1. 沉默的失败:为什么你的 CUDA 程序"没报错但结果不对"

CUDA 的 API 大多长这样:**返回一个 `cudaError_t` 错误码,但你通常没接**。

```cpp
cudaMalloc(&d_in, bytes);   // 如果显存不够,它返回错误码,但你忽略了!
```

于是显存没申请成功,`d_in` 是野指针,后面 kernel 访问它,结果要么崩、要么悄悄算出垃圾数据。**CUDA 不会主动抛异常打断你**,错误码不查就石沉大海。资深读者请把这当成和"没检查 `malloc` 返回 NULL""没看 RPC 返回状态"同级别的纪律问题。

解决方案:写一个**检查宏**,包住每个 CUDA 调用。这是所有正经 CUDA 项目的标配:

```cpp
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",              \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
```

用法:把每个返回错误码的调用包起来,出错时它会打印**文件、行号、出错的那行代码、人类可读的错误描述**,然后退出。

```cpp
CUDA_CHECK(cudaMalloc(&d_in, bytes));
CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
```

> 宏里那几个技巧:`do{...}while(0)` 让宏能像普通语句一样安全地用在 `if` 后面;`#call` 把调用代码转成字符串;`cudaGetErrorString` 把错误码翻译成人话。这是 CUDA 圈的通用模板,记下来直接用。

---

## 2. kernel 启动是特殊的:它不返回错误码

这里有个大坑:**kernel 启动 `kernel<<<...>>>()` 没有返回值**,你没法用 `CUDA_CHECK` 直接包它。那 kernel 的错误(比如线程配置非法、kernel 内非法访存)怎么抓?

分两类错误,两步抓:

```cpp
kernel<<<grid, block>>>(args);
// ① 抓"启动错误"(同步的):比如 block 超过 1024、grid 配置非法
CUDA_CHECK(cudaGetLastError());
// ② 抓"执行错误"(异步的):比如 kernel 内越界访存。必须先同步,等它真跑完
CUDA_CHECK(cudaDeviceSynchronize());
```

- `cudaGetLastError()` 返回**最近一次错误并清空错误状态**,紧跟在 kernel 启动后,能抓到"配置类"的启动失败(立即可知)。
- kernel 是异步的,**内部的运行错误要等真正执行后才暴露**,所以要 `cudaDeviceSynchronize()` 强制 CPU 等 GPU 跑完,再检查。

> 调试技巧:写新 kernel 时,在启动后**总是**跟上这两行。否则你可能看到"程序正常退出但结果是 0",其实 kernel 早就因越界挂了,只是你没问它。

---

## 3. 异步性如何骗了你的秒表

回顾 Lesson 2:`kernel<<<>>>()` 启动后,**CPU 不等 GPU,立刻往下执行**。现在看上一课那段"计时":

```cpp
auto t0 = clock_now();
kernel<<<grid, block>>>(...);   // CPU 发完命令就返回,GPU 才刚开始算
auto t1 = clock_now();          // 这里 GPU 大概率还没算完!
double ms = t1 - t0;            // 测到的是"启动命令耗时",不是"kernel 耗时"
```

`t1 - t0` 测到的几乎只是**启动开销**(几微秒),而非 kernel 真正的计算时间。你会得到一个荒谬地小的数字,误以为 kernel 飞快。

要让 CPU 计时器正确,必须在停表前**同步**:

```cpp
auto t0 = clock_now();
kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();        // 等 GPU 真正算完
auto t1 = clock_now();          // 现在 t1-t0 才包含 kernel 执行时间
```

但即便加了同步,CPU 计时仍有缺点:它把**启动开销 + CPU↔GPU 同步抖动**也算进去了,在测很短的 kernel 时误差很大。GPU 上有更精准的工具——**CUDA Event**。

---

## 4. cudaEvent:在 GPU 时间线上打点

`cudaEvent` 的本质:往 GPU 的**命令流里插一个时间戳标记**。它记录的是事件在 GPU 上**真正被执行到**的时刻,所以测的是纯 GPU 时间,不受 CPU 抖动影响。

标准用法:

```cpp
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start));        // 在流里打"开始"标记
kernel<<<grid, block>>>(...);              // 要计时的 kernel
CUDA_CHECK(cudaEventRecord(stop));         // 打"结束"标记

CUDA_CHECK(cudaEventSynchronize(stop));    // 等 stop 这个标记真正被执行到
float ms = 0.0f;
CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));  // 两标记间的 GPU 毫秒数

CUDA_CHECK(cudaEventDestroy(start));
CUDA_CHECK(cudaEventDestroy(stop));
```

要测得准,还有**两条铁律**:

1. **Warmup(预热)**:第一次跑 kernel 包含一次性开销——CUDA 上下文初始化、JIT 编译、缓存冷。所以**先空跑几次再开始计时**,否则第一次的数字虚高。
2. **多次取平均**:单次测量有噪声。跑几十上百次取平均(或最小值),数字才稳定。

```cpp
// 预热
for (int i = 0; i < 5; ++i) kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();
// 正式计时,跑 100 次取平均
cudaEventRecord(start);
for (int i = 0; i < 100; ++i) kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
float avg_ms = ms / 100.0f;
```

> 这套"warmup + 循环 + event 计时"是后面每一课 benchmark 的**标准模板**(Lesson 7 多版本归约对比、Lesson 8 转置优化都靠它)。先在这课练熟。

---

## 5. 从时间到带宽:把性能落成数字

测出 kernel 耗时后,对**访存受限**的 kernel(回顾 Lesson 1 roofline:大部分逐元素/拷贝类 kernel 都在带宽斜坡上),最有意义的指标是**有效带宽**:

```
有效带宽 (GB/s) = 读写的总字节数 / 耗时(秒) / 1e9
```

举例:向量加法 `C = A + B`,`n` 个 float。每个元素**读 A、读 B、写 C 共 3 次**访存:

```
总字节数 = 3 * n * sizeof(float)
若 n = 2^24 ≈ 1678 万,float 4 字节 → 3 * 16.7M * 4 ≈ 201 MB
若 kernel 耗时 0.1 ms → 带宽 = 0.201 GB / 0.0001 s ≈ 2010 GB/s
```

怎么判断这个数字好不好?**和硬件峰值带宽比**。比如某卡显存峰值带宽 ~3000 GB/s,你测到 2010 GB/s,达到峰值的 ~67%,对一个朴素逐元素 kernel 已相当不错。

> 这是性能优化的"体检指标":
> - 实测带宽接近峰值 → kernel 已经把内存喂满,**别瞎优化计算**,这类 kernel 已经到顶。
> - 实测带宽远低于峰值 → 大概率访存没合并(coalescing 差)或并行度不足,有优化空间。
>
> 大数据里你会算"作业吞吐 / 集群理论吞吐",这里就是"kernel 带宽 / 显存峰值带宽",同一种思维。

(算力受限 kernel 则看 GFLOP/s,Module 2 GEMM 时再用,这里先掌握带宽。)

---

## 6. 动手实验

### 实验 A:让错误现形(必做)
跑 `code/error_check.cu`:它故意做两件错事——申请超大显存、用非法的 block 配置启动 kernel。观察 `CUDA_CHECK` 和 `cudaGetLastError` 如何精确报出文件、行号和错误原因。然后把检查宏去掉,看错误如何"消失"(但结果错了)。

```bash
nvcc error_check.cu -o error_check && ./error_check
```

### 实验 B:正确给向量加法计时并算带宽(必做)
补全 `code/timing_bandwidth.cu` 的 TODO:用 cudaEvent + warmup + 循环测出向量加法 kernel 的平均耗时,并算出有效带宽,和你显卡的峰值带宽(可用 `nvidia-smi -q` 或查规格)对比。

```bash
nvcc timing_bandwidth.cu -o timing_bandwidth && ./timing_bandwidth
```

对照实验:把 cudaEvent 的结果和"不加同步的 CPU 计时"打印在一起,亲眼看 CPU 计时小得多么离谱。

---

## 练习题

1. 为什么 kernel 启动后要同时用 `cudaGetLastError()` 和 `cudaDeviceSynchronize()`?它们各抓哪类错误?
2. 不加 `cudaDeviceSynchronize` 的 CPU 计时,测到的数字代表什么?为什么会非常小?
3. 为什么计时前要 warmup?第一次运行慢在哪几件事上?
4. 向量缩放 `x = a*x`(原地),`n` 个 float,每元素的访存字节数是多少?如果 `n=2^24`、耗时 0.08ms,有效带宽多少?
5. 你测到某 kernel 带宽只有峰值的 15%,下一步该怀疑什么、怎么查?

<details>
<summary>参考答案</summary>

1. `cudaGetLastError()` 紧跟启动后,抓**同步的启动配置错误**(如 block>1024、共享内存超额);`cudaDeviceSynchronize()` 强制等 kernel 执行完,才能暴露**异步的运行期错误**(如越界访存)。两者配合才能既抓启动失败又抓执行失败。
2. 测到的基本只是 **kernel 启动命令的开销**(CPU 把启动请求塞进队列就返回了,几微秒),不含 GPU 实际计算时间,所以荒谬地小。
3. 第一次运行包含 **CUDA 上下文/驱动初始化、可能的 JIT 编译、指令与数据缓存冷启动**等一次性开销,会让首次计时虚高,故先空跑预热。
4. 原地缩放每元素**读 1 次 + 写 1 次 = 2 次**访存,`2 * n * 4` 字节 = `2 * 16.7M * 4 ≈ 134 MB`;带宽 = `0.134 GB / 0.00008 s ≈ 1675 GB/s`。
5. 优先怀疑**访存不合并(coalescing)**——检查相邻线程(同 warp 内 threadIdx.x 连续)是否访问连续地址;其次是**并行度/occupancy 不足**、**warp divergence**。用 Nsight Compute(`ncu`)看 memory throughput、coalescing 相关指标定位(Lesson 8 会演示)。

</details>

---

## 小结

- CUDA API 默认**沉默失败**,用 `CUDA_CHECK` 宏包住每个调用,出错即打印文件/行号/原因并退出。
- kernel 启动**不返回错误码**:用 `cudaGetLastError()`(抓启动配置错误)+ `cudaDeviceSynchronize()`(抓异步执行错误)。
- kernel 异步 → **不同步的 CPU 计时是错的**;正确计时用 `cudaEvent`(GPU 时间线打点)+ **warmup + 多次取平均**。
- **有效带宽 = 总访存字节 / 耗时 / 1e9**,和显存峰值对比判断 kernel 好坏——这是访存受限 kernel 的核心体检指标。

## 自测验收(过了再进 Lesson 6)
- [ ] 能默写 `CUDA_CHECK` 宏并说清每个细节为什么这么写。
- [ ] 能解释为什么 kernel 后要跟 `cudaGetLastError` + `cudaDeviceSynchronize`。
- [ ] 能讲清异步性怎么让 CPU 计时失真,会用 cudaEvent + warmup 正确计时。
- [ ] 实验 B 跑通,算出向量加法的有效带宽并和峰值对比。

下一课:**Lesson 6 — 共享内存与线程协作**。前几课线程都是各干各的、互不通信。从下一课起,同一个 block 内的线程将通过**共享内存 + `__syncthreads()`** 协作,这是所有高性能 kernel(归约、转置、GEMM)的基石。
