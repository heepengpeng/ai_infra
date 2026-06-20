# M1 · Lesson 4:CUDA 内存模型与数据传输

> Lesson 1 说过"GPU 的瓶颈常在内存而非算力"。本课把那张内存层级图从"直觉"变成"能写进代码的规则":每一层内存是什么、谁能访问、`cudaMalloc`/`cudaMemcpy` 在搬哪条数据,以及为什么 CPU↔GPU 之间的传输往往是整个程序最大的性能杀手。
> 预计用时:2 小时。
> 前置:Lesson 2(五步流程、host/device 指针)、Lesson 3(行优先布局)。

## 学习目标

1. 说清 CUDA 五类内存(register / local / shared / constant / global)各自的作用域、速度、容量。
2. 彻底理解 host 内存与 device 显存是**两个独立地址空间**,以及 `cudaMalloc`/`cudaMemcpy`/`cudaFree` 各自在干什么。
3. 理解 **PCIe 传输是瓶颈**:为什么要尽量减少 H2D/D2H 拷贝、把数据"留在"GPU 上。
4. 会用 **pinned(锁页)内存** 加速传输,并理解它为什么更快、代价是什么。

---

## 1. 两个地址空间:host 指针和 device 指针不是一回事

这是新手最大的认知障碍,先彻底钉死:

```
   CPU 世界(Host)                    GPU 世界(Device)
 ┌───────────────────┐   PCIe 总线   ┌───────────────────┐
 │  主机内存 RAM      │ ◄──────────► │  显存 VRAM         │
 │  malloc/new 得到   │   (慢!)      │  cudaMalloc 得到   │
 │  h_ptr 指向这里    │              │  d_ptr 指向这里    │
 └───────────────────┘              └───────────────────┘
```

- `malloc`/`new` 返回的指针指向**主机 RAM**,CPU 能解引用,**kernel 里不能用**。
- `cudaMalloc` 返回的指针指向**显存 VRAM**,kernel 能解引用,**CPU 端不能直接解引用**(会段错误)。
- 两边的指针**数值上是各自地址空间的地址,不可混用**。`d_ptr` 在 host 端 `*d_ptr` 必崩。

> 类比你熟悉的分布式:host 和 device 就像两台机器,`cudaMemcpy` 就是它们之间的**网络传输**。你在 Spark 里绝不会随便跨节点搬数据,这里同理——**跨 PCIe 搬数据是要付费的**。

`cudaMemcpy(dst, src, bytes, kind)` 的第四个参数 `kind` 就是在声明"这是哪台机器到哪台机器":

| kind | 方向 | 场景 |
|---|---|---|
| `cudaMemcpyHostToDevice` | RAM → VRAM | 输入上传(H2D) |
| `cudaMemcpyDeviceToHost` | VRAM → RAM | 结果取回(D2H) |
| `cudaMemcpyDeviceToDevice` | VRAM → VRAM | GPU 内部拷贝,很快 |

---

## 2. Device 端的五层内存:写 kernel 时你在用哪一层

Lesson 1 给了从快到慢的直觉图,这里补上**编程接口**——你写的每个变量到底落在哪层:

```
                  作用域            速度      容量          怎么得到
─────────────────────────────────────────────────────────────────────
寄存器 Register    单线程私有         最快      极少(几十~255/线程)  kernel 里的局部标量
本地内存 Local     单线程私有         慢(在显存) 较大           寄存器放不下时溢出 / 局部数组
共享内存 Shared    block 内共享       很快(片上) 几十~228KB/block  __shared__ 声明(Lesson 6)
常量内存 Constant  全体只读           快(有缓存) 64KB            __constant__ + cudaMemcpyToSymbol
全局内存 Global    全体读写           慢(显存)   GB 级           cudaMalloc 得到的 d_ptr
```

逐个说清关键点:

- **Register(寄存器)**:kernel 里写的普通局部变量(`int tid`、`float sum`)默认放寄存器,访问零延迟。但每个 SM 的寄存器总量有限,用太多会限制能同时驻留的线程数(occupancy,Module 2 详讲)。
- **Local memory(本地内存)**:名字骗人——它**物理上在全局显存里**,慢。当寄存器不够(变量太多、或线程里开了**需要动态下标的局部数组**)时,编译器把变量"溢出"到 local。要警惕:看着是局部变量,实际可能在慢速显存里。
- **Shared memory(共享内存)**:片上、同 block 内线程共享,是手动管理的高速缓存,后面 Lesson 6/7/8 的优化主角。这里先知道它存在。
- **Constant memory(常量内存)**:只读、有专用缓存,适合放所有线程都读同一份的小数据(如卷积核系数)。
- **Global memory(全局内存)**:就是 `cudaMalloc` 拿到的那块 GB 级显存,所有线程可读写,但延迟几百周期。**绝大多数数据都在这里,优化的核心就是减少对它的访问次数**(呼应 Lesson 1)。

> 性能心法不变:**把全局内存的数据搬进寄存器/共享内存,复用,再写回。** 这五层内存就是这句话的"工具箱"。

---

## 3. cudaMalloc / cudaMemcpy / cudaFree:再看一眼五步流程

Lesson 2 的五步流程,现在你能看懂每一步在内存层面发生了什么:

```cpp
float *d_in, *d_out;
cudaMalloc(&d_in,  bytes);   // ① 在 VRAM 里划一块,d_in 指向它。注意传 &d_in!
cudaMalloc(&d_out, bytes);

cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);  // ② RAM → VRAM,过 PCIe

kernel<<<grid, block>>>(d_in, d_out, n);                // ③ GPU 在 VRAM 上算

cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);// ④ VRAM → RAM,过 PCIe

cudaFree(d_in); cudaFree(d_out);                        // ⑤ 释放显存,否则泄漏
```

两个易错点:

- **`cudaMalloc(&d_in, bytes)` 要传指针的地址(`&d_in`)**。因为它要**修改**你的指针让它指向新显存,所以传二级指针。漏写 `&` 是经典编译/运行错误。
- **显存会泄漏**。`cudaMalloc` 不像栈变量自动回收,忘了 `cudaFree`,长跑的服务会把显存耗尽(做推理服务时尤其要警惕)。

---

## 4. PCIe 是瓶颈:为什么"别老往返搬数据"

这是本课最重要的工程结论。给一组**量级**感受(具体数值因硬件而异,记数量级):

| 路径 | 典型带宽 | 相对快慢 |
|---|---|---|
| GPU 全局显存(VRAM 内部) | ~1000–3000 GB/s | 基准 |
| PCIe 4.0 x16(H2D / D2H) | ~25 GB/s | 慢几十到上百倍 |
| 网络(对比 Spark shuffle) | ~1–12 GB/s | 更慢 |

也就是说,**CPU↔GPU 拷贝比 GPU 内部访存慢一两个数量级**。一个常见的"伪加速"翻车现场:

```
朴素写法(每次操作都来回搬):
  H2D 拷 A → kernel 加法 → D2H 拷回 → H2D 再拷 → kernel 乘法 → D2H 拷回
  传输时间 >> 计算时间,GPU 反而比 CPU 还慢!

正确写法(数据上去就别下来):
  H2D 拷 A 一次 → kernel1 → kernel2 → kernel3(全在 GPU 上链式跑)→ D2H 拷结果一次
```

> 黄金法则:**数据一旦上了 GPU,就让它尽量待在 GPU 上,把多个 kernel 串起来,最后只取回最终结果。** 这正是大模型推理引擎的设计哲学——权重常驻显存,中间激活值不落盘、不回 CPU。你在大数据里"减少 shuffle、算子链式执行"的经验,在这里一一对应。

判断一个任务值不值得上 GPU,也用这个视角:**计算量要大到能摊薄传输开销**。小数据搬上搬下,传输开销吃光收益(Lesson 1 的"小任务不适合 GPU"就是这个意思)。

---

## 5. Pinned(锁页)内存:让传输更快

默认 `malloc` 出来的是**可分页内存(pageable)**,操作系统可能把它换出到磁盘。GPU 的 DMA 引擎不能直接搬可能被换走的内存,所以 `cudaMemcpy` 实际是:**先把可分页内存拷到一块临时锁页缓冲区,再从那里 DMA 给 GPU**——多了一道暗中的拷贝。

**Pinned memory(锁页内存)** 用 `cudaMallocHost` 申请,它被锁定在物理 RAM 里不会被换出,DMA 可以直接搬,省掉那道中转:

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, bytes);   // 锁页内存,传输更快
// ... 当作普通 host 内存用,填数据、cudaMemcpy ...
cudaFreeHost(h_pinned);             // 注意:配对的释放是 cudaFreeHost,不是 free!
```

收益与代价:

- ✅ H2D/D2H 带宽通常能提升 ~1.5–2 倍(省掉中转拷贝)。
- ✅ 是后续 `cudaMemcpyAsync` + stream **传输与计算重叠**的前提(进阶,Module 2/4 会用到)。
- ⚠️ 锁页内存挤占物理 RAM,**申请过多会拖慢甚至挂死整个系统**,别滥用。
- ⚠️ 申请/释放比 `malloc` 慢,适合**复用**(比如推理服务里固定的输入缓冲区)。

> 推理工程里,输入/输出缓冲区常用 pinned memory 并复用,配合多 stream 让"传下一批输入"和"算当前批"重叠,把 PCIe 这段时间藏起来。这是后面服务化的常见手法,现在先理解原理。

---

## 6. 动手实验

### 实验 A:测量传输 vs 计算的占比(必做)
跑 `code/mem_transfer.cu`,它做一个简单的向量缩放,并用计时(粗略)分别报告 H2D、kernel、D2H 三段耗时。观察:**对这种轻计算任务,传输时间是不是远大于计算时间?**(精确计时方法是下一课 Lesson 5 的主题,这里先用粗略的 wall-clock 建立直觉。)

```bash
nvcc mem_transfer.cu -o mem_transfer && ./mem_transfer
```

### 实验 B:pinned vs pageable 对比(必做)
补全 `code/pinned_vs_pageable.cu` 的 TODO:用 `malloc` 和 `cudaMallocHost` 各申请一块大缓冲区,各做多次 H2D 拷贝并计时,对比带宽。预期 pinned 明显更快。

---

## 练习题

1. 为什么 kernel 里不能直接解引用 `cudaMalloc` 得到的指针所对应的 host 端值?host 指针在 kernel 里又会怎样?
2. 一个程序里 `cudaMemcpy` H2D + D2H 的总时间是 kernel 时间的 10 倍,你会怎么优化?
3. `cudaMallocHost` 申请的内存用 `free` 释放会怎样?反过来 `malloc` 的内存用 `cudaFreeHost` 呢?
4. 局部变量明明是"局部"的,为什么有时会落到慢速的 local memory(显存)里?这对性能意味着什么?

<details>
<summary>参考答案</summary>

1. `d_ptr` 是显存地址,host CPU 的地址空间里没有这块映射,直接解引用 → 段错误。反过来 host 指针(RAM 地址)在 kernel 里解引用同样非法,GPU 访问不到主机 RAM(统一内存 UVM 是另一回事,后面再说)。
2. 典型瓶颈在传输:① 减少来回次数,数据上 GPU 后把多个 kernel 串起来,只在首尾各传一次;② 用 pinned memory 提升单次带宽;③ 进阶用多 stream + `cudaMemcpyAsync` 让传输与计算重叠;④ 重新评估这个任务计算量是否大到值得上 GPU。
3. 二者**必须配对**:`cudaMallocHost`↔`cudaFreeHost`,`malloc`↔`free`。混用是未定义行为,轻则报错重则破坏堆,务必配对。
4. 当寄存器不够用(变量太多)、或定义了**用运行期变量做下标的局部数组**(寄存器无法按动态下标寻址)时,编译器把这些变量溢出到 local memory,它物理在全局显存,延迟高。意味着看似"局部"的访问其实在走慢速显存,是隐蔽的性能坑。

</details>

---

## 小结

- host RAM 与 device VRAM 是**两个独立地址空间**,指针不可混用;`cudaMemcpy` 就是它们之间的"网络传输"。
- device 端五层内存:register / local(实在显存) / shared / constant / global,**优化核心是少碰 global、多复用 shared/register**。
- `cudaMalloc(&d_ptr, ...)` 传二级指针;显存要 `cudaFree`,否则泄漏。
- **PCIe 传输比 GPU 内部访存慢一两个数量级**,黄金法则:数据上 GPU 后尽量留在 GPU,kernel 链式执行,首尾各传一次。
- **pinned 内存**(`cudaMallocHost`/`cudaFreeHost`)省掉中转拷贝、提速传输,但挤占 RAM,适合复用。

## 自测验收(过了再进 Lesson 5)
- [ ] 能解释 host 指针和 device 指针为什么不能混用。
- [ ] 能说出五类内存的作用域与快慢,并知道局部数组可能落到 local。
- [ ] 能讲清"为什么数据要尽量留在 GPU 上"以及黄金法则。
- [ ] 两个实验跑通,亲眼看到传输 vs 计算占比、pinned vs pageable 带宽差异。
- [ ] 知道 `cudaMallocHost` 必须配 `cudaFreeHost`。

下一课:**Lesson 5 — 错误处理与正确计时**。本课实验里那个"粗略计时"其实暗藏陷阱——因为 kernel 是异步的,CPU 计时器测到的根本不是 GPU 真实耗时。下一课教你用 `cudaEvent` 测准,并把它和带宽计算挂钩。
