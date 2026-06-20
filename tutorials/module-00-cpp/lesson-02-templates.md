# M0 · Lesson 2:模板与函数重载

> 打开 vLLM、CUTLASS、TensorRT-LLM 的 `.cu` / `.h` 源码,你会被满屏的 `template<typename T>`、`<<<...>>>(...)<T>` 吓到。本课就是为了让你看到它们时**不慌**:知道它在干什么、为什么非这么写不可。
> 预计用时:1.5 小时。
> 前置:Lesson 1(指针、引用)。

## 学习目标

学完本课你应该能:
1. 解释**函数重载**(overload)的规则,知道编译器靠什么挑函数。
2. 看懂并自己写出 `template<typename T>` 函数模板,理解"编译期生成代码"的含义。
3. 说清为什么算子库偏爱模板(而不是为每种类型各写一份),并能联系到 FP32/FP16/BF16/INT8 多精度。
4. 区分模板和 Python 鸭子类型/泛型的异同。

---

## 1. 起点:重载——同名函数,按类型区分

Python 里一个函数名只能绑一个函数,靠"鸭子类型"在运行时随机应变:

```python
def add(a, b):
    return a + b   # int、float、str、list 都能进来,运行时才知道行不行
```

C++ 是静态类型语言,但它允许**同一个名字定义多个函数**,只要参数列表不同。编译器根据你**传进来的参数类型/个数**,在编译期挑出唯一匹配的那个。这叫**函数重载(function overloading)**。

看 `code/overload_demo.cpp`:

```7:11:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/overload_demo.cpp
// 三个同名函数,编译器按"参数类型"挑哪一个 —— 这叫重载(overload)
int  add(int a, int b)        { return a + b; }
double add(double a, double b){ return a + b; }
// 参数个数不同也算重载
int  add(int a, int b, int c) { return a + b + c; }
```

```bash
g++ -std=c++17 -Wall overload_demo.cpp -o overload_demo && ./overload_demo
```

```
add(1, 2)      = 3
add(1.5, 2.5)  = 4.0
add(1, 2, 3)   = 6
```

关键规则:

> **重载靠参数的"类型 + 个数"区分,与返回类型无关。** 即:你不能只靠返回类型不同来重载(`int f()` 和 `double f()` 是冲突的)。

重载解决了"同一逻辑、不同类型"的命名问题。但它有个致命缺点:**每种类型你都得手写一遍**。`int`、`double`、`float`、`long`……加法逻辑一模一样,却要复制四份。这显然蠢。模板就是来消灭这种重复的。

---

## 2. 函数模板:把"类型"变成参数

模板的核心思想一句话:

> **把类型本身抽象成一个参数 `T`,写一份"代码蓝图";编译器在你调用时,按实际类型自动"填空"生成具体函数。**

```16:19:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/template_demo.cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}
```

- `template <typename T>`:声明"接下来这段代码里,`T` 是一个类型占位符"。`typename` 也可写 `class`,等价。
- 调用 `add(3, 4)` 时,编译器看到实参是 `int`,于是**推导** `T = int`,生成一份 `int add(int, int)`;
- 调用 `add(1.5, 2.5)` 时推导 `T = double`,**再生成一份** `double add(double, double)`。

```24:28:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/template_demo.cpp
    printf("add<int>    : %d\n",   add(3, 4));         // T 推导为 int
    printf("add<double> : %.2f\n", add(1.5, 2.5));     // T 推导为 double
    // 显式指定类型(当无法从参数推导,或想强制时)
    printf("add<double> : %.2f\n", add<double>(1, 2)); // 强制走 double 版
```

```bash
g++ -std=c++17 -Wall template_demo.cpp -o template_demo && ./template_demo
```

**这里有个极重要的认知**——模板是**编译期**的代码生成,不是运行时的多态:

```
你写的(1 份蓝图)              编译器实际生成的(N 份具体函数)

template<typename T>           int    add(int, int)      { return a+b; }
T add(T a, T b)        ──────► double add(double,double) { return a+b; }
{ return a + b; }              float  add(float, float)  { return a+b; }
                               ... (你用到几种类型,就生成几份)
```

这个"按需生成"的过程叫**模板实例化(instantiation)**。生成后的每一份都是普通的、类型固定的、零运行时开销的函数——这正是它和 Python 泛型/Java 泛型的根本区别。

| | C++ 模板 | Python(鸭子类型) |
|---|---|---|
| 何时确定类型 | **编译期**生成专用代码 | 运行时动态判断 |
| 运行时开销 | 零(等同手写专用函数) | 有(动态分发、装箱) |
| 类型错误何时暴露 | 编译期(用了 `T` 不支持的操作就编译失败) | 运行时才崩 |
| 产物 | 每种类型一份机器码(代码膨胀) | 一份字节码 |

> 一句话:**C++ 模板 = "编译期的复制粘贴 + 自动填类型"**。它换来了零运行时开销,代价是编译变慢、二进制变大、报错信息冗长。

---

## 3. 模板不只接受"类型",还能接受编译期常量

模板参数除了 `typename T`,还能是**编译期就确定的整数等常量**(非类型模板参数,NTTP)。这在高性能算子里极常见——比如把 tile 大小、向量长度写进模板,让编译器据此展开循环、优化寄存器分配。

看练习文件 `code/vector_template.cpp` 的类模板:

```14:25:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/vector_template.cpp
template <typename T, int N>      // 注意:模板参数不仅能是类型,还能是编译期常量 N
struct Vec {
    T data[N];

    // 逐元素相加,返回新 Vec。const& 传参避免拷贝(回顾 Lesson 1 引用)
    Vec<T, N> operator+(const Vec<T, N>& other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) {
            // TODO: 写出逐元素相加  result.data[i] = ...
            result.data[i] = T{};   // 占位,先填 0,练习时改成正确实现
        }
        return result;
    }
```

`Vec<float, 3>` 和 `Vec<int, 4>` 是编译器生成的**两个完全不同的类型**,各自的 `N` 在编译期就是常数,循环 `for (i < N)` 可以被编译器完全展开。CUDA 里 `template <typename T, int BLOCK_SIZE>` 的 kernel 就是用这招把 block 尺寸"焊死"进代码换取性能。

---

## 4. 为什么算子库离不开模板?——多精度才是真正动机

现在回答标题问题。大模型推理里,**同一个算子要支持多种数据精度**:

| 精度 | C++ 类型(典型) | 用在哪 |
|---|---|---|
| FP32 | `float` | 训练、对精度敏感处 |
| FP16 | `half` / `__half` | 推理主力 |
| BF16 | `__nv_bfloat16` | 大模型推理/训练 |
| INT8 | `int8_t` | 量化推理(Module 5) |
| FP8 | `__nv_fp8_e4m3` 等 | 最新量化 |

一个 `elementwise_add`、`softmax`、`gemm`,逻辑对所有精度几乎一致,区别只在数据类型。如果不用模板,你要为 5 种精度各复制粘贴一份 kernel,改一处 bug 要改五处。用模板:

```cpp
// 这就是你将在 CUDA 课和 vLLM 源码里反复见到的写法
template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = a[tid] + b[tid];
}

// 实例化:用哪种精度就实例化哪种,编译器各生成一份最优代码
add_kernel<float><<<grid, block>>>(...);   // FP32 版
add_kernel<half><<<grid, block>>>(...);    // FP16 版
```

> 所以 `template<typename T>` 不是炫技,而是**多精度算子库的工程刚需**:一份逻辑、N 种精度、零运行时开销、改一处全生效。看懂它,你就看懂了 vLLM/CUTLASS 源码骨架的一大半。

注意:CUDA 的 `__global__` kernel 同样可以是模板,只是实例化发生在 GPU 代码侧,机制完全一致。这个伏笔在 Module 1/2 会兑现。

---

## 5. 动手实验

### 实验 A:重载与模板对照(必做)
分别跑通 `overload_demo.cpp` 和 `template_demo.cpp`,确认输出。然后在 `template_demo.cpp` 里加一行 `add(1, 2.5)`(一个 int 一个 double),编译,**观察报错**——想清楚为什么模板推导 `T` 时会冲突。

<details><summary>提示:为什么 add(1, 2.5) 报错</summary>

`add(T, T)` 要求两个参数是**同一个 `T`**。`1` 想让 `T=int`,`2.5` 想让 `T=double`,编译器无法同时满足,推导失败。解决办法:显式指定 `add<double>(1, 2.5)`(让 1 转成 double),或把模板写成两个类型参数 `template<typename A, typename B>`。
</details>

### 实验 B:补全类模板(必做)
补全 `code/vector_template.cpp` 里 `operator+` 的 TODO(逐元素相加),编译运行,验证输出 `11 22 33` 和 `2 3 4 5`。体会:**同一份 `Vec<T,N>` 蓝图,被 `<float,3>` 和 `<int,4>` 实例化成了两个独立类型**。

```bash
g++ -std=c++17 -Wall vector_template.cpp -o vector_template && ./vector_template
```

---

## 练习题

1. 下面两个函数能同时存在(构成合法重载)吗?

```cpp
int    f(int x);
double f(int x);
```

<details><summary>参考答案</summary>

**不能**。重载只看参数列表(类型 + 个数),**不看返回类型**。这两个函数参数完全相同,编译器会报"重定义/无法重载"。如果调用 `f(3)`,编译器也无从判断你想要哪个返回类型。
</details>

2. 模板函数 `add<T>` 如果传入一个没有定义 `operator+` 的自定义类型,错误会在什么时候、什么位置报出来?

<details><summary>参考答案</summary>

在**编译期、且在实例化那一刻**报错。模板本身定义时不检查 `T` 是否支持 `+`(它还不知道 `T` 是谁);只有当你用某个具体类型实例化(如 `add(myObj1, myObj2)`)、编译器把 `T` 填进去生成代码时,才发现 `return a + b;` 这一行对该类型非法,于是报错。这也是模板报错信息往往又长又绕的原因——它会把整条实例化链都吐出来。
</details>

3. 为什么说 C++ 模板是"零运行时开销",而 Python 的泛型/鸭子类型有运行时开销?

<details><summary>参考答案</summary>

C++ 模板在**编译期**就为每种用到的类型生成了一份类型固定的专用机器码,运行时直接执行,没有任何"判断当前是什么类型"的动作,和你手写专用函数完全一样快。Python 在**运行时**才知道对象类型,每次操作都要动态查找类型、分发方法、可能还要装箱拆箱,这些都是额外开销。代价方面,模板换来速度但导致代码膨胀(每种类型一份)和编译变慢。
</details>

4. 为什么大模型算子库要把数据类型做成模板参数 `template<typename T>`,而不是统一用 `float` 然后运行时转换?

<details><summary>参考答案</summary>

因为推理的核心诉求就是用低精度(FP16/INT8/FP8)换取**显存占用减半/带宽翻倍/吞吐提升**(回顾 Module 1 的"decode 是带宽受限")。如果内部统一转成 `float` 计算,就丧失了低精度带来的全部收益,而且多了来回转换的开销。把 `T` 做成模板参数,能让每种精度都生成**原生按该精度计算**的最优代码,显存和带宽都按实际精度算,这才是量化加速能落地的前提。
</details>

---

## 小结

- **重载**:同名函数靠"参数类型 + 个数"区分(与返回类型无关),解决"同逻辑不同类型"的命名,但要手写多份。
- **模板**:把类型抽象成参数 `T`,写一份蓝图,编译器按调用类型**编译期实例化**出多份专用代码——零运行时开销。
- 模板参数除了类型,还能是**编译期常量**(如 tile/block 大小),用于性能优化。
- 算子库爱用模板的真正动机:**一份逻辑支持 FP32/FP16/BF16/INT8/FP8 多精度**,这是量化推理的工程基础。
- 模板 ≠ Python 泛型:前者编译期生成、零开销、报错在编译期;后者运行时动态、有开销。

## 自测验收(过了再进 Lesson 3)
- [ ] 能说清重载的判定规则,并解释"为什么不能只靠返回类型重载"。
- [ ] 能默写一个 `template<typename T> T add(T,T)` 并跑通。
- [ ] 能解释"模板实例化"是什么、发生在什么时候。
- [ ] 能讲清算子库用模板支持多精度的动机,并联系到量化。

下一课:**Lesson 3 — RAII、智能指针与 move 语义**,解决 Lesson 1 留下的"手动管理内存太危险"的痛点。
