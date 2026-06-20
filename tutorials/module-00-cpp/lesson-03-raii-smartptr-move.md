# M0 · Lesson 3:RAII、智能指针与 move 语义

> Lesson 1 留了个痛点:手动 `new/delete` 太容易漏、太容易错。本课交付现代 C++ 最重要的一招——**RAII**,以及它的产物**智能指针**和**move 语义**。学完你会明白:为什么现代 C++ 代码里几乎看不到裸露的 `delete`,以及 vLLM 源码里那些 `std::shared_ptr<...>`、`std::move(...)` 到底在干什么。
> 预计用时:2.5 小时。
> 前置:Lesson 1(指针/栈堆)、Lesson 2(模板,智能指针就是类模板)。

## 学习目标

学完本课你应该能:
1. 用一句话说清 **RAII** 是什么,并写出一个构造申请、析构释放的 RAII 类。
2. 区分 `unique_ptr`(独占)和 `shared_ptr`(引用计数),知道各自该用哪个。
3. 解释 **move 语义** 解决了什么问题,看懂 `std::move(x)` 后 `x` 发生了什么。
4. 把 RAII/智能指针的思想迁移到"自动管理 GPU 显存"的场景。

---

## 1. 核心思想:RAII——把资源绑在对象生命周期上

先看 Python 你早就在用的东西:

```python
with open("f.txt") as f:
    data = f.read()
# 出了 with 块,文件自动关闭,哪怕中间抛异常也会关
```

`with` 的本质是:**进入时获取资源(开文件),离开时自动释放(关文件)**,你不用手写 `f.close()`。

C++ 没有 `with`,但它有更通用、更底层的机制,而且是**全自动**的——**RAII(Resource Acquisition Is Initialization,资源获取即初始化)**:

> **RAII = 在对象的构造函数里获取资源,在析构函数里释放资源。** 而 C++ 保证:**栈上对象一旦离开作用域,其析构函数必被自动调用**(正常返回、异常退出都一样)。于是"释放"这件事被绑定到对象生命周期上,自动发生,你永远不会忘。

看 `code/raii_demo.cpp`:

```9:25:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/raii_demo.cpp
class Buffer {
public:
    explicit Buffer(int n) : size_(n) {
        data_ = new float[n];                 // 构造 = 申请资源
        printf("  [Buffer] 申请 %d 个 float\n", n);
    }
    ~Buffer() {
        delete[] data_;                        // 析构 = 释放资源(自动调用!)
        printf("  [Buffer] 释放 %d 个 float\n", size_);
    }
    float* data() { return data_; }
private:
    float* data_;
    int    size_;
};
```

```bash
g++ -std=c++17 -Wall raii_demo.cpp -o raii_demo && ./raii_demo
```

输出:

```
进入 use_buffer
  [Buffer] 申请 8 个 float
buf[0] = 3.14
离开 use_buffer(此刻 buf 自动析构,无需手写 delete)
  [Buffer] 释放 8 个 float
回到 main,Buffer 早已被自动清理
```

注意 `[Buffer] 释放` 那行:**没有任何人手写 `delete`**,它是 `buf` 离开 `use_buffer` 作用域时,析构函数被自动调用打印的。这就是 RAII 的魔力。

```
栈对象生命周期(自动)
  进入作用域 ──► 构造函数(申请)  ┐
                                  │  对象存活,资源可用
  离开作用域 ──► 析构函数(释放)  ┘  ← 自动!异常也会触发
```

> RAII 是现代 C++ 安全性的基石:文件、锁、内存、显存、网络连接……一切"用完要还"的资源,都用 RAII 类包起来,就再也不会泄漏。CUDA 里把 `cudaMalloc`/`cudaFree` 包成一个 RAII 类,正是工程里的标准做法。

---

## 2. 智能指针:标准库给你现成的 RAII 内存管家

你不必每次都手写 `Buffer` 那样的类。C++ 标准库的 `<memory>` 已经为"管理堆内存"提供了两个 RAII 类模板:**智能指针**。它们用起来像指针(支持 `*`、`->`),但会**自动 `delete`**。

### 2.1 `unique_ptr`:独占所有权

`unique_ptr<T>` 表示"这块内存**只有我一个主人**"。它不可拷贝(拷贝会产生两个主人,矛盾),离开作用域自动释放。

```13:18:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/smartptr_demo.cpp
        std::unique_ptr<Tensor> a = std::make_unique<Tensor>(1);
        printf("a->id = %d\n", a->id);
        // unique_ptr 不能拷贝(独占),只能 move 转移所有权(下个 demo 详讲)
        // 出作用域时自动 delete,无需手写
```

- 用 `std::make_unique<T>(构造参数...)` 创建,**永远优先用 `make_unique`** 而不是裸 `new`。
- 零运行时开销:它就是个裹着裸指针的薄壳,大小和裸指针一样,没有引用计数。
- 这是你的**默认选择**:99% 想"堆上放个东西、用完自动回收"的场景,用 `unique_ptr`。

### 2.2 `shared_ptr`:引用计数,共享所有权

`shared_ptr<T>` 表示"这块内存**可以有多个主人**,等最后一个主人走了才释放"。它内部维护一个**引用计数**:

```20:33:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/smartptr_demo.cpp
        std::shared_ptr<Tensor> p = std::make_shared<Tensor>(2);
        printf("use_count = %ld\n", p.use_count());   // 1
        {
            std::shared_ptr<Tensor> q = p;            // 拷贝 = 计数 +1
            printf("use_count = %ld (q 也指向它)\n", p.use_count());  // 2
        }
        printf("use_count = %ld (q 出作用域,计数 -1)\n", p.use_count()); // 1
        // 计数归 0 时才真正 delete
```

```bash
g++ -std=c++17 -Wall smartptr_demo.cpp -o smartptr_demo && ./smartptr_demo
```

> 这不就是 **Python 的内存模型**吗?对!Python 的对象正是靠引用计数(CPython)管理,`b = a` 让计数 +1,变量销毁让计数 -1,归零回收。`shared_ptr` 等于把 Python 那套"引用计数 GC"按需、局部地搬进 C++,但**只对你显式声明的 shared_ptr 生效**,且有计数的运行时开销。

选择原则:

| | `unique_ptr` | `shared_ptr` |
|---|---|---|
| 所有权 | 独占(单一主人) | 共享(多主人) |
| 可拷贝 | 否(只能 move) | 是(计数 +1) |
| 开销 | 零(同裸指针) | 有(原子引用计数) |
| 何时用 | **默认首选** | 确实需要多处共享同一对象时 |

> 经验法则:**默认 `unique_ptr`,只有当一份资源真的要被多个对象共享、且谁最后释放说不清时,才升级到 `shared_ptr`。** 别无脑用 `shared_ptr`——它的原子计数在高频路径上是真实开销。

---

## 3. move 语义:转移所有权,而不是深拷贝

这是现代 C++ 最容易让人迷糊、但对性能至关重要的概念。先讲它解决什么问题。

### 3.1 问题:拷贝一个大对象很贵

假设你有一个装着 10 亿个 float 的 `vector`(几 GB)。把它"传给"另一个变量:

```cpp
std::vector<float> a = make_huge_vector();   // 几 GB
std::vector<float> b = a;                    // 拷贝构造:把几 GB 逐字节复制一遍!
```

`b = a` 是**深拷贝**:重新申请几 GB 内存,逐字节复制。但很多时候你根本不想要副本——你只是想把 `a` 的内容"搬"到 `b`,之后 `a` 不用了。复制纯属浪费。

### 3.2 解决:move——"偷走"内部资源,O(1)

`vector` 内部其实就是"一个指向堆缓冲区的指针 + size"。所谓 move,就是**把这个内部指针直接转交给 `b`,再把 `a` 的指针置空**——不碰那几 GB 数据本身,只搬一个指针,O(1)。

```9:24:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/move_demo.cpp
    std::vector<int> big(5, 7);              // 5 个 7
    printf("big.size() = %zu\n", big.size());

    // ① 拷贝构造:深拷贝,big 仍然有效(两块独立内存)
    std::vector<int> copied = big;
    printf("拷贝后: big.size()=%zu, copied.size()=%zu\n",
           big.size(), copied.size());

    // ② move 构造:把 big 的内部缓冲区"偷"给 moved,big 被掏空(置为空壳)
    //    没有逐元素拷贝,只是把内部指针转移过去 —— O(1),极快
    std::vector<int> moved = std::move(big);
    printf("move 后: big.size()=%zu (被掏空), moved.size()=%zu\n",
           big.size(), moved.size());
```

```bash
g++ -std=c++17 -Wall move_demo.cpp -o move_demo && ./move_demo
```

输出关键两行:

```
拷贝后: big.size()=5, copied.size()=5        ← 拷贝:两个都有 5 个元素(独立)
move 后: big.size()=0 (被掏空), moved.size()=5  ← move:big 被掏空,资源转给了 moved
```

```
拷贝(深拷贝,贵)                 move(转移所有权,O(1))

a:[ptr]→[■■■■■ 几GB]            move 前  a:[ptr]→[■■■■■ 几GB]
b:[ptr]→[■■■■■ 副本]                     b: 空

复制了整块数据                    move 后  a:[null]   (被掏空)
                                          b:[ptr]→[■■■■■ 几GB]  (指针转交)
                                  数据没动,只转交了指针
```

### 3.3 `std::move` 和右值引用 `&&`

- `std::move(x)` **本身不移动任何东西**!它只是个类型转换:把 `x` 标记成"我接下来不要它了,你可以掏空它"——即转成一个**右值引用**。真正的"偷"动作发生在接收方的**移动构造函数 `T(T&& other)`** 里。
- `T&&`(两个 `&`)叫**右值引用**,专门用来绑定"即将废弃的临时对象",是 move 的语法载体。你现在记住:**看到 `&&` 和 `std::move`,脑子里就该浮现"所有权转移,源对象将被掏空"**。

> 一个铁律:**对一个变量用了 `std::move` 之后,就不要再使用它的值了**(它已被掏空,处于"有效但未指定"状态)。这是新手常见 bug。

### 3.4 为什么这对推理 Infra 极其重要

推理引擎里到处是"大块缓冲区/显存句柄/KV Cache 块"在不同模块间传递。如果每次传递都深拷贝几 GB,性能直接崩。move 让这些传递都变成 O(1) 的指针转交。而且——

> **`unique_ptr` 不能拷贝,只能 `move`**。这正是它"独占所有权"的体现:你不能复制一个独占指针(那就成了两个主人),但可以把所有权 `std::move` 转交出去(原 `unique_ptr` 变空)。vLLM/引擎源码里 `return std::move(ptr);`、`某容器.push_back(std::move(obj))` 满地都是,看懂它你就不晕了。

---

## 4. 动手实验

### 实验 A:观察 RAII 自动析构(必做)
跑 `raii_demo.cpp`,**重点看输出顺序**:确认"释放"那行是在离开 `use_buffer` 时自动打印的。再把 `Buffer buf(8);` 放进一个 `if(true){...}` 块里,观察析构提前到块结束时发生。

### 实验 B:智能指针引用计数(必做)
跑 `smartptr_demo.cpp`,跟着打印的 `use_count` 走一遍,确认 `q = p` 让计数变 2、`q` 出作用域变回 1、归零才析构。把 `shared_ptr` 换成 `unique_ptr` 再试 `q = p`——**观察编译报错**,理解"unique 不可拷贝"。

### 实验 C:手写 move(选做但强烈推荐)
补全 `move_demo.cpp` 里的 TODO:写一个含 `new[]` 的 `MyArray` 类,分别实现拷贝构造(深拷贝)和移动构造 `MyArray(MyArray&& o) noexcept`(偷指针 + 把 `o` 置空)。各调用一次,打印地址,亲眼确认 move 后两个对象的内部指针相同、源对象被置空。

---

## 练习题

1. RAII 类的析构函数,在"函数中途抛出异常提前退出"时还会被调用吗?为什么这点很重要?

<details><summary>参考答案</summary>

**会**。C++ 保证:当异常导致栈展开(stack unwinding)时,已构造的栈对象的析构函数会被**逐个自动调用**。这正是 RAII 比"手写 `delete`"优越的核心原因——手写 `delete` 在异常路径上极易被跳过导致泄漏,而 RAII 把释放绑在析构上,任何退出路径(正常 return / 异常)都保证执行。这就是为什么现代 C++ 提倡"资源一律用 RAII 包裹"。
</details>

2. 什么时候该用 `unique_ptr`,什么时候才升级到 `shared_ptr`?

<details><summary>参考答案</summary>

默认用 `unique_ptr`:它表达"单一明确的所有者",零开销,语义清晰。只有当**同一份资源确实需要被多个对象同时持有,且无法静态确定谁最后释放**时,才用 `shared_ptr`(靠引用计数决定释放时机)。滥用 `shared_ptr` 的代价:原子引用计数的运行时开销、所有权变模糊难以推理、还可能因循环引用导致泄漏(需 `weak_ptr` 打破)。
</details>

3. 执行 `std::vector<int> b = std::move(a);` 后,`a` 处于什么状态?能继续用吗?

<details><summary>参考答案</summary>

`a` 被掏空,处于"**有效但未指定(valid but unspecified)**"状态——对标准库容器而言通常是空(`size()==0`),对象本身仍合法(可以析构、可以重新赋值),但你**不应再依赖它原来的值**。可以给它重新赋值后再用,但直接读它原来的内容是逻辑错误。
</details>

4. 为什么 `std::move(x)` 这个名字有误导性?它实际做了什么?

<details><summary>参考答案</summary>

误导在于它叫 "move" 却**不移动任何数据**。`std::move(x)` 本质只是一个**类型转换**:把 `x` 转成右值引用(`T&&`),从而"告知"重载决议——可以选择移动构造/移动赋值版本。真正的资源转移动作发生在接收它的移动构造函数 / 移动赋值运算符内部。可以把 `std::move(x)` 理解为"我授权你可以掏空 x"的一个标记,动手掏的是别人。
</details>

---

## 小结

- **RAII**:构造申请资源、析构释放资源;栈对象离开作用域必自动析构(异常也触发)。这是 C++ 自动、安全管理资源的根基。
- **智能指针**是现成的 RAII 内存管家:`unique_ptr`(独占、零开销、**默认首选**)、`shared_ptr`(引用计数、共享、有开销)。优先 `make_unique`/`make_shared`。
- **move 语义**:把内部资源(指针)O(1) 转交,而非深拷贝大块数据;`std::move` 只是标记成右值,真正转移在移动构造里;move 后源对象被掏空,勿再用。
- 这三件套合起来,让现代 C++ 代码几乎不再出现裸 `delete`;它们直接迁移到"RAII 封装显存""move 转移 KV Cache 块"等推理 Infra 场景。

## 自测验收(过了再进 Lesson 4)
- [ ] 能写出一个构造申请、析构释放的 RAII 类并解释自动析构。
- [ ] 能说清 `unique_ptr` vs `shared_ptr` 的区别和选择原则。
- [ ] 能解释 move 与拷贝的本质差异,并默写"move 后源对象被掏空"。
- [ ] 能讲清 `std::move` 为什么"名不副实"。
- [ ] 实验 A/B 跑通,亲眼见过自动析构和引用计数变化。

下一课:**Lesson 4 — 编译链接全过程与 CMake 入门**,把前三课的代码组织成真正的多文件项目,为后面 C++/CUDA 工程打底。
