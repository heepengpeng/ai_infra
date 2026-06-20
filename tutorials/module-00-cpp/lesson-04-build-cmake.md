# M0 · Lesson 4:编译链接全过程与 CMake 入门

> 前三课你都是 `g++ 一个文件.cpp` 一把梭。但真实项目(vLLM、TensorRT-LLM)有成百上千个文件、还混着 `.cu`。本课打通两件事:**一份源码到可执行文件中间到底发生了什么**(编译/链接四步),以及**怎么用 CMake 管理多文件 C++/CUDA 项目**。这是你日后 clone 一个引擎仓库、`cmake .. && make` 能跑起来的底层认知。
> 预计用时:2 小时。
> 前置:Lesson 1–3;装好 `g++` 和 `cmake`(`cmake --version` 能打印即可,没装见文末附录)。

## 学习目标

学完本课你应该能:
1. 说清源码变可执行文件的**四个阶段**(预处理 → 编译 → 汇编 → 链接),知道 `.o` 和可执行文件的区别。
2. 解释**头文件(声明)与实现(定义)分离**的意义,看懂 `#include` 和 `#pragma once`。
3. 读懂并写出一个最小 `CMakeLists.txt`,用 `cmake -S . -B build && cmake --build build` 构建多文件项目。
4. 理解 CMake 在 C++/CUDA 工程里的角色,为后面 clone 推理引擎打基础。

---

## 1. 先有个直觉:Python 没有"编译链接"这一关

Python 你 `python main.py` 就跑了,`import utils` 解释器运行时自动找模块。没有"先编译成机器码、再把多个文件拼起来"的步骤——因为 Python 是解释执行的。

C++ 是**编译型**语言:源码必须先被翻译成 CPU 能直接执行的机器码,装进一个可执行文件,才能运行。这中间是一条流水线:

```
你的源码 .cpp / .h
      │
      ▼   ① 预处理 (Preprocess)   g++ -E
展开 #include、宏替换后的纯 .cpp
      │
      ▼   ② 编译 (Compile)        g++ -S
汇编代码 .s
      │
      ▼   ③ 汇编 (Assemble)       g++ -c   ← 产出 .o(目标文件)
机器码目标文件 .o(还不能直接跑,缺别处的函数地址)
      │
      ▼   ④ 链接 (Link)           g++ (默认最后一步)
可执行文件(a.out / app)← 把多个 .o + 库拼成一个完整程序
```

平时 `g++ main.cpp -o app` 是把这四步**一条龙**做完了。理解每一步在干嘛,才能看懂报错到底卡在哪(编译错误 vs 链接错误是两类完全不同的问题)。

| 阶段 | 干什么 | 典型报错 |
|---|---|---|
| 预处理 | 展开 `#include`、处理 `#define`/`#pragma` | 头文件找不到 `No such file` |
| 编译 | 语法检查、生成汇编 | 语法错、类型错、未声明的标识符 |
| 汇编 | 汇编 → 机器码 `.o` | 极少直接报错 |
| 链接 | 把多个 `.o` 和库拼起来,解析符号 | `undefined reference`(找不到函数实现)、`duplicate symbol`(重复定义) |

> 实战中最让新手困惑的就是 **`undefined reference`**:它**不是**编译错误,而是链接错误——意思是"我知道有这个函数(见过声明),但找遍了所有 `.o` 和库都没找到它的**实现**"。记住这个区分,能省你大量调试时间。

---

## 2. 为什么要"头文件 + 实现"分离?

C++ 工程的标准组织:**头文件 `.h`** 放声明(接口长什么样),**实现文件 `.cpp`** 放函数体(具体怎么做)。用的人只 `#include` 头文件。看本课的小项目 `code/build_demo/`:

`mathops.h`——只声明,不实现:

```1:13:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/build_demo/mathops.h
// M0 Lesson 4:头文件 = 接口声明(告诉别人"有这些函数,长这样")。
// 头文件只放声明,不放实现(模板除外),避免重复定义。
#pragma once   // 防止同一头文件被重复 include 导致重定义(等价老式 include guard)

namespace mathops {

// 只声明,不实现。实现放在 mathops.cpp
int    add(int a, int b);
double mean(const double* data, int n);

}  // namespace mathops
```

`mathops.cpp`——实现:

```4:17:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/build_demo/mathops.cpp
#include "mathops.h"   // 包含自己的头,让编译器核对声明与实现一致

namespace mathops {

int add(int a, int b) {
    return a + b;
}

double mean(const double* data, int n) {
    if (n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += data[i];
    return sum / n;
}

}  // namespace mathops
```

`main.cpp`——使用者只 include 头:

```1:11:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/build_demo/main.cpp
#include <cstdio>
#include "mathops.h"

int main() {
    printf("add(3, 4) = %d\n", mathops::add(3, 4));

    double xs[4] = {1.0, 2.0, 3.0, 4.0};
    printf("mean = %.2f\n", mathops::mean(xs, 4));
    return 0;
}
```

为什么这么分?

- **编译解耦**:`main.cpp` 只需知道 `add` 的**样子**(声明),不需要它的源码就能编译。改了 `mathops.cpp` 的实现,只需重编 `mathops.cpp`,`main.cpp` 不用动——大项目里这是编译速度的关键。
- **`#include` 本质就是文本粘贴**:预处理阶段把头文件内容原样贴到 `#include` 那一行。所以头文件里放实现会导致"被多个 cpp 各贴一份 → 链接时 `duplicate symbol`"。
- **`#pragma once`**:防止同一个头被一个 cpp 间接 include 多次(比如 a.h 和 b.h 都 include 了 c.h)造成重复。

> 类比 Python:头文件像 `.pyi` 类型存根(只有签名),实现像 `.py`(有函数体)。但 C++ 的分离是**强制的、编译期的**,不是可选的类型提示。

### 手动走一遍四步(理解链接)

在 `code/build_demo/` 目录:

```bash
# ③ 各自编译成 .o(注意 -c:只编译不链接)
g++ -std=c++17 -c mathops.cpp -o mathops.o
g++ -std=c++17 -c main.cpp    -o main.o
# ④ 链接:把两个 .o 拼成可执行文件
g++ mathops.o main.o -o app
./app
```

输出:

```
add(3, 4) = 7
mean = 2.50
```

试一下:如果你**漏掉** `mathops.o`,只 `g++ main.o -o app`,会得到经典的 `undefined reference to 'mathops::add(int, int)'`——因为 `main.o` 里只有"调用 add"的指令,add 的实体在 `mathops.o` 里,没给链接器它就找不到。这就是第 1 节说的链接错误,亲手制造一次印象最深。

---

## 3. 文件一多,手敲 g++ 就崩了——CMake 登场

两个文件还能手敲。但 vLLM 有上千个文件、几十个库依赖、还要分 Debug/Release、还混 CUDA `.cu`……手写 `g++` 命令完全不现实。

**CMake 是 C++/CUDA 世界事实上的标准构建系统**。它的定位:你用 `CMakeLists.txt` **声明**"我有哪些源文件、要编成什么、依赖哪些库",CMake 负责**生成**底层构建脚本(Makefile / Ninja)并调用编译器,自动处理编译顺序、增量编译、依赖关系。

> 类比:CMake 之于 C++,约等于 `pyproject.toml` / `setup.py` 之于 Python 项目——你声明项目结构和依赖,工具负责把它装配起来。区别是 CMake 管的是"编译链接"这件 Python 没有的事。

看本项目的 `CMakeLists.txt`:

```7:24:/Users/hp/010Code/007Python/001Learn/ai_infra/tutorials/module-00-cpp/code/build_demo/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(build_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)          # 等价 g++ -std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 把多个源文件编成一个可执行文件 app。
# CMake 会自动处理"先各自编译成 .o,再链接成可执行文件"的全过程。
add_executable(app
    main.cpp
    mathops.cpp
)
```

逐行:

- `cmake_minimum_required`:声明最低 CMake 版本。
- `project(... LANGUAGES CXX)`:工程名 + 用到的语言。**以后加 CUDA 就是 `LANGUAGES CXX CUDA`**。
- `set(CMAKE_CXX_STANDARD 17)`:用 C++17,等价于命令行 `-std=c++17`。
- `add_executable(app main.cpp mathops.cpp)`:声明"把这两个源文件编译链接成一个叫 `app` 的可执行文件"。你**不用**告诉它编译顺序、不用手写 `.o`,CMake 全包了。

### 标准构建三连(背下来,几乎所有 C++ 仓库都这套)

```bash
cmake -S . -B build      # 配置:读 CMakeLists.txt,在 build/ 里生成构建系统
cmake --build build      # 构建:真正调用编译器,产物在 build/
./build/app              # 运行
```

输出同样是 `add(3, 4) = 7` / `mean = 2.50`。

- `-S .`:源码目录(有 CMakeLists.txt 的地方)。
- `-B build`:构建产物放 `build/`(**out-of-source build**,所有中间文件隔离在此,`rm -rf build` 即可彻底清理,源码目录保持干净)。

> 你以后 clone vLLM 或任何 C++ 引擎,看到 `CMakeLists.txt` 就知道:`cmake -S . -B build && cmake --build build` 是入口。读懂顶层 `CMakeLists.txt` 就能快速搞清"这个项目由哪些模块/库组成"——这是读引擎源码的第一步。

---

## 4. 一眼看懂 CUDA 项目的 CMake(伏笔)

现在还不用动手(没 GPU),但提前认个脸。一个 C++/CUDA 混合项目的 CMake 大致长这样:

```cmake
cmake_minimum_required(VERSION 3.18)        # CUDA 支持需较新版本
project(my_cuda_proj LANGUAGES CXX CUDA)    # ← 多了 CUDA

set(CMAKE_CUDA_STANDARD 17)

# .cu 文件和 .cpp 文件可以混在一起,CMake 自动用 nvcc 编 .cu、用 g++ 编 .cpp
add_executable(app
    main.cpp
    kernel.cu        # ← CUDA kernel 源文件
)

# 指定目标 GPU 架构(比如 80=A100, 89=4090);后面 CUDA 课会讲
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "80")
```

> 关键认知:**CMake 把 `.cpp` 交给 g++、把 `.cu` 交给 nvcc,然后统一链接**,你不用手动协调两个编译器。这就是为什么 Module 1 之后我们能从容组织"C++ 主程序 + CUDA kernel"的多文件项目——本课打的就是这个地基。

---

## 5. 动手实验

### 实验 A:手动四步与链接错误(必做)
进入 `code/build_demo/`,按第 2 节手动 `-c` 编出两个 `.o` 再链接成 `app`,跑通。然后**故意**只用 `main.o` 链接(`g++ main.o -o app`),观察 `undefined reference` 报错,理解它属于链接阶段。

### 实验 B:CMake 构建(必做)
在 `code/build_demo/` 用三连命令构建并运行:

```bash
cmake -S . -B build && cmake --build build && ./build/app
```

然后改 `mathops.cpp` 里 `add` 的实现(比如 `return a + b + 100;`),**只**重新跑 `cmake --build build`,观察它只重编了 `mathops.cpp` 没碰 `main.cpp`(增量编译)。

### 实验 C:新增一个文件(选做)
在项目里加 `code/build_demo/strutil.h` + `strutil.cpp`(随便写个函数),在 `main.cpp` 里调用它。试着只靠"在 `add_executable` 里加一行 `strutil.cpp`",让 CMake 把它纳入构建。体会"加文件 = 改一行 CMake"的便利。

---

## 练习题

1. `undefined reference to 'foo'` 和 `'foo' was not declared in this scope`,分别是哪个阶段的错误?根因有何不同?

<details><summary>参考答案</summary>

- `'foo' was not declared in this scope` 是**编译错误**:编译当前 cpp 时,根本没见过 `foo` 的声明(没 include 对应头文件,或拼错名字)。编译器连"有这么个东西"都不知道。
- `undefined reference to 'foo'` 是**链接错误**:编译期见过声明(知道 `foo` 长什么样),但链接时在所有 `.o` 和库里找不到 `foo` 的**实现**(忘了编译/链接含实现的那个 cpp,或库没链上)。
两者根因不同:前者缺"声明",后者缺"定义/实现"。
</details>

2. 为什么不能把函数的**实现**(函数体)直接写在头文件里,然后被多个 `.cpp` include?(普通函数)

<details><summary>参考答案</summary>

因为 `#include` 本质是文本粘贴:每个 include 了该头的 cpp 都会得到一份完整的函数实现,各自编译出一份同名符号的 `.o`,链接时就会 `duplicate symbol`(重复定义)报错。这违反 C++ 的"单一定义规则"(ODR)。所以普通函数:头里放声明,cpp 里放唯一一份实现。(例外:`inline` 函数、**模板**——模板必须放头文件里,因为实例化要在使用点看到完整定义,这也是 Lesson 2 算子库头文件巨大的原因之一。)
</details>

3. CMake 推荐 "out-of-source build"(把产物放 `build/`),好处是什么?

<details><summary>参考答案</summary>

所有中间产物(`.o`、缓存、生成的 Makefile、可执行文件)都集中在 `build/` 目录,与源码完全隔离。好处:① 源码目录始终干净(便于 git 管理,`build/` 直接 gitignore);② 想彻底重新构建,`rm -rf build` 一键清理,不会残留;③ 可以为不同配置(Debug/Release、不同编译器)建多个 build 目录互不干扰。
</details>

4. 你 clone 了一个陌生的 C++ 推理引擎仓库,想快速跑起来并搞清它的结构,第一步该看什么?

<details><summary>参考答案</summary>

先看**顶层 `CMakeLists.txt`**(和 README 的 build 说明)。顶层 CMake 会告诉你:工程用什么语言(有没有 CUDA)、依赖哪些第三方库(`find_package`)、由哪些子模块/库构成(`add_subdirectory`、`add_library`/`add_executable`)、入口可执行文件是谁。读懂它就掌握了项目的"骨架地图"。然后用标准三连 `cmake -S . -B build && cmake --build build` 尝试构建。这是读任何 C++ 工程的通用入口。
</details>

---

## 小结

- 源码到可执行文件经过**四阶段**:预处理 → 编译 → 汇编(产出 `.o`)→ 链接(拼成可执行)。`g++ x.cpp` 是一条龙跑完。
- **编译错误 vs 链接错误**是两类:`not declared`(缺声明,编译期)≠ `undefined reference`(缺实现,链接期)。
- **头文件放声明、cpp 放实现**;`#include` 是文本粘贴,`#pragma once` 防重复;实现放头会导致重复定义(模板例外,必须放头)。
- **CMake** 是 C++/CUDA 标准构建系统:`CMakeLists.txt` 声明结构,三连命令 `cmake -S . -B build && cmake --build build && ./build/app` 构建运行。
- CUDA 项目只需 `LANGUAGES CXX CUDA`,CMake 自动用 nvcc 编 `.cu`、g++ 编 `.cpp` 再统一链接——这是后续组织 CUDA 工程的基础。

## 自测验收(过了恭喜你打通 Module 0)
- [ ] 能说清编译四阶段,以及 `.o` 和可执行文件的区别。
- [ ] 能区分 `not declared` 和 `undefined reference` 两类错误,并亲手制造过后者。
- [ ] 能解释头/实现分离的原因和 `#include` 的本质。
- [ ] 能独立写一个最小 `CMakeLists.txt`,用三连命令构建多文件项目。
- [ ] 能看懂 CUDA 版 CMake 多出来的 `LANGUAGES CUDA` 和 `CUDA_ARCHITECTURES`。

---

## 附录:装 CMake

- **macOS**:`brew install cmake`
- **Ubuntu/Debian**:`sudo apt install cmake`
- **验证**:`cmake --version`(本课需要 ≥ 3.16;CUDA 项目建议 ≥ 3.18)。

**Module 0 到此结束。** 你已经具备"能写 CUDA 算子、能读 vLLM/TensorRT-LLM 源码"所需的够用 C++ 基础:指针与内存模型、模板与多精度、RAII/智能指针/move、编译链接与 CMake。

下一站:**Module 1 · Lesson 1 — GPU 为什么快**,正式进入 CUDA 主线。你在本模块攒下的指针、内存、模板、构建认知,马上就会在 `cudaMalloc`、`template<typename T> __global__`、`.cu` 工程里全部兑现。
