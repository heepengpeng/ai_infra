# M0 Lesson 4:最小 CMake 工程。
# 构建命令(在本目录下):
#   cmake -S . -B build      # 配置:读 CMakeLists,生成构建系统到 build/
#   cmake --build build      # 编译:真正调用 g++ 编译链接
#   ./build/app              # 运行
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
