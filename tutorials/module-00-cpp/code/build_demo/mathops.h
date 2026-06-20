// M0 Lesson 4:头文件 = 接口声明(告诉别人"有这些函数,长这样")。
// 头文件只放声明,不放实现(模板除外),避免重复定义。
#pragma once   // 防止同一头文件被重复 include 导致重定义(等价老式 include guard)

namespace mathops {

// 只声明,不实现。实现放在 mathops.cpp
int    add(int a, int b);
double mean(const double* data, int n);

}  // namespace mathops
