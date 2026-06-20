// M0 Lesson 4:实现文件 = 函数体的真正定义。
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
