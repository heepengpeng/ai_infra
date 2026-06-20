// M0 Lesson 2:函数模板——一份代码,适配任意类型。
// 编译运行: g++ -std=c++17 -Wall template_demo.cpp -o template_demo && ./template_demo

#include <cstdio>
#include <string>

// 函数模板:T 是"类型占位符",编译期按调用时的实参类型生成具体函数
template <typename T>
T add(T a, T b) {
    return a + b;
}

// 模板也能配合上一节的指针:一个"对数组逐元素 ×2"的通用函数
// (这正是 CUDA 逐元素 kernel 的 CPU 版雏形)
template <typename T>
void scale_inplace(T* data, int n, T factor) {
    for (int i = 0; i < n; ++i) data[i] *= factor;
}

int main() {
    printf("add<int>    : %d\n",   add(3, 4));         // T 推导为 int
    printf("add<double> : %.2f\n", add(1.5, 2.5));     // T 推导为 double
    // 显式指定类型(当无法从参数推导,或想强制时)
    printf("add<double> : %.2f\n", add<double>(1, 2)); // 强制走 double 版

    float arr[4] = {1, 2, 3, 4};
    scale_inplace<float>(arr, 4, 10.0f);               // T = float
    printf("scaled: %.0f %.0f %.0f %.0f\n", arr[0], arr[1], arr[2], arr[3]);
    return 0;
}
