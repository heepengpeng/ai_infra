// M0 Lesson 2:函数重载——同名函数,按参数类型区分。
// 编译运行: g++ -std=c++17 -Wall overload_demo.cpp -o overload_demo && ./overload_demo

#include <cstdio>

// 三个同名函数,编译器按"参数类型"挑哪一个 —— 这叫重载(overload)
int  add(int a, int b)        { return a + b; }
double add(double a, double b){ return a + b; }
// 参数个数不同也算重载
int  add(int a, int b, int c) { return a + b + c; }

int main() {
    printf("add(1, 2)      = %d\n",   add(1, 2));        // 选 int 版
    printf("add(1.5, 2.5)  = %.1f\n", add(1.5, 2.5));    // 选 double 版
    printf("add(1, 2, 3)   = %d\n",   add(1, 2, 3));     // 选三参数版
    return 0;
}
