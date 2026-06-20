// M0 Lesson 1:指针与引用的最小演示。
// 编译运行: g++ -std=c++17 -Wall pointers_demo.cpp -o pointers_demo && ./pointers_demo

#include <cstdio>

void add_one_by_value(int x)   { x += 1; }       // 改的是副本,外面看不到
void add_one_by_pointer(int* p){ *p += 1; }      // 通过地址改原值
void add_one_by_ref(int& r)    { r += 1; }       // 引用 = 原变量的别名

int main() {
    int a = 10;

    add_one_by_value(a);
    printf("after by_value:   a = %d\n", a);   // 仍是 10

    add_one_by_pointer(&a);                    // 显式取地址传进去
    printf("after by_pointer: a = %d\n", a);   // 11

    add_one_by_ref(a);                         // 直接传变量,语法上像值传递
    printf("after by_ref:     a = %d\n", a);   // 12

    // 指针本身也是一个变量,它存的是地址。打印看看地址和值的区别。
    int* p = &a;
    printf("p (a 的地址) = %p, *p (a 的值) = %d\n", (void*)p, *p);

    return 0;
}
