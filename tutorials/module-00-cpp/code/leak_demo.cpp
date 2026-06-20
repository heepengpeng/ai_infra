// M0 Lesson 1 练习:故意制造一个内存泄漏,用工具抓出来。
// 编译运行(普通):  g++ -std=c++17 leak_demo.cpp -o leak_demo && ./leak_demo
// 抓泄漏(Linux/支持的环境):
//   g++ -std=c++17 -fsanitize=address -g leak_demo.cpp -o leak_demo && ./leak_demo
//   程序退出时 ASan 会报 "detected memory leaks"。

#include <cstdio>

void leaky() {
    int* p = new int[1000];   // 申请了 4000 字节
    p[0] = 42;
    printf("leaky: p[0] = %d\n", p[0]);
    // TODO: 这里"忘了" delete[] p; —— 函数返回后 p 出作用域,
    //       但它指向的 4000 字节堆内存永远没人能再访问、也没被释放 = 泄漏。
    //       把 delete[] p; 加回来,再用 ASan 跑一次,对比报告。
}

int main() {
    for (int i = 0; i < 3; ++i) leaky();   // 漏 3 次
    return 0;
}
