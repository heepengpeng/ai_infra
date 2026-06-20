// M0 Lesson 4:使用方只 include 头文件,看不到也不需要实现细节。
#include <cstdio>
#include "mathops.h"

int main() {
    printf("add(3, 4) = %d\n", mathops::add(3, 4));

    double xs[4] = {1.0, 2.0, 3.0, 4.0};
    printf("mean = %.2f\n", mathops::mean(xs, 4));
    return 0;
}
