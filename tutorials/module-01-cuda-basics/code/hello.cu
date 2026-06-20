// M1 Lesson 2 实验 A:第一个 kernel。
// 编译运行: nvcc hello.cu -o hello && ./hello

#include <cstdio>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d (block=%d, threadInBlock=%d)\n",
           tid, blockIdx.x, threadIdx.x);
}

int main() {
    // 练习:把 <<<2, 4>>> 改成 <<<3, 5>>>,先预测输出再验证。
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
