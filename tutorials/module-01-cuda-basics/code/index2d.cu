// M1 Lesson 3 实验 B:打印每个线程的二维坐标与换算出的一维下标。
// 目标:用一个小网格(grid 2x2, block 4x4)手动核对索引公式。
// 编译运行: nvcc index2d.cu -o index2d && ./index2d

#include <cstdio>

__global__ void print_index(int width) {
    // TODO(1): 用二维公式算出 col 和 row
    //   col = blockIdx.x * blockDim.x + threadIdx.x;
    //   row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO(2): 算出行优先的一维下标 idx = row * width + col;

    // TODO(3): printf 打印 (blockIdx.x,blockIdx.y) (threadIdx.x,threadIdx.y)
    //          以及 row, col, idx,方便你手动核对
}

int main() {
    const int width = 8;        // 假想矩阵宽度,用于换算一维下标
    dim3 block(4, 4);           // 每个 block 16 个线程
    dim3 grid(2, 2);            // 共 4 个 block → 8x8 = 64 个线程
    print_index<<<grid, block>>>(width);
    cudaDeviceSynchronize();    // kernel 异步,必须同步才能看到 printf 输出
    return 0;
}
