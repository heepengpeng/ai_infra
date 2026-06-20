// M1 Lesson 3 实验 A:二维矩阵逐元素加法 C = A + B。
// 编译运行: nvcc matadd2d.cu -o matadd2d && ./matadd2d

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void mat_add(const float* A, const float* B, float* C,
                        int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → col
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y → row
    // 两个维度都向上取整启动,所以行、列都要做边界检查
    if (row < height && col < width) {
        int idx = row * width + col;                   // 行优先:二维 → 一维
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int width = 1024, height = 768;
    const int n = width * height;
    const size_t bytes = n * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_A[i] = (float)i; h_B[i] = 2.0f * i; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    mat_add<<<grid, block>>>(d_A, d_B, d_C, width, height);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 自校验:每个元素应满足 C = A + B = 3*i
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_C[i] - 3.0f * i) > 1e-3f) { ++errors; }
    }
    printf("grid=(%d,%d) block=(%d,%d), n=%d, errors=%d -> %s\n",
           grid.x, grid.y, block.x, block.y, n, errors,
           errors == 0 ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
