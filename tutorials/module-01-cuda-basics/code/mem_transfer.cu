// M1 Lesson 4 实验 A:粗略测量 H2D / kernel / D2H 三段耗时占比。
// 注意:这里用 CPU 端 wall-clock 粗测,kernel 异步性会让计时不准 —— 这正是
//       Lesson 5 要解决的问题。本课只为建立"传输 >> 轻计算"的直觉。
// 编译运行: nvcc mem_transfer.cu -o mem_transfer && ./mem_transfer

#include <cstdio>
#include <cstdlib>
#include <chrono>

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0) {
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

__global__ void scale_kernel(float* x, float a, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) x[tid] = a * x[tid];   // 极轻计算:一次乘法
}

int main() {
    const int n = 1 << 24;              // ~1600 万个 float ≈ 64 MB
    const size_t bytes = n * sizeof(float);

    float* h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;

    float* d;
    cudaMalloc(&d, bytes);

    // ① H2D
    auto t0 = clk::now();
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    double t_h2d = ms_since(t0);

    // ② kernel(cudaDeviceSynchronize 等它真正算完再停表)
    int block = 256, grid = (n + block - 1) / block;
    t0 = clk::now();
    scale_kernel<<<grid, block>>>(d, 2.0f, n);
    cudaDeviceSynchronize();
    double t_kernel = ms_since(t0);

    // ③ D2H
    t0 = clk::now();
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    double t_d2h = ms_since(t0);

    printf("data = %.1f MB\n", bytes / 1e6);
    printf("H2D    : %7.3f ms\n", t_h2d);
    printf("kernel : %7.3f ms\n", t_kernel);
    printf("D2H    : %7.3f ms\n", t_d2h);
    printf("-> 传输(H2D+D2H) 占比 %.1f%%\n",
           100.0 * (t_h2d + t_d2h) / (t_h2d + t_kernel + t_d2h));

    cudaFree(d);
    free(h);
    return 0;
}
