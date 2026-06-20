// M1 Lesson 4 实验 B:对比 pageable(malloc) 与 pinned(cudaMallocHost) 的 H2D 带宽。
// 编译运行: nvcc pinned_vs_pageable.cu -o pinned_vs_pageable && ./pinned_vs_pageable

#include <cstdio>
#include <cstdlib>
#include <chrono>

using clk = std::chrono::high_resolution_clock;

// 多次拷贝取平均,返回平均带宽 GB/s
static double bench_h2d(float* h_src, float* d_dst, size_t bytes, int iters) {
    // 预热一次(首次传输有额外开销)
    cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    double seconds = ms / 1000.0;
    return (double)bytes * iters / seconds / 1e9;   // GB/s
}

int main() {
    const size_t bytes = (size_t)256 * 1024 * 1024;   // 256 MB
    const int iters = 20;

    float* d;
    cudaMalloc(&d, bytes);

    // ---- pageable:普通 malloc ----
    float* h_pageable = (float*)malloc(bytes);
    double bw_pageable = bench_h2d(h_pageable, d, bytes, iters);

    // ---- pinned:cudaMallocHost ----
    float* h_pinned = nullptr;
    // TODO(1): 用 cudaMallocHost(&h_pinned, bytes) 申请锁页内存

    // TODO(2): 调用 bench_h2d 测 pinned 带宽,存到 double bw_pinned
    double bw_pinned = 0.0;

    printf("pageable H2D : %6.2f GB/s\n", bw_pageable);
    printf("pinned   H2D : %6.2f GB/s\n", bw_pinned);
    printf("speedup      : %.2fx\n", bw_pinned / bw_pageable);

    // TODO(3): 正确释放 —— free(h_pageable); cudaFreeHost(h_pinned); cudaFree(d);
    free(h_pageable);
    return 0;
}
