// bandwidth_test.cu
// 实测 GPU 全局显存峰值带宽:用纯 copy kernel(零计算),多次取最优。
// 编译: nvcc -O3 bandwidth_test.cu -o bandwidth_test
// 运行: ./bandwidth_test
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 极简错误检查宏(M1 L5 的套路)。
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));               \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// 纯访存 kernel:读 4 字节 + 写 4 字节。grid-stride loop 覆盖任意大数组。
__global__ void copy_kernel(const float* __restrict__ in,
                            float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        out[i] = in[i];
    }
}

int main() {
    // 256 MB 级别的数组,远大于 L2,确保测到真实 HBM 带宽。
    const size_t n = 64ull * 1024 * 1024;   // 64M 个 float
    const size_t bytes = n * sizeof(float);
    const int runs = 50;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    int block = 256;
    // grid 取一个足以填满 GPU 的固定值,配合 grid-stride loop。
    int grid = 4096;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up:首次启动含一次性开销,不计入测量。
    copy_kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float best_ms = 1e30f;
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        copy_kernel<<<grid, block>>>(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best_ms) best_ms = ms;
    }

    // copy 每个元素搬运 = 读 4 + 写 4 = 8 字节。
    double moved_bytes = 2.0 * bytes;
    double gbps = moved_bytes / (best_ms / 1000.0) / 1e9;

    std::printf("[bandwidth] N = %zu floats (%.1f MB)\n", n, bytes / 1e6);
    std::printf("[bandwidth] best time = %.3f ms over %d runs\n", best_ms, runs);
    std::printf("[bandwidth] moved %.1f MB (read+write)\n", moved_bytes / 1e6);
    std::printf("[bandwidth] effective bandwidth = %.1f GB/s\n", gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
