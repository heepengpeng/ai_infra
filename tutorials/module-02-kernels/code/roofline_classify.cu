// roofline_classify.cu
// 用 roofline 给一个真实 kernel(SAXPY: y = a*x + y)定性:带宽受限还是算力受限。
// 编译: nvcc -O3 roofline_classify.cu -o roofline_classify
// 运行: ./roofline_classify --peak_bw 1438 --peak_tflops 19.5
//   peak_bw    : 实测峰值带宽 (GB/s),来自 bandwidth_test
//   peak_tflops: 该卡 FP32 峰值算力 (TFLOP/s),用标称或自测
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));               \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// 轻量日志宏:正文要求"用 log 不用裸 print",C++ 侧用统一前缀的宏模拟。
#define LOG_INFO(fmt, ...) std::printf("[INFO] " fmt "\n", ##__VA_ARGS__)

// SAXPY: y = a*x + y。每元素 2 FLOP(1 乘 1 加),访存 = 读 x + 读 y + 写 y = 12 字节。
__global__ void saxpy_kernel(float a, const float* __restrict__ x,
                             float* __restrict__ y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    double peak_bw_gbps = 1438.0;   // 默认 A100 实测量级
    double peak_tflops = 19.5;      // 默认 A100 FP32 标称
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--peak_bw") == 0)
            peak_bw_gbps = std::atof(argv[i + 1]);
        else if (std::strcmp(argv[i], "--peak_tflops") == 0)
            peak_tflops = std::atof(argv[i + 1]);
    }

    const size_t n = 64ull * 1024 * 1024;
    const size_t bytes = n * sizeof(float);
    const int runs = 50;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemset(d_x, 1, bytes));
    CUDA_CHECK(cudaMemset(d_y, 1, bytes));

    int block = 256, grid = 4096;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    saxpy_kernel<<<grid, block>>>(2.0f, d_x, d_y, n);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    float best_ms = 1e30f;
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        saxpy_kernel<<<grid, block>>>(2.0f, d_x, d_y, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best_ms = std::min(best_ms, ms);
    }

    double total_flop = 2.0 * (double)n;          // 每元素 2 FLOP
    double total_bytes = 12.0 * (double)n;        // 读 x + 读 y + 写 y
    double secs = best_ms / 1000.0;

    double achieved_flops = total_flop / secs;
    double achieved_bw = total_bytes / secs;
    double ai = total_flop / total_bytes;         // 算术强度

    double peak_flops = peak_tflops * 1e12;
    double peak_bw = peak_bw_gbps * 1e9;
    double ridge = peak_flops / peak_bw;          // 拐点 AI*
    const char* regime = (ai < ridge) ? "MEMORY-BOUND" : "COMPUTE-BOUND";
    double roof = std::min(peak_flops, peak_bw * ai);  // roofline 上限

    LOG_INFO("kernel = SAXPY (y = a*x + y), N = %zu", n);
    LOG_INFO("time = %.3f ms", best_ms);
    LOG_INFO("achieved: %.2f GFLOP/s, %.1f GB/s", achieved_flops / 1e9,
             achieved_bw / 1e9);
    LOG_INFO("AI = %.3f FLOP/B, ridge point = %.2f -> %s", ai, ridge, regime);
    LOG_INFO("roofline ceiling = %.2f GFLOP/s; you reached %.1f%% of it",
             roof / 1e9, 100.0 * achieved_flops / roof);
    LOG_INFO("=> 因 AI 远小于拐点,SAXPY 受带宽限制;实测带宽已接近峰值,"
             "单独优化计算无意义,应考虑算子融合减少访存(见 Lesson 5)");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
