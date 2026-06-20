// M1 Lesson 5 实验 B:用 cudaEvent 正确给向量加法计时,并算有效带宽。
// 编译运行: nvcc timing_bandwidth.cu -o timing_bandwidth && ./timing_bandwidth

#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",              \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) C[tid] = A[tid] + B[tid];
}

int main() {
    const int n = 1 << 24;                 // ~1678 万
    const size_t bytes = n * sizeof(float);

    float *h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h, bytes, cudaMemcpyHostToDevice));

    int block = 256, grid = (n + block - 1) / block;

    // ---- warmup:空跑几次吸收一次性开销 ----
    for (int i = 0; i < 5; ++i) vec_add<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- cudaEvent 正式计时 ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 100;
    // TODO(1): cudaEventRecord(start)
    for (int i = 0; i < iters; ++i) vec_add<<<grid, block>>>(d_A, d_B, d_C, n);
    // TODO(2): cudaEventRecord(stop)
    // TODO(3): cudaEventSynchronize(stop)
    float total_ms = 0.0f;
    // TODO(4): cudaEventElapsedTime(&total_ms, start, stop)
    float avg_ms = total_ms / iters;

    // 向量加法每元素读 A、读 B、写 C = 3 次访存
    double gb = 3.0 * bytes / 1e9;
    double bw = gb / (avg_ms / 1000.0);

    // ---- 对照:错误的 CPU 计时(不同步)----
    auto t0 = std::chrono::high_resolution_clock::now();
    vec_add<<<grid, block>>>(d_A, d_B, d_C, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("cudaEvent 平均: %.4f ms  ->  有效带宽 %.1f GB/s\n", avg_ms, bw);
    printf("错误的CPU计时 : %.4f ms  (没同步,基本只是启动开销)\n", cpu_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h);
    return 0;
}
