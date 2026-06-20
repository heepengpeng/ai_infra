// gemm_tiled.cu
// 对比 naive GEMM 与 shared memory 分块(tiling)GEMM 的性能。
// 计算 C = A*B,A: MxK, B: KxN, C: MxN(本例取方阵 M=N=K)。
// 编译: nvcc -O3 gemm_tiled.cu -o gemm_tiled
// 运行: ./gemm_tiled 1024
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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

#define LOG_INFO(fmt, ...) std::printf("[INFO] " fmt "\n", ##__VA_ARGS__)

#define TILE 32

// ---- naive:一线程一输出,每次内层循环 2 次全局访存,零复用 ----
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ---- tiled:一个 block 协作把 TILExTILE 小块搬进 shared memory 复用 ----
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    int num_phase = (K + TILE - 1) / TILE;
    for (int ph = 0; ph < num_phase; ++ph) {
        int a_col = ph * TILE + tx;
        int b_row = ph * TILE + ty;
        // 边界外补 0,保证非整除尺寸也正确。
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// CPU 参考实现,用于正确性对拍(规模大时较慢,够用即可)。
static void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float tol = 1e-2f) {
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol * (1.0f + std::fabs(b[i])))
            return false;
    return true;
}

// 计时一个 kernel,返回最优毫秒数。
template <typename Launch>
static float time_kernel(Launch launch, int runs = 20) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    launch();  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(s));
        launch();
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return best;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int M = N, K = N;
    LOG_INFO("GEMM M=N=K=%d", N);

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
    for (auto& x : hA) x = (float)(rand() % 7 - 3) / 3.0f;
    for (auto& x : hB) x = (float)(rand() % 7 - 3) / 3.0f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    double gflop = 2.0 * M * N * K / 1e9;

    // 只在中小规模做 CPU 对拍,避免大矩阵 CPU 太慢。
    bool do_check = (N <= 1024);
    if (do_check) gemm_cpu(hA, hB, hRef, M, N, K);

    float t_naive = time_kernel(
        [&] { gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok_naive = !do_check || allclose(hC, hRef);
    LOG_INFO("naive : %.2f ms, %.1f GFLOP/s  (%s)", t_naive,
             gflop / (t_naive / 1e3), ok_naive ? "correct" : "WRONG");

    float t_tiled = time_kernel(
        [&] { gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok_tiled = !do_check || allclose(hC, hRef);
    LOG_INFO("tiled : %.2f ms, %.1f GFLOP/s (%s)", t_tiled,
             gflop / (t_tiled / 1e3), ok_tiled ? "correct" : "WRONG");

    LOG_INFO("speedup tiled/naive = %.1fx", t_naive / t_tiled);

    // TODO 1: 把 TILE 改成 16 / 64,重新编译对比,验证 AI≈0.25*TILE 的趋势。
    //         (注意 64x64=4096 线程超过单 block 1024 上限,需配合 Lesson 3 的寄存器分块。)
    // TODO 2: 用非整除尺寸验证边界,例如 ./gemm_tiled 1000,确认仍然 correct。

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
