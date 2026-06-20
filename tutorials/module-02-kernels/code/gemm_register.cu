// gemm_register.cu
// 三版 GEMM 性能对比:naive -> shared tiling -> 寄存器分块(2D block-tiling)。
// 可选用 -DUSE_CUBLAS 链接 cuBLAS 作为标杆。
// 要求 M=N=K 且为 64 的倍数(为聚焦优化思想,省去边界处理)。
// 编译: nvcc -O3 gemm_register.cu -o gemm_register
//       nvcc -O3 -DUSE_CUBLAS gemm_register.cu -o gemm_register -lcublas
// 运行: ./gemm_register 2048
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

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

__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int ph = 0; ph < K / TILE; ++ph) {
        As[ty][tx] = A[row * K + ph * TILE + tx];
        Bs[ty][tx] = B[(ph * TILE + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// 经典 2D block-tiling:一个 block 算 BM×BN 的输出,一个线程算 TM×TN 个输出。
// 复用层级:global -> shared(As/Bs)-> 寄存器(regM/regN/threadResults)。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 把指针移到本 block 负责的子块起点。
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int numThreads = (BM * BN) / (TM * TN);  // = blockDim.x

    // 搬运用的展开索引(把 numThreads 个线程铺到 BMxBK / BKxBN 的搬运任务上)。
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA = numThreads / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB = numThreads / BN;

    float threadResults[TM * TN] = {0.0f};  // 累加器,在寄存器
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // global -> shared(此版用标量搬运,向量化留作 TODO)
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] =
                A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] =
                B[(innerRowB + off) * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        // 沿 BK 逐步:读一列 regM、一行 regN 到寄存器,做 TM*TN 次外积累加。
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] =
                threadResults[i * TN + j];
}

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
        if (std::fabs(a[i] - b[i]) > tol * (1.0f + std::fabs(b[i]))) return false;
    return true;
}

template <typename Launch>
static float time_kernel(Launch launch, int runs = 20) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    launch();
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
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
    if (N % 64 != 0) {
        LOG_INFO("N 必须是 64 的倍数(本示例省略了边界处理),收到 N=%d", N);
        return 1;
    }
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

    double gflop = 2.0 * M * N * K / 1e9;
    bool do_check = (N <= 1024);
    if (do_check) gemm_cpu(hA, hB, hRef, M, N, K);

    // naive
    dim3 b1(TILE, TILE), g1(N / TILE, M / TILE);
    float t_naive = time_kernel([&] { gemm_naive<<<g1, b1>>>(dA, dB, dC, M, N, K); });
    LOG_INFO("naive    : %6.2f ms, %5.0f GFLOP/s", t_naive, gflop / (t_naive / 1e3));

    // tiled
    float t_tiled = time_kernel([&] { gemm_tiled<<<g1, b1>>>(dA, dB, dC, M, N, K); });
    LOG_INFO("tiled    : %6.2f ms, %5.0f GFLOP/s", t_tiled, gflop / (t_tiled / 1e3));

    // register-blocked: BM=BN=64, BK=8, TM=TN=4 -> 256 线程/block
    const int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    dim3 b2((BM * BN) / (TM * TN));
    dim3 g2(N / BN, M / BM);
    float t_reg = time_kernel(
        [&] { gemm_register<BM, BN, BK, TM, TN><<<g2, b2>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok = !do_check || allclose(hC, hRef);
    LOG_INFO("register : %6.2f ms, %5.0f GFLOP/s   (%s)", t_reg,
             gflop / (t_reg / 1e3), ok ? "correct" : "WRONG");

#ifdef USE_CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS 是列主序;用 C^T = B^T * A^T 的技巧直接对行主序数据计算。
    float t_cublas = time_kernel([&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N,
                    dA, K, &beta, dC, N);
    });
    LOG_INFO("cublas   : %6.2f ms, %5.0f GFLOP/s", t_cublas,
             gflop / (t_cublas / 1e3));
    LOG_INFO("register reaches %.1f%% of cuBLAS", 100.0 * t_cublas / t_reg);
    cublasDestroy(handle);
#endif

    // TODO: 把 TM,TN 改成 8x8 重测;再试 16x16 观察寄存器溢出(用 nvcc --ptxas-options=-v
    //       查看 registers/spill stores),理解复用率与寄存器压力的权衡。

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
