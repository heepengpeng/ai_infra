// M1 Lesson 8:矩阵转置三版对比 —— naive / tiled(无padding) / tiled+padding。
// 编译运行: nvcc -O3 transpose.cu -o transpose && ./transpose
// 进阶 profile:
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./transpose

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",              \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define TILE 32

// ---- naive:写端非合并 ----
__global__ void transpose_naive(const float* A, float* B, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
        B[col * height + row] = A[row * width + col];   // 写 B 跨行 → 不合并
}

// ---- tiled 无 padding:全局合并改善,但 shared 按列读有 bank conflict ----
__global__ void transpose_tiled(const float* A, float* B, int width, int height) {
    __shared__ float tile[TILE][TILE];                  // 无 padding → 冲突
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];  // 合并读
    __syncthreads();

    int xt = blockIdx.y * TILE + threadIdx.x;
    int yt = blockIdx.x * TILE + threadIdx.y;
    if (xt < height && yt < width)
        B[yt * height + xt] = tile[threadIdx.x][threadIdx.y];  // 合并写
}

// ---- tiled + padding:消除 bank conflict ----
__global__ void transpose_padded(const float* A, float* B, int width, int height) {
    // TODO(1): 把 shared 声明改成 [TILE][TILE + 1] 以消除 bank conflict
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];
    __syncthreads();

    int xt = blockIdx.y * TILE + threadIdx.x;
    int yt = blockIdx.x * TILE + threadIdx.y;
    if (xt < height && yt < width)
        B[yt * height + xt] = tile[threadIdx.x][threadIdx.y];
}

typedef void (*kernel_t)(const float*, float*, int, int);

static float bench(kernel_t k, const float* d_A, float* d_B,
                   int width, int height) {
    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
    for (int i = 0; i < 5; ++i) k<<<grid, block>>>(d_A, d_B, width, height); // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; float ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    k<<<grid, block>>>(d_A, d_B, width, height);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

static int verify(const float* A, const float* B, int width, int height) {
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c)
            if (fabsf(B[c * height + r] - A[r * width + c]) > 1e-4f) return 1;
    return 0;
}

int main() {
    const int width = 4096, height = 4096;
    const int n = width * height;
    const size_t bytes = n * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h_A[i] = (float)i;

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    struct { const char* name; kernel_t k; } cfg[] = {
        {"naive          ", transpose_naive},
        {"tiled (no pad) ", transpose_tiled},
        {"tiled + padding", transpose_padded},
    };

    double gb = 2.0 * bytes / 1e9;   // 读 + 写
    for (auto& c : cfg) {
        float ms = bench(c.k, d_A, d_B, width, height);
        CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
        int bad = verify(h_A, h_B, width, height);
        printf("%s : %7.4f ms | %6.1f GB/s | %s\n",
               c.name, ms, gb / (ms / 1000.0), bad ? "WRONG" : "OK");
    }

    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}
