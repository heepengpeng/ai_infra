// M1 Lesson 6 实验 B:制造再消除 bank conflict。
// 同一个"按列写、按行读"的小核,对比共享数组 [32][32](冲突) 与 [32][33](padding 消除)。
// 编译运行: nvcc bank_conflict.cu -o bank_conflict && ./bank_conflict
// 进阶: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bank_conflict

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",              \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// 有冲突版本:行宽 32,按列访问时 32 个线程全落同一 bank
__global__ void conflict_kernel(const float* in, float* out, int rounds) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;
    float acc = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        tile[ty][tx] = in[ty * 32 + tx] + r;
        __syncthreads();
        acc += tile[tx][ty];        // 按列读:32 路 bank conflict
        __syncthreads();
    }
    out[ty * 32 + tx] = acc;
}

// 无冲突版本:行宽 33(padding),按列读时落到 32 个不同 bank
__global__ void padded_kernel(const float* in, float* out, int rounds) {
    __shared__ float tile[32][33];   // TODO 思考:为什么 +1 就能消除冲突?
    int tx = threadIdx.x, ty = threadIdx.y;
    float acc = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        tile[ty][tx] = in[ty * 32 + tx] + r;
        __syncthreads();
        acc += tile[tx][ty];
        __syncthreads();
    }
    out[ty * 32 + tx] = acc;
}

static float time_kernel(void (*k)(const float*, float*, int),
                         const float* d_in, float* d_out, int rounds) {
    dim3 block(32, 32);
    for (int i = 0; i < 5; ++i) k<<<1, block>>>(d_in, d_out, rounds);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; float ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    k<<<1, block>>>(d_in, d_out, rounds);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    const int n = 32 * 32;
    const size_t bytes = n * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    const int rounds = 20000;   // 放大循环让差异可测
    float t_conf = time_kernel(conflict_kernel, d_in, d_out, rounds);
    float t_pad  = time_kernel(padded_kernel,  d_in, d_out, rounds);

    printf("conflict [32][32] : %.4f ms\n", t_conf);
    printf("padded   [32][33] : %.4f ms\n", t_pad);
    printf("speedup           : %.2fx\n", t_conf / t_pad);

    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
