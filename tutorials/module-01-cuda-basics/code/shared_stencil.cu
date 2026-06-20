// M1 Lesson 6 实验 A:用共享内存做 1D 三点滑动平均,演示"搬入-同步-计算-写回"。
// 试验:把 __syncthreads() 注释掉再跑,观察结果偶发出错(数据竞争)。
// 编译运行: nvcc shared_stencil.cu -o shared_stencil && ./shared_stencil

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

#define BLOCK 256

__global__ void smooth3(const float* in, float* out, int n) {
    __shared__ float s[BLOCK + 2];                 // +2 给左右 halo 留位
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    if (gid < n) s[t + 1] = in[gid];               // 主体搬入(让出下标 0 给左 halo)
    if (t == 0)               s[0]         = (gid > 0)     ? in[gid - 1] : in[0];
    if (t == blockDim.x - 1)  s[BLOCK + 1] = (gid + 1 < n) ? in[gid + 1] : in[n - 1];

    __syncthreads();                               // 确保白板(含 halo)写完再读

    if (gid < n) out[gid] = (s[t] + s[t + 1] + s[t + 2]) / 3.0f;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 100);

    // CPU 参考结果
    for (int i = 0; i < n; ++i) {
        float l = h_in[i > 0 ? i - 1 : 0];
        float r = h_in[i + 1 < n ? i + 1 : n - 1];
        h_ref[i] = (l + h_in[i] + r) / 3.0f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (n + BLOCK - 1) / BLOCK;

    cudaEvent_t start, stop; float ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 5; ++i) smooth3<<<grid, BLOCK>>>(d_in, d_out, n);   // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    smooth3<<<grid, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; ++i) if (fabsf(h_out[i] - h_ref[i]) > 1e-3f) ++errors;
    printf("smooth3: %.4f ms, errors=%d -> %s\n", ms, errors,
           errors == 0 ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
