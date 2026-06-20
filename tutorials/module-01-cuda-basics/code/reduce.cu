// M1 Lesson 7:并行归约求和,四个递进版本 + CPU 参考,统一 benchmark。
// 编译运行: nvcc -O3 reduce.cu -o reduce && ./reduce
// 注:每个 kernel 输出"每个 block 一个部分和",最后部分和拿回 CPU 收尾。

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

// ---- v0:交错寻址,存在 warp divergence ----
__global__ void reduce_v0(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + t;
    s[t] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (t % (2 * stride) == 0) s[t] += s[t + stride];   // 隔位活跃 → 发散
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

// ---- v1:连续寻址,消除 divergence ----
__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + t;
    s[t] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];              // 活跃线程连续成段
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

// ---- v2:加载时先加一次,免费做掉第一层 + 减半 grid ----
__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + t;
    float v = (gid < n) ? in[gid] : 0.0f;
    if (gid + blockDim.x < n) v += in[gid + blockDim.x];     // 搬入时顺手加
    s[t] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

// ---- v3:shared 归约到每 warp 一个值,再用 warp shuffle 收尾 ----
__device__ float warp_reduce(float val) {
    // TODO: 补全 warp 内树形归约,offset = 16,8,4,2,1
    //   for (int offset = 16; offset > 0; offset >>= 1)
    //       val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float warp_sums[BLOCK / 32];
    int t = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + t;
    float v = (gid < n) ? in[gid] : 0.0f;
    if (gid + blockDim.x < n) v += in[gid + blockDim.x];

    v = warp_reduce(v);                       // 每个 warp 内先归约
    int lane = t & 31, wid = t >> 5;
    if (lane == 0) warp_sums[wid] = v;        // 每个 warp 的和写进 shared
    __syncthreads();

    // 用第 0 个 warp 把各 warp 的和再归约一次
    if (wid == 0) {
        v = (t < BLOCK / 32) ? warp_sums[lane] : 0.0f;
        v = warp_reduce(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// ---------- host 端:统一计时 + 校验 ----------
typedef void (*kernel_t)(const float*, float*, int);

static float run(kernel_t k, const float* d_in, float* d_partial,
                 float* h_partial, int n, int grid, double* out_sum) {
    for (int i = 0; i < 5; ++i) k<<<grid, BLOCK>>>(d_in, d_partial, n);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; float ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    k<<<grid, BLOCK>>>(d_in, d_partial, n);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < grid; ++i) sum += h_partial[i];   // CPU 收尾
    *out_sum = sum;
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = n * sizeof(float);
    float* h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;          // 和应为 n
    double ref = (double)n;

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    int grid_full = (n + BLOCK - 1) / BLOCK;          // v0/v1 用
    int grid_half = (n + BLOCK * 2 - 1) / (BLOCK * 2);// v2/v3 用
    CUDA_CHECK(cudaMalloc(&d_partial, grid_full * sizeof(float)));
    float* h_partial = (float*)malloc(grid_full * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    struct { const char* name; kernel_t k; int grid; } cfg[] = {
        {"v0 interleaved", reduce_v0, grid_full},
        {"v1 contiguous ", reduce_v1, grid_full},
        {"v2 load-add    ", reduce_v2, grid_half},
        {"v3 warp shuffle", reduce_v3, grid_half},
    };

    double read_gb = (double)bytes / 1e9;   // 归约主要读一遍输入
    for (auto& c : cfg) {
        double sum = 0.0;
        float ms = run(c.k, d_in, d_partial, h_partial, n, c.grid, &sum);
        printf("%s : %7.4f ms | %6.1f GB/s | sum=%.0f %s\n",
               c.name, ms, read_gb / (ms / 1000.0), sum,
               fabs(sum - ref) < 1.0 ? "OK" : "WRONG(补全 v3 的 TODO?)");
    }

    cudaFree(d_in); cudaFree(d_partial);
    free(h); free(h_partial);
    return 0;
}
