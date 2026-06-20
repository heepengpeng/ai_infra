// M1 Lesson 2 实验 B:数组平方。补全 4 个 TODO 后编译运行。
// 编译运行: nvcc square.cu -o square && ./square
// 期望输出: 0^2=0, 1^2=1, 2^2=4, ... 9^2=81

#include <cstdio>

__global__ void square_kernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO(1): 加边界检查并写出平方运算  if (tid < n) { out[tid] = ...; }
}

int main() {
    const int n = 10;
    const size_t bytes = n * sizeof(float);

    float h_in[n], h_out[n];
    for (int i = 0; i < n; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // TODO(2): 把 h_in 拷到 d_in (cudaMemcpyHostToDevice)

    int block = 256;
    int grid = (n + block - 1) / block;
    // TODO(3): 启动 square_kernel<<<grid, block>>>(...)

    // TODO(4): 把 d_out 拷回 h_out (cudaMemcpyDeviceToHost)

    for (int i = 0; i < n; ++i) printf("%.0f^2 = %.0f\n", h_in[i], h_out[i]);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
