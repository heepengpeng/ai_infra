// M1 Lesson 5 实验 A:让 CUDA 错误现形。
// 这里故意制造错误,演示 CUDA_CHECK / cudaGetLastError 如何精确报错。
// 编译运行: nvcc error_check.cu -o error_check && ./error_check

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

__global__ void noop_kernel() {}

int main() {
    // 演示一:用非法 block 配置启动 kernel(每 block > 1024 线程)。
    // kernel 启动不返回错误码,要靠 cudaGetLastError 抓这种"启动配置错误"。
    noop_kernel<<<1, 2048>>>();                 // 2048 > 1024,非法!
    cudaError_t e = cudaGetLastError();
    printf("启动非法 block 后 cudaGetLastError = %s\n", cudaGetErrorString(e));

    // 演示二:申请远超显存容量的内存,CUDA_CHECK 会精确报出文件/行号/原因。
    // 注意:下面这行预期会触发 CUDA_CHECK 退出。想继续看后面,把它注释掉。
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (size_t)1024 * 1024 * 1024 * 1024));  // 1 TB

    printf("不会执行到这里\n");
    return 0;
}
