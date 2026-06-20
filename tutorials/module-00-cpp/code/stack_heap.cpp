// M0 Lesson 1:栈内存 vs 堆内存,以及手动内存管理。
// 编译运行: g++ -std=c++17 -Wall stack_heap.cpp -o stack_heap && ./stack_heap
//
// 进阶(很重要):用 AddressSanitizer 抓内存错误
//   g++ -std=c++17 -fsanitize=address -g stack_heap.cpp -o stack_heap && ./stack_heap

#include <cstdio>
#include <cstdlib>   // malloc/free
#include <new>       // new/delete 在 <new>,但通常直接可用

int main() {
    // ① 栈上分配:出了作用域自动回收,你不用管
    int stack_arr[4] = {1, 2, 3, 4};
    printf("stack_arr[0] = %d (地址在栈上: %p)\n", stack_arr[0], (void*)stack_arr);

    // ② 堆上分配(C 风格 malloc/free)—— CUDA 的 cudaMalloc 就是这个味道
    int* heap_c = (int*)malloc(4 * sizeof(int));
    for (int i = 0; i < 4; ++i) heap_c[i] = i * 10;
    printf("heap_c[2] = %d (地址在堆上: %p)\n", heap_c[2], (void*)heap_c);
    free(heap_c);            // 必须手动还,否则内存泄漏

    // ③ 堆上分配(C++ 风格 new/delete)
    int* heap_cpp = new int[4];
    for (int i = 0; i < 4; ++i) heap_cpp[i] = i * 100;
    printf("heap_cpp[3] = %d\n", heap_cpp[3]);
    delete[] heap_cpp;       // new[] 必须配 delete[],不能配 delete

    return 0;
}
