// M0 Lesson 3:RAII —— 用对象的生命周期自动管理资源。
// 编译运行: g++ -std=c++17 -Wall raii_demo.cpp -o raii_demo && ./raii_demo

#include <cstdio>

// 一个最小 RAII 类:构造时申请资源,析构时自动释放。
// 把它想成 cudaMalloc/cudaFree 的封装雏形。
class Buffer {
public:
    explicit Buffer(int n) : size_(n) {
        data_ = new float[n];                 // 构造 = 申请资源
        printf("  [Buffer] 申请 %d 个 float\n", n);
    }
    ~Buffer() {
        delete[] data_;                        // 析构 = 释放资源(自动调用!)
        printf("  [Buffer] 释放 %d 个 float\n", size_);
    }
    float* data() { return data_; }
private:
    float* data_;
    int    size_;
};

void use_buffer() {
    printf("进入 use_buffer\n");
    Buffer buf(8);                 // 栈上对象,出作用域自动析构
    buf.data()[0] = 3.14f;
    printf("buf[0] = %.2f\n", buf.data()[0]);
    printf("离开 use_buffer(此刻 buf 自动析构,无需手写 delete)\n");
}

int main() {
    use_buffer();
    printf("回到 main,Buffer 早已被自动清理\n");
    return 0;
}
