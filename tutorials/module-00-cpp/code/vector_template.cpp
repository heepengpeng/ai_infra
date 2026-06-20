// M0 Lesson 2 练习:实现一个极简的类模板 Vec<T>(定长向量,逐元素加法)。
// 这是"类模板"的最小练手,模拟算子库里 Tensor<T> / Array<T> 的味道。
// 编译运行: g++ -std=c++17 -Wall vector_template.cpp -o vector_template && ./vector_template

#include <cstdio>

template <typename T, int N>      // 注意:模板参数不仅能是类型,还能是编译期常量 N
struct Vec {
    T data[N];

    // 逐元素相加,返回新 Vec。const& 传参避免拷贝(回顾 Lesson 1 引用)
    Vec<T, N> operator+(const Vec<T, N>& other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) {
            // TODO: 写出逐元素相加  result.data[i] = ...
            result.data[i] = data[i] + other.data[i];   // 占位,先填 0,练习时改成正确实现
        }
        return result;
    }

    void print() const {
        for (int i = 0; i < N; ++i) printf("%g ", (double)data[i]);
        printf("\n");
    }
};

int main() {
    Vec<float, 3> a{1.0f, 2.0f, 3.0f};
    Vec<float, 3> b{10.0f, 20.0f, 30.0f};
    Vec<float, 3> c = a + b;
    c.print();   // 正确实现后应输出: 11 22 33

    // 同一份模板,换成 int、换成长度 4,照样工作
    Vec<int, 4> x{1, 2, 3, 4};
    Vec<int, 4> y{1, 1, 1, 1};
    (x + y).print();   // 正确实现后应输出: 2 3 4 5
    return 0;
}
