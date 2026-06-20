// M0 Lesson 3:move 语义 —— 转移所有权而非深拷贝大块数据。
// 编译运行: g++ -std=c++17 -Wall move_demo.cpp -o move_demo && ./move_demo

#include <cstdio>
#include <utility>     // std::move
#include <vector>

int main() {
    printf("== vector 的拷贝 vs move ==\n");

    std::vector<int> big(5, 7);              // 5 个 7
    printf("big.size() = %zu\n", big.size());

    // ① 拷贝构造:深拷贝,big 仍然有效(两块独立内存)
    std::vector<int> copied = big;
    printf("拷贝后: big.size()=%zu, copied.size()=%zu\n",
           big.size(), copied.size());

    // ② move 构造:把 big 的内部缓冲区"偷"给 moved,big 被掏空(置为空壳)
    //    没有逐元素拷贝,只是把内部指针转移过去 —— O(1),极快
    std::vector<int> moved = std::move(big);
    printf("move 后: big.size()=%zu (被掏空), moved.size()=%zu\n",
           big.size(), moved.size());

    // TODO(练习): 用一个含 new[] 的自定义类 MyArray,分别实现
    //   - 拷贝构造(深拷贝)
    //   - 移动构造 MyArray(MyArray&& other) noexcept
    //   打印两者行为差异,体会 move 为什么对大 buffer/显存句柄至关重要。

    return 0;
}
