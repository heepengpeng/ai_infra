// M0 Lesson 3:标准库智能指针 unique_ptr / shared_ptr。
// 编译运行: g++ -std=c++17 -Wall smartptr_demo.cpp -o smartptr_demo && ./smartptr_demo

#include <cstdio>
#include <memory>      // unique_ptr / shared_ptr / make_unique / make_shared

struct Tensor {
    int id;
    Tensor(int i) : id(i) { printf("  Tensor(%d) 构造\n", id); }
    ~Tensor()             { printf("  Tensor(%d) 析构\n", id); }
};

int main() {
    printf("== unique_ptr:独占所有权 ==\n");
    {
        std::unique_ptr<Tensor> a = std::make_unique<Tensor>(1);
        printf("a->id = %d\n", a->id);
        // unique_ptr 不能拷贝(独占),只能 move 转移所有权(下个 demo 详讲)
        // 出作用域时自动 delete,无需手写
    }
    printf("(unique_ptr 作用域已结束,Tensor(1) 自动析构)\n\n");

    printf("== shared_ptr:引用计数,多个指针共享 ==\n");
    {
        std::shared_ptr<Tensor> p = std::make_shared<Tensor>(2);
        printf("use_count = %ld\n", p.use_count());   // 1
        {
            std::shared_ptr<Tensor> q = p;            // 拷贝 = 计数 +1
            printf("use_count = %ld (q 也指向它)\n", p.use_count());  // 2
        }
        printf("use_count = %ld (q 出作用域,计数 -1)\n", p.use_count()); // 1
        // 计数归 0 时才真正 delete
    }
    printf("(shared_ptr 计数归零,Tensor(2) 自动析构)\n");
    return 0;
}
