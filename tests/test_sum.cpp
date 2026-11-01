#include "tensor.cuh"
#include "sum.cuh"

#include <iostream>
#include <vector>
#include <memory>

int main()
{
    // 验证构造可行
    graph::TensorImpl x({2, 4, 3, 3}, base::TypeMeta(base::ScalarType::Float), base::Device::CPU);

    // 验证切片，按索引赋值可行
    float v = x[0][1][2][1].get(); // 索引数 == 维数 → 元素
    x[0][1][2][1].set(3.14f);      // 写元素
    x[0][1][2][2].set(3.14f);
    x[0][1][1][0].set(3.14f);
    x[0][1][1][1].set(3.14f);

    graph::TensorImpl s = x[0].slice();                           // 索引数 < 维数 → 切片（共享 storage，requires_grad=false）
    std::cout << "s[1][2][1]: " << s[1][2][1].get() << std::endl; // 在切片上继续索引

    std::cout << "x: " << std::endl;
    x.print();

    // 验证cpu的sum前向可行
    graph::tensorPtr t = std::make_shared<graph::TensorImpl>(
        base::s_vec{2, 4, 3, 3}, base::TypeMeta(base::ScalarType::Float),
        base::Device::CPU, true);

    (*t)[0][1][2][1].set(3.14f);
    (*t)[0][1][2][2].set(3.14f);
    (*t)[0][1][1][0].set(3.14f);
    (*t)[0][1][1][1].set(3.14f);

    operation::SumOpr sumopr;
    auto y = sumopr({t}, 1);
    std::cout << "sum(t,dim=1): " << std::endl;
    y->print();
    std::cout << "y.grad_fn "
              << ((y->autograd().grad_fn_ != nullptr) ? "exist" : "not exist") << std::endl;

    // 验证cpu的sum反向可行
    graph::Engine engine;
    auto z = sumopr({y}, 1);

    // std::shared_ptr<graph::AccumulateGrad> accum = std::make_shared<graph::AccumulateGrad>(t);
    // t->autograd().accumulator_ = accum;

    engine.backward(z);
    // std::cout << "t.grad_: " << std::endl;
    // t->autograd().grad_->print();

    return 0;
}