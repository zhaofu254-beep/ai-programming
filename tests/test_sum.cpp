#include "tensor.cuh"
#include "sum.cuh"

#include <iostream>
#include <memory>

int main()
{
    std::cout << "验证构造、切片、索引、赋值、print可行" << std::endl;
    // 验证构造可行
    graph::TensorImpl x({2, 4, 3, 3}, base::TypeMeta(base::ScalarType::Float), base::Device::CPU);

    // 验证切片，按索引赋值可行
    x[0][1][2][1].set(3.14f);          // 写元素
    float value = x[0][1][2][1].get(); // 索引数 == 维数 → 元素

    graph::TensorImpl slice = x[0].slice();                               // 索引数 < 维数 → 切片（共享 storage，requires_grad=false）
    std::cout << "slice[1][2][1]: " << slice[1][2][1].get() << std::endl; // 在切片上继续索引
    std::cout << "x[0][1][2][1]: " << value << std::endl;
    std::cout << "x: " << std::endl;
    x.print();
    std::cout << std::endl;

    std::cout << "验证CPU上的sum操作" << std::endl;
    // 验证cpu的sum前向可行
    graph::tensorPtr t = std::make_shared<graph::TensorImpl>(
        base::s_vec{2, 4, 3, 3}, base::TypeMeta(base::ScalarType::Float),
        base::Device::CPU, true);

    (*t)[0][1][2][1].set(3.14f);
    (*t)[0][1][2][2].set(3.14f);
    (*t)[0][1][1][0].set(3.14f);
    (*t)[0][1][1][1].set(3.14f);

    // 叶子张量需在算子之前设置 accumulator，反向时梯度才能累积到 t.grad_
    t->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(t);

    operation::SumOpr sumoprCPU(1);
    auto y = sumoprCPU({t});
    std::cout << "sum(t,dim=1): " << std::endl;
    y->print();
    std::cout << "y.grad_fn "
              << ((y->autograd().grad_fn_ != nullptr) ? "exist" : "not exist") << std::endl;

    // 验证cpu的sum反向可行
    graph::Engine engine1;
    auto z = sumoprCPU({y});

    engine1.backward(z);
    std::cout << "t.grad_: "
              << ((t->autograd().grad_ != nullptr) ? "exist" : "not exist") << std::endl;
    std::cout << "t.grad_: " << std::endl;
    t->autograd().grad_->print();

    //=====================================================================================
    std::cout << std::endl;
    std::cout << "验证GPU上的项目" << std::endl;
    // 验证gpu的sum（需要分small，mid，last三种情况验证）前向可行
    graph::tensorPtr s = std::make_shared<graph::TensorImpl>(
        base::s_vec{4, 4, 256, 256, 4}, base::TypeMeta(base::ScalarType::Float),
        base::Device::CUDA, true);

    (*s)[0][0][1][2][1].set(3.14f);
    (*s)[1][0][1][2][2].set(3.14f);
    (*s)[0][0][1][1][0].set(3.14f);
    (*s)[1][0][1][1][1].set(3.14f);

    // 叶子张量需在算子之前设置 accumulator，反向时梯度才能累积到 t.grad_
    s->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(s);

    operation::SumOpr sumopr1(0);
    auto s1 = sumopr1({s}); // small,s1.shape = [ 4, 256, 256, 4 ]
    std::cout << "small dim reduce fine, s1.grad_ "
              << ((s1->autograd().grad_fn_ != nullptr) ? "exist" : "not exist")
              << std::endl;

    operation::SumOpr sumopr2(2);
    auto s2 = sumopr2({s1}); // short tail,s2.shape = [ 4, 256, 4 ]
    std::cout << "short tail reduce fine, s2.grad_ "
              << ((s2->autograd().grad_fn_ != nullptr) ? "exist" : "not exist")
              << std::endl;

    operation::SumOpr sumopr3(2);
    auto s3 = sumopr3({s2}); // last dim,s3.shape = [ 4, 256 ]
    std::cout << "last dim reduce fine, s3.grad_ "
              << ((s3->autograd().grad_fn_ != nullptr) ? "exist" : "not exist")
              << std::endl;

    operation::SumOpr sumopr4(1);
    auto s4 = sumopr4({s3}); // long tail,s4.shape = [ 4 ]
    std::cout << "long tail reduce fine, s4.grad_ "
              << ((s4->autograd().grad_fn_ != nullptr) ? "exist" : "not exist")
              << std::endl;

    // 验证cpu的sum反向可行
    graph::Engine engine2;

    engine2.backward(s4);
    std::cout << "backward fine, s.grad_ : "
              << ((s->autograd().grad_ != nullptr) ? "exist" : "not exist") << std::endl;

    return 0;
}