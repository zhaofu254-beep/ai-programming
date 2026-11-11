#include "tensor.cuh"
#include "softmax.cuh"

#include <iostream>
#include <memory>

int main()
{
    graph::tensorPtr x = std::make_shared<graph::TensorImpl>(
        base::s_vec{4, 4}, base::ScalarType::Float,
        base::Device::CUDA, true);

    for (int i = 0; i < 4; i++)
        (*x)[i][i].set(1.0f);

    x->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(x);

    std::cout << "x:" << std::endl;
    x->print();
    std::cout << std::endl;

    operation::SoftmaxOpr softmax;
    auto y = softmax({x});

    std::cout << "Forward fine:" << std::endl
              << "y:" << std::endl;
    y->print();
    std::cout << std::endl;

    graph::Engine engine;

    graph::tensorPtr grad = std::make_shared<graph::TensorImpl>(
        base::s_vec{4, 4}, base::ScalarType::Float,
        base::Device::CUDA, false);
    grad->fill(0);
    for (int i = 0; i < 4; i++)
        (*grad)[i][3 - i].set(1.0f);

    engine.backward(y, grad);

    std::cout << "Backward fine:" << std::endl
              << "x.grad:" << std::endl;
    x->autograd().grad_->print();
    std::cout << std::endl;

    return 0;
}