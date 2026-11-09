#include "tensor.cuh"
#include "conv.cuh"

#include <iostream>
#include <memory>

int main()
{ // 验证前向
    graph::tensorPtr x = std::make_shared<graph::TensorImpl>(
        base::s_vec{2, 2, 8, 8}, base::ScalarType::Float,
        base::Device::CUDA, true);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (*x)[i][j % 2][j + i + 1][j + i + 2].set(j * 0.1f);
        }
    }
    graph::tensorPtr kernel = std::make_shared<graph::TensorImpl>(
        base::s_vec{2, 2, 3, 3}, base::ScalarType::Float,
        base::Device::CUDA, true);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2 * 3 * 3; j++)
        {
            (*kernel)[i][j / 9][(j % 9) / 3][j % 3].set(1.0f);
        }
    }
    graph::tensorPtr b = std::make_shared<graph::TensorImpl>(
        base::s_vec{2}, base::ScalarType::Float,
        base::Device::CUDA, true);
    b->fill(1.0f);

    kernel->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(kernel);
    x->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(x);
    b->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(b);

    operation::ConvOpr conv;
    auto y = conv({kernel, x, b});
    std::cout << "Conv forward is OK" << std::endl;
    y->print();
    std::cout << std::endl;

    // 验证后向
    graph::Engine engine;
    engine.backward(y);
    std::cout << "backward fine." << std::endl
              << " kernel.grad_ : " << std::endl;
    kernel->autograd().grad_->print();
    std::cout << "x.grad_" << std::endl;
    x->autograd().grad_->print();
    std::cout << "b.grad_" << std::endl;
    b->autograd().grad_->print();
    std::cout << std::endl;

    return 0;
}