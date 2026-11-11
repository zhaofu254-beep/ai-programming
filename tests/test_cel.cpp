#include "tensor.cuh"
#include "crossentropy.cuh"

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

    graph::tensorPtr label = std::make_shared<graph::TensorImpl>(
        base::s_vec{4}, base::ScalarType::Int,
        base::Device::CUDA, false);

    label->fill(0);

    operation::CrossEntropyOpr loss;
    auto l = loss({x, label});

    std::cout << "Forward fine:" << std::endl
              << "l:" << std::endl;
    l->print();
    std::cout << std::endl;

    graph::Engine engine;

    engine.backward(l);

    std::cout << "Backward fine:" << std::endl
              << "x.grad:" << std::endl;
    x->autograd().grad_->print();
    std::cout << std::endl;

    return 0;
}