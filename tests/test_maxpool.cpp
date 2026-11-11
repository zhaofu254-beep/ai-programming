#include "tensor.cuh"
#include "maxpool.cuh"

#include <iostream>
#include <memory>

int main()
{
    graph::tensorPtr x = std::make_shared<graph::TensorImpl>(
        base::s_vec{1, 1, 9, 9}, base::ScalarType::Float,
        base::Device::CUDA, true);
    for (int j = 0; j < 3; j++)
    {
        (*x)[0][0][j + 1][1u << j].set(j * 0.1f);
    }
    x->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(x);
    x->print();

    operation::MaxPoolOpr pool;
    auto y = pool({x});
    std::cout << "MaxPool forward fine, y.shape:" << std::endl;
    for (auto s : y->shape())
        std::cout << s << ' ';
    std::cout << std::endl;

    std::cout << "y:" << std::endl;
    y->print();
    std::cout << std::endl;

    graph::Engine engine;
    engine.backward(y);
    std::cout << "backward fine." << std::endl
              << " x.grad_ : " << std::endl;
    x->autograd().grad_->print();
    std::cout << std::endl;

    return 0;
}
