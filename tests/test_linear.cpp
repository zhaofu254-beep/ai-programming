#include "tensor.cuh"
#include "linear.cuh"

#include <iostream>
#include <memory>

int main()
{
    // 验证正向可行
    int64_t batch_size = 2, feature_in = 2;
    graph::tensorPtr x = std::make_shared<graph::TensorImpl>(
        base::s_vec{batch_size, feature_in}, base::ScalarType::Float, base::Device::CUDA,
        false); // 输入feature一般不计算grad

    (*x)[0][0].set(1);
    (*x)[0][1].set(2);
    (*x)[1][0].set(2);
    (*x)[1][1].set(-1);

    int64_t feature_hidden = 3;
    graph::tensorPtr W1 = std::make_shared<graph::TensorImpl>(
        base::s_vec{feature_hidden, feature_in}, base::ScalarType::Float, base::Device::CUDA, true);
    (*W1)[0][0].set(-1);
    (*W1)[0][1].set(1);
    (*W1)[1][0].set(2);
    (*W1)[1][1].set(1);
    (*W1)[2][0].set(0);
    (*W1)[2][1].set(1);
    graph::tensorPtr b = std::make_shared<graph::TensorImpl>(
        base::s_vec{feature_hidden}, base::ScalarType::Float, base::Device::CUDA, true);
    (*b)[0].set(1);
    (*b)[1].set(2);
    (*b)[2].set(-1);

    W1->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(W1);
    b->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(b);

    auto linear1 = operation::LinearOpr();
    auto y = linear1({x, W1, b});
    std::cout << "y:" << std::endl;
    y->print();

    // 验证反向可行
    int64_t feature_out = 4;

    graph::tensorPtr W2 = std::make_shared<graph::TensorImpl>(
        base::s_vec{feature_out, feature_hidden}, base::ScalarType::Float, base::Device::CUDA, true);
    (*W2)[0][0].set(0);
    (*W2)[0][1].set(1);
    (*W2)[0][2].set(-1);
    (*W2)[1][0].set(1);
    (*W2)[1][1].set(2);
    (*W2)[1][2].set(0);
    (*W2)[2][0].set(-1);
    (*W2)[2][1].set(1);
    (*W2)[2][2].set(1);
    (*W2)[3][0].set(1);
    (*W2)[3][1].set(0);
    (*W2)[3][2].set(1);

    W2->autograd().accumulator_ = std::make_shared<graph::AccumulateGrad>(W2);

    auto linear2 = operation::LinearOpr(false);
    auto z = linear2({y, W2});
    std::cout << "z:" << std::endl;
    z->print();

    graph::Engine engine;
    engine.backward(z);
    std::cout << "backward fine." << std::endl
              << " W1.grad_ : " << std::endl;
    W1->autograd().grad_->print();
    std::cout << "b.grad_" << std::endl;
    b->autograd().grad_->print();
    std::cout << "W2.grad_" << std::endl;
    W2->autograd().grad_->print();
    std::cout << std::endl;

    return 0;
}