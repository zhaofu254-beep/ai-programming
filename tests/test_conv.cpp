#include "tensor.cuh"
#include "conv.cuh"

#include <iostream>
#include <memory>

int main()
{
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

    operation::ConvOpr conv;
    auto y = conv({kernel, x, b});
    std::cout << "Conv forward is OK" << std::endl;
    y->print();
    return 0;
}