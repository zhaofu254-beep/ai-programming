#include "tensor.cuh"
#include "sum.cuh"

#include <iostream>
#include <vector>
#include <memory>

int main()
{
    // 验证构造可行
    graph::TensorImpl t(base::s_vec{2, 4, 3, 3},
                        base::TypeMeta(base::ScalarType::Float), base::Device::CPU);
    t.fill(0);

    graph::TensorImpl x({2, 4, 3, 3}, base::TypeMeta(base::ScalarType::Float), base::Device::CUDA); // 或 CPU

    float v = x[0][1][2].get(); // 索引数 == 维数 → 元素
    x[0][1][2].set(3.14f);      // 写元素

    graph::TensorImpl s = x[0].slice(); // 索引数 < 维数 → 切片（共享 storage，requires_grad=false）
    s[1][2].get();                      // 在切片上继续索引

    x.print(); // 打印，CPU/CUDA 均可用，非连续视图（permute 后）也正确
    // printf("验证赋值与读取在CPU上可行\n");
    // t[0][1][1][2] = 3.14f;
    // printf("t[0][1][1][2] == %f\n", static_cast<float>(t[0][1][1][2]));
    // t[0][1][2][1] = -3.14f;
    // printf("t[0][1][2][1] == %f\n", static_cast<float>(t[0][1][2][1]));

    // 验证填充可行
    graph::TensorImpl g(
        {2, 4, 3, 3}, base::TypeMeta(base::ScalarType::Float), base::Device::CUDA);
    g.fill(1.0f);

    // 验证移动设备可行
    std::shared_ptr<graph::TensorImpl> t_gpu =
        std::make_shared<graph::TensorImpl>(t.to(base::Device::CUDA));

    return 0;
}