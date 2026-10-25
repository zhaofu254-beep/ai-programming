#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include <map>
#include <array>
#include <ranges>
#include <cstdlib>
#include <cuda_runtime.h>

#include "error.cuh"
#include "tensor.cuh"

namespace operation
{
    std::array<base::s_vec, 3> coalesce_dimensions(
        base::s_vec shape, base::s_vec stride, std::vector<bool> masked_dims);

    class SumOpr;
    class SumNode : public graph::Node
    {
    private:
        base::s_vec inputStrides, inputShape;
        const int rdim_;

        SumNode(std::vector<graph::tensorPtr> adds, int rdim)
            : Node(1, 1), rdim_(rdim)
        {
            next_edges_[0] = {adds[0]->autograd().grad_fn_, 0};
        }
        friend SumOpr;

    public:
        std::vector<graph::tensorPtr> backward()
        {
            if (!prev_grads_[0])
                throw std::invalid_argument("No grad to be backward.");
            // TODO: 反向需把输出梯度沿 rdim_ 广播（expand）回输入，尚未实现。
            return {};
        }
    };
    class SumOpr
    {
    private:
        std::shared_ptr<graph::Node> makeNode(
            std::vector<graph::tensorPtr> inputs, int rdim)
        {
            return std::shared_ptr<graph::Node>(new SumNode(inputs, rdim));
        }

    public:
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(
            std::vector<graph::tensorPtr> inputs, int rdim = -1);
    };

}