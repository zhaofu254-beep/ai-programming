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
        const base::s_vec inputShape_;
        const int rdim_;

        SumNode(graph::tensorPtr input, graph::tensorPtr output, int rdim)
            : Node(1, 1), rdim_(rdim), inputShape_(input->shape())
        {
            // 叶子张量的 grad_fn_ 为 null，此时边应指向其 accumulator_（AccumulateGrad）
            std::shared_ptr<graph::Node> input_grad_fn = input->autograd().grad_fn_;
            if (!input_grad_fn)
                input_grad_fn = input->autograd().accumulator_;
            next_edges_[0] = {input_grad_fn, 0};
        }
        friend SumOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class SumOpr
    {
    private:
        std::shared_ptr<SumNode> makeNode(
            graph::tensorPtr input, graph::tensorPtr output, int rdim)
        {
            return std::shared_ptr<SumNode>(new SumNode(input, output, rdim));
        }

    public:
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(
            std::vector<graph::tensorPtr> inputs, int rdim = -1);
    };

}