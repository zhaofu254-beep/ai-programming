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
        const base::s_vec inputStride_, inputShape_, outputStride_, outputShape_;
        const int rdim_;

        SumNode(graph::tensorPtr input, graph::tensorPtr output, int rdim)
            : Node(1, 1), rdim_(rdim), inputShape_(input->shape()),
              inputStride_(input->stride()), outputShape_(output->shape()),
              outputStride_(output->stride())
        {
            next_edges_[0] = {input->autograd().grad_fn_, 0};
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