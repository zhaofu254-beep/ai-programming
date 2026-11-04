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
    namespace detail
    {
        std::array<base::s_vec, 3> coalesce_dimensions(
            base::s_vec shape, base::s_vec stride, std::vector<bool> masked_dims);

        graph::tensorPtr contigReduce(graph::tensorPtr t, int64_t rdim = -1);
    }

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

    class SumOpr : public Opr
    {
    private:
        const int rdim_;
        std::shared_ptr<SumNode> makeNode(
            graph::tensorPtr input, graph::tensorPtr output, int rdim)
        {
            return std::shared_ptr<SumNode>(new SumNode(input, output, rdim));
        }

    public:
        SumOpr() = delete;
        SumOpr(int rdim = -1) : Opr(), rdim_(rdim) {};
        bool broadcastingSupport() { return false; }
        int getRdim() const { return rdim_; }
        graph::tensorPtr operator()(
            std::vector<graph::tensorPtr> inputs);
    };

}