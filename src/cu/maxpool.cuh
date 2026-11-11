#pragma once

#include "tensor.cuh"

const int KERNEL_SIZE = 2;

namespace operation
{
    class MaxPoolOpr;
    class MaxPoolNode : public graph::Node
    {
    private:
        graph::tensorPtr mask_;
        MaxPoolNode(graph::tensorPtr input, graph::tensorPtr mask)
            : Node(1, 1), mask_(mask)
        {
            std::shared_ptr<graph::Node> input_fn = input->autograd().grad_fn_;
            if (!input_fn)
                input_fn = input->autograd().accumulator_;
            next_edges_[0] = {input_fn, 0};
        }
        friend MaxPoolOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class MaxPoolOpr : public Opr
    {
    private:
        std::shared_ptr<MaxPoolNode> makeNode(graph::tensorPtr input, graph::tensorPtr mask)
        {
            return std::shared_ptr<MaxPoolNode>(new MaxPoolNode(input, mask));
        }

    public:
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
    };
}