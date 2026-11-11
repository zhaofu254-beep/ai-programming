#pragma once

#include "tensor.cuh"

namespace operation
{
    class CrossEntropyOpr;
    class CrossEntropyNode : public graph::Node
    {
    private:
        graph::tensorPtr prob_, label_;
        CrossEntropyNode(graph::tensorPtr input, graph::tensorPtr prob, graph::tensorPtr label)
            : Node(1, 1), prob_(prob), label_(label)
        {
            std::shared_ptr<graph::Node> input_fn = input->autograd().grad_fn_;
            if (!input_fn)
                input_fn = input->autograd().accumulator_;
            next_edges_[0] = {input_fn, 0};
        }
        friend CrossEntropyOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class CrossEntropyOpr : public Opr
    {
    private:
        std::shared_ptr<CrossEntropyNode> makeNode(
            graph::tensorPtr input, graph::tensorPtr prob, graph::tensorPtr label)
        {
            return std::shared_ptr<CrossEntropyNode>(new CrossEntropyNode(input, prob, label));
        }

    public:
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
    };
}