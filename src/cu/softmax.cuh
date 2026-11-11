#pragma once

#include "tensor.cuh"

namespace operation
{
    namespace detail
    {
        graph::tensorPtr Exp(graph::tensorPtr input);
        graph::tensorPtr softmax(graph::tensorPtr input);
    }

    class SoftmaxOpr;
    class SoftmaxNode : public graph::Node
    {
    private:
        graph::tensorPtr forwardResult_;
        SoftmaxNode(graph::tensorPtr input, graph::tensorPtr forwardResult)
            : Node(1, 1), forwardResult_(forwardResult)
        {
            std::shared_ptr<graph::Node> input_fn = input->autograd().grad_fn_;
            if (!input_fn)
                input_fn = input->autograd().accumulator_;
            next_edges_[0] = {input_fn, 0};
        }
        friend SoftmaxOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class SoftmaxOpr : public Opr
    {
    private:
        std::shared_ptr<SoftmaxNode> makeNode(graph::tensorPtr input, graph::tensorPtr forwardResult)
        {
            return std::shared_ptr<SoftmaxNode>(new SoftmaxNode(input, forwardResult));
        }

    public:
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
    };
}