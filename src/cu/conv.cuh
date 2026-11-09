#pragma once

#include "tensor.cuh"

namespace operation
{
    class ConvOpr;
    class ConvNode : public graph::Node
    {
    private:
        graph::tensorPtr kernel, X;
        std::vector<graph::tensorPtr> XcolSave_;
        bool bias_ = true;
        ConvNode(graph::tensorPtr k, graph::tensorPtr x, graph::tensorPtr b,
                 std::vector<graph::tensorPtr> XcolSave, bool bias = true)
            : Node((bias) ? 3 : 2, 1), kernel(k), X(x), bias_(bias), XcolSave_(XcolSave)
        {
            // LinearNode的0, 1, 2 edge分别对应W, X, b；叶子用 accumulator_ 回退
            std::shared_ptr<graph::Node> k_fn = k->autograd().grad_fn_;
            if (!k_fn)
                k_fn = k->autograd().accumulator_;
            std::shared_ptr<graph::Node> x_fn = x->autograd().grad_fn_;
            if (!x_fn)
                x_fn = x->autograd().accumulator_;
            next_edges_[0] = {k_fn, 0};
            next_edges_[1] = {x_fn, 0};
            if (bias)
            {
                std::shared_ptr<graph::Node> b_fn = b->autograd().grad_fn_;
                if (!b_fn)
                    b_fn = b->autograd().accumulator_;
                next_edges_[2] = {b_fn, 0};
            }
        }
        friend ConvOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class ConvOpr : public Opr
    {
    private:
        bool bias_ = true;
        std::shared_ptr<ConvNode> makeNode(
            graph::tensorPtr kernel, graph::tensorPtr X, graph::tensorPtr b,
            std::vector<graph::tensorPtr> XcolSave, bool bias = true)
        {
            return std::shared_ptr<ConvNode>(new ConvNode(kernel, X, b, XcolSave, bias));
        }

    public:
        ConvOpr(bool bias = true) : bias_(bias) {}
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
    };
}