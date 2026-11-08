#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include <map>
#include <array>
#include "tensor.cuh"
#include "sum.cuh"

namespace operation
{
    namespace detail
    {
        graph::tensorPtr GEMM(graph::tensorPtr A, graph::tensorPtr B,
                              graph::tensorPtr C, bool bias,
                              float alpha, float beta,
                              bool transposeA = false, bool transposeB = false);
    }

    class LinearOpr;
    class LinearNode : public graph::Node
    {
    private:
        graph::tensorPtr W, X;
        bool bias_;
        LinearNode(graph::tensorPtr w, graph::tensorPtr x, graph::tensorPtr b, bool bias = true)
            : Node((bias) ? 3 : 2, 1), W(w), X(x), bias_(bias)
        {
            // LinearNode的0, 1, 2 edge分别对应W, X, b；叶子用 accumulator_ 回退
            std::shared_ptr<graph::Node> w_fn = w->autograd().grad_fn_;
            if (!w_fn) w_fn = w->autograd().accumulator_;
            std::shared_ptr<graph::Node> x_fn = x->autograd().grad_fn_;
            if (!x_fn) x_fn = x->autograd().accumulator_;
            next_edges_[0] = {w_fn, 0};
            next_edges_[1] = {x_fn, 0};
            if (bias)
            {
                std::shared_ptr<graph::Node> b_fn = b->autograd().grad_fn_;
                if (!b_fn) b_fn = b->autograd().accumulator_;
                next_edges_[2] = {b_fn, 0};
            }
        }
        friend LinearOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class LinearOpr : public Opr
    {
    private:
        bool bias_ = true;
        std::shared_ptr<LinearNode> makeNode(
            graph::tensorPtr W, graph::tensorPtr X, graph::tensorPtr b, bool bias = true)
        {
            return std::shared_ptr<LinearNode>(new LinearNode(W, X, b, bias));
        }

    public:
        LinearOpr(bool bias = true) : bias_(bias) {}
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
    };

}