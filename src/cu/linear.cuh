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
#include "sum.cuh"

namespace operation
{
    class LinearOpr;
    class LinearNode : public graph::Node
    {
    private:
        graph::tensorPtr W, X;
        LinearNode(graph::tensorPtr w, graph::tensorPtr x, graph::tensorPtr b)
            : Node(3, 1), W(w), X(x)
        {
            // LinearNode的0, 1, 2 edge分别对应W, X, b
            next_edges_[0] = {w->autograd().grad_fn_, 0};
            next_edges_[1] = {x->autograd().grad_fn_, 0};
            next_edges_[2] = {b->autograd().grad_fn_, 0};
        }
        friend LinearOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };

    class LinearOpr : public Opr
    {
    private:
        std::shared_ptr<LinearNode> makeNode(
            graph::tensorPtr W, graph::tensorPtr X, graph::tensorPtr b)
        {
            return std::make_shared<LinearNode>(W, X, b);
        }

    public:
        bool broadcastingSupport() { return false; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr>)
        {
        }
    };

}