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
        base::s_vec inputStrides, inputShape, dims;

        SumNode(std::vector<graph::tensorPtr> adds) : Node(1, 1)
        {
            next_edges_[0] = {adds[0]->autograd().grad_fn_, 0};
        }
        friend SumOpr;

    public:
        std::vector<graph::tensorPtr> backward()
        {
            if (!prev_grads_[0])
                throw std::invalid_argument("No grad to be backward.");
        }
    };
    class SumOpr
    {
    private:
        base::s_vec dims_;
        std::shared_ptr<graph::Node> makeNode(std::vector<graph::tensorPtr> inputs)
        {
            return std::shared_ptr<graph::Node>(new SumNode(inputs));
        }

    public:
        bool broadcastingSupport() { return false; }
        void setDims(base::s_vec dims) { dims_ = dims; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
    };

    class ElimentWiseOpr;
    class ElimentWiseNode : public graph::Node
    {
    private:
        base::s_vec input1Shape, input2Shape, outputShape;

        ElimentWiseNode(std::vector<graph::tensorPtr> inputs) : Node(2, 1)
        {
            for (int i = 0; i < inputs.size(); i++)
            {
                next_edges_[i] = {inputs[i]->autograd().grad_fn_, 0};
            }
        }
        friend ElimentWiseOpr;

    public:
        std::vector<graph::tensorPtr> backward();
    };
    class ElimentWiseOpr : public Opr
    {
    private:
        std::shared_ptr<graph::Node> makeNode(std::vector<graph::tensorPtr> inputs)
        {
            return std::shared_ptr<graph::Node>(new ElimentWiseNode(inputs));
        }

    public:
        bool broadcastingSupport() { return true; }
        graph::tensorPtr operator()(std::vector<graph::tensorPtr> inputs);
        // {
        //     if (inputs.size() != 2)
        //         throw(std::invalid_argument("Require Two arguments"));

        //     graph::tensorPtr result = std::make_shared<graph::TensorImpl>(
        //         inputs[0]->shape(), inputs[0]->type(), inputs[0]->device());
        //     for (auto add : inputs)
        //         result->accumulate(add);
        //     result->autograd().grad_fn_ = makeNode(inputs);
        // }
    };

}