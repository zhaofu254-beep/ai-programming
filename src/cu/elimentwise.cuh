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