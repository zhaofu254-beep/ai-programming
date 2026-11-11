#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>

#include "tensor.cuh"
#include "sum.cuh"
#include "softmax.cuh"
#include "crossentropy.cuh"

namespace operation
{
    const float epsilon = 1e-8;
    template <typename T>
    graph::tensorPtr crossEntropyCPU(graph::tensorPtr prob, graph::tensorPtr label)
    {

        graph::tensorPtr loss = std::make_shared<graph::TensorImpl>(
            label->shape(), prob->type(), prob->device(), prob->requires_grad());

        int64_t N = label->numel();
        int64_t L = prob->shape().back();

        T *probP = static_cast<T *>(prob->data());
        int *labelP = static_cast<int *>(label->data());
        T *lossP = static_cast<T *>(loss->data());

        for (int64_t idx = 0; idx < N * L; idx++)
        {

            int64_t n = idx / L, l = idx % L;
            if (l == labelP[n])
                lossP[n] = -logf(probP[idx] + epsilon) / (static_cast<T>(N));
        }

        return loss;
    }

    template <typename T>
    __global__ void crossEntropyKernel(const T *prob, const int *label, T *loss,
                                       int64_t N, int64_t L)
    {
        int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;

        int64_t lbl = label[n];
        T p = prob[n * L + lbl];

        if constexpr (std::is_same_v<T, float>)
            loss[n] = -logf(p + epsilon) / static_cast<T>(N);
        else if constexpr (std::is_same_v<T, double>)
            loss[n] = -log(static_cast<double>(p) + epsilon) / static_cast<T>(N);
        else
            loss[n] = static_cast<T>(-::log(static_cast<double>(p) + epsilon) /
                                     static_cast<T>(N));
    }

    template <typename T>
    graph::tensorPtr crossEntropyCUDA(graph::tensorPtr prob, graph::tensorPtr label)
    {
        graph::tensorPtr loss = std::make_shared<graph::TensorImpl>(
            label->shape(), prob->type(), prob->device(), prob->requires_grad());

        int64_t N = label->numel();
        int64_t L = prob->shape().back();

        const int BLOCK_SIZE = 256;
        int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL_FUNC(crossEntropyKernel<<<grid_size, BLOCK_SIZE>>>(
                              static_cast<T *>(prob->data()),
                              static_cast<int *>(label->data()),
                              static_cast<T *>(loss->data()),
                              N, L););

        return loss;
    }

    namespace detail
    {
        graph::tensorPtr crossEntropy(graph::tensorPtr prob, graph::tensorPtr label)
        {
            if (!prob->is_continue())
                throw(std::invalid_argument("tensor must be continuous!"));

            graph::tensorPtr loss;
            base::dispatch(prob->type().scalar_type_, prob->device(),
                           [&]<typename T, base::Device Dev>()
                           {
                               if constexpr (Dev == base::Device::CUDA)
                               {
                                   loss = crossEntropyCUDA<T>(prob, label);
                               }
                               else if constexpr (Dev == base::Device::CPU)
                               {
                                   // CPU 统一用一套算法了。
                                   loss = crossEntropyCPU<T>(prob, label);
                               }
                               else
                                   throw std::runtime_error(
                                       "CEL: Unkown Device");
                           });
            return contigReduce(loss);
        }
    }

    graph::tensorPtr CrossEntropyOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
        auto input = inputs[0], label = inputs[1];
        auto prob = detail::softmax(input);
        auto loss = detail::crossEntropy(prob, label);

        std::shared_ptr<CrossEntropyNode> node = makeNode(input, prob, label);
        loss->autograd().grad_fn_ = node;
        return loss;
    }
}

namespace operation
{
    // 反向传播
    template <typename T>
    graph::tensorPtr backCPU(graph::tensorPtr prob, graph::tensorPtr label, float prev)
    {
        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            prob->shape(), prob->type(), prob->device(), prob->requires_grad());

        int64_t N = label->numel();
        int64_t L = prob->shape().back();

        T *probP = static_cast<T *>(prob->data());
        int *labelP = static_cast<int *>(label->data());
        T *nextP = static_cast<T *>(next->data());

        for (int64_t idx = 0; idx < N * L; idx++)
        {
            int64_t n = idx / L, l = idx % L;

            T y = (l == labelP[n]) ? 1.0f : 0.0f;
            nextP[idx] = prev * (probP[idx] - y);
        }

        return next;
    }

    template <typename T>
    __global__ void backKernel(const T *prob, const int *label,
                               T *next, const float prev, // scalar loss对各个batch的梯度一定是相同的一个num
                               const int64_t N, const int64_t L)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N * L)
            return;

        int64_t n = idx / L, l = idx % L;

        T y = (l == label[n]) ? 1.0f : 0.0f;
        next[idx] = prev * (prob[idx] - y);
    }

    template <typename T>
    graph::tensorPtr backCUDA(graph::tensorPtr prob, graph::tensorPtr label, float prev)
    {
        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            prob->shape(), prob->type(), prob->device(), prob->requires_grad());

        int64_t N = label->numel();
        int64_t L = prob->shape().back();

        const int BLOCK_SIZE = 256;
        int grid_size = (prob->numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL_FUNC(backKernel<<<grid_size, BLOCK_SIZE>>>(
                              static_cast<T *>(prob->data()),
                              static_cast<int *>(label->data()),
                              static_cast<T *>(next->data()),
                              prev, N, L););

        return next;
    }

    graph::tensorPtr back(graph::tensorPtr prob, graph::tensorPtr label, float prev)
    {
        if (!prob->is_continue())
            throw(std::invalid_argument("tensor must be continuous!"));

        graph::tensorPtr next;
        base::dispatch(prob->type().scalar_type_, prob->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               next = backCUDA<T>(prob, label, prev);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               next = backCPU<T>(prob, label, prev);
                           }
                           else
                               throw std::runtime_error(
                                   "CELBack: Unkown Device");
                       });
        return next;
    }

    std::vector<graph::tensorPtr> CrossEntropyNode::backward()
    {
        float prev = 1.0f / label_->numel();

        auto next = back(prob_, label_, prev);
        return {next};
    }
}
