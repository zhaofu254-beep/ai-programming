#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>

#include "tensor.cuh"
#include "sum.cuh"
#include "softmax.cuh"

namespace operation
{
    // exp
    template <typename T>
    graph::tensorPtr expCPU(graph::tensorPtr input)
    {
        int64_t N = input->numel();

        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), input->requires_grad());

        for (int idx = 0; idx < N; idx++)
            static_cast<T *>(output->data())[idx] =
                std::exp(static_cast<T *>(input->data())[idx]);

        return output;
    }

    template <typename T>
    __global__ void expKernel(const T *input, int64_t N, T *output)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;
        if constexpr (std::is_same_v<T, float>)
            output[idx] = expf(input[idx]);
        else if constexpr (std::is_same_v<T, double>)
            output[idx] = exp(input[idx]);
        else
            output[idx] = static_cast<T>(::exp(static_cast<double>(input[idx])));
    }

    template <typename T>
    graph::tensorPtr expCUDA(graph::tensorPtr input)
    {
        int64_t N = input->numel();

        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), input->requires_grad());

        const size_t BLOCK_SIZE = 256;
        size_t grid_size = (output->numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL_FUNC(expKernel<<<grid_size, BLOCK_SIZE>>>(
            static_cast<T *>(input->data()), N,
            static_cast<T *>(output->data())));

        return output;
    }
    namespace detail
    {
        graph::tensorPtr Exp(graph::tensorPtr input)
        {
            if (!input->is_continue())
                throw(std::invalid_argument("tensor must be continuous!"));

            graph::tensorPtr output;
            base::dispatch(input->type().scalar_type_, input->device(),
                           [&]<typename T, base::Device Dev>()
                           {
                               if constexpr (Dev == base::Device::CUDA)
                               {
                                   output = expCUDA<T>(input);
                               }
                               else if constexpr (Dev == base::Device::CPU)
                               {
                                   // CPU 统一用一套算法了。
                                   output = expCPU<T>(input);
                               }
                               else
                                   throw std::runtime_error(
                                       "Softmax: Unkown Device");
                           });
            return output;
        }
    }
}

namespace operation
{
    // tailDevide
    const float epsilon = 1e-8;

    template <typename T>
    graph::tensorPtr tailDevideCPU(graph::tensorPtr input, graph::tensorPtr frac)
    {
        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), input->requires_grad());

        int64_t M = input->shape().back();
        int64_t N = input->numel();
        for (int idx = 0; idx < N; idx++)
        {
            auto n = idx / M;
            static_cast<T *>(output->data())[idx] =
                static_cast<T *>(input->data())[idx] /
                (static_cast<T *>(frac->data())[n] + epsilon);
        }

        return output;
    }

    template <typename T>
    __global__ void tailDevideKernel(const T *input, const T *frac,
                                     T *output, int64_t N, int64_t M)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;
        auto n = idx / M;
        output[idx] = input[idx] / (frac[n] + epsilon);
    }

    template <typename T>
    graph::tensorPtr tailDevideCUDA(graph::tensorPtr input, graph::tensorPtr frac)
    {
        int64_t M = input->shape().back();
        int64_t N = input->numel();

        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), input->requires_grad());

        const size_t BLOCK_SIZE = 256;
        size_t grid_size = (output->numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL_FUNC(tailDevideKernel<<<grid_size, BLOCK_SIZE>>>(
            static_cast<T *>(input->data()),
            static_cast<T *>(frac->data()),
            static_cast<T *>(output->data()),
            N, M));

        return output;
    }

    graph::tensorPtr tailDevide(graph::tensorPtr input, graph::tensorPtr frac)
    {
        if (!input->is_continue())
            throw(std::invalid_argument("tensor must be continuous!"));

        graph::tensorPtr output;
        base::dispatch(input->type().scalar_type_, input->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               output = tailDevideCUDA<T>(input, frac);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               output = tailDevideCPU<T>(input, frac);
                           }
                           else
                               throw std::runtime_error(
                                   "TailDevide: Unkown Device");
                       });
        return output;
    }
}

namespace operation
{
    // 前向传播
    namespace detail
    {
        graph::tensorPtr softmax(graph::tensorPtr input)
        {
            auto expInput = Exp(input);
            auto tailSum = contigReduce(expInput);
            return tailDevide(expInput, tailSum);
        }
    }

    graph::tensorPtr SoftmaxOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
        auto output = detail::softmax(inputs[0]);

        std::shared_ptr<SoftmaxNode> node = makeNode(inputs[0], output);
        output->autograd().grad_fn_ = node;
        return output;
    }
}

namespace operation
{
    // 反向传播
    template <typename T>
    graph::tensorPtr backCPU(graph::tensorPtr prev, graph::tensorPtr input)
    {
        graph::tensorPtr giyi = detail::contigReduce(
            std::make_shared<graph::TensorImpl>(
                (*prev) * (*input)));
        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), false);

        int64_t N = next->numel();
        int64_t M = next->shape().back();

        T *prevP = static_cast<T *>(prev->data());
        T *inputP = static_cast<T *>(input->data());
        T *giyiP = static_cast<T *>(giyi->data());
        T *nextP = static_cast<T *>(next->data());

        for (int idx = 0; idx < N; idx++)
        {
            int n = idx / M;
            nextP[idx] = (prevP[idx] - giyiP[n]) * inputP[idx];
        }

        return next;
    }

    template <typename T>
    __global__ void backKernel(const T *prev, const T *input, const T *giyi,
                               T *next, int64_t N, int64_t M)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;
        int n = idx / M;
        next[idx] = (prev[idx] - giyi[n]) * input[idx];
    }

    template <typename T>
    graph::tensorPtr backCUDA(graph::tensorPtr prev, graph::tensorPtr input)
    {
        graph::tensorPtr giyi = detail::contigReduce(
            std::make_shared<graph::TensorImpl>(
                (*prev) * (*input)));
        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), false);

        int64_t N = next->numel();
        int64_t M = next->shape().back();
        const size_t BLOCK_SIZE = 256;
        size_t grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL_FUNC(backKernel<<<grid_size, BLOCK_SIZE>>>(
                              static_cast<T *>(prev->data()),
                              static_cast<T *>(input->data()),
                              static_cast<T *>(giyi->data()),
                              static_cast<T *>(next->data()), N, M););

        return next;
    }

    graph::tensorPtr back(graph::tensorPtr prev, graph::tensorPtr input)
    {
        if (!input->is_continue())
            throw(std::invalid_argument("tensor must be continuous!"));

        graph::tensorPtr next;
        base::dispatch(input->type().scalar_type_, input->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               next = backCUDA<T>(prev, input);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               next = backCPU<T>(prev, input);
                           }
                           else
                               throw std::runtime_error(
                                   "Softmax backward: Unkown Device");
                       });
        return next;
    }

    std::vector<graph::tensorPtr> SoftmaxNode::backward()
    {
        auto input = forwardResult_;
        auto prev = prev_grads_[0];

        return {back(prev, input)};
    }
}