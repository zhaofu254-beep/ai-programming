#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "tensor.cuh"
#include "sum.cuh"
#include "maxpool.cuh"

namespace operation
{ // 实现前向传播
    template <typename T>
    std::array<graph::tensorPtr, 2> maxpoolCPU(graph::tensorPtr input)
    {
        auto new_shape = input->shape();
        auto H = new_shape[new_shape.size() - 2];
        auto W = new_shape[new_shape.size() - 1];
        auto h_out = (H + 1) / 2;
        auto w_out = (W + 1) / 2;
        new_shape[new_shape.size() - 2] = h_out;
        new_shape[new_shape.size() - 1] = w_out;

        graph::tensorPtr mask = std::make_shared<graph::TensorImpl>(
            input->shape(), base::ScalarType::Bool, input->device(), false);
        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            new_shape, input->type(), input->device(), input->requires_grad());

        auto inputPtr = static_cast<T *>(input->data());
        auto outputPtr = static_cast<T *>(output->data());
        auto maskPtr = static_cast<bool *>(mask->data());

        for (int n = 0; n < input->numel() / (H * W); n++)
        {
            size_t layer_offset = n * H * W;
            for (int i = 0; i < h_out; i++)
            {
                size_t h_in = KERNEL_SIZE * i;
                for (int j = 0; j < w_out; j++)
                {
                    size_t w_in = KERNEL_SIZE * j;
                    T max_value = inputPtr[layer_offset + h_in * W + w_in];
                    size_t maxIdx = 0;

                    if (w_in + 1 < W)
                    {
                        T v = inputPtr[layer_offset + h_in * W + w_in + 1];
                        if (v > max_value)
                        {
                            max_value = v;
                            maxIdx = 1;
                        }
                    }
                    if (h_in + 1 < H)
                    {
                        T v = inputPtr[layer_offset + (h_in + 1) * W + w_in];
                        if (v > max_value)
                        {
                            max_value = v;
                            maxIdx = 2;
                        }
                    }
                    if (w_in + 1 < W && h_in + 1 < H)
                    {
                        T v = inputPtr[layer_offset + (h_in + 1) * W + w_in + 1];
                        if (v > max_value)
                        {
                            max_value = v;
                            maxIdx = 3;
                        }
                    }

                    maskPtr[n * H * W + (h_in + maxIdx / 2) * W + w_in + maxIdx % 2] = true;
                    outputPtr[n * h_out * w_out + i * w_out + j] = max_value;
                }
            }
        }
        return {output, mask};
    }

    template <typename T>
    __global__ void maxpoolKernel(const T *input, size_t H, size_t W, size_t HW, size_t N,
                                  size_t h_out, size_t w_out, size_t hw_out,
                                  T *output, bool *mask)
    {
        // N = batch_size * C_in;
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t n = idx / hw_out, p = idx % hw_out;
        size_t h_out_idx = p / w_out, w_out_idx = p % w_out;

        if (n >= N || h_out_idx >= h_out || w_out_idx >= w_out)
            return;

        size_t h_in_idx = 2 * h_out_idx, w_in_idx = 2 * w_out_idx;
        size_t layer_offset = n * HW;

        T max_value = input[layer_offset + h_in_idx * W + w_in_idx];
        size_t maxIdx = 0;

        if (w_in_idx + 1 < W)
        {
            T v = input[layer_offset + h_in_idx * W + w_in_idx + 1];
            if (v > max_value)
            {
                max_value = v;
                maxIdx = 1;
            }
        }
        if (h_in_idx + 1 < H)
        {
            T v = input[layer_offset + (h_in_idx + 1) * W + w_in_idx];
            if (v > max_value)
            {
                max_value = v;
                maxIdx = 2;
            }
        }
        if (w_in_idx + 1 < W && h_in_idx + 1 < H)
        {
            T v = input[layer_offset + (h_in_idx + 1) * W + w_in_idx + 1];
            if (v > max_value)
            {
                max_value = v;
                maxIdx = 3;
            }
        }

        mask[n * HW + (h_in_idx + maxIdx / 2) * W + w_in_idx + maxIdx % 2] = true;
        output[n * hw_out + h_out_idx * w_out + w_out_idx] = max_value;
    }

    template <typename T>
    std::array<graph::tensorPtr, 2> maxpoolCUDA(graph::tensorPtr input)
    {
        auto new_shape = input->shape();
        auto H = new_shape[new_shape.size() - 2];
        auto W = new_shape[new_shape.size() - 1];
        auto h_out = (H + 1) / 2;
        auto w_out = (W + 1) / 2;
        new_shape[new_shape.size() - 2] = h_out;
        new_shape[new_shape.size() - 1] = w_out;

        graph::tensorPtr mask = std::make_shared<graph::TensorImpl>(
            input->shape(), base::ScalarType::Bool, input->device(), false);
        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            new_shape, input->type(), input->device(), input->requires_grad());

        const size_t BLOCK_SIZE = 256;
        size_t grid_size = (output->numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL_FUNC(maxpoolKernel<T><<<grid_size, BLOCK_SIZE>>>(
                              static_cast<T *>(input->data()),
                              H, W, H * W, input->numel() / (H * W),
                              h_out, w_out, h_out * w_out,
                              static_cast<T *>(output->data()),
                              static_cast<bool *>(mask->data())););
        return {output, mask};
    }

    std::array<graph::tensorPtr, 2> maxpool(graph::tensorPtr input)
    {
        if (!input->is_continue())
            throw(std::invalid_argument("tensor must be continuous!"));

        std::array<graph::tensorPtr, 2> outAndMask;
        base::dispatch(input->type().scalar_type_, input->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               outAndMask = maxpoolCUDA<T>(input);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               outAndMask = maxpoolCPU<T>(input);
                           }
                           else
                               throw std::runtime_error(
                                   "MaxPool: Unkown Device");
                       });
        return outAndMask;
    }

    graph::tensorPtr MaxPoolOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
        auto input = inputs[0];
        auto results = maxpool(input);
        std::shared_ptr<MaxPoolNode> node = makeNode(input, results[1]);
        auto output = results[0];
        output->autograd().grad_fn_ = node;
        return output;
    }
}

namespace operation
{
    // 实现mask
    template <typename T>
    graph::tensorPtr maskCPU(graph::tensorPtr input, graph::tensorPtr mask_)
    {

        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), false);

        int64_t N = input->numel();
        T *inputP = static_cast<T *>(input->data());
        bool *maskP = static_cast<bool *>(mask_->data());
        T *outputP = static_cast<T *>(output->data());

        for (int i = 0; i < N; i++)
        {
            outputP[i] = (maskP[i]) ? inputP[i] : 0.0f;
        }

        return output;
    }

    template <typename T>
    __global__ void maskKernel(const T *input, const int64_t N,
                               const bool *mask_, T *output)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        output[idx] = mask_[idx] ? input[idx] : 0.0f;
    }

    template <typename T>
    graph::tensorPtr maskCUDA(graph::tensorPtr input, graph::tensorPtr mask_)
    {
        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), false);

        int64_t N = input->numel();
        const int BLOCK_SIZE = 256;
        int block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CHECK_KERNEL_FUNC(maskKernel<<<block_num, BLOCK_SIZE>>>(
                              static_cast<T *>(input->data()), N,
                              static_cast<bool *>(mask_->data()),
                              static_cast<T *>(output->data())););

        return output;
    }

    graph::tensorPtr mask(graph::tensorPtr input, graph::tensorPtr mask_)
    {
        if (mask_->type().scalar_type_ != base::ScalarType::Bool)
            throw(std::invalid_argument("Wrong mask type"));
        if (input->shape() != mask_->shape())
            throw(std::invalid_argument("Wrong shape"));

        graph::tensorPtr output = std::make_shared<graph::TensorImpl>(
            input->shape(), input->type(), input->device(), false);

        base::dispatch(input->type().scalar_type_, input->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               output = maskCUDA<T>(input, mask_);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               output = maskCPU<T>(input, mask_);
                           }
                           else
                               throw std::runtime_error(
                                   "MaxPool: Unkown Device");
                       });
        return output;
    }
}

namespace operation
{ // 实现backward
    template <typename T>
    graph::tensorPtr poolBackCPU(graph::tensorPtr prev, graph::tensorPtr mask)
    {
        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            mask->shape(), prev->type(), prev->device(), false);
        next->fill(0);

        auto &mask_shape = mask->shape();
        auto H = mask_shape[mask_shape.size() - 2];
        auto W = mask_shape[mask_shape.size() - 1];
        auto HW = H * W;
        auto N = mask->numel() / HW;
        auto h_out = (H + 1) / 2;
        auto w_out = (W + 1) / 2;
        auto hw_out = h_out * w_out;

        T *prevP = static_cast<T *>(prev->data());
        bool *maskP = static_cast<bool *>(mask->data());
        T *nextP = static_cast<T *>(next->data());

        for (int n = 0; n < N; n++)
        {
            for (int h_in_idx = 0; h_in_idx < H; h_in_idx++)
            {
                for (int w_in_idx = 0; w_in_idx < W; w_in_idx++)
                {
                    if (!maskP[n * HW + h_in_idx * W + w_in_idx])
                        continue;

                    auto h_out_idx = h_in_idx / 2,
                         w_out_idx = w_in_idx / 2;

                    nextP[n * HW + h_in_idx * W + w_in_idx] =
                        prevP[n * hw_out + h_out_idx * w_out + w_out_idx];
                }
            }
        }

        return next;
    }

    template <typename T>
    __global__ void poolBackKernel(const T *prev,
                                   const bool *mask, T *next,
                                   size_t W, size_t H, size_t HW, size_t N,
                                   size_t w_out, size_t hw_out)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t n = idx / HW, p = idx % HW;
        size_t h_in_idx = p / W, w_in_idx = p % W;
        size_t h_out_idx = h_in_idx / KERNEL_SIZE,
               w_out_idx = w_in_idx / KERNEL_SIZE;

        if (!mask[idx] || n >= N || h_in_idx >= H || w_in_idx >= W)
            return;

        next[idx] = prev[n * hw_out + h_out_idx * w_out + w_out_idx];
    }

    template <typename T>
    graph::tensorPtr poolBackCUDA(graph::tensorPtr prev, graph::tensorPtr mask)
    {

        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            mask->shape(), prev->type(), prev->device(), false);

        auto &mask_shape = mask->shape();
        auto H = mask_shape[mask_shape.size() - 2];
        auto W = mask_shape[mask_shape.size() - 1];
        auto HW = H * W;
        auto N = mask->numel() / HW;
        auto h_out = (H + 1) / 2;
        auto w_out = (W + 1) / 2;
        auto hw_out = h_out * w_out;

        const size_t BLOCK_SIZE = 256;
        size_t grid_size = (next->numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CHECK_KERNEL_FUNC(poolBackKernel<<<grid_size, BLOCK_SIZE>>>(
            static_cast<T *>(prev->data()),
            static_cast<bool *>(mask->data()),
            static_cast<T *>(next->data()),
            W, H, HW, N, w_out, hw_out));

        return next;
    }

    graph::tensorPtr poolBack(graph::tensorPtr prev, graph::tensorPtr mask)
    {
        if (mask->type().scalar_type_ != base::ScalarType::Bool)
            throw(std::invalid_argument("Wrong mask type"));

        graph::tensorPtr next = std::make_shared<graph::TensorImpl>(
            mask->shape(), prev->type(), prev->device(), false);

        base::dispatch(prev->type().scalar_type_, prev->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               next = poolBackCUDA<T>(prev, mask);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               next = poolBackCPU<T>(prev, mask);
                           }
                           else
                               throw std::runtime_error(
                                   "MaxPool: Unkown Device");
                       });
        return next;
    }

    std::vector<graph::tensorPtr> MaxPoolNode::backward()
    {
        graph::tensorPtr prev = prev_grads_[0];
        return {poolBack(prev, mask_)};
    }
}
