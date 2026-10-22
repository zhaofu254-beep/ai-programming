#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <set>
#include <algorithm>

#include "tensor.cuh"
#include "sum.cuh"

namespace operation
{
    // 实现算子

    // coalesce_dimensions —— 合并相邻且连续（stride[i] == stride[i+1]*shape[i+1]）的非归约维；
    // 返回 {new_shape, new_stride, new_masked_dims_idx}（归约维在新坐标系中的下标）。
    std::array<base::s_vec, 3> coalesce_dimensions(
        base::s_vec shape, base::s_vec stride, std::vector<bool> masked_dims)
    {
        base::s_vec new_shape, new_stride, new_masked_dims_idx;
        int64_t n = static_cast<int64_t>(shape.size());

        int64_t i = n - 1;
        while (i >= 0)
        {
            if (masked_dims[i])
            { // 归约维不能合并，单独保留
                new_shape.push_back(shape[i]);
                new_stride.push_back(stride[i]);
                new_masked_dims_idx.push_back(static_cast<int64_t>(new_shape.size()) - 1);
                i -= 1;
            }
            else
            { // 从内向外合并连续的非归约维
                int64_t merged_size = shape[i];
                int64_t innermost_stride = stride[i];
                int64_t j = i - 1;
                while (j >= 0 && !masked_dims[j] &&
                       stride[j] == stride[j + 1] * shape[j + 1])
                {
                    merged_size *= shape[j];
                    j -= 1;
                }
                new_shape.push_back(merged_size);
                new_stride.push_back(innermost_stride);
                i = j;
            }
        }

        // 上面从内向外 push，reverse 后变为从外向内；同步重映射归约维下标
        std::reverse(new_shape.begin(), new_shape.end());
        std::reverse(new_stride.begin(), new_stride.end());
        int64_t m = static_cast<int64_t>(new_shape.size());
        for (auto &idx : new_masked_dims_idx)
            idx = m - 1 - idx;

        return {std::move(new_shape), std::move(new_stride)};
    }

    template <typename T>
    graph::tensorPtr smallContigReduceCPU(graph::tensorPtr t, const int64_t rdim)
    {
        base::s_vec out_shape;
        int64_t ndim = t->shape().size();
        out_shape.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i)
            if (i != rdim)
                out_shape.push_back(t->shape()[i]);
        graph::tensorPtr reduced = std::make_shared<graph::TensorImpl>(
            out_shape, t->type(), t->device(), t->requires_grad());
        base::s_vec coor(ndim, 0);
        int64_t reduced_idx = 0;
        T *reduced_data = static_cast<T *>(reduced->data().data());
        T *t_data = static_cast<T *>(t->data().data());
        for (int n = 0; n < reduced->numel(); n++)
        {
            reduced_idx += reduced->stride()[ndim - 2];
            coor[ndim - 2] += 1;
            for (int i = ndim - 2; i >= 0; i--)
            {
                if (coor[i] >= reduced->shape()[i])
                {
                    coor[i] = 0;
                    coor[i - 1] += 1;
                    reduced_idx -= reduced->shape()[i] * reduced->stride()[i];
                    reduced_idx += reduced->stride()[i - 1];
                }
                else
                    break;
            }
            int64_t t_idx = 0;
            for (int i = 0; i < ndim - 1; i++)
                t_idx += coor[i] * t->stride()[i];
            for (int i = 0; i < t->shape()[rdim]; i++)
            {
                reduced_data[reduced_idx] += t_data[t_idx];
                t_idx += t->stride()[rdim];
            }
        }
        return reduced;
    }

    // smallContigReduceKernel —— 单维归约：每个线程负责一个输出位置，串行累加归约维。
    // 工作在合并后的维度上（input_dim/input_stride/rdim 均为 coalesce 之后的结果）。
    const int MAX_SMALL_DIM_SIZE = 8;
    struct SmallReduceKernelDimStride
    { // 用于向kernel传参，当维度个数大于8时，默认不使用smallDimReduce
        int64_t input_dim[MAX_SMALL_DIM_SIZE];
        int64_t input_stride[MAX_SMALL_DIM_SIZE];
        int64_t output_stride[MAX_SMALL_DIM_SIZE];
    };
    template <typename T>
    __global__ void smallContigReduceKernel(T *output, const T *input,
                                            const int64_t out_numel,
                                            SmallReduceKernelDimStride s,
                                            const int64_t rdim,
                                            const int ndim)
    {
        int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= out_numel)
            return;

        // 把 idx 按行主序分解为输出坐标，映射回输入偏移（跳过归约维）
        int64_t rem = idx;
        int64_t in_off = 0;
        for (int od = 0; od < ndim - 1; ++od)
        {
            int64_t c = rem / s.output_stride[od];
            rem -= c * s.output_stride[od];
            int d = (od < rdim) ? od : od + 1;
            in_off += c * s.input_stride[d];
        }

        T v = 0;
        for (int64_t j = 0; j < s.input_dim[rdim]; ++j)
            v += input[in_off + j * s.input_stride[rdim]];

        output[idx] = v;
    }

    // smallContigReduceCUDA —— 单维归约入口：先合并连续维，再按标量类型派发内核
    template <typename T>
    graph::tensorPtr smallContigReduceCUDA(graph::tensorPtr t, const int64_t rdim)
    {
        int64_t ndim = static_cast<int64_t>(t->shape().size());
        if (rdim < 0 || rdim >= ndim)
            throw std::out_of_range("smallDimReduce: invalid rdim");

        // 输出逻辑形状 = 去掉第 rdim 维
        base::s_vec out_shape;
        out_shape.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i)
            if (i != rdim)
                out_shape.push_back(t->shape()[i]);

        graph::tensorPtr reduced = std::make_shared<graph::TensorImpl>(
            out_shape, t->type(), t->device(), t->requires_grad());

        int64_t out_numel = reduced->numel();
        if (out_numel == 0) // 空张量直接返回
            return reduced;

        // 标记归约维，合并连续非归约维，减少内核里的整除/取模次数
        std::vector<bool> masked_dims(ndim, false);
        masked_dims[rdim] = true;
        auto [new_shape, new_stride, new_reduce_idx] =
            coalesce_dimensions(t->shape(), t->stride(), masked_dims);
        int64_t new_rdim = new_reduce_idx[0]; // 归约维在合并后坐标系中的下标

        // 合并后的输出 stride（行主序，去掉归约维）
        base::s_vec out_stride(new_shape.size() - 1);
        int64_t step = 1;
        for (int64_t i = static_cast<int64_t>(new_shape.size()) - 1; i >= 0; --i)
        {
            if (i == new_rdim)
                continue;
            int64_t od = (i < new_rdim) ? i : i - 1;
            out_stride[od] = step;
            step *= new_shape[i];
        }

        // 构造kernel参数
        SmallReduceKernelDimStride s;
        for (int64_t i = static_cast<int64_t>(new_shape.size()) - 1; i >= 0; --i)
        {
            s.input_dim[i] = new_shape[i];
            s.input_stride[i] = new_stride[i];
            if (i < static_cast<int64_t>(new_shape.size()) - 1)
                s.output_stride[i] = out_stride[i];
        }

        const int BLOCK_SIZE = 1024;
        int grid_size = static_cast<int>(
            (out_numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
        CHECK_KERNEL_FUNC(smallContigReduceKernel<T>
                          <<<grid_size, BLOCK_SIZE>>>(
                              static_cast<T *>(reduced->data().data()),
                              static_cast<const T *>(t->data().data()),
                              out_numel, s, new_rdim,
                              static_cast<int>(new_shape.size())));

        return reduced;
    }

    graph::tensorPtr smallContigReduce(graph::tensorPtr t, const int64_t rdim)
    {
        graph::tensorPtr reduced;
        base::dispatch(t->type().scalar_type_, t->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               reduced = smallContigReduceCUDA(t, rdim);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               reduced = smallContigReduceCPU(t, rdim);
                           }
                           else
                               throw std::runtime_error(
                                   "smallContigReduce: Unkown Device");
                       });
    }

    template <typename T>
    __global__ void longTailContigMidReduceKernel(T *output, const T *input,
                                                  const int64_t in_numel,
                                                  const int64_t out_numel,
                                                  const int64_t inputshape[3],
                                                  const int64_t inputstride[3],
                                                  const int64_t outputstride[2],
                                                  const int oneLineBlockNum,
                                                  const int accumulateStride,
                                                  const int accumulateNum)
    {
        // 归约中间维度
        // 在inputshape[2]>=32, inputshape[1]>=256时连续tensor使用
        // gridDim.x沿着outputshape[1]铺开，gridDim.y代表处理同一批归约的线程的block集合
        // 一个block中的各个warp对应同样连续的32个位置，分别间隔256
        // blockDim.y=maxBlockSize/warp_size,blockDIm.x=warp_size

        int64_t bx = blockIdx.x, by = blockIdx.y;
        int64_t out0 = bx / oneLineBlockNum, out1 = bx % oneLineBlockNum;
        int64_t lane = threadIdx.x, warp_idx = threadIdx.y;

        T value = 0;

        if (out1 * blockDim.x + lane < outputshape[1])
        {

            int64_t start_idx = out0 * inputstride[0] + out1 * blockDim.x +
                                (warp_idx * blockDIm.x + lane) * inputstride[1];

            int64_t idx = start_idx;
            for (i = 0; i < accumulateNum; i++)
            {
                if (idx < in_numel)
                    value += input[idx];
                idx += accumulateStride;
            }
        }

        extern __shared__ T laneValues[];

        atomicAdd(&laneValues[lane], value);
        __syncthreads();

        if (warp_idx == 0)
        {
            if (out1 * blockDim.x + lane < outputshape[1])
            {
                atomicAdd(&output[out0 * inputshape[0] + out1 * blockDim.x + lane],
                          laneValues[lane]);
            }
        }
    }

    template <typename T>
    graph::tensorPtr longTailContigReduceCUDA(graph::tensorPtr t, const int64_t rdim)
    {
        std::vector<bool> mask(0, t->shape().size());
        mask[rdim] = 1;
        auto [coalesced_shape, new_rdim] = coalesce_dimensions(t->shape(), t->stride(), mask);

        const int WARP_ACCUMU_NUM = 256;
        const int WARP_SIZE = 32;
        int BLOCK_DIM_Y = std::min(32, (coalesced_shape[1] + WARP_ACCUMU_NUM - 1) / WARP_ACCUMU_NUM);

        int oneLineBlockNum = (coalesced_shape[2] + WARP_SIZE - 1) / WARP_SIZE;
        int reduceBlockNum = (coalesced_shape[1] + BLOCK_SIZE / WARP_ACCUMU_NUM - 1) /
                             (BLOCK_SIZE / WARP_ACCUMU_NUM);

        dim3 block = {WARP_SIZE, BLOCK_DIM_Y};
        dim3 grid = {coalesced_shape[0] * coalesced_shape[2] + oneLineBlockNum, reduceBlockNum};

        base::s_vec new_tensor_shape.reserve(t->shape().size() - 1);
        for (int i = 0; i < t->shape().size(); i++)
            if (i != rdim)
                new_tensor_shape.push_back(t->shape()[i]);
        graph::tensorPtr r = std::make_shared<graph::TensorImpl>(
            new_tensor_shape, t->type(), t->device(), t->requires_grad());

        const int64_t inputshape[3];
        const int64_t inputstride[3];
        const int64_t outputstride[2];
        for (int i = 0; i < 3; i++)
            inputshape[i] = coalesced_shape[i];
        inputstride[2] = 1;
        inputstride[1] = coalesced_shape[2];
        inputstride[0] = coalesced_shape[1] * coalesced_shape[2];
        outputstride[1] = 1;
        outputstride[0] = coalesced_shape[2];
        const int64_t in_numel = t->numel(), out_numel = r->numel();

        longTailContigMidReduceKernel<<<grid, block, WARP_SIZE>>>(r->data().data(),
                                                                  t->data().data(),
                                                                  in_numel, out_numel,
                                                                  inputshape, inputstride,
                                                                  outputstride,
                                                                  oneLineBlockNum,
                                                                  inputstride[1],
                                                                  WARP_ACCUMU_NUM);
        return r;
    }

    graph::tensorPtr longTailContigReduce(graph::tensorPtr t, const int64_t rdim)
    {
        graph::tensorPtr reduced;
        base::dispatch(t->type().scalar_type_, t->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               reduced = longTailContigReduceCUDA(t, rdim);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               reduced = smallContigReduceCPU(t, rdim);
                           }
                           else
                               throw std::runtime_error(
                                   "smallContigReduce: Unkown Device");
                       });
    }
    graph::tensorPtr SumOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
    }
}