#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <set>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>

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

        return {std::move(new_shape), std::move(new_stride), std::move(new_masked_dims_idx)};
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
                               reduced = smallContigReduceCUDA<T>(t, rdim);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               reduced = smallContigReduceCPU<T>(t, rdim);
                           }
                           else
                               throw std::runtime_error(
                                   "smallContigReduce: Unkown Device");
                       });
        return reduced;
    }

    // longTailContigMidReduceKernel —— 归约"中间维"（连续张量，归约维非尾维）：
    // 每个 warp 沿连续尾维铺 32 个位置，block 内多个 warp 沿归约维按 accumulateNum 分块，
    // gridDim.y 再沿归约维分批。inputshape = [outer, reduce, tail]。
    template <typename T>
    __global__ void longTailContigMidReduceKernel(T *output, const T *input,
                                                  const int64_t inputshape[3],
                                                  const int64_t inputstride[3],
                                                  const int64_t outputstride[2],
                                                  const int oneLineBlockNum,
                                                  const int accumulateNum)
    {
        int64_t bx = blockIdx.x, by = blockIdx.y;
        int64_t out0 = bx / oneLineBlockNum, out1 = bx % oneLineBlockNum;
        int64_t lane = threadIdx.x, warp_idx = threadIdx.y;

        int64_t tail_pos = out1 * 32 + lane;
        int64_t reduce_start = (by * blockDim.y + warp_idx) * accumulateNum;

        extern __shared__ T laneValues[]; // blockDim.y * 32 个 T
        laneValues[warp_idx * 32 + lane] = 0;
        __syncthreads();

        if (tail_pos < inputshape[2] && reduce_start < inputshape[1])
        {
            int64_t idx = out0 * inputstride[0] +
                          reduce_start * inputstride[1] +
                          tail_pos * inputstride[2];
            T value = 0;
            int n = accumulateNum;
            if (reduce_start + accumulateNum > inputshape[1])
                n = static_cast<int>(inputshape[1] - reduce_start);
            for (int i = 0; i < n; ++i)
            {
                value += input[idx];
                idx += inputstride[1]; // 归约维 stride = tail
            }
            laneValues[warp_idx * 32 + lane] = value;
        }
        __syncthreads();

        if (warp_idx == 0 && tail_pos < inputshape[2])
        {
            T sum = 0;
            for (int w = 0; w < blockDim.y; ++w)
                sum += laneValues[w * 32 + lane];
            int64_t out_idx = out0 * outputstride[0] + tail_pos * outputstride[1];
            atomicAdd(&output[out_idx], sum);
        }
    }

    template <typename T>
    graph::tensorPtr longTailContigReduceCUDA(graph::tensorPtr t, const int64_t rdim)
    {
        int64_t ndim = static_cast<int64_t>(t->shape().size());
        if (rdim < 0 || rdim >= ndim)
            throw std::out_of_range("longTailContigReduce: invalid rdim");
        if (rdim == ndim - 1)
            throw std::invalid_argument(
                "longTailContigReduce: rdim should not be the last dim");

        // 标记归约维并合并连续非归约维（连续张量 + 中间归约 -> [outer, reduce, tail]）
        std::vector<bool> mask(ndim, false);
        mask[rdim] = true;
        auto coalesced = coalesce_dimensions(t->shape(), t->stride(), mask);
        base::s_vec shape = coalesced[0];
        while (shape.size() < 3)
            shape.insert(shape.begin(), 1); // 归约最外层维时补一个 outer=1

        const int WARP_ACCUMU_NUM = 256;
        const int WARP_SIZE = 32;

        int64_t outer = shape[0], reduce = shape[1], tail = shape[2];
        int oneLineBlockNum = static_cast<int>((tail + WARP_SIZE - 1) / WARP_SIZE);
        int BLOCK_DIM_Y = std::min(32, static_cast<int>(
                                           (reduce + WARP_ACCUMU_NUM - 1) / WARP_ACCUMU_NUM));
        int reduceBlockNum = static_cast<int>(
            (reduce + BLOCK_DIM_Y * WARP_ACCUMU_NUM - 1) /
            (BLOCK_DIM_Y * WARP_ACCUMU_NUM));

        dim3 block(WARP_SIZE, BLOCK_DIM_Y);
        dim3 grid(static_cast<unsigned>(outer * oneLineBlockNum),
                  static_cast<unsigned>(reduceBlockNum));

        // 输出形状 = 去掉 rdim
        base::s_vec out_shape;
        out_shape.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i)
            if (i != rdim)
                out_shape.push_back(t->shape()[i]);
        graph::tensorPtr r = std::make_shared<graph::TensorImpl>(
            out_shape, t->type(), t->device(), t->requires_grad());
        r->fill(0); // atomicAdd 累加前先清零

        const int64_t inputshape[3] = {outer, reduce, tail};
        const int64_t inputstride[3] = {reduce * tail, tail, 1};
        const int64_t outputstride[2] = {tail, 1};
        int shared_bytes = BLOCK_DIM_Y * WARP_SIZE * sizeof(T);

        CHECK_KERNEL_FUNC(longTailContigMidReduceKernel<T>
                          <<<grid, block, shared_bytes>>>(
                              static_cast<T *>(r->data().data()),
                              static_cast<const T *>(t->data().data()),
                              inputshape, inputstride, outputstride,
                              oneLineBlockNum, WARP_ACCUMU_NUM));

        return r;
    }

    graph::tensorPtr longTailContigReduce(graph::tensorPtr t, const int64_t rdim)
    {
        if (!t->data().is_continue())
            throw(std::invalid_argument("tensor must be continuous!"));

        graph::tensorPtr reduced;
        base::dispatch(t->type().scalar_type_, t->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               reduced = longTailContigReduceCUDA<T>(t, rdim);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               // CPU 统一用一套算法了。
                               reduced = smallContigReduceCPU<T>(t, rdim);
                           }
                           else
                               throw std::runtime_error(
                                   "smallContigReduce: Unkown Device");
                       });
        return reduced;
    }

    // shortTailContigMidReduceKernel —— 归约"中间维"、连续尾维较短（<=16）时的 fold 归约：
    // 把 warp 的 32 个 lane 拆成 (2^fold2exponent) 个 fold × tail_size 个尾位置，
    // 用多余 fold 覆盖归约维，warp 内 shuffle 合并 fold，跨 warp/block 用 atomicAdd 合并。
    template <typename T>
    __global__ void shortTailContigMidReduceKernel(T *output, const T *input,
                                                   const int64_t inputshape[3],
                                                   const int64_t inputstride[3],
                                                   const int64_t outputstride[2],
                                                   const int fold2exponent,
                                                   const int accumulateNum)
    {
        int64_t bx = blockIdx.x, by = blockIdx.y;
        int64_t lane = threadIdx.x, warp_idx = threadIdx.y;

        const int tail2exp = 5 - fold2exponent;
        const int tail_size = 1 << tail2exp;
        int fold_idx = lane >> tail2exp;
        int tail_idx = lane & (tail_size - 1);

        int64_t warpCover = (int64_t)accumulateNum << fold2exponent;
        int64_t reduce_start = by * blockDim.y * warpCover +
                               warp_idx * warpCover +
                               fold_idx * accumulateNum;

        T v = 0;
        if (tail_idx < inputshape[2] && reduce_start < inputshape[1])
        {
            int64_t idx = bx * inputstride[0] +
                          reduce_start * inputstride[1] +
                          tail_idx * inputstride[2];
            int n = accumulateNum;
            if (reduce_start + accumulateNum > inputshape[1])
                n = static_cast<int>(inputshape[1] - reduce_start);
            for (int i = 0; i < n; ++i)
            {
                v += input[idx];
                idx += inputstride[1]; // 归约维 stride = tail
            }
        }

        // 全 warp 参与：合并同一尾位置、不同 fold 的部分和
        for (int delta = tail_size; delta < 32; delta <<= 1)
            v += __shfl_down_sync(0xffffffffu, v, delta);

        if (fold_idx == 0 && tail_idx < inputshape[2])
        {
            int64_t out_idx = bx * outputstride[0] + tail_idx * outputstride[1];
            atomicAdd(&output[out_idx], v);
        }
    }

    template <typename T>
    graph::tensorPtr shortTailContigReduceCUDA(graph::tensorPtr t, const int64_t rdim)
    {
        int64_t ndim = static_cast<int64_t>(t->shape().size());
        if (rdim < 0 || rdim >= ndim)
            throw std::out_of_range("shortTailContigReduce: invalid rdim");
        if (rdim == ndim - 1)
            throw std::invalid_argument(
                "shortTailContigReduce: rdim should not be the last dim");

        // 标记归约维并合并连续非归约维（连续张量 + 中间归约 -> [outer, reduce, tail]）
        std::vector<bool> mask(ndim, false);
        mask[rdim] = true;
        auto coalesced = coalesce_dimensions(t->shape(), t->stride(), mask);
        base::s_vec shape = coalesced[0];
        while (shape.size() < 3)
            shape.insert(shape.begin(), 1);

        const int WARP_SIZE = 32;
        const int WARP_NUM = 32;
        const int WARP_ACCUMU_NUM = 256;

        int64_t outer = shape[0], reduce = shape[1], tail = shape[2];

        // tail_size = 最小 2 的幂 >= tail；fold2exp = log2(32/tail_size)
        int tail_size = 1, tail2exp = 0;
        while (tail_size < tail) { tail_size <<= 1; ++tail2exp; }
        int fold2exp = 5 - tail2exp;

        int warpCover = WARP_ACCUMU_NUM << fold2exp; // 每 warp 覆盖的归约元素数
        int blockCover = WARP_NUM * warpCover;
        int reduceBlockNum = static_cast<int>(
            (reduce + blockCover - 1) / blockCover);

        dim3 block(WARP_SIZE, WARP_NUM);
        dim3 grid(static_cast<unsigned>(outer),
                  static_cast<unsigned>(reduceBlockNum));

        // 输出形状 = 去掉 rdim
        base::s_vec out_shape;
        out_shape.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i)
            if (i != rdim)
                out_shape.push_back(t->shape()[i]);
        graph::tensorPtr r = std::make_shared<graph::TensorImpl>(
            out_shape, t->type(), t->device(), t->requires_grad());
        r->fill(0); // atomicAdd 累加前先清零

        const int64_t inputshape[3] = {outer, reduce, tail};
        const int64_t inputstride[3] = {reduce * tail, tail, 1};
        const int64_t outputstride[2] = {tail, 1};

        CHECK_KERNEL_FUNC(shortTailContigMidReduceKernel<T>
                          <<<grid, block>>>(
                              static_cast<T *>(r->data().data()),
                              static_cast<const T *>(t->data().data()),
                              inputshape, inputstride, outputstride,
                              fold2exp, WARP_ACCUMU_NUM));

        return r;
    }

    graph::tensorPtr shortTailContigReduce(graph::tensorPtr t, const int64_t rdim)
    {
        if (!t->data().is_continue())
            throw(std::invalid_argument("tensor must be continuous!"));

        graph::tensorPtr reduced;
        base::dispatch(t->type().scalar_type_, t->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               reduced = shortTailContigReduceCUDA<T>(t, rdim);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               reduced = smallContigReduceCPU<T>(t, rdim);
                           }
                           else
                               throw std::runtime_error(
                                   "shortTailContigReduce: Unkown Device");
                       });
        return reduced;
    }

    graph::tensorPtr SumOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
    }
}