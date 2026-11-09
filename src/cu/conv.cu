#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cstdint>

#include "tensor.cuh"
#include "sum.cuh"
#include "linear.cuh"
#include "conv.cuh"

namespace operation // 实现im2col
{

    // CPU 版 im2col：单个样本 X_n(C_in, H, W) -> Xcol(H*W, C_in*9)
    void im2colCPU(const float *x, int C_in, int H, int W, float *x_col)
    {
        const int KK = 3 * 3;
        for (int oh = 0; oh < H; oh++)
            for (int ow = 0; ow < W; ow++)
            {
                int row = oh * W + ow; // 输出位置
                for (int c = 0; c < C_in; c++)
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                        {
                            int ih = oh - 1 + kh; // padding=1
                            int iw = ow - 1 + kw;
                            float v = 0.0f; // zero padding
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                v = x[c * H * W + ih * W + iw];
                            x_col[row * (C_in * KK) + (c * KK + kh * 3 + kw)] = v;
                        }
            }
    }

    // CUDA版：一个线程处理一个输出位置
    __global__ void im2colKernel(const float *x, int C_in, int H, int W,
                                 float *x_col)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total = H * W * C_in * 9;
        if (idx >= total)
            return;

        // 如CPU函数中 row * (C_in * KK) + (c * KK + kh * 3 + kw)
        int kw = idx % 3;           // KK = 9
        int kh = (idx / 3) % 3;     // kw < 3
        int c = (idx / 9) % C_in;   // kh * 3 + kw < 9, c < C_in
        int row = idx / (9 * C_in); // row = oh*W + ow
        int oh = row / W;
        int ow = row % W;

        // 和CPU一样
        int ih = oh - 1 + kh; // padding = 1
        int iw = ow - 1 + kw;
        float v = 0.0f; // zero padding
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            v = x[(c * H + ih) * W + iw];

        x_col[row * (C_in * 9) + (c * 9 + kh * 3 + kw)] = v;
    }

    void img2colCUDA(const float *x, int C_in, int H, int W,
                     float *x_col)
    {
        int HW = H * W;
        int K = C_in * 9;
        int total = HW * K;
        int block = 256;
        int grid = (total + block - 1) / block;
        CHECK_KERNEL_FUNC(im2colKernel<<<grid, block>>>(x, C_in, H, W, x_col););
    }

    graph::tensorPtr img2col(graph::tensorPtr Ximg, int C_in, int H, int W, int n)
    { // n表示Ximg的第n个batch
        int HW = H * W;
        int K = C_in * 9;

        graph::tensorPtr Xcol = std::make_shared<graph::TensorImpl>(
            base::s_vec{HW, K}, Ximg->type(), Ximg->device(), false);
        float *xcol_ptr = static_cast<float *>(Xcol->data());
        const float *x_ptr = static_cast<const float *>(Ximg->data());
        const float *xn = x_ptr + static_cast<int64_t>(n) * C_in * H * W;

        if (Ximg->device() == base::Device::CUDA)
        {
            img2colCUDA(xn, C_in, H, W, xcol_ptr);
        }
        else
        {
            im2colCPU(xn, C_in, H, W, xcol_ptr);
        }
        return Xcol;
    }
}

namespace operation // 实现transpose：(H*W, C_out) 转置拷贝到 (C_out, H*W)
{

    void transposeCopyCPU(const float *src, float *dst, int C_out, int HW)
    {
        for (int co = 0; co < C_out; ++co)
            for (int hw = 0; hw < HW; ++hw)
                dst[co * HW + hw] = src[hw * C_out + co];
    }

    const size_t TILE_DIM = 32;
    __global__ void transposeCopyKernel(const float *src, float *dst,
                                        int C_out, int HW)
    {
        __shared__ float S[TILE_DIM][TILE_DIM + 1];
        int bx = blockIdx.x * TILE_DIM;
        int by = blockIdx.y * TILE_DIM;

        int nx1 = bx + threadIdx.x;
        int ny1 = by + threadIdx.y;

        if (nx1 < C_out && ny1 < HW)
        {
            S[threadIdx.y][threadIdx.x] = src[ny1 * C_out + nx1];
        }
        __syncthreads();

        int nx2 = bx + threadIdx.y;
        int ny2 = by + threadIdx.x;

        if (nx2 < C_out && ny2 < HW)
        {
            dst[nx2 * HW + ny2] = S[threadIdx.x][threadIdx.y];
        }
    }

    void transposeCopyCUDA(const float *src, float *dst,
                           int C_out, int HW)
    {
        dim3 grid((C_out + TILE_DIM - 1) / TILE_DIM, (HW + TILE_DIM - 1) / TILE_DIM);
        dim3 block(TILE_DIM, TILE_DIM);

        CHECK_KERNEL_FUNC(transposeCopyKernel<<<grid, block>>>(src, dst, C_out, HW););
    }

    void transposeCopy(graph::tensorPtr Yn, graph::tensorPtr Y, int C_out, int H, int W, int n)
    { // n是Yn在Y中的位置
        int HW = H * W;
        float *y_ptr = static_cast<float *>(Y->data());
        float *yn_ptr = static_cast<float *>(Yn->data());
        float *y_dst = y_ptr + static_cast<int64_t>(n) * C_out * HW;
        if (Y->device() == base::Device::CUDA)
        {
            transposeCopyCUDA(yn_ptr, y_dst, C_out, HW);
        }
        else
        {
            transposeCopyCPU(yn_ptr, y_dst, C_out, HW);
        }
    }
}

namespace operation // Conv 正向传播
{
    graph::tensorPtr Conv(graph::tensorPtr kernel, graph::tensorPtr X,
                          graph::tensorPtr b, bool bias, std::vector<graph::tensorPtr> &XcolSave)
    {
        // 输入合法性检查
        if (!X->is_continue())
            throw std::invalid_argument("Conv: X must be contiguous");
        if (X->shape().size() != 4)
            throw std::invalid_argument("Conv: X shape must be (N, C_in, H, W)");
        if (!kernel->is_continue())
            throw std::invalid_argument("Conv: kernel must be contiguous");
        if (kernel->shape().size() != 4 ||
            kernel->shape()[2] != 3 || kernel->shape()[3] != 3)
            throw std::invalid_argument("Conv: kernel shape must be (C_out, C_in, 3, 3)");
        if (X->device() != kernel->device() ||
            (bias && X->device() != b->device()))
            throw std::invalid_argument("Conv: tensors must be on the same device");

        int N = static_cast<int>(X->shape()[0]);
        int C_in = static_cast<int>(X->shape()[1]);
        int H = static_cast<int>(X->shape()[2]);
        int W = static_cast<int>(X->shape()[3]);
        int C_out = static_cast<int>(kernel->shape()[0]);
        if (kernel->shape()[1] != C_in)
            throw std::invalid_argument("Conv: kernel C_in does not match X");
        if (bias)
        {
            if (b->shape() != base::s_vec{C_out})
                throw std::invalid_argument("Conv: bias shape must be (C_out)");
            if (!b->is_continue())
                throw std::invalid_argument("Conv: bias must be contiguous");
        }

        // 把 kernel (C_out, C_in, 3, 3) 视为二维 (C_out, C_in*9)
        // kernel 连续存储，共享 storage
        base::s_vec w_shape{C_out, C_in * 9};
        graph::tensorPtr Wflat = std::make_shared<graph::TensorImpl>(kernel->view(w_shape));

        // (N, C_out, H, W)
        bool requires_grad = X->requires_grad() || kernel->requires_grad() || (bias && b->requires_grad());
        graph::tensorPtr Y = std::make_shared<graph::TensorImpl>(
            base::s_vec{N, C_out, H, W}, X->type(), X->device(), requires_grad);

        // 在 N 维上循环
        for (int n = 0; n < N; ++n)
        {
            // im2col: X[n] -> Xcol (H*W, C_in*9)
            auto Xcol = img2col(X, C_in, H, W, n);
            XcolSave.push_back(Xcol);

            graph::tensorPtr Yn = detail::GEMM(Xcol, Wflat, b, bias,
                                               1.0f, 1.0f, false, false);

            transposeCopy(Yn, Y, C_out, H, W, n);
        }

        return Y;
    }

    graph::tensorPtr ConvOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
        if (inputs.size() != ((bias_) ? 3 : 2))
            throw std::invalid_argument("Conv requires 3 inputs: Kernel, X , b (if bias)");

        graph::tensorPtr kernel = inputs[0], X = inputs[1],
                         b = (bias_ ? inputs[2] : nullptr);

        std::vector<graph::tensorPtr> XcolSave;
        XcolSave.reserve(X->shape()[0]);
        graph::tensorPtr Y = Conv(kernel, X, b, bias_, XcolSave);

        std::shared_ptr<ConvNode> node = makeNode(kernel, X, b, std::move(XcolSave), bias_);
        Y->autograd().grad_fn_ = node;

        return Y;
    }
}

namespace operation // 这个实现col2img
{
    // 与 im2col 对应：x (H*W, C_in*9)，
    // x_col (C_in, H, W)
    void col2imgCPU(const float *x, int C_in, int H, int W, float *x_col)
    {
        const int KK = 3 * 3;
        for (int c = 0; c < C_in; c++)
            for (int ih = 0; ih < H; ih++)
                for (int iw = 0; iw < W; iw++)
                {
                    float sum = 0.0f;
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                        {
                            // ih = oh - 1 + kh  =>  oh = ih + 1 - kh
                            int oh = ih + 1 - kh;
                            int ow = iw + 1 - kw;
                            if (oh >= 0 && oh < H && ow >= 0 && ow < W) // padding 丢弃
                            {
                                int row = oh * W + ow;
                                sum += x[row * (C_in * KK) + (c * KK + kh * 3 + kw)];
                                // 与x_col[row * (C_in * KK) + (c * KK + kh * 3 + kw)] = v 一致
                            }
                        }
                    x_col[c * H * W + ih * W + iw] = sum;
                    // 与v = x[c * H * W + ih * W + iw];一致
                }
    }

    // CUDA 版：每个线程负责一个图像像素 (c, ih, iw)的输出，剩下的和CPU没什么区别
    __global__ void col2imgKernel(const float *x, int C_in, int H, int W,
                                  float *x_col)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total = C_in * H * W;
        if (idx >= total)
            return;

        int iw = idx % W;
        int ih = (idx / W) % H;
        int c = idx / (W * H);

        float sum = 0.0f;
        for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
            {
                int oh = ih + 1 - kh;
                int ow = iw + 1 - kw;
                if (oh >= 0 && oh < H && ow >= 0 && ow < W)
                {
                    int row = oh * W + ow;
                    sum += x[row * (C_in * 9) + (c * 9 + kh * 3 + kw)];
                }
            }
        x_col[c * H * W + ih * W + iw] = sum;
    }

    void col2imgCUDA(const float *x, int C_in, int H, int W,
                     float *x_col)
    {
        int total = C_in * H * W;
        const int BLOCK_DIM = 256;
        int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
        CHECK_KERNEL_FUNC(col2imgKernel<<<grid, BLOCK_DIM>>>(x, C_in, H, W, x_col););
    }

    // 把第 n 个样本的列梯度 Xcol(H*W, C_in*9) 累加回图像梯度 Ximg(N, C_in, H, W) 的 [n] 处
    void col2img(graph::tensorPtr Xcol, graph::tensorPtr Ximg, int C_in, int H, int W, int n)
    {
        const float *x_ptr = static_cast<const float *>(Xcol->data());
        float *ximg_ptr = static_cast<float *>(Ximg->data());
        float *xn = ximg_ptr + static_cast<int64_t>(n) * C_in * H * W;

        if (Ximg->device() == base::Device::CUDA)
            col2imgCUDA(x_ptr, C_in, H, W, xn);
        else
            col2imgCPU(x_ptr, C_in, H, W, xn);
    }

}

namespace operation // 这个实现backward
{

    std::vector<graph::tensorPtr> ConvNode::backward()
    {
        // 准备参数
        auto prev = prev_grads_[0];
        int N = static_cast<int>(X->shape()[0]);
        int C_in = static_cast<int>(X->shape()[1]);
        int H = static_cast<int>(X->shape()[2]);
        int W = static_cast<int>(X->shape()[3]);
        int C_out = static_cast<int>(kernel->shape()[0]);

        base::s_vec kernel_shape{C_out, C_in * 9};
        graph::tensorPtr kernelFlat = std::make_shared<graph::TensorImpl>(kernel->view(kernel_shape));
        graph::tensorPtr kernelFlat_grad = std::make_shared<graph::TensorImpl>(
            kernel_shape, kernel->type(), kernel->device(), false);
        kernelFlat_grad->fill(0);

        graph::tensorPtr X_grad = (X->requires_grad())
                                      ? std::make_shared<graph::TensorImpl>(
                                            X->shape(), X->type(), X->device(), false)
                                      : nullptr;

        for (int n = 0; n < N; n++)
        {
            int64_t prev_offset = n * C_out * H * W;
            graph::tensorPtr prevn = std::make_shared<graph::TensorImpl>(
                std::make_shared<base::TensorPart>(
                    prev->part().storage(), prev->type(), base::s_vec{C_out, H * W},
                    base::s_vec{H * W, 1}, prev_offset),
                false);

            // 计算Wflat_grad
            if (kernel->requires_grad())
            {
                auto xcol = XcolSave_[n];
                auto kernelFlat_grad_n = detail::GEMM(
                    prevn, xcol, nullptr, false, 1.0f, 0.0f, false, true);

                kernelFlat_grad->accumulate(kernelFlat_grad_n);
            }

            // 计算X_grad_n
            if (X->requires_grad())
            {
                // ∂L/∂Xcol = prevn^T @ kernelFlat = (H*W, C_out) @ (C_out, C_in*9) = (H*W, C_in*9)
                auto xcoln_grad = detail::GEMM(
                    prevn, kernelFlat, nullptr, false, 1.0f, 0.0f, true, true);
                col2img(xcoln_grad, X_grad, C_in, H, W, n);
            }
        }

        base::s_vec w_kernel_shape{C_out, C_in, 3, 3};
        graph::tensorPtr kernel_grad = (kernel->requires_grad())
                                           ? std::make_shared<graph::TensorImpl>(
                                                 kernelFlat_grad->view(w_kernel_shape))
                                           : nullptr;

        if (bias_)
        {
            base::s_vec prev_flat_shape = {prev->shape()[0], C_out, H * W};
            auto prev_flat_grad =
                std::make_shared<graph::TensorImpl>(prev->view(prev_flat_shape));
            auto b_batch_grad = detail::contigReduce(prev_flat_grad, -1);
            auto b_grad = detail::contigReduce(b_batch_grad, 0);
            return {kernel_grad, X_grad, b_grad};
        }
        else
            return {kernel_grad, X_grad};
    }
}