#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cstdint>

#include "tensor.cuh"
#include "sum.cuh"
#include "linear.cuh"

namespace operation
{
    //   transposeA=false: A 存为 [M,K]，op(A)=A；transposeA=true: A 存为 [K,M]，op(A)=A^T。
    //   transposeB=false: B 存为 [N,K]，op(B)=B^T；transposeB=true: B 存为 [K,N]，op(B)=B。
    template <typename T>
    graph::tensorPtr GEMMCPU(graph::tensorPtr A, graph::tensorPtr B,
                             graph::tensorPtr C, bool bias,
                             float alpha, float beta, base::s_vec &sOut,
                             bool transposeA, bool transposeB)
    {
        bool requires_grad = A->requires_grad() || B->requires_grad() ||
                             (bias && C->requires_grad());
        graph::tensorPtr D = std::make_shared<graph::TensorImpl>(
            sOut, A->type(), A->device(), requires_grad);
        D->fill(0);

        const T *a = static_cast<const T *>(A->data().data());
        const T *b = static_cast<const T *>(B->data().data());
        T *d = static_cast<T *>(D->data().data());

        int64_t M = sOut[0], N = sOut[1];
        int64_t K = transposeA ? A->shape()[0] : A->shape()[1];
        int64_t aCols = A->shape()[1], bCols = B->shape()[1];

        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j)
            {
                T sum = 0;
                for (int64_t k = 0; k < K; ++k)
                {
                    T av = transposeA ? a[k * aCols + i] : a[i * aCols + k];
                    T bv = transposeB ? b[k * bCols + j] : b[j * bCols + k];
                    sum += av * bv;
                }
                d[i * N + j] = alpha * sum;
            }

        // 按 C->numel() 取模广播
        if (bias)
        {
            const T *c = static_cast<const T *>(C->data().data());
            for (int64_t i = 0; i < D->numel(); ++i)
                d[i] += beta * c[i % C->numel()];
        }

        return D;
    }

    // 加 bias 的 CUDA kernel
    __global__ void addBiasKernel(float *d, const float *c, int64_t numel, int64_t cNumel, float beta)
    {
        int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= numel)
            return;
        d[idx] += beta * c[idx % cNumel];
    }

    template <typename T>
    graph::tensorPtr GEMMCUDA(graph::tensorPtr A, graph::tensorPtr B,
                              graph::tensorPtr C, bool bias,
                              float alpha, float beta,
                              base::s_vec s1, base::s_vec s2, base::s_vec sOut,
                              bool transposeA, bool transposeB)
    {
        int64_t M = sOut[0], N = sOut[1];
        int64_t K = transposeA ? A->shape()[0] : A->shape()[1];

        bool requires_grad = A->requires_grad() || B->requires_grad() ||
                             (bias && C->requires_grad());
        graph::tensorPtr D = std::make_shared<graph::TensorImpl>(
            sOut, A->type(), A->device(), requires_grad);

        const float *a = static_cast<const float *>(A->data().data());
        const float *b = static_cast<const float *>(B->data().data());
        float *d = static_cast<float *>(D->data().data());

        // 行主序 D = op(A) @ op(B)，列主序 D^T = op(B)^T @ op(A)^T
        cublasOperation_t transa = transposeB ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transb = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
        int lda = static_cast<int>(B->shape()[1]); // 第一操作数 B 的 leading dim
        int ldb = static_cast<int>(A->shape()[1]); // 第二操作数 A 的 leading dim

        cublasHandle_t handle;
        cublasCreate(&handle);
        float beta0 = 0.0f;
        // 先算矩阵乘法
        cublasSgemm(handle, transa, transb,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, b, lda, a, ldb, //( A @ B )^T = B^T @ A^T，以此欺骗gemm
                    &beta0, d, static_cast<int>(N));
        cublasDestroy(handle);

        // 如果加 bias：D += beta * C
        if (bias)
        {
            const float *c = static_cast<const float *>(C->data().data());
            int64_t numel = D->numel();
            int64_t cNumel = C->numel();
            int block = 256;
            int grid = static_cast<int>((numel + block - 1) / block);
            if (grid > 0)
            {
                CHECK_KERNEL_FUNC(addBiasKernel<<<grid, block>>>(d, c, numel, cNumel, beta););
            }
        }

        return D;
    }

    namespace detail
    {
        graph::tensorPtr GEMM(graph::tensorPtr A, graph::tensorPtr B,
                              graph::tensorPtr C, bool bias,
                              float alpha, float beta,
                              bool transposeA, bool transposeB)
        {
            if (bias && C == nullptr)
                throw(std::invalid_argument("bias tensor is lost"));

            auto dvc = A->device();
            if (dvc != B->device())
                throw(std::invalid_argument("Must be on same Device!"));
            if (bias && dvc != C->device())
                throw(std::invalid_argument("Must be on same Device!"));

            if (!A->data().is_continue() || !B->data().is_continue())
                throw(std::invalid_argument("Tensor Not Continue."));
            if (bias && !C->data().is_continue())
                throw(std::invalid_argument("Tensor Not Continue."));

            auto Ashape = A->shape(), Bshape = B->shape();
            base::s_vec Cshape = bias ? C->shape() : base::s_vec{};
            int64_t reduceA, reduceB;

            if (transposeA)
            {
                if (Ashape.size() != 2)
                    throw(std::invalid_argument("Transpose can only work on 2d tensor."));
                reduceA = A->shape().front();
            }
            else
            {
                reduceA = A->shape().back();
            }

            if (transposeB)
            {
                if (Bshape.size() != 2)
                    throw(std::invalid_argument("Transpose can only work on 2d tensor."));
                reduceB = B->shape().front();
            }
            else
            {
                reduceB = B->shape().back();
            }

            if (reduceA != reduceB)
                throw(std::invalid_argument("A and B is not Multiplable"));

            auto s1 = Ashape, s2 = Bshape;
            if (transposeA)
                s1.erase(s1.begin());
            else
                s1.pop_back();
            if (transposeB)
                s2.erase(s2.begin());
            else
                s2.pop_back();

            base::s_vec sOut;
            sOut.reserve(s1.size() + s2.size());
            sOut.insert(sOut.end(), s1.begin(), s1.end());
            sOut.insert(sOut.end(), s2.begin(), s2.end());
            if (bias && Cshape != sOut && Cshape != s2)
                throw(std::invalid_argument("C can not be added with A@B"));

            graph::tensorPtr output;
            base::dispatch(A->type().scalar_type_, dvc,
                           [&]<typename T, base::Device Dev>()
                           {
                               if constexpr (Dev == base::Device::CPU)
                                   output = GEMMCPU<T>(A, B, C, bias, alpha, beta, sOut, transposeA, transposeB);
                               else if constexpr (Dev == base::Device::CUDA)
                                   output = GEMMCUDA<T>(A, B, C, bias, alpha, beta, s1, s2, sOut, transposeA, transposeB);
                               else
                                   throw std::runtime_error("Unknown Device");
                           });
            return output;
        }
    }
}

namespace operation
{
    // LinearOpr::operator() —— Y = X @ W^T + b前向传播
    // inputs = [X, W, b]；transposeB=false 表示 W 存为 [out_features, in_features]，算 X @ W^T。
    graph::tensorPtr LinearOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
        if (inputs.size() != ((bias_) ? 3 : 2))
            throw std::invalid_argument("Linear requires 3 inputs: X, W, b");

        graph::tensorPtr X = inputs[0], W = inputs[1],
                         b = (bias_ ? inputs[2] : nullptr);

        graph::tensorPtr Y = detail::GEMM(X, W, b, bias_, 1.0f, 1.0f,
                                          false, false);

        std::shared_ptr<LinearNode> node = makeNode(W, X, b, bias_);
        Y->autograd().grad_fn_ = node;

        return Y;
    }
}

namespace operation
{
    std::vector<graph::tensorPtr> LinearNode::backward()
    {
        auto prev = prev_grads_[0];
        if (prev == nullptr)
            throw(std::invalid_argument("No grad to be backward"));

        graph::tensorPtr W_grad = nullptr, X_grad = nullptr, b_grad = nullptr;

        if (W->requires_grad())
        {
            W_grad = detail::GEMM(prev, X, nullptr, false, 1.0, 0.0, true, true);
        }

        if (X->requires_grad())
        {
            X_grad = detail::GEMM(prev, W, nullptr, false, 1.0, 0.0, false, true);
        }

        if (bias_)
        {
            b_grad = detail::contigReduce(prev, 0);
        }

        return {W_grad, X_grad, b_grad};
    }
}