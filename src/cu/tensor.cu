#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <set>
#include <algorithm>

#include <tensor.cuh>

namespace base
{
    int TypeMeta::get_size(ScalarType type_)
    { // get size from type
        switch (type_)
        {
        case ScalarType::Byte:
            return 1;
            break;
        case ScalarType::Char:
            return 1;
            break;
        case ScalarType::Short:
            return 2;
            break;
        case ScalarType::Int:
            return 4;
            break;
        case ScalarType::Long:
            return 8;
            break;
        case ScalarType::Float:
            return 4;
            break;
        case ScalarType::Double:
            return 8;
            break;
        case ScalarType::Bool:
            return 1;
            break;
        default:
            throw std::invalid_argument("Unknown ScalarType");
        }
    }

    Storage::Storage(const Storage &s, Device dvc)
    {
        if (dvc == s.device())
        {
            impl_ = s.impl_;
            return;
        }
        if (dvc == Device::CPU && s.device() == Device::CUDA)
        {
            void *ptr = malloc(s.nbytes());
            CHECK(cudaMemcpy(ptr, s.data(), s.nbytes(), cudaMemcpyDeviceToHost));
            impl_ = std::make_shared<StoragePart>(s.nbytes(), ptr, ptr, dvc);
            return;
        }
        else if (dvc == Device::CUDA && s.device() == Device::CPU)
        {
            void *ptr;
            CHECK(cudaMalloc(&ptr, s.nbytes()));
            CHECK(cudaMemcpy(ptr, s.data(), s.nbytes(), cudaMemcpyHostToDevice));
            impl_ = std::make_shared<StoragePart>(s.nbytes(), ptr, ptr, dvc);
            return;
        }
        else
            throw(std::invalid_argument("Wrong Device"));
    }

    Storage TensorPart::makeFromShape(int64_t size, Device dvc, TypeMeta tp)
    {
        void *ptr;
        if (dvc == Device::CPU)
        {
            ptr = malloc(size * tp.itemsize_);
            if (!ptr)
                throw std::bad_alloc();
        }
        else if (dvc == Device::CUDA)
            CHECK(cudaMalloc(&ptr, size * tp.itemsize_));
        else
            throw std::invalid_argument("Unknown Device");
        return Storage(size * tp.itemsize_, ptr, ptr, dvc);
    }

    template <typename T>
    void fill_cpu(void *t, float f, int numel_)
    {
        T *t_ = static_cast<T *>(t);
        for (int i = 0; i < numel_; i++)
        {
            t_[i] = static_cast<T>(f);
        }
    }

    template <typename T>
    __global__ void fill_cuda_kernel(T *t, T f, int numel_)
    {
        int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numel_)
            t[i] = f;
    }

    template <typename T>
    void fill_cuda(void *t, float f, int numel_)
    {
        T f_ = static_cast<T>(f);
        T *t_ = static_cast<T *>(t);

        static const int BLOCK_SIZE = 32 * 4;
        int blockNum = (numel_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CHECK_KERNEL_FUNC(fill_cuda_kernel<T>
                          <<<blockNum, BLOCK_SIZE>>>(t_, f_, numel_));
    }

    void TensorPart::fill(float f)
    {
        dispatch(
            this->type_.scalar_type_, this->device(), [&]<typename T, Device Dev>()
            {
            if constexpr(Dev==Device::CPU)
            fill_cpu<T>(this->data(), f, numel_);
            else
            fill_cuda<T>(this->data(),f,numel_); });
    }

    int64_t TensorPart::numel_count(const s_vec &shape)
    {
        int64_t size = 1;
        for (auto x : shape)
        {
            if (x < 0)
                throw std::invalid_argument("Negative dim");
            size *= x;
        }
        return size;
    }

    TensorPart TensorPart::permute(const std::vector<int8_t> &dimOrder)
    {
        int dimNum = dimOrder.size();
        if (dimNum != shape_.size())
            throw(std::invalid_argument("Invalid Permute"));
        s_vec shape_new = s_vec(dimNum, 0);
        s_vec strides_new = s_vec(dimNum, 0);
        auto visited = std::vector<bool>(dimNum, false);
        for (size_t i = 0; i < dimOrder.size(); ++i)
        {
            int8_t x = dimOrder[i];
            if (x < 0 || x >= dimNum || visited[x])
                throw(std::invalid_argument("Invalid Permute"));
            shape_new[i] = shape_[x];
            strides_new[i] = strides_[x];
            visited[x] = true;
        }
        return TensorPart(strg_, type_, std::move(shape_new),
                          std::move(strides_new), storage_offset_);
    }

    void TensorPart::cpu_continuous_copy(void *src, void *target,
                                         int64_t numel, int64_t itemsize,
                                         const s_vec &shape, const s_vec &s_stride,
                                         const s_vec &t_stride)
    {
        char *d = static_cast<char *>(target);
        const char *s = static_cast<const char *>(src);
        s_vec idx(shape.size(), 0);

        for (int64_t n = 0; n < numel; ++n)
        {
            int64_t src_off = 0, dst_off = 0;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                src_off += idx[i] * s_stride[i];
                dst_off += idx[i] * t_stride[i];
            }
            std::memcpy(d + dst_off * itemsize, s + src_off * itemsize, itemsize);

            // carry
            for (int64_t i = shape.size() - 1; i >= 0; --i)
            {
                if (++idx[i] < shape[i])
                    break;
                idx[i] = 0;
            }
        }
    }

    // Addition
    template <typename T>
    void add_cpu_tmplt(void *result, const void *mul1, const void *add2, int numel_)
    {
        const T *a1 = static_cast<const T *>(mul1);
        const T *a2 = static_cast<const T *>(add2);
        T *r = static_cast<T *>(result);
        for (int i = 0; i < numel_; i++)
            r[i] = a1[i] + a2[i];
    }

    template <typename T>
    __global__ void add_cuda_kernel(T *result, const T *mul1, const T *add2, int numel_)
    {
        int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numel_)
            result[i] = mul1[i] + add2[i];
    }

    template <typename T>
    void add_cuda_tmplt(void *result, const void *mul1, const void *add2, int numel_)
    {
        static const int BLOCK_SIZE = 32 * 4;
        int blockNum = (numel_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

        const T *a1 = static_cast<const T *>(mul1);
        const T *a2 = static_cast<const T *>(add2);
        T *r = static_cast<T *>(result);

        CHECK_KERNEL_FUNC(add_cuda_kernel<T>
                          <<<blockNum, BLOCK_SIZE>>>(r, a1, a2, numel_));
    }

    TensorPart TensorPart::operator+(const TensorPart &other) const
    {
        if (shape_ != other.shape_)
            throw(std::invalid_argument("Can't add two tensors in different shape"));

        TensorPart result(shape_, type_, this->device());

        dispatch(
            this->type_.scalar_type_, this->device(), [&]<typename T, Device Dev>()
            {
            if constexpr(Dev==Device::CPU)
            add_cpu_tmplt<T>(result.data(), this->data(), other.data(), numel_);
            else
            add_cuda_tmplt<T>(result.data(), this->data(), other.data(), numel_); });

        return result;
    } // Addition

    // self addition

    template <typename T>
    void self_add_cpu(void *self, const void *other, int numel_)
    {
        T *s = static_cast<T *>(self);
        const T *o = static_cast<const T *>(other);
        for (int i = 0; i < numel_; i++)
            s[i] += o[i];
    }

    template <typename T>
    __global__ void self_add_cuda_kernel(T *self, const T *other, int numel_)
    {
        int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numel_)
            self[i] += other[i];
    }

    template <typename T>
    void self_add_cuda(void *self, const void *other, int numel_)
    {
        static const int BLOCK_SIZE = 32 * 4;
        int blockNum = (numel_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

        T *s = static_cast<T *>(self);
        const T *o = static_cast<const T *>(other);

        CHECK_KERNEL_FUNC(self_add_cuda_kernel<T>
                          <<<blockNum, BLOCK_SIZE>>>(s, o, numel_);)
    }

    void TensorPart::operator+=(const TensorPart &other)
    {
        if (shape_ != other.shape_)
            throw(std::invalid_argument("Can't add two tensors in different shape"));

        dispatch(
            this->type_.scalar_type_, this->device(), [&]<typename T, Device Dev>()
            {
            if constexpr(Dev==Device::CPU)
            self_add_cpu<T>(this->data(), other.data(), numel_);
            else
            self_add_cuda<T>(this->data(), other.data(), numel_); });

    } // Addition

    // Multiplication
    template <typename T>
    void mul_cpu_tmplt(void *result, const void *mul1, const void *mul2, int numel_)
    {
        const T *m1 = static_cast<const T *>(mul1);
        const T *m2 = static_cast<const T *>(mul2);
        T *r = static_cast<T *>(result);
        for (int i = 0; i < numel_; i++)
            r[i] = m1[i] * m2[i];
    }

    template <typename T>
    __global__ void mul_cuda_kernel(T *result, const T *mul1, const T *mul2, int numel_)
    {
        int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numel_)
            result[i] = mul1[i] * mul2[i];
    }

    template <typename T>
    void mul_cuda_tmplt(void *result, const void *mul1, const void *mul2, int numel_)
    {
        static const int BLOCK_SIZE = 32 * 4;
        int blockNum = (numel_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

        const T *m1 = static_cast<const T *>(mul1);
        const T *m2 = static_cast<const T *>(mul2);
        T *r = static_cast<T *>(result);

        CHECK_KERNEL_FUNC(mul_cuda_kernel<T>
                          <<<blockNum, BLOCK_SIZE>>>(r, m1, m2, numel_));
    }

    TensorPart TensorPart::operator*(const TensorPart &other) const
    {
        if (shape_ != other.shape_)
            throw(std::invalid_argument("Can't mul two tensors in different shape"));

        TensorPart result(shape_, type_, this->device());

        dispatch(
            this->type_.scalar_type_, this->device(), [&]<typename T, Device Dev>()
            {
            if constexpr(Dev==Device::CPU)
            mul_cpu_tmplt<T>(result.data(), this->data(), other.data(), numel_);
            else
            mul_cuda_tmplt<T>(result.data(), this->data(), other.data(), numel_); });

        return result;
    } // Multiplication

} // namespace base

namespace graph
{
    // ---------- IndexProxy 与 print 的实现 ----------

    TensorImpl::IndexProxy TensorImpl::operator[](int64_t i)
    {
        const auto &shape = data().shape();
        if (shape.empty())
            throw std::out_of_range("Empty tensor");
        if (i < 0 || i >= shape[0])
            throw std::out_of_range("Index out of range");
        int64_t off = i * data().strides()[0] * data().type().itemsize_;
        return IndexProxy(this, off, 1);
    }

    TensorImpl::IndexProxy TensorImpl::IndexProxy::operator[](int64_t i) const
    {
        const auto &shape = t_->data().shape();
        int64_t ndim = static_cast<int64_t>(shape.size());
        if (dim_ >= ndim)
            throw std::out_of_range("Too many indices");
        if (i < 0 || i >= shape[dim_])
            throw std::out_of_range("Index out of range");
        int64_t off = byte_offset_ +
                      i * t_->data().strides()[dim_] * t_->data().type().itemsize_;
        return IndexProxy(t_, off, dim_ + 1);
    }

    float TensorImpl::IndexProxy::get() const
    {
        const base::TensorPart &tp = t_->data();
        int64_t ndim = static_cast<int64_t>(tp.shape().size());
        if (dim_ != ndim)
            throw std::out_of_range(
                "Partial index: call slice() for a view, or index all dims for an element");

        const void *src = static_cast<const char *>(tp.data()) + byte_offset_;
        float tmp = 0.0f;
        if (tp.device() == base::Device::CPU)
            tmp = *static_cast<const float *>(src);
        else
            CHECK(cudaMemcpy(&tmp, src, sizeof(float), cudaMemcpyDeviceToHost));
        return tmp;
    }

    void TensorImpl::IndexProxy::set(float v) const
    {
        base::TensorPart &tp = t_->data();
        int64_t ndim = static_cast<int64_t>(tp.shape().size());
        if (dim_ != ndim)
            throw std::out_of_range(
                "Partial index: call slice() for a view, or index all dims for an element");

        void *dst = static_cast<char *>(tp.data()) + byte_offset_;
        if (tp.device() == base::Device::CPU)
            *static_cast<float *>(dst) = v;
        else
            CHECK(cudaMemcpy(dst, &v, sizeof(float), cudaMemcpyHostToDevice));
    }

    TensorImpl TensorImpl::IndexProxy::slice() const
    {
        const base::TensorPart &tp = t_->data();
        const base::s_vec &shape = tp.shape();
        const base::s_vec &strides = tp.strides();
        int64_t ndim = static_cast<int64_t>(shape.size());
        if (dim_ >= ndim)
            throw std::out_of_range("Cannot slice a fully-indexed (scalar) position");

        base::s_vec new_shape(shape.begin() + dim_, shape.end());
        base::s_vec new_strides(strides.begin() + dim_, strides.end());
        int64_t itemsize = tp.type().itemsize_;
        int64_t new_offset = tp.storage_offset() + byte_offset_ / itemsize;

        auto part = std::make_shared<base::TensorPart>(
            tp.storage(), tp.type(), std::move(new_shape), std::move(new_strides), new_offset);
        // 无 grad 的视图
        return TensorImpl(part, /*requires_grad=*/false);
    }

    // print 辅助：递归打印（按 shape 嵌套方括号，逗号分隔）
    namespace
    {
        void print_rec(const float *base, const base::s_vec &shape,
                       const base::s_vec &strides, int64_t storage_offset,
                       base::s_vec &idx, int64_t dim)
        {
            std::cout << "[";
            for (int64_t i = 0; i < shape[dim]; ++i)
            {
                idx[dim] = i;
                if (dim + 1 == static_cast<int64_t>(shape.size()))
                {
                    int64_t off = storage_offset;
                    for (size_t d = 0; d < shape.size(); ++d)
                        off += idx[d] * strides[d];
                    std::cout << base[off];
                }
                else
                {
                    print_rec(base, shape, strides, storage_offset, idx, dim + 1);
                }
                if (i + 1 < shape[dim])
                    std::cout << ", ";
            }
            std::cout << "]";
        }
    } // namespace

    void TensorImpl::print() const
    {
        const base::TensorPart &tp = *data_;
        const base::s_vec &shape = tp.shape();
        const base::s_vec &strides = tp.strides();
        int64_t storage_offset = tp.storage_offset();
        int64_t nbytes = tp.storage().nbytes();
        bool is_cuda = (tp.device() == base::Device::CUDA);

        // 材质化 host 端原始存储（区分设备）
        void *host_buf = nullptr;
        const float *base = nullptr;
        if (is_cuda)
        {
            host_buf = malloc(nbytes);
            CHECK(cudaMemcpy(host_buf, tp.storage().data(), nbytes, cudaMemcpyDeviceToHost));
            base = static_cast<const float *>(host_buf);
        }
        else
        {
            base = static_cast<const float *>(tp.storage().data());
        }

        if (shape.empty())
        {
            // 0 维标量
            std::cout << base[storage_offset] << std::endl;
        }
        else
        {
            base::s_vec idx(shape.size(), 0);
            print_rec(base, shape, strides, storage_offset, idx, 0);
            std::cout << std::endl;
        }

        if (is_cuda)
            free(host_buf);
    }

} // namespace graph

namespace graph
{
    // 其余graph功能实现

    // Node::accumulate_grads —— 把梯度累加到 prev_grads_[idx]（支持同一张量被多次使用）
    void Node::accumulate_grads(tensorPtr prev, int idx)
    {
        if (!prev_grads_[idx])
            prev_grads_[idx] = prev;
        else
            prev_grads_[idx]->accumulate(prev);
    }

    // AccumulateGrad::backward —— 把入向梯度正确写回叶子 .grad_
    std::vector<tensorPtr> AccumulateGrad::backward()
    {
        if (!t_->requires_grad() || !prev_grads_[0])
            return {};
        tensorPtr &grad = t_->autograd().grad_;
        if (!grad)
            grad = prev_grads_[0];
        else
            grad->accumulate(prev_grads_[0]);
        return {};
    }

    // Engine::count_node_dfs —— 统计每个节点的入度（父边数），用 visited 去重避免 DAG 重复遍历
    void Engine::count_node_dfs(std::shared_ptr<Node> n)
    {
        nodes_count.clear();
        if (!n) // 叶子 / 未接图，无入度可统计
            return;
        std::stack<std::shared_ptr<Node>> stack;
        std::set<std::shared_ptr<Node>> seen;
        stack.push(n);
        seen.insert(n);
        while (!stack.empty())
        {
            std::shared_ptr<Node> node = stack.top();
            stack.pop();
            for (auto &next_node : node->next_edges())
            {
                nodes_count[next_node.first] += 1;       // 每条父边使 child 入度 +1
                if (seen.insert(next_node.first).second) // 首次见到才入栈
                    stack.push(next_node.first);
            }
        }
    }

    // Engine::backward —— 拓扑序反向传播；去掉重复默认实参；用 accumulate_grads 累加路由
    void Engine::backward(tensorPtr t, tensorPtr grad)
    {
        std::shared_ptr<Node> root = t->autograd().grad_fn_;
        if (!root) // 叶子张量或未接入计算图，无梯度可传播
            return;

        if (!grad)
        {
            grad = std::make_shared<TensorImpl>(
                t->shape(), t->type(), t->device(), false);
            grad->fill(1);
        }

        count_node_dfs(root);

        std::stack<std::shared_ptr<Node>> stack;
        root->accumulate_grads(grad, 0); // 根节点输出梯度（种子），入度天然为 0
        stack.push(root);

        while (!stack.empty())
        {
            std::shared_ptr<Node> node = stack.top();
            stack.pop();

            int edges_num = node->edges_num();
            std::vector<tensorPtr> grads = node->backward();
            if (grads.empty())
                continue;

            for (size_t i = 0; i < edges_num && i < edges_num; ++i)
            {
                graph::edge &edge = node->getEdge(i);
                std::shared_ptr<Node> child = edge.first;
                child->accumulate_grads(grads[i], edge.second);
                if (--nodes_count[child] == 0)
                    stack.push(child);
            }
        }
    }
}

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

        return {new_shape, new_stride, new_masked_dims_idx};
    }

    template <typename T>
    graph::tensorPtr smallDimReduceCPU(graph::tensorPtr t, const int64_t rdim)
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

    // smallDimReduceKernel —— 单维归约：每个线程负责一个输出位置，串行累加归约维。
    // 工作在合并后的维度上（input_dim/input_stride/rdim 均为 coalesce 之后的结果）。
    const int MAX_SMALL_DIM_SIZE = 8;
    struct SmallReduceKernelDimStride
    { // 用于向kernel传参，当维度个数大于8时，默认不使用smallDimReduce
        int64_t input_dim[MAX_SMALL_DIM_SIZE];
        int64_t input_stride[MAX_SMALL_DIM_SIZE];
        int64_t output_stride[MAX_SMALL_DIM_SIZE];
    };
    template <typename T>
    __global__ void smallDimReduceKernel(T *output, const T *input,
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

    // smallDimReduceCUDA —— 单维归约入口：先合并连续维，再按标量类型派发内核
    template <typename T>
    graph::tensorPtr smallDimReduceCUDA(graph::tensorPtr t, const int64_t rdim)
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
        CHECK_KERNEL_FUNC(smallDimReduceKernel<T>
                          <<<grid_size, BLOCK_SIZE>>>(
                              static_cast<T *>(reduced->data().data()),
                              static_cast<const T *>(t->data().data()),
                              out_numel, s, new_rdim,
                              static_cast<int>(new_shape.size())));

        return reduced;
    }

    graph::tensorPtr smallDimReduce(graph::tensorPtr t, const int64_t rdim)
    {
        graph::tensorPtr reduced;
        base::dispatch(t->type().scalar_type_, t->device(),
                       [&]<typename T, base::Device Dev>()
                       {
                           if constexpr (Dev == base::Device::CUDA)
                           {
                               reduced = smallDimReduce(t, rdim);
                           }
                           else if constexpr (Dev == base::Device::CPU)
                           {
                               reduced = smallDimReduceCPU(t, rdim);
                           }
                           else
                               throw std::runtime_error(
                                   "smallDimReduce: Unkown Device");
                       });
    }

    graph::tensorPtr SumOpr::operator()(std::vector<graph::tensorPtr> inputs)
    {
    }
}