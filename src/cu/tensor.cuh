#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include <map>
#include <array>
#include <cstdlib>
#include <optional>

#include "error.cuh"

namespace base
{
    // This vector is widely used, especially for shape and stride;
    using s_vec = std::vector<int64_t>;
    // Uniform Destructor
    using DeleterFnPtr = void (*)(void *);
    enum class Device : int8_t // enum device
    {
        CPU = 0,
        CUDA,
    };

    enum class ScalarType : int8_t // enum types
    {
        Byte = 0,
        Char,
        Short,
        Int,
        Long,
        Float,
        Double,
        Bool,
    };

    // do switch on ScalarType and send C++ type to callable
    template <typename F>
    void dispatch_scalar_type(ScalarType t, F &&f)
    {
        switch (t)
        {
        case ScalarType::Bool:
            f.template operator()<bool>();
            return;
        case ScalarType::Byte:
            f.template operator()<char>();
            return;
        case ScalarType::Short:
            f.template operator()<int16_t>();
            return;
        case ScalarType::Int:
            f.template operator()<int32_t>();
            return;
        case ScalarType::Long:
            f.template operator()<int64_t>();
            return;
        case ScalarType::Float:
            f.template operator()<float>();
            return;
        case ScalarType::Double:
            f.template operator()<double>();
            return;
        }
        throw std::invalid_argument("Unknown ScalarType");
    }

    template <typename F>
    void dispatch(ScalarType t, Device d, F &&f)
    {
        if (d == Device::CPU)
        {
            dispatch_scalar_type(t, [&]<typename T>()
                                 { f.template operator()<T, Device::CPU>(); });
        }
        else if (d == Device::CUDA)
        {
            dispatch_scalar_type(t, [&]<typename T>()
                                 { f.template operator()<T, Device::CUDA>(); });
        }
        else
        {
            throw std::invalid_argument("Unknown Device");
        }
    }

    struct TypeMeta // arrange types and its size
    {
        int32_t itemsize_;
        ScalarType scalar_type_;
        TypeMeta(ScalarType type) : itemsize_(get_size(type)), scalar_type_(type) {}

    private:
        static int32_t get_size(ScalarType type_); // get size from type
    };

    class DataUniquePtr
    // Managing low-level memory
    {
    private:
        void *data_;
        std::unique_ptr<void, DeleterFnPtr> ctx_;

    public:
        DataUniquePtr(void *data, void *ctx, DeleterFnPtr del)
            : data_(data), ctx_(ctx, del) {}
        void *data() { return data_; }
        void *data() const { return data_; }
        // It is not copiable but movable.
        DataUniquePtr(const DataUniquePtr &) = delete;
        DataUniquePtr &operator=(const DataUniquePtr &) = delete;
        DataUniquePtr(DataUniquePtr &&ptr) noexcept = default;
        DataUniquePtr &operator=(DataUniquePtr &&) noexcept = default;
    };

    class DataPtr
    // manage device and DataUniquePtr
    {
    private:
        DataUniquePtr ptr_;
        Device device_;

    public:
        DataPtr(const DataPtr &) = delete;
        DataPtr &operator=(const DataPtr &) = delete;
        DataPtr(DataPtr &&) noexcept = default;
        DataPtr &operator=(DataPtr &&) noexcept = default;
        static void cpu_deleter(void *ctx) { free(ctx); }
        static void cuda_deleter(void *ctx) { CHECK(cudaFree(ctx)); }
        static DeleterFnPtr get_deleter(Device dvc)
        { // use different destructor on deferent device
            switch (dvc)
            {
            case Device::CPU:
                return cpu_deleter;
            case Device::CUDA:
                return cuda_deleter;
            default:
                throw std::invalid_argument("Unknown Device");
            }
        }
        DataPtr(void *data, void *ctx, Device dvc)
            : ptr_(data, ctx, get_deleter(dvc)), device_(dvc) {}
        void *data() { return ptr_.data(); }
        void *data() const { return ptr_.data(); }
        Device device() const { return device_; }
    };

    class StoragePart
    // unable tobe copied or moved
    {
    private:
        int64_t nbytes_;
        DataPtr ptr_;

    public:
        StoragePart(int64_t nbytes, void *data, void *ctx, Device dvc)
            : nbytes_(nbytes), ptr_(data, ctx, dvc) {}

        void *data() { return ptr_.data(); }
        const void *data() const { return ptr_.data(); }
        Device device() const { return ptr_.device(); }
        int64_t nbytes() const { return nbytes_; }
    };

    class Storage
    // copiable but movable
    {
    private:
        std::shared_ptr<StoragePart> impl_;

    public:
        Storage(int64_t nbytes, void *data, void *ctx, Device dvc)
            : impl_(std::make_shared<StoragePart>(nbytes, data, ctx, dvc)) {}

        // View:made from exit Storage,share the same memory
        explicit Storage(std::shared_ptr<StoragePart> impl)
            : impl_(std::move(impl)) {}
        Storage(const Storage &s, Device dvc);
        void *data() { return impl_->data(); }
        const void *data() const { return impl_->data(); }
        Device device() const { return impl_->device(); }
        int64_t nbytes() const { return impl_->nbytes(); }

        // copiable,count += 1
        Storage(const Storage &) = default;
        Storage &operator=(const Storage &) = default;
        Storage(Storage &&) noexcept = default;
        Storage &operator=(Storage &&) noexcept = default;

        std::shared_ptr<StoragePart> impl() const { return impl_; }
    };

    class TensorPart
    {
    private:
        int64_t numel_;
        Storage strg_;
        TypeMeta type_;
        s_vec strides_;
        s_vec shape_;
        int64_t storage_offset_ = 0;
        bool canViewByView(s_vec from, s_vec to);
        static Storage makeFromShape(int64_t size, Device dvc, TypeMeta tp);
        static int64_t numel_count(const s_vec &shape);
        static void cpu_continuous_copy(void *src, void *target,
                                        int64_t numel, int64_t itemsize,
                                        const s_vec &shape, const s_vec &s_stride,
                                        const s_vec &t_stride);

    public:
        TensorPart(s_vec shape, TypeMeta tp, Device dvc = Device::CPU)
            : numel_(numel_count(shape)), strg_(makeFromShape(numel_, dvc, tp)),
              type_(tp), shape_(std::move(shape))
        {
            strides_.resize(shape_.size());
            int64_t step = 1;
            for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i)
            {
                strides_[i] = step;
                step *= shape_[i];
            }
        }
        TensorPart(Storage strg, TypeMeta tp, s_vec shape,
                   s_vec strides, int64_t offset)
            : numel_(numel_count(shape)),
              strg_(std::move(strg)),
              type_(tp),
              shape_(std::move(shape)),
              strides_(std::move(strides)),
              storage_offset_(offset) {}

        void fill(float f);

        // Storage is copiable, tensor is copiable;
        TensorPart(const TensorPart &) = default;
        TensorPart &operator=(const TensorPart &) = default;
        TensorPart(TensorPart &&) noexcept = default;
        TensorPart &operator=(TensorPart &&) noexcept = default;

        int64_t numel() const { return numel_; }
        int64_t storage_offset() const { return storage_offset_; }
        const s_vec &shape() const { return shape_; }
        const s_vec &strides() const { return strides_; }
        int64_t stride(int64_t d) const { return strides_[d]; }
        Device device() const { return strg_.device(); }
        TypeMeta type() const { return type_; }
        Storage storage() const { return strg_; }

        void *data()
        {
            return static_cast<char *>(strg_.data()) +
                   storage_offset_ * type_.itemsize_;
        }
        const void *data() const
        {
            return static_cast<const char *>(strg_.data()) +
                   storage_offset_ * type_.itemsize_;
        }
        void *at(const std::vector<int64_t> &idx)
        {
            int64_t pos = 0;
            if (strides_.size() != idx.size())
                throw std::invalid_argument("Invalid idx");
            for (size_t i = 0; i < idx.size(); ++i)
            {
                int64_t x = idx[i];
                if (x < 0 || x >= shape_[i])
                    throw std::invalid_argument("Out Of Range");
                pos += x * strides_[i] * type_.itemsize_;
            }
            return static_cast<char *>(data()) + pos;
        }

        bool is_continue()
        {
            int64_t dimProduct = 1;
            for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i)
            {
                if (strides_[i] != dimProduct)
                    return false;
                dimProduct *= shape_[i];
            }
            return true;
        }
        TensorPart to(Device dvc)
        {
            if (dvc == strg_.device())
                return *this;
            return TensorPart(Storage(strg_, dvc), type_, shape_, strides_,
                              storage_offset_);
        }
        TensorPart permute(const std::vector<int8_t> &dimOrder);
        TensorPart view(s_vec new_shape);
        // TensorPart continuous();
        // TensorPart reshape(s_vec shape);

        TensorPart operator+(const TensorPart &other) const;
        void operator+=(const TensorPart &other);
        TensorPart operator*(const TensorPart &other) const;
    };

} // namespace base

namespace graph
// This namespace focus on graph build
{
    class TensorImpl;
    class Node;
    using edge = std::pair<std::shared_ptr<Node>, int>;
    using tensorPtr = std::shared_ptr<TensorImpl>;

    class Node
    {
    protected:
        // i = edge[1]表示该node是edge[0]的第i个输出下游，通常为0
        std::vector<edge> next_edges_;
        // prev_grads_[i]表示下游传回来的需要的第i个grad，通常size = 1
        std::vector<tensorPtr> prev_grads_;

    public:
        Node(std::vector<edge> next_nodes, int grads_num)
            : next_edges_(next_nodes), prev_grads_(grads_num) {}
        Node(int edges_num, int grads_num)
            : next_edges_(edges_num), prev_grads_(grads_num) {}

        edge &getEdge(int i)
        {
            return next_edges_.at(i);
        }
        std::vector<edge> &next_edges() { return next_edges_; }
        const std::vector<edge> &next_edges() const { return next_edges_; }
        int edges_num() { return next_edges_.size(); }
        void accumulate_grads(tensorPtr prev, int idx);
        virtual std::vector<tensorPtr> backward() = 0;
    };

    class AccumulateGrad : public Node
    {
    private:
        tensorPtr t_;

    public:
        AccumulateGrad(tensorPtr t) : Node({}, 1), t_(t) {}
        std::vector<tensorPtr> backward();
    };

    struct AutogradMeta
    {
        std::shared_ptr<Node> grad_fn_ = nullptr;
        std::shared_ptr<AccumulateGrad> accumulator_ = nullptr;
        tensorPtr grad_ = nullptr;

        AutogradMeta() = default;
        AutogradMeta(std::shared_ptr<Node> grad_fn) : grad_fn_(grad_fn) {}
        AutogradMeta(std::shared_ptr<AccumulateGrad> accumulator)
            : accumulator_(accumulator) {}
    };

    class TensorImpl
    {
    private:
        std::shared_ptr<base::TensorPart> part_;
        bool requires_grad_ = true;
        AutogradMeta autograd_;

    public:
        TensorImpl(base::s_vec shape, base::TypeMeta tp,
                   base::Device dvc = base::Device::CPU,
                   bool requires_grad = true)
            : part_(std::make_shared<base::TensorPart>(shape, tp, dvc)),
              requires_grad_(requires_grad) {}

        TensorImpl(std::shared_ptr<base::TensorPart> part, bool requires_grad = true)
            : part_(std::move(part)), requires_grad_(requires_grad) {}

        base::TensorPart &part() { return *part_; }
        const base::TensorPart &part() const { return *part_; }
        void *data()
        {
            return part_->data();
        }
        const void *data() const { return part_->data(); }
        const base::s_vec &shape() const { return part_->shape(); }
        const base::s_vec &strides() const { return part_->strides(); }
        base::Device device() const { return part_->device(); }
        base::TypeMeta type() const { return part_->type(); }
        int64_t numel() { return part_->numel(); }
        bool is_continue() { return part_->is_continue(); }
        TensorImpl view(base::s_vec new_shape)
        {
            auto part = std::make_shared<base::TensorPart>(part_->view(new_shape));
            return TensorImpl(part, requires_grad_);
        }

        AutogradMeta &autograd()
        {
            return autograd_;
        }

        void fill(float f)
        {
            part_->fill(f);
        }

        TensorImpl to(base::Device dvc)
        {
            if (dvc == part_->device())
                return *this;
            return TensorImpl(std::make_shared<base::TensorPart>(part_->to(dvc)));
        }

        // 使用索引递归访问：
        // 索引数 == 维度数 -> 元素（get/set）；索引数 < 维度数 -> 切片（slice）
        class IndexProxy
        {
        private:
            TensorImpl *t_;
            int64_t byte_offset_; // 相对 data() 的字节偏移（data() 已含 storage_offset）
            int64_t dim_;         // 已消费的维数

        public:
            IndexProxy(TensorImpl *t, int64_t off, int64_t dim)
                : t_(t), byte_offset_(off), dim_(dim) {}

            IndexProxy operator[](int64_t i) const;
            float get() const;
            void set(float v) const;
            TensorImpl slice() const;
        };

        IndexProxy operator[](int64_t i);
        void print() const;

        bool requires_grad() const { return requires_grad_; }

        void accumulate(tensorPtr t)
        {
            (*part_) += t->part();
        }
        TensorImpl operator*(TensorImpl t)
        {
            auto newPartPtr = std::make_shared<base::TensorPart>((*part_) * (*t.part_));
            return TensorImpl(newPartPtr, requires_grad_ && t.requires_grad_);
        }
    };

    class Engine
    {
    private:
        std::map<std::shared_ptr<Node>, int> nodes_count;
        void count_node_dfs(std::shared_ptr<Node> n);

    public:
        Engine() {}
        void backward(tensorPtr t, tensorPtr grad = nullptr);
    };
}

namespace operation
{
    class Opr
    {
    public:
        virtual bool broadcastingSupport() = 0;
        virtual graph::tensorPtr operator()(std::vector<graph::tensorPtr>) = 0;
    };
}