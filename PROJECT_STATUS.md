# Project Status

A minimal, PyTorch-style deep learning framework implemented from scratch in C++ / CUDA.

This project builds a **define-by-run dynamic computational graph** with automatic differentiation
("autograd"), targeting a course assignment on AI programming. It is deliberately written from
first principles (no `libtorch` / `ATen`): the storage layer, the tensor view abstraction, the
autograd engine, and the numerical kernels are all implemented directly on top of the CUDA
runtime (`cuda_runtime.h`) and cuBLAS (`cublas_v2.h`).

The project currently provides, end-to-end and verified:

- a general **Tensor** abstraction (shape / stride / storage-offset views, CPU + CUDA),
- a full **autograd engine** (`Node` / `Edge` / `AccumulateGrad` / `Engine`),
- the **Sum** operator (single-dim reduction, forward + backward),
- the **Linear** operator (`Y = X @ W^T + b`, forward + backward), built on a shared **GEMM**
  primitive (CPU reference implementation + cuBLAS).

---

## 1. Architecture

The code is organized into three logical layers, each living in its own C++ namespace. The
separation mirrors, in spirit, the layering of PyTorch's `c10` / `at::` / `torch::autograd`.

| Namespace     | Responsibility                                                   | Files                        |
| ------------- | ---------------------------------------------------------------- | ---------------------------- |
| `base`        | Device-aware memory ownership, type erasure, tensor view metadata | `tensor.cuh`, `tensor.cu`    |
| `graph`       | Autograd nodes, the differentiation engine, the public `TensorImpl` | `tensor.cuh`, `tensor.cu`  |
| `operation`   | Concrete operators (sum, linear) and reusable numerical kernels    | `sum.cuh/.cu`, `linear.cuh/.cu` |

```text
src/cu/
├── tensor.cuh / tensor.cu   # base + graph namespaces
├── sum.cuh / sum.cu         # operation::SumOpr/SumNode + reduction kernels
├── linear.cuh / linear.cu   # operation::LinearOpr/LinearNode + GEMM
└── error.cuh                # CHECK / CHECK_KERNEL_FUNC macros
tests/
├── test_sum.cpp
└── test_linear.cpp
CMakeLists.txt
```

**Layering principle.** `base` knows nothing about gradients; `graph` knows nothing about specific
operators; `operation` composes `base` (for data movement) and `graph` (for wiring the
computational graph). Operators depend on a small set of shared helpers declared under
`operation::detail` (`coalesce_dimensions`, `contigReduce`, `GEMM`).

---

## 2. Tensor core (`base`)

### 2.1 Memory ownership

Memory ownership is layered, from raw allocation to shared, viewable storage:

- `DataUniquePtr` — a raw `void*` plus a `std::unique_ptr<void, DeleterFnPtr>` that owns the
  underlying allocation with a **device-specific deleter** (`free` vs `cudaFree`).
- `DataPtr` — a `DataUniquePtr` tagged with a `Device` (CPU or CUDA).
- `StoragePart` — a byte count plus a `DataPtr`.
- `Storage` — a `std::shared_ptr<StoragePart>`. Sharing a `Storage` is what makes tensor **views**
  cheap: multiple tensors can alias the same buffer.

A `Storage` can be copied across devices via the copy constructor `Storage(const Storage&, Device)`,
which allocates on the target device and performs a `cudaMemcpy` (Host↔Device) as appropriate.

### 2.2 Tensor views (`TensorPart`)

`TensorPart` is the analog of PyTorch's `TensorImpl`. It decouples the *logical* tensor from its
*physical* layout using three pieces of metadata:

- `shape_` — the logical sizes per dimension,
- `strides_` — the element stride per dimension,
- `storage_offset_` — the offset (in elements) into the shared `Storage`.

The flat byte address of the logical `(i0, i1, …)` element is therefore

```text
data() + storage_offset_ * itemsize  +  Σ_k i_k * strides_[k] * itemsize
```

This is what enables zero-copy operations such as `permute` (which only reorders `shape_`/`strides_`)
and `slice` (which trims a leading dimension and advances `storage_offset_`). The class also
implements contiguous elementwise `+`, `+=`, `*` and `fill`.

### 2.3 Type dispatch

`ScalarType` (Byte/Char/Short/Int/Long/Float/Double/Bool) is mapped to a C++ type through
`dispatch_scalar_type<F>(ScalarType, F&&)`, which invokes `F` with the matching compile-time type.
`dispatch<F>(ScalarType, Device, F&&)` layers device on top of that, producing the two-parameter
`F::operator()<T, Device>()` used throughout the kernels. This is a lightweight, compile-time
"dispatcher" that keeps the device/type branching out of the numerical code.

### 2.4 Element access (`TensorImpl::IndexProxy`)

Tensor indexing is exposed through a small proxy object with explicit, non-implicit semantics:

- **full index** (`x[0][1][2][3]`) → `.get()` / `.set(v)` read/write a scalar (`cudaMemcpy` is used
  when the tensor lives on CUDA);
- **partial index** (`x[0]`) → `.slice()` returns a `TensorImpl` *view* that shares the same storage
  (`requires_grad = false`).

No implicit scalar/view conversions are provided in C++; the intent is to keep the C++ API explicit
and reserve "pythonic" ergonomics for a future binding layer.

---

## 3. Autograd engine (`graph`)

### 3.1 Core types

- `TensorImpl` — the user-facing tensor. It wraps a `std::shared_ptr<base::TensorPart>` (`data_`),
  a `requires_grad_` flag, and an `AutogradMeta` sidecar.
- `AutogradMeta` — `grad_fn_` (the node that produced this tensor), `accumulator_` (for leaf
  tensors), and `grad_` (the accumulated gradient).
- `Node` — the base class for a differentiation step (`grad_fn`). It holds
  - `next_edges_`: a list of `Edge`s, one per input, and
  - `prev_grads_`: the incoming gradient(s) for this node (indexed by output number).
  It declares `virtual std::vector<tensorPtr> backward() = 0;`.
- `Edge` — `std::pair<std::shared_ptr<Node>, int>`. The `int` is the **output number** of the child
  node this edge connects to (mirrors PyTorch's `Edge = (Node*, output_nr)`).
- `AccumulateGrad` — a leaf node that writes the incoming gradient into its tensor's `.grad_`.

### 3.2 The backward engine (`Engine`)

`Engine::backward(root, seed)` performs a topological backward pass over the graph:

1. `count_node_dfs(root)` computes, for every reachable node, its **in-degree** (number of parent
   edges). A `std::set` guards against double traversal in DAGs.
2. The root is seeded with `seed` (an all-ones tensor if none is given) and pushed onto a stack.
3. Nodes are popped in dependency order. For each node, `backward()` returns one gradient per input
   edge; each gradient is accumulated into the child via `child->accumulate_grads(grad, output_nr)`.
   When a child's in-degree reaches zero, it is pushed.

Robustness rules enforced in the engine:

- **null edges are skipped** (a leaf that was never attached to an `AccumulateGrad`);
- **null gradients are skipped** (an input with `requires_grad == false`, for which a node's
  `backward()` returns `nullptr`).

This makes the engine safe to run on graphs that contain `requires_grad == false` tensors.

### 3.3 Wiring convention

When an operator creates a node, it connects each input via:

```cpp
std::shared_ptr<Node> fn = input->autograd().grad_fn_;
if (!fn) fn = input->autograd().accumulator_;   // leaf fallback
```

That is, a **leaf** tensor (whose `grad_fn_` is null) contributes its `AccumulateGrad` node instead,
so that gradients correctly accumulate into `tensor.grad_`.

---

## 4. Operators (`operation`)

### 4.1 Reduction (`operation::detail`)

The reduction stack is the most mature part of the numerical layer. `contigReduce(t, rdim)` is a
single entry point that dispatches on tensor shape to four strategies, all for **contiguous**
single-dimension reduction:

| Strategy | Condition | Approach |
| --- | --- | --- |
| `smallContigReduce` | reduced dim < 256 | one thread per output element, serial accumulation |
| `longTailContigReduce` | reduced dim ≥ 256, tail ≥ 32 | each warp tiles the contiguous tail; warps spread along the reduction axis |
| `shortTailContigReduce` | reduced dim ≥ 256, tail < 32 | "fold" the 32 lanes into `tail_size × (32/tail_size)`, combine with `__shfl_down_sync` |
| `lastDimContigReduce` | reduction is the last dim | grid-stride over the last dim + warp/block reduction |

`coalesce_dimensions(shape, stride, masked_dims)` merges adjacent contiguous non-reduced dimensions
into a single dimension (PyTorch-style), which shrinks the number of expensive integer
divisions/modulos inside the kernels and is reused by the broadcasting logic planned for future
elementwise ops.

The CPU path is a single naive reference implementation (`ContigReduceCPU`) shared by all strategies.

### 4.2 Sum

`SumOpr` reduces an input tensor along `rdim` (defaulting to the last dimension).

- Forward: `contigReduce(input, rdim)`.
- Backward (`SumNode::backward`): **broadcast** the output gradient along `rdim`, i.e. materialize
  `∂L/∂input = expand(∂L/∂output, rdim)`. Both a CPU loop and a CUDA kernel are provided, and the
  gradient is passed back through the graph.

### 4.3 GEMM (`operation::detail::GEMM`)

`GEMM(A, B, C, bias, alpha, beta, transposeA, transposeB)` computes

```text
D = alpha * (op(A) @ op(B)) + beta * C
```

where `op(X)` is `X` or `Xᵀ` according to the transpose flags, and `C` (when `bias` is set) is
broadcast-added. Two backends exist:

- **`GEMMCPU`** — a straightforward triple loop over the 2-D logical matrices, handling both
  transpose flags via index remapping.
- **`GEMMCUDA`** — a single `cublasSgemm` call. Because the tensors are **row-major** and cuBLAS is
  **column-major**, the multiplication is expressed as `Dᵀ = op(B)ᵀ @ op(A)ᵀ` with the transpose
  flags mapped accordingly (`transa = transposeB ? OP_N : OP_T`, `transb = transposeA ? OP_T : OP_N`,
  leading dimensions taken from `shape()[1]`). The bias term is added by a tiny elementwise kernel.

### 4.4 Linear

`LinearOpr` implements `Y = X @ W^T + b`:

- Forward: `GEMM(X, W, b, bias, 1, 1, transposeA=false, transposeB=false)`, where `W` is stored as
  `[out_features, in_features]` and the `transposeB=false` convention means `op(B) = Bᵀ = Wᵀ`.
- Backward (`LinearNode::backward`):
  - `∂L/∂W = prevᵀ @ X` → `GEMM(prev, X, nullptr, false, 1, 0, true, true)`,
  - `∂L/∂X = prev @ W` → `GEMM(prev, W, nullptr, false, 1, 0, false, true)`,
  - `∂L/∂b = Σ_batch prev` → `contigReduce(prev, 0)` (only when bias is enabled).

`requires_grad` is propagated to the output as the **logical OR** of all inputs, so that a
non-differentiable input (e.g. the network's input features) does not silently disable the
gradients of the learnable parameters.

---

## 5. Key design decisions & pitfalls

- **Views over copies.** `shape`/`stride`/`storage_offset` + shared `Storage` are the central
  abstraction; `permute` and `slice` are zero-copy. This matches `torch.storage` and keeps the door
  open for later reshape/transpose/broadcast.
- **Host pointers never reach device kernels.** CUDA `__global__` array parameters decay to
  pointers, so small dim/stride metadata is passed **by value in a struct** (e.g.
  `SmallReduceKernelDimStride`, `ContigReduceDimStride`, `BackwardKernelDimStride`). Passing a
  `std::vector::data()` host pointer to a kernel is a classic `cudaError 700` (illegal memory
  access) that was diagnosed and fixed in several places.
- **`requires_grad` must be the OR of all inputs.** Propagating it from only one input (e.g. the
  first) breaks autograd whenever a later input needs a gradient.
- **Leaves need an `AccumulateGrad`.** A leaf's `grad_fn_` is null; operators must fall back to
  `accumulator_` when wiring edges, otherwise the leaf gradient never accumulates.
- **Explicit transpose convention.** The GEMM/Linear `transposeB=false` case deliberately means
  "`B` is stored as `[N, K]`, compute `op(B)=Bᵀ`" so that `W` can be stored in its natural
  `[out, in]` layout and used directly in `X @ Wᵀ`.
- **Reduction strategy selection.** Small vs. large vs. short-tail vs. last-dim reductions are
  separate kernels because their optimal memory access patterns differ; the dispatcher keeps the
  public API uniform.

---

## 6. Build & test

CMake produces three static libraries linked against the CUDA runtime (and cuBLAS for `linear`):

```text
tensor   src/cu/tensor.cu            -> CUDA::cudart
sum      src/cu/sum.cu               -> CUDA::cudart
linear   src/cu/linear.cu            -> CUDA::cudart CUDA::cublas tensor sum
```

`tests/test_sum.cpp` exercises indexing, printing, CPU/GPU reduction (small / short-tail / last-dim
/ long-tail) and the full forward/backward of `sum`. `tests/test_linear.cpp` exercises the
forward/backward of `Linear` with a `requires_grad=false` input and `requires_grad=true` parameters.

---

## 7. Current status & roadmap

**Done and verified:**

- Tensor core (`base`): memory ownership, views, type dispatch, element indexing, contiguous
  elementwise ops.
- Autograd (`graph`): `Node`/`Edge`/`AccumulateGrad`/`Engine` with topological backward, null-edge /
  null-gradient robustness, and correct `requires_grad` propagation.
- `Sum`: single-dim reduction (four CUDA strategies + CPU reference) with broadcast backward.
- `Linear`: GEMM (CPU + cuBLAS) with correct forward and backward, including the `bias` on/off paths.

**Planned / not yet implemented:**

- More operators (activation functions beyond `relu`/`sigmoid` scaffolding, `matmul` generalization,
  elementwise broadcasting).
- Multi-dimension reduction and general (non-contiguous) reduction.
- A public, unified dispatcher layer (`Opr` hierarchy is partially scaffolded).
- `zero_grad`, in-place version counters, and gradient hooks.
- Python bindings (the explicit C++ `IndexProxy` API was designed with this in mind).
