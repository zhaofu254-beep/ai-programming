# 爱编程CUDA自主编写深度学习简易框架

## Part 1: framework

相较于上一版本，本项目进行了较大规模的底层设计改造，风格上更贴近于PyTorch。核心区别是在`src/cu/tensor.cuh`中，计算图，前向传播，反向传播与张量存储的进一步解耦。具体而言：

1. **`namespace base `**
   - 只负责基本的`TensorPart`数据结构本身的搭建，与第一版相同。

2. **`namespace graph`**
   - `TensorImpl`: 包含了一个指向tensor本身的`std::shared_ptr<TensorPart>`，以及存储了grad的相关信息的`struct AutoGrad`；
   - `Node`: 计算图的基本构建单位，记录了上游`Node`的信息以及Backward所需要的tensor；
   - `Engine`: 负责驱动反向传播。

3. **`namespace operation`**
   - 定义了基类`Opr`，所有的具体的操作类都继承自它，并创造相应的`Node`子类，这也是包含所有后续操作类的命名空间。
   - `operation`的子命名空间`namespace detail`存储需要被经常使用的函数。不同操作位于不同文件不同环境之下，避免彼此干扰。但是有些操作需要在别的地方复用，就加入`detail`中

## Part 2: Sum operation

由于前向传播与反向传播都会经常性涉及求和的问题，因此创建了`SumOpr`，并把`contigReduce`写入`detail`。

`contigReduce`分为了五种情况：

- CPU版：tensor位于CPU上时使用。
- `smallContigReduce`: 归约维度小于等于`MAX_SMALL_DIM_SIZE = 256`。
- `lastDimContigReduce`: 归约维度大于`MAX_SMALL_DIM_SIZE = 256`且在最后一维。
- `longTailContigReduce`: 归约维度在中间且尾维度大于`MAX_SHORT_TAIL_SIZE = 32`
- `shortTailContigReduce`: 归约维度在中间且尾维度小于等于`MAX_SHORT_TAIL_SIZE = 32`

## Part 3: Linear operation

使用`cublasSgemm`编写`GEMM`写进`detail`。`LinearOpr`分为有无`bias`两种情况。

## Part 4: Conv operation

仅限于`kernel_size=3, padding-1`的情况。逐个`batch`处理，使用`img2col`配合`GEMM`进行前向传播，再使用`col2img`处理反向传播。也分为有无`bias`两种情况。

## Part 5: Maxpool operation

仅限于`kernel_size=2, stride=2`的情况。前向传播时记录每个位置是否是max值，供反向传播时使用。

## Part 6: Softmax operation

前向传播时记录 $\sum_{j}e^j$，供反向传播时使用。`softmax`写进`ditail`供`CEL`使用。

## Part 7: CEL

实现了处理生成的`prob`的`crossEntropy`，写进`detail`。`CrossEntropyOpr`直接内置了`softmax`操作。

## Part 8: test

每个具体的`Opr`都有相应的`test`文件，运行结果保存在`tests/results.txt`中。
