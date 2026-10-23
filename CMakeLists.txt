cmake_minimum_required(VERSION 3.20)
project(hw2 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# tensor.cu编译为静态库
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_library(tensor STATIC src/cu/tensor.cu)
add_library(sum STATIC src/cu/sum.cu)

target_include_directories(tensor PUBLIC src/cu)

target_link_libraries(tensor PUBLIC CUDA::cudart)

#编译测试的可执行文件
# add_executable(test_activation tests/test_activation.cpp)

target_link_libraries(test_activation PRIVATE tensor)
