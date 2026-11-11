cmake_minimum_required(VERSION 3.20)
project(hw2 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# tensor.cu编译为静态库
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_library(tensor STATIC src/cu/tensor.cu)
add_library(sum STATIC src/cu/sum.cu)
add_library(linear STATIC src/cu/linear.cu)
add_library(conv STATIC src/cu/conv.cu)
add_library(maxpool STATIC src/cu/maxpool.cu)
add_library(softmax STATIC src/cu/softmax.cu)
add_library(cel STATIC src/cu/crossentropy.cu)

target_include_directories(tensor PUBLIC src/cu)
target_include_directories(sum PUBLIC src/cu)
target_include_directories(linear PUBLIC src/cu)
target_include_directories(conv PUBLIC src/cu)
target_include_directories(maxpool PUBLIC src/cu)
target_include_directories(softmax PUBLIC src/cu)
target_include_directories(cel PUBLIC src/cu)


target_link_libraries(tensor PUBLIC CUDA::cudart)
target_link_libraries(sum PUBLIC CUDA::cudart)
target_link_libraries(linear PUBLIC CUDA::cudart CUDA::cublas tensor sum)
target_link_libraries(conv PUBLIC CUDA::cudart tensor sum linear)
target_link_libraries(maxpool PUBLIC CUDA::cudart tensor sum)
target_link_libraries(softmax PUBLIC CUDA::cudart tensor sum)
target_link_libraries(cel PUBLIC CUDA::cudart tensor sum softmax)

#编译测试的可执行文件
add_executable(test_sum tests/test_sum.cpp)
add_executable(test_linear tests/test_linear.cpp)
add_executable(test_conv tests/test_conv.cpp)
add_executable(test_maxpool tests/test_maxpool.cpp)
add_executable(test_softmax tests/test_softmax.cpp)
add_executable(test_cel tests/test_cel.cpp)

target_link_libraries(test_sum PRIVATE tensor sum)
target_link_libraries(test_linear PRIVATE tensor linear)
target_link_libraries(test_conv PRIVATE tensor linear conv)
target_link_libraries(test_maxpool PRIVATE tensor sum maxpool)
target_link_libraries(test_softmax PRIVATE tensor sum softmax)
target_link_libraries(test_cel PRIVATE tensor sum softmax cel)
