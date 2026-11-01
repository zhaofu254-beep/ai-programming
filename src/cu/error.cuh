#pragma once
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                              \
    do                                                                           \
    {                                                                            \
        const cudaError_t error_code = call;                                     \
        if (error_code != cudaSuccess)                                           \
        {                                                                        \
            printf("CUDA ERROR:\n");                                             \
            printf("    File:      %s\n", __FILE__);                             \
            printf("    Line:      %d\n", __LINE__);                             \
            printf("    Error code:      %d\n", error_code);                     \
            printf("    Error test:      %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define CHECK_KERNEL_FUNC(...)          \
    do                                  \
    {                                   \
        __VA_ARGS__;                    \
        CHECK(cudaGetLastError());      \
        CHECK(cudaDeviceSynchronize()); \
    } while (0)
