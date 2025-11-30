#pragma once

#include <cuda_runtime.h>

// Защита от компиляции cl.exe (MSVC)
#ifdef __NVCC__

__global__ void transpose_naive_kernel(const float* in, float* out, int rows, int cols);

__global__ void transpose_shared_kernel(const float* in, float* out, int rows, int cols);

#endif // __NVCC__

// Хост-функции (можно объявлять всегда)
#ifdef __cplusplus
extern "C" {
#endif

void transpose_gpu_naive(const float* h_in, float* h_out, int rows, int cols);
void transpose_gpu_shared(const float* h_in, float* out, int rows, int cols);

#ifdef __cplusplus
}
#endif