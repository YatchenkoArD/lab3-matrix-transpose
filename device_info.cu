#include <cuda_runtime.h>
#include <stdio.h>

int is_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        return 1;
    }
    return 0;
}

void print_device_info() {
    // Проверка доступности CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    printf("=== Транспонирование матрицы (CPU vs GPU) ===\n");
    if (err == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err == cudaSuccess) {
            printf("CUDA доступна: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);
        } else {
            printf("CUDA доступна: устройство найдено\n\n");
        }
    } else {
        printf("Предупреждение: CUDA недоступна, GPU версии пропущены\n\n");
    }
}

void cudaDeviceReset_safe() {
    cudaDeviceReset();
}

