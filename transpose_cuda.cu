#include "transpose_cuda.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel для транспонирования матрицы (naive версия с 2D блоками)
__global__ void transpose_naive_kernel(const float* in, float* out, int rows, int cols) {
    // Вычисляем глобальные индексы в исходной матрице
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Проверяем границы (для обработки размеров, не кратных размеру блока)
    if (row < rows && col < cols) {
        // Транспонирование: out[j][i] = in[i][j]
        // out имеет размер cols x rows
        out[col * rows + row] = in[row * cols + col];
    }
}

// CUDA kernel для транспонирования матрицы (оптимизированная версия с shared memory)
__global__ void transpose_shared_kernel(const float* in, float* out, int rows, int cols) {
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;
    
    // Shared memory с padding для избежания bank conflicts
    // Используем [TILE_DIM+1][TILE_DIM] вместо [TILE_DIM][TILE_DIM]
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    // Вычисляем глобальные индексы в исходной матрице
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    
    // Читаем тайл в shared memory коалесцированно (обычным порядком)
    // Так как в блоке только BLOCK_ROWS потоков по Y, нужен цикл для чтения всего тайла
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int read_row = row + i;
        if (read_row < rows && col < cols) {
            // Коалесцированное чтение: соседние потоки читают соседние элементы
            tile[threadIdx.y + i][threadIdx.x] = in[read_row * cols + col];
        }
    }
    
    // Синхронизируем все потоки блока
    __syncthreads();
    
    // Вычисляем глобальные индексы для записи (транспонированные)
    // Меняем местами: col становится row, row становится col
    // out имеет размер cols x rows, поэтому out[col_out * rows + row_out]
    int col_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int row_out = blockIdx.x * BLOCK_ROWS + threadIdx.y;
    
    // Записываем из shared memory в глобальную память транспонированно
    // Используем tile[threadIdx.x][threadIdx.y] для транспонирования
    // Это избегает bank conflicts благодаря padding в shared memory
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int write_row = row_out + i;
        if (write_row < cols && col_out < rows) {
            out[write_row * rows + col_out] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// Host функция для транспонирования на GPU
void transpose_gpu_naive(const float* h_in, float* h_out, int rows, int cols) {
    int total_elements = rows * cols;
    size_t size = total_elements * sizeof(float);
    
    // Выделяем память на GPU
    float* d_in = NULL;
    float* d_out = NULL;
    
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_in, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaMalloc failed for input: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&d_out, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaMalloc failed for output: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        return;
    }
    
    // Копируем входную матрицу на GPU
    err = cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaMemcpy failed (HostToDevice): %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Настраиваем параметры запуска kernel с 2D блоками
    // Используем блоки 16x16 или 32x32
    const int TILE_SIZE = 32;  // Можно изменить на 16
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    
    // Вычисляем количество блоков в grid (с округлением вверх)
    int gridX = (cols + TILE_SIZE - 1) / TILE_SIZE;
    int gridY = (rows + TILE_SIZE - 1) / TILE_SIZE;
    dim3 gridSize(gridX, gridY);
    
    // Запускаем kernel
    transpose_naive_kernel<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    
    // Проверяем ошибки запуска kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Синхронизируем выполнение
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Копируем результат обратно на CPU
    err = cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaMemcpy failed (DeviceToHost): %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Освобождаем память GPU
    err = cudaFree(d_in);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaFree failed for input: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaFree(d_out);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_naive: cudaFree failed for output: %s\n", cudaGetErrorString(err));
    }
}

// Host функция для транспонирования на GPU с использованием shared memory
void transpose_gpu_shared(const float* h_in, float* h_out, int rows, int cols) {
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;
    
    int total_elements = rows * cols;
    size_t size = total_elements * sizeof(float);
    
    // Выделяем память на GPU
    float* d_in = NULL;
    float* d_out = NULL;
    
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_in, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaMalloc failed for input: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&d_out, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaMalloc failed for output: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        return;
    }
    
    // Копируем входную матрицу на GPU
    err = cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaMemcpy failed (HostToDevice): %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Настраиваем параметры запуска kernel
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / BLOCK_ROWS);
    
    // Запускаем kernel
    transpose_shared_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, rows, cols);
    
    // Проверяем ошибки запуска kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Синхронизируем выполнение
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Копируем результат обратно на CPU
    err = cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaMemcpy failed (DeviceToHost): %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    
    // Освобождаем память GPU
    err = cudaFree(d_in);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaFree failed for input: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaFree(d_out);
    if (err != cudaSuccess) {
        fprintf(stderr, "transpose_gpu_shared: cudaFree failed for output: %s\n", cudaGetErrorString(err));
    }
}

// CUDA kernel для транспонирования матрицы (старая версия для обратной совместимости)
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    
    if (idx < total_elements) {
        int row = idx / cols;
        int col = idx % cols;
        output[col * rows + row] = input[row * cols + col];
    }
}

int transpose_cuda(const float* input, float* output, int rows, int cols) {
    int total_elements = rows * cols;
    size_t size = total_elements * sizeof(float);
    
    // Выделяем память на GPU
    float* d_input = NULL;
    float* d_output = NULL;
    
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for input: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for output: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return 1;
    }
    
    // Копируем данные на GPU
    err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed (HostToDevice): %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    // Настраиваем параметры запуска kernel
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Запускаем kernel
    transpose_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, rows, cols);
    
    // Проверяем ошибки kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    // Синхронизируем
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    // Копируем результаты обратно на CPU
    err = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed (DeviceToHost): %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
    
    // Освобождаем память GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
