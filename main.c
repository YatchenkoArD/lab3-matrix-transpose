#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transpose_cpu.h"
#include "transpose_cuda.cuh"
#include "utils.h"

// Объявления функций из device_info.cu
extern int is_cuda_available();
extern void print_device_info();
extern void cudaDeviceReset_safe();

int main(int argc, char* argv[]) {
    // Тестовые матрицы: {rows, cols}
    // Включаем как квадратные, так и неквадратные матрицы
    int test_matrices[][2] = {
        {256, 256},    // квадратная
        {512, 512},    // квадратная
        {1024, 1024},  // квадратная
        {2048, 2048},  // квадратная
        {4096, 4096},  // квадратная
        {8192, 8192},  // квадратная
        {256, 512},    // неквадратная (узкая)
        {512, 256},    // неквадратная (широкая)
        {1024, 2048}, // неквадратная (узкая)
        {2048, 1024},  // неквадратная (широкая)
        {512, 1024},   // неквадратная (узкая)
        {1024, 512}    // неквадратная (широкая)
    };
    int num_tests = sizeof(test_matrices) / sizeof(test_matrices[0]);
    const int num_runs = 10;  // Количество прогонов для усреднения
    
    // Открываем файл для сохранения результатов
    FILE* csv_file = fopen("results.csv", "w");
    if (csv_file == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть файл results.csv для записи\n");
        return 1;
    }
    
    // Заголовок CSV с разделителями для лучшей читаемости в Excel
    fprintf(csv_file, "rows,cols,cpu_time_ms,gpu_naive_time_ms,gpu_shared_time_ms,speedup_naive,speedup_shared,correct\n");
    
    // Проверка доступности CUDA и вывод информации об устройстве
    print_device_info();
    int cuda_available = is_cuda_available();
    
    // Выводим заголовок таблицы
    printf("Rows x Cols | CPU (мс) | GPU naive (мс) | GPU shared (мс) | Ускорение\n");
    printf("------------|----------|----------------|-----------------|----------\n");
    
    // Цикл по тестовым матрицам
    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        int rows = test_matrices[test_idx][0];
        int cols = test_matrices[test_idx][1];
        
        // Создаем матрицы
        float* input = create_matrix(rows, cols);
        float* cpu_output = create_matrix(cols, rows);
        float* gpu_naive_output = create_matrix(cols, rows);
        float* gpu_shared_output = create_matrix(cols, rows);
        
        if (input == NULL || cpu_output == NULL || 
            gpu_naive_output == NULL || gpu_shared_output == NULL) {
            fprintf(stderr, "Ошибка: не удалось выделить память для матрицы %d x %d\n", rows, cols);
            if (input) free_matrix(input);
            if (cpu_output) free_matrix(cpu_output);
            if (gpu_naive_output) free_matrix(gpu_naive_output);
            if (gpu_shared_output) free_matrix(gpu_shared_output);
            continue;
        }
        
        // Замеряем время CPU (несколько прогонов)
        double cpu_time_sum = 0.0;
        for (int run = 0; run < num_runs; run++) {
            double start = get_time_ms();
            transpose_cpu(input, cpu_output, rows, cols);
            double end = get_time_ms();
            cpu_time_sum += (end - start);
        }
        double cpu_time_avg = cpu_time_sum / num_runs;
        
        // Замеряем время GPU naive (несколько прогонов)
        double gpu_naive_time_avg = 0.0;
        if (cuda_available) {
            double gpu_naive_time_sum = 0.0;
            for (int run = 0; run < num_runs; run++) {
                double start = get_time_ms();
                transpose_gpu_naive(input, gpu_naive_output, rows, cols);
                double end = get_time_ms();
                gpu_naive_time_sum += (end - start);
            }
            gpu_naive_time_avg = gpu_naive_time_sum / num_runs;
        }
        
        // Замеряем время GPU shared (несколько прогонов)
        double gpu_shared_time_avg = 0.0;
        if (cuda_available) {
            double gpu_shared_time_sum = 0.0;
            for (int run = 0; run < num_runs; run++) {
                double start = get_time_ms();
                transpose_gpu_shared(input, gpu_shared_output, rows, cols);
                double end = get_time_ms();
                gpu_shared_time_sum += (end - start);
            }
            gpu_shared_time_avg = gpu_shared_time_sum / num_runs;
        }
        
        // Проверка корректности (сравниваем GPU shared с CPU)
        int correct = 0;
        if (cuda_available) {
            correct = matrices_equal(cpu_output, gpu_shared_output, cols, rows, 1e-5f);
        }
        
        // Вычисляем ускорение
        double speedup_naive = (gpu_naive_time_avg > 0) ? cpu_time_avg / gpu_naive_time_avg : 0.0;
        double speedup_shared = (gpu_shared_time_avg > 0) ? cpu_time_avg / gpu_shared_time_avg : 0.0;
        
        // Выводим строку таблицы
        if (cuda_available) {
            printf("%4d x %-4d | %8.3f | %14.3f | %15.3f | %.2fx\n",
                   rows, cols, cpu_time_avg, gpu_naive_time_avg, gpu_shared_time_avg,
                   speedup_shared);
        } else {
            printf("%4d x %-4d | %8.3f | %14s | %15s | %s\n",
                   rows, cols, cpu_time_avg, "N/A", "N/A", "N/A");
        }
        
        // Записываем в CSV (каждое значение в отдельной ячейке)
        fprintf(csv_file, "%d,%d,%.6f,%.6f,%.6f,%.2f,%.2f,%d\n",
                rows,
                cols,
                cpu_time_avg,
                gpu_naive_time_avg,
                gpu_shared_time_avg,
                speedup_naive,
                speedup_shared,
                correct);
        
        // Освобождаем память
        free_matrix(input);
        free_matrix(cpu_output);
        free_matrix(gpu_naive_output);
        free_matrix(gpu_shared_output);
    }
    
    fclose(csv_file);
    printf("\nРезультаты сохранены в results.csv\n");
    
    // Сброс CUDA устройства
    cudaDeviceReset_safe();
    
    return 0;
}
