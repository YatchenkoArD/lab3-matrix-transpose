#include "utils.h"
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

// Статическая переменная для инициализации генератора случайных чисел
static int rand_initialized = 0;

float* create_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return NULL;
    }
    
    size_t size = (size_t)rows * (size_t)cols * sizeof(float);
    float* matrix = (float*)malloc(size);
    
    if (matrix == NULL) {
        return NULL;
    }
    
    // Инициализируем генератор случайных чисел один раз
    if (!rand_initialized) {
        srand((unsigned int)time(NULL));
        rand_initialized = 1;
    }
    
    // Заполняем случайными значениями от 0 до 1
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
    
    return matrix;
}

void free_matrix(float* mat) {
    if (mat != NULL) {
        free(mat);
    }
}

void print_matrix(const float* mat, int rows, int cols) {
    // Печатаем только если размеры <= 10
    if (rows > 10 || cols > 10) {
        return;
    }
    
    if (mat == NULL) {
        printf("Matrix is NULL\n");
        return;
    }
    
    printf("Matrix (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int matrices_equal(const float* a, const float* b, int rows, int cols, float eps) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    
    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; i++) {
        float diff = a[i] - b[i];
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > eps) {
            return 0;
        }
    }
    
    return 1;
}

double get_time_ms(void) {
#ifdef _WIN32
    // Windows: используем QueryPerformanceCounter
    static LARGE_INTEGER frequency = {0};
    LARGE_INTEGER counter;
    
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    // Linux/Unix: используем clock_gettime
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}
