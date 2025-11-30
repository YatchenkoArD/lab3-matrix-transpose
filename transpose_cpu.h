#ifndef TRANSPOSE_CPU_H
#define TRANSPOSE_CPU_H

#include <stdlib.h>

/**
 * Транспонирует матрицу на CPU
 * @param in - входная матрица (размером rows x cols)
 * @param out - выходная транспонированная матрица (размером cols x rows)
 * @param rows - количество строк входной матрицы
 * @param cols - количество столбцов входной матрицы
 */
void transpose_cpu(const float* in, float* out, int rows, int cols);

#endif // TRANSPOSE_CPU_H

