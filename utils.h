#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Выделяет память под матрицу в row-major порядке и заполняет случайными float 0..1
 * @param rows - количество строк
 * @param cols - количество столбцов
 * @return указатель на выделенную память или NULL при ошибке
 */
float* create_matrix(int rows, int cols);

/**
 * Освобождает память матрицы
 * @param mat - указатель на матрицу
 */
void free_matrix(float* mat);

/**
 * Красиво печатает матрицу, только если rows и cols <= 10
 * @param mat - матрица
 * @param rows - количество строк
 * @param cols - количество столбцов
 */
void print_matrix(const float* mat, int rows, int cols);

/**
 * Проверяет, равны ли две матрицы с погрешностью eps
 * @param a - первая матрица
 * @param b - вторая матрица
 * @param rows - количество строк
 * @param cols - количество столбцов
 * @param eps - допустимая погрешность
 * @return 1 если все элементы равны с погрешностью eps, 0 иначе
 */
int matrices_equal(const float* a, const float* b, int rows, int cols, float eps);

/**
 * Возвращает текущее время в миллисекундах
 * @return время в миллисекундах
 */
double get_time_ms(void);

#endif // UTILS_H
