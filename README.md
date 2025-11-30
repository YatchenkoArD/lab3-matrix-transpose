# Проект транспонирования матрицы (CPU vs GPU)

Проект на чистом языке C для сравнения производительности транспонирования матрицы на CPU и GPU (CUDA).

## Структура проекта

- `main.c` - основная программа с замерами времени и сохранением результатов
- `transpose_cpu.h` / `transpose_cpu.c` - CPU версия транспонирования
- `transpose_cuda.cuh` / `transpose_cuda.cu` - GPU версия транспонирования (CUDA)
- `utils.h` / `utils.c` - вспомогательные функции

## Требования

- Компилятор C (gcc, clang)
- NVIDIA CUDA Toolkit
- Видеокарта NVIDIA с поддержкой CUDA

## Сборка

```bash
make
```

Или вручную:

```bash
gcc -Wall -O2 -std=c11 -c transpose_cpu.c -o transpose_cpu.o
gcc -Wall -O2 -std=c11 -c utils.c -o utils.o
nvcc -O2 -arch=sm_75 -std=c++11 -c transpose_cuda.cu -o transpose_cuda.o
nvcc -O2 -arch=sm_75 -std=c++11 -x cu -c main.c -o main.o -I. -D__CUDACC__
nvcc -O2 -arch=sm_75 -std=c++11 -o transpose transpose_cpu.o utils.o transpose_cuda.o main.o -lcudart
```

**Примечание:** Замените `sm_75` на архитектуру вашей видеокарты (например, `sm_86` для RTX 3050).

## Использование

### Запуск с заданными размерами матрицы:

```bash
./transpose <rows> <cols>
```

Пример:
```bash
./transpose 1000 1000
```

### Запуск с фиксированным набором размеров:

```bash
./transpose --default
```

Это выполнит тесты для следующих размеров:
- 100 x 100
- 500 x 500
- 1000 x 1000
- 2000 x 2000
- 100 x 1000
- 1000 x 100

## Результаты

Программа сохраняет результаты в файл `results.csv` со следующими колонками:
- `rows` - количество строк исходной матрицы
- `cols` - количество столбцов исходной матрицы
- `cpu_time_ms` - время выполнения на CPU (миллисекунды)
- `gpu_time_ms` - время выполнения на GPU (миллисекунды)
- `speedup` - ускорение GPU относительно CPU
- `correct` - корректность результатов (1 = совпадают, 0 = не совпадают)

## Пример вывода

```
=== Транспонирование матрицы (CPU vs GPU) ===

CUDA доступна: NVIDIA GeForce RTX 3050 (Compute 8.6)

Тест 1: матрица 1000 x 1000
----------------------------------------
CPU время: 12.345678 мс
GPU время: 0.123456 мс
Ускорение: 100.00x
Результаты совпадают: ✓

Результаты сохранены в results.csv
```

