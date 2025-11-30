# Makefile для проекта транспонирования матрицы

# Компиляторы
CC = gcc
NVCC = nvcc

# Флаги компиляции
CFLAGS = -Wall -Wextra -O2 -std=c11
# RTX 3050 имеет compute capability 8.6 (sm_86)
# Можно использовать sm_86 или более общий код для всех архитектур
NVCCFLAGS = -O2 -arch=sm_86 -std=c++11

# Имена файлов
CPU_SOURCES = transpose_cpu.c utils.c
CUDA_SOURCES = transpose_cuda.cu device_info.cu
MAIN_SOURCE = main.c

# Объектные файлы
CPU_OBJECTS = $(CPU_SOURCES:.c=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
MAIN_OBJECT = main.o

# Имя исполняемого файла
TARGET = transpose

# Правило по умолчанию
all: $(TARGET)

# Сборка основного исполняемого файла
$(TARGET): $(CPU_OBJECTS) $(CUDA_OBJECTS) $(MAIN_OBJECT)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(CPU_OBJECTS) $(CUDA_OBJECTS) $(MAIN_OBJECT) -lcudart

# Компиляция CPU файлов
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Компиляция CUDA файлов
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Компиляция main.c (нужен CUDA runtime)
main.o: main.c
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@ -I. -D__CUDACC__

# Очистка
clean:
	rm -f $(TARGET) $(CPU_OBJECTS) $(CUDA_OBJECTS) $(MAIN_OBJECT) device_info.o results.csv

# Запуск с тестами по умолчанию
test: $(TARGET)
	./$(TARGET) --default

.PHONY: all clean test

