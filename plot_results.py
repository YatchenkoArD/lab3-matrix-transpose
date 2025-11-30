#!/usr/bin/env python3
"""
Скрипт для построения графиков результатов транспонирования матрицы
Читает results.csv и строит графики для квадратных и неквадратных матриц
"""

import matplotlib.pyplot as plt
import csv

# Данные для квадратных матриц
square_sizes = []
square_cpu = []
square_naive = []
square_shared = []

# Данные для неквадратных матриц
rect_rows = []
rect_cols = []
rect_cpu = []
rect_naive = []
rect_shared = []

with open('results.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # пропускаем заголовок
    for row in reader:
        if row and len(row) >= 7:
            rows = int(row[0])
            cols = int(row[1])
            cpu_time = float(row[2])
            naive_time = float(row[3])
            shared_time = float(row[4])
            
            if rows == cols:
                # Квадратная матрица
                square_sizes.append(rows)
                square_cpu.append(cpu_time)
                square_naive.append(naive_time)
                square_shared.append(shared_time)
            else:
                # Неквадратная матрица
                rect_rows.append(rows)
                rect_cols.append(cols)
                rect_cpu.append(cpu_time)
                rect_naive.append(naive_time)
                rect_shared.append(shared_time)

# График 1: Квадратные матрицы
if square_sizes:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog(square_sizes, square_cpu, 'o-', label='CPU', linewidth=2, markersize=8)
    plt.loglog(square_sizes, square_naive, 's-', label='GPU naive', linewidth=2, markersize=8)
    plt.loglog(square_sizes, square_shared, '^-', label='GPU shared memory', linewidth=2, markersize=8)
    plt.xlabel('Размер квадратной матрицы N×N')
    plt.ylabel('Время выполнения (мс)')
    plt.title('Квадратные матрицы: CPU vs CUDA')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    # График 2: Неквадратные матрицы (по общему количеству элементов)
    if rect_rows:
        plt.subplot(1, 2, 2)
        rect_elements = [r * c for r, c in zip(rect_rows, rect_cols)]
        plt.loglog(rect_elements, rect_cpu, 'o-', label='CPU', linewidth=2, markersize=8)
        plt.loglog(rect_elements, rect_naive, 's-', label='GPU naive', linewidth=2, markersize=8)
        plt.loglog(rect_elements, rect_shared, '^-', label='GPU shared memory', linewidth=2, markersize=8)
        plt.xlabel('Количество элементов (rows × cols)')
        plt.ylabel('Время выполнения (мс)')
        plt.title('Неквадратные матрицы: CPU vs CUDA')
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('transpose_plot.png', dpi=300, bbox_inches='tight')
    print("График сохранён как transpose_plot.png")
    plt.show()
else:
    print("Нет данных для построения графика")
