@echo off
REM Кастомный скрипт компиляции с правильными флагами для nvcc
REM Использует -x cu для файлов, использующих CUDA API

echo === Компиляция проекта транспонирования матрицы ===
echo.

REM Инициализация окружения Visual Studio
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo Компиляция с использованием nvcc...
echo.

REM Компиляция main.c с флагом -x cu (важно для CUDA типов)
nvcc -O3 -arch=sm_86 -x cu -c main.c -o main.o
if %ERRORLEVEL% NEQ 0 goto :error

REM Компиляция utils.c с флагом -x cu (если использует CUDA)
nvcc -O3 -arch=sm_86 -x cu -c utils.c -o utils.o
if %ERRORLEVEL% NEQ 0 goto :error

REM Компиляция transpose_cpu.c (может быть без -x cu, если не использует CUDA)
nvcc -O3 -arch=sm_86 -x cu -c transpose_cpu.c -o transpose_cpu.o
if %ERRORLEVEL% NEQ 0 goto :error

REM Компиляция transpose_cuda.cu (CUDA файл)
nvcc -O3 -arch=sm_86 -c transpose_cuda.cu -o transpose_cuda.o
if %ERRORLEVEL% NEQ 0 goto :error

REM Компиляция device_info.cu (CUDA файл с функциями для работы с устройством)
nvcc -O3 -arch=sm_86 -c device_info.cu -o device_info.o
if %ERRORLEVEL% NEQ 0 goto :error

echo Линковка...
nvcc -O3 -arch=sm_86 main.o utils.o transpose_cpu.o transpose_cuda.o device_info.o -o lab3_transpose.exe
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo === Компиляция успешно завершена! ===
echo Исполняемый файл: lab3_transpose.exe
echo.
echo Запуск программы...
lab3_transpose.exe
goto :end

:error
echo.
echo === ОШИБКА при компиляции ===
pause
exit /b 1

:end

