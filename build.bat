@echo off
REM Batch-скрипт для компиляции проекта на Windows
REM Требуется: CUDA Toolkit 12.6, компилятор C (MSVC или MinGW)

echo === Компиляция проекта транспонирования матрицы ===
echo.

REM Проверка наличия nvcc
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: nvcc не найден в PATH
    echo Убедитесь, что CUDA Toolkit установлен и добавлен в PATH
    pause
    exit /b 1
)

echo CUDA Toolkit найден
nvcc --version
echo.

REM Проверка наличия компилятора C
where gcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Используется GCC
    set CC=gcc
    goto :compile
)

where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Используется MSVC
    set CC=cl
    goto :compile
)

echo ОШИБКА: Компилятор C не найден
echo Установите MinGW-w64 или Visual Studio
pause
exit /b 1

:compile
echo.
echo Компиляция CPU файлов...
%CC% -Wall -O2 -std=c11 -c transpose_cpu.c -o transpose_cpu.o
if %ERRORLEVEL% NEQ 0 goto :error

%CC% -Wall -O2 -std=c11 -c utils.c -o utils.o
if %ERRORLEVEL% NEQ 0 goto :error

echo Компиляция CUDA файлов...
nvcc -O2 -arch=sm_86 -std=c++11 -c transpose_cuda.cu -o transpose_cuda.o
if %ERRORLEVEL% NEQ 0 goto :error

nvcc -O2 -arch=sm_86 -std=c++11 -c device_info.cu -o device_info.o
if %ERRORLEVEL% NEQ 0 goto :error

nvcc -O2 -arch=sm_86 -std=c++11 -x cu -c main.c -o main.o -I. -D__CUDACC__
if %ERRORLEVEL% NEQ 0 goto :error

echo Линковка...
nvcc -O2 -arch=sm_86 -std=c++11 -o transpose.exe transpose_cpu.o utils.o transpose_cuda.o device_info.o main.o -lcudart
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo === Компиляция успешно завершена! ===
echo Исполняемый файл: transpose.exe
echo.
echo Для запуска: transpose.exe
goto :end

:error
echo.
echo === ОШИБКА при компиляции ===
pause
exit /b 1

:end

