@echo off
REM Упрощенный скрипт компиляции - использует только nvcc
REM Работает даже без отдельного компилятора C

echo === Компиляция проекта (только nvcc) ===
echo.

REM Проверка nvcc
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: nvcc не найден
    pause
    exit /b 1
)

echo Компиляция всех файлов через nvcc...
echo.

REM Компиляция через nvcc (может компилировать и C файлы)
nvcc -O2 -arch=sm_86 -std=c++11 -x cu transpose_cpu.c -c -o transpose_cpu.o -I.
if %ERRORLEVEL% NEQ 0 goto :error

nvcc -O2 -arch=sm_86 -std=c++11 -x cu utils.c -c -o utils.o -I.
if %ERRORLEVEL% NEQ 0 goto :error

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
echo === Успешно! ===
echo Запуск: transpose.exe
goto :end

:error
echo.
echo === ОШИБКА ===
pause
exit /b 1

:end

