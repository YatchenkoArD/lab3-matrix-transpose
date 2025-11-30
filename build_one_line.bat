@echo off
REM Упрощенная однострочная компиляция
REM nvcc сам найдет компилятор, если он в PATH

set "TEMP=C:\Windows\Temp"
set "TMP=C:\Windows\Temp"

echo === Компиляция проекта ===
nvcc -O3 -arch=sm_86 -Xcompiler "/MD /wd4819" -lcudart main.c utils.c transpose_cpu.c transpose_cuda.cu device_info.cu -o lab3_transpose.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Компиляция успешна! Запуск программы... ===
    lab3_transpose.exe
) else (
    echo.
    echo === ОШИБКА компиляции ===
    echo Попробуйте использовать build_custom.bat или сначала выполните:
    echo call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

