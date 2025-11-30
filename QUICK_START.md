# Быстрый старт

## Что уже есть ✅
- CUDA Toolkit 12.6 установлен
- nvcc доступен

## Что нужно проверить

### 1. Компилятор C

Откройте PowerShell и выполните:

```powershell
gcc --version
```

**Если gcc найден** → можно компилировать сразу!

**Если gcc НЕ найден**, установите один из вариантов:

#### Вариант A: MinGW-w64 (рекомендуется)
1. Скачайте: https://sourceforge.net/projects/mingw-w64/
2. Или через MSYS2: https://www.msys2.org/
3. Добавьте `C:\mingw64\bin` в PATH

#### Вариант B: Visual Studio
1. Скачайте Visual Studio Community (бесплатно)
2. При установке выберите "Desktop development with C++"
3. Используйте "Developer Command Prompt for VS"

## Компиляция

### Самый простой способ:

```powershell
.\build.bat
```

### Или вручную (если gcc установлен):

```powershell
# Компиляция
gcc -Wall -O2 -std=c11 -c transpose_cpu.c -o transpose_cpu.o
gcc -Wall -O2 -std=c11 -c utils.c -o utils.o
nvcc -O2 -arch=sm_86 -std=c++11 -c transpose_cuda.cu -o transpose_cuda.o
nvcc -O2 -arch=sm_86 -std=c++11 -x cu -c main.c -o main.o -I. -D__CUDACC__
nvcc -O2 -arch=sm_86 -std=c++11 -o transpose.exe transpose_cpu.o utils.o transpose_cuda.o main.o -lcudart
```

## Запуск

```powershell
.\transpose.exe
```

## Важно!

Ваша видеокарта RTX 3050 имеет архитектуру **sm_86**, которая уже указана в Makefile.

Если возникнут ошибки компиляции, проверьте:
1. CUDA в PATH
2. Компилятор C установлен
3. Архитектура GPU правильная (sm_86 для RTX 3050)

