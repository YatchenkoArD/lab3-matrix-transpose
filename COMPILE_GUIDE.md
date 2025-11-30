# Руководство по компиляции проекта

## Требования

1. **CUDA Toolkit 12.0+** (рекомендуется 12.6)
2. **Компилятор C/C++** (один из вариантов):
   - **MinGW-w64** (рекомендуется для Windows)
   - **Visual Studio** с компонентами C++
   - **MSVC** (Microsoft Visual C++)

## Проверка установки

### Проверка CUDA:
```powershell
nvcc --version
```
Должно показать версию CUDA Toolkit.

### Проверка компилятора C:

**Для MinGW:**
```powershell
gcc --version
```

**Для MSVC:**
```powershell
cl
```

## Компиляция

### Вариант 1: Использование batch-скрипта (Windows) - РЕКОМЕНДУЕТСЯ

**С Visual Studio:**
```powershell
.\build_custom.bat
```

**Автоматический выбор компилятора:**
```powershell
.\build.bat
```

**Упрощенная сборка:**
```powershell
.\build_simple.bat
```

**Однострочная команда:**
```powershell
.\build_one_line.bat
```

### Вариант 2: Использование Makefile (Linux/Mac)

```bash
make
```

### Вариант 3: Ручная компиляция

#### Если используете MinGW (gcc):

```powershell
# Компиляция CPU файлов
gcc -Wall -O2 -std=c11 -c transpose_cpu.c -o transpose_cpu.o
gcc -Wall -O2 -std=c11 -c utils.c -o utils.o

# Компиляция CUDA файлов
nvcc -O2 -arch=sm_86 -std=c++11 -c transpose_cuda.cu -o transpose_cuda.o
nvcc -O2 -arch=sm_86 -std=c++11 -c device_info.cu -o device_info.o
nvcc -O2 -arch=sm_86 -std=c++11 -x cu -c main.c -o main.o -I. -D__CUDACC__

# Линковка
nvcc -O2 -arch=sm_86 -std=c++11 -o lab3_transpose.exe transpose_cpu.o utils.o transpose_cuda.o device_info.o main.o -lcudart
```

#### Если используете MSVC:

Сначала откройте "Developer Command Prompt for VS" или выполните:
```powershell
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Затем:
```powershell
# Компиляция CPU файлов
cl /c /O2 /std:c11 transpose_cpu.c
cl /c /O2 /std:c11 utils.c

# Компиляция CUDA файлов
nvcc -O2 -arch=sm_86 -std=c++11 -c transpose_cuda.cu -o transpose_cuda.obj
nvcc -O2 -arch=sm_86 -std=c++11 -c device_info.cu -o device_info.obj
nvcc -O2 -arch=sm_86 -std=c++11 -x cu -c main.c -o main.obj -I. -D__CUDACC__

# Линковка
nvcc -O2 -arch=sm_86 -std=c++11 -o lab3_transpose.exe transpose_cpu.obj utils.obj transpose_cuda.obj device_info.obj main.obj -lcudart
```

#### Однострочная команда (nvcc компилирует все):

```powershell
nvcc -O3 -arch=sm_86 -Xcompiler "/MD /wd4819" -lcudart main.c utils.c transpose_cpu.c transpose_cuda.cu device_info.cu -o lab3_transpose.exe
```

## Архитектура GPU

Ваша видеокарта: **NVIDIA GeForce RTX 3050**
- Compute Capability: **8.6**
- Архитектура в командах: **sm_86**

Если у вас другая видеокарта, замените `sm_86` на соответствующую архитектуру:

| GPU | Compute Capability | Архитектура |
|-----|-------------------|-------------|
| RTX 3050 | 8.6 | sm_86 |
| RTX 3060/3070/3080 | 8.6 | sm_86 |
| RTX 3090 | 8.6 | sm_86 |
| RTX 4090 | 8.9 | sm_89 |
| GTX 1660 | 7.5 | sm_75 |
| GTX 1080 | 6.1 | sm_61 |

## Установка MinGW-w64 (если нужно)

1. Скачайте с https://www.mingw-w64.org/downloads/
2. Или используйте MSYS2: https://www.msys2.org/
3. Добавьте `C:\mingw64\bin` в PATH

## Установка Visual Studio (если нужно)

1. Скачайте Visual Studio Community: https://visualstudio.microsoft.com/
2. При установке выберите "Desktop development with C++"
3. Откройте "Developer Command Prompt for VS" для компиляции

## Запуск программы

После успешной компиляции:

```powershell
.\lab3_transpose.exe
```

Программа автоматически выполнит тесты для квадратных и неквадратных матриц различных размеров.

## Очистка

Удалить скомпилированные файлы:

```powershell
del *.o *.obj lab3_transpose.exe 2>nul
```

Или используйте Makefile:
```bash
make clean
```

## Возможные проблемы

### Ошибка: "nvcc не найден"
- Добавьте CUDA в PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
- Перезапустите терминал

### Ошибка: "gcc не найден"
- Установите MinGW-w64
- Добавьте в PATH: `C:\mingw64\bin`

### Ошибка: "compute_86" не поддерживается
- Обновите CUDA Toolkit до версии 12.0 или выше
- Или используйте более старую архитектуру (например, sm_75)

### Ошибка линковки: "неразрешенный внешний символ"
- Убедитесь, что все файлы включены в компиляцию (включая `device_info.cu`)
- Проверьте, что `cudart.lib` доступен
- Проверьте, что CUDA правильно установлена

### Ошибка: "Could not set up the environment for Microsoft Visual Studio"
- Используйте `call` перед `vcvars64.bat` в скриптах
- Или используйте готовый скрипт `build_custom.bat`
- Или выполните команды в два шага: сначала `vcvars64.bat`, потом компиляцию
