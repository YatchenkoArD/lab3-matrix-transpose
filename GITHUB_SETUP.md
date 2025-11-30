# Инструкция по отправке проекта на GitHub

## Шаг 1: Инициализация Git репозитория

Откройте терминал в папке проекта и выполните:

```bash
git init
```

## Шаг 2: Настройка Git (если еще не настроен)

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "ваш.email@example.com"
```

## Шаг 3: Добавление файлов

```bash
# Проверьте, какие файлы будут добавлены
git status

# Добавьте все файлы (кроме тех, что в .gitignore)
git add .

# Или добавьте файлы по отдельности:
git add *.c *.cu *.cuh *.h *.bat *.py *.md Makefile .gitignore
```

## Шаг 4: Создание первого коммита

```bash
git commit -m "Initial commit: CUDA matrix transpose project"
```

## Шаг 5: Создание репозитория на GitHub

1. Зайдите на [GitHub.com](https://github.com)
2. Нажмите кнопку **"+"** в правом верхнем углу → **"New repository"**
3. Заполните:
   - **Repository name:** `lab3-matrix-transpose` (или другое имя)
   - **Description:** `CUDA matrix transpose: CPU vs GPU performance comparison`
   - **Visibility:** Public или Private (на ваш выбор)
   - **НЕ** создавайте README, .gitignore или license (они уже есть)
4. Нажмите **"Create repository"**

## Шаг 6: Подключение к GitHub и отправка

GitHub покажет инструкции, но вот команды:

```bash
# Добавьте удаленный репозиторий (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/lab3-matrix-transpose.git

# Или если используете SSH:
# git remote add origin git@github.com:YOUR_USERNAME/lab3-matrix-transpose.git

# Переименуйте ветку в main (если нужно)
git branch -M main

# Отправьте код на GitHub
git push -u origin main
```

## Шаг 7: Проверка

Зайдите на страницу вашего репозитория на GitHub - там должен быть весь код!

## Дополнительные команды

### Просмотр статуса:
```bash
git status
```

### Просмотр изменений:
```bash
git diff
```

### Просмотр истории коммитов:
```bash
git log
```

### Добавление изменений и новый коммит:
```bash
git add .
git commit -m "Описание изменений"
git push
```

## Что будет в репозитории

✅ **Будет включено:**
- Исходный код (`.c`, `.cu`, `.h`, `.cuh`)
- Скрипты сборки (`.bat`, `Makefile`)
- Документация (`.md`)
- Python скрипт для графиков
- `.gitignore`

❌ **НЕ будет включено** (благодаря `.gitignore`):
- Скомпилированные файлы (`.o`, `.obj`, `.exe`)
- Результаты выполнения (`results.csv`, `transpose_plot.png`)
- Временные файлы CUDA
- Системные файлы

## Полезные ссылки

- [GitHub Docs](https://docs.github.com/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)

