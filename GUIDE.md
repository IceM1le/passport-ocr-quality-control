# Гайд по запуску и сдаче

## 1. Проверить проект

```powershell
python -m unittest discover -s tests -v
```

## 2. Проверить train отдельно от test

```powershell
python postprocess.py data/train_predictions.jsonl train_answers.jsonl
python evaluate.py train_answers.jsonl data/train_labels.jsonl
python validate_train.py train_answers.jsonl data/train_labels.jsonl
```

Не сравнивай ответы на test с `train_labels.jsonl`: идентификаторы и наборы документов различаются.

## 3. Сформировать файл сдачи

```powershell
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

Проверь, что в выводе `Processed: 2000`, а `answers.jsonl` содержит 2000 уникальных `doc_id`. В принятых строках должны быть все 10 полей; у отклонённых строк ключ `fields` отсутствует.

## 4. Перед отправкой в GitHub

```powershell
git status
git diff --stat
git diff
```

Убедись, что не попали `.venv/`, `.idea/`, исходные `data/*.jsonl`, локальные логи и секреты. `answers.jsonl` — требуемый результат задания; он содержит распознанные персональные поля, поэтому репозиторий оставь приватным, если владелец данных не разрешил публикацию.
