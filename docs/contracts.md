# Договорённости К1–К5

Зафиксированы 17.10 в 13:30. Менять только через Агыбая.

## К1. Ответ API (владелец: Нариман)

Код: [`backend/app/schemas.py`](../backend/app/schemas.py).

`POST /api/v1/analyze`: multipart, поле `file` (`.mov`, также принимаем `.mp4`), заголовок `X-API-Key`.

```json
{ "job_id": "3f2a9c1e04b1", "status": "processing" }
```

`GET /api/v1/jobs/{job_id}` с заголовком `X-API-Key`.

Пока идёт обработка:
```json
{ "job_id": "3f2a9c1e04b1", "status": "processing", "progress": 0.5, "stage": "rules" }
```

Готово (обязательные поля строго по ТЗ; поля из «расширения» необязательные):
```json
{
  "job_id": "3f2a9c1e04b1",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Название типа нарушения",
      "timecode_sec": 22.4,
      "description": "Что видно и почему это может быть нарушением. Основание: <документ>, <пункт>.",
      "evidence_frames": [{ "timestamp_sec": 22.4, "image_base64": "/9j/4AAQ..." }],

      "type_id": "snake_case_id",
      "t_start": 20.0,
      "t_end": 31.0,
      "confidence": 0.87,
      "rule": { "document": "...", "clause": "п. N", "quote": "...", "url": "https://adilet.zan.kz/..." }
    }
  ]
}
```

Ошибка:
```json
{ "job_id": "3f2a9c1e04b1", "status": "failed", "error": "Не удалось обработать видео (...)" }
```

Инварианты (проверяются схемой):
- `verdict = violation` ⇔ список `violations` не пустой;
- `evidence_frames` содержит от 1 до 3 кадров JPEG в base64 без префикса `data:`;
- основание дублируется последней фразой в `description`: жюри проверяет JSON автоматически, а `rule` может не читать.

Коды ответа: `401` — нет ключа или ключ неверный; `415` — формат файла не тот; `413` — файл слишком большой; `400` — пустой файл; `404` — задания нет.

## К2. Интерфейс пайплайна (владелец: Агыбай)

```python
# ml/pipeline.py
def analyze(video_path: str, on_progress=None, time_budget_sec=None) -> dict: ...
```

- Возвращает dict формата `AnalysisResult` (К1 без `job_id` и `status`).
- `on_progress(progress: float 0..1, stage: str)` — для статуса в UI.
- `time_budget_sec`: при превышении вернуть лучший результат на текущий момент (например, без VLM-проверки), а не падать.
- Бэкенд с `PIPELINE=mock` вызывает [`backend/app/mock_pipeline.py`](../backend/app/mock_pipeline.py) с той же сигнатурой.

## К3. Каталог нарушений (владельцы: Алпамыс и Олжас)

[`kb/violations.yaml`](../kb/violations.yaml). Внутри файла описаны правила и шаблон записи.

## К4. Разметка (владелец: Алпамыс)

[`kb/annotations.csv`](../kb/annotations.csv). Колонки описаны в [`kb/README.md`](../kb/README.md).

## К5. Отчёт качества (владелец: Нурхан)

`ml/eval.py` прогоняет `analyze()` по видео из `data/open/videos` и сравнивает результат с `open_labels.xlsx` и `kb/annotations.csv`.

Что считает:
- точность вердикта;
- полноту по нарушениям: пропуск недопустим;
- ложные срабатывания;
- долю событий с таймкодом в пределах ±2 с;
- путаницу типов;
- время обработки одного видео.

Отдельной строкой — результат на отложенном ракурсе.
