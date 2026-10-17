# Datathon 26 · видео-контроль кассовой дисциплины

Сервис для ForteBank Datathon 26. Пользователь загружает видеозапись работы кассира (`.mov`). Сервис локально, без облачных AI, ищет визуальные признаки нарушений и для каждого показывает:
- тип нарушения;
- таймкод;
- 1–3 кадра-доказательства;
- описание того, что видно на видео;
- ссылку на пункт НПА РК.

## Структура и владельцы

| Папка | Что | Владелец |
|---|---|---|
| `backend/` | FastAPI, очередь заданий, ключи, Docker | Нариман |
| `frontend/` | React UI | Рамазан |
| `ml/` | `pipeline.py` (К2), детекция, правила, VLM, `eval.py` | Агыбай, Нурхан |
| `kb/` | `violations.yaml`, `annotations.csv`, `sources.md` | Алпамыс, Олжас |
| `docs/` | [договорённости К1–К5](docs/contracts.md), архитектура, лицензии, ограничения | все |

Видео и веса моделей в git не кладём. Датасет каждый копирует себе в `data/open/` (`data/open/videos/*.mov`, `data/open/open_labels.xlsx`); папка `data/` уже в `.gitignore`.

## Переменные окружения (`.env`)

Создайте файл `.env` в корне проекта на основе `.env.example`:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `API_KEYS` | `jury_forte26_7f9c2d1b8e,ui_forte26_4a6e8b0d2c` | Список разрешённых ключей авторизации через запятую |
| `JURY_KEY` | `jury_forte26_7f9c2d1b8e` | Ключ доступа для жюри (`X-API-Key`) |
| `UI_KEY` | `ui_forte26_4a6e8b0d2c` | Ключ доступа для UI фронтенда |
| `DATA_DIR` | `./var` | Директория сохранения заданий, загруженных видео и артефактов |
| `WORKERS` | `2` | Количество параллельных воркеров обработки очереди |
| `PIPELINE` | `mock` | Режим работы: `mock` (быстрое тестирование контракта) или `real` (ML) |
| `JOB_TIMEOUT_SEC`| `240` | Сторож времени: при превышении отдаётся быстрый результат вместо `failed` |
| `FRONTEND_DIST` | `frontend/dist` | Директория собранного статического UI (раздаётся на `/`) |
| `MAX_UPLOAD_MB` | `200` | Максимальный размер загружаемого видеофайла в мегабайтах |

---

## Быстрый старт (локально)

### 1. Установка окружения и запуск
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn --factory backend.app.main:app_from_env --host 0.0.0.0 --port 8000
```

### 2. Примеры вызовов API (curl)

**Проверка работоспособности (публичный):**
```bash
curl -s http://localhost:8000/health
# Ответ: {"status":"ok","pipeline":"mock"}
```

**Отправка видео на анализ (Контракт К1):**
```bash
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H "X-API-Key: jury_forte26_7f9c2d1b8e" \
  -F "file=@data/open/videos/X-1.mov"
# Ответ: {"job_id":"a1b2c3d4e5f6","status":"processing"}
```

**Получение результатов анализа:**
```bash
curl -s http://localhost:8000/api/v1/jobs/a1b2c3d4e5f6 \
  -H "X-API-Key: jury_forte26_7f9c2d1b8e"
```

Пример успешного ответа (`status: "done"`):
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Оставление денег в лотке или на стойке без присмотра",
      "timecode_sec": 42.5,
      "description": "Купюры оставлены на клиентской полке без контроля. Нарушение п. 24 Правил ведения кассовых операций (Постановление НБРК № 144).",
      "rule": {
        "document": "Постановление Правления НБРК № 144",
        "clause": "п. 24",
        "quote": "Купюры оставлены на клиентской полке без контроля.",
        "url": "https://adilet.zan.kz/"
      },
      "evidence_frames": [
        {
          "timestamp_sec": 42.5,
          "image_base64": "/9j/4AAQSkZJRgABAQ..."
        }
      ]
    }
  ]
}
```

---

## Запуск в Docker (включая офлайн-режим)

### Стандартный запуск через Compose:
```bash
docker compose up --build
```

### Запуск в полностью изолированной сети (без интернета):
Контейнер сконфигурирован с `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1` (согласно ТЗ п. 7, полная переносимость). Используется внутренняя сеть Docker (`--internal`), блокирующая внешний интернет, но сохраняющая доступность API на хосте:
```bash
# 1. Создать изолированную сеть без внешнего шлюза в интернет
docker network create --internal offline_net

# 2. Запустить контейнер в изолированной сети
docker run --rm \
  --network offline_net \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/var:/data \
  datathon26-api:latest
```

---

## Предотвращение засыпания и мониторинг (`scripts/keepalive.sh`)

Скрипт предотвращает переход macOS в режим сна (через `caffeinate`) и выполняет мониторинг `/health` каждые 5 минут:
```bash
./scripts/keepalive.sh http://127.0.0.1:8000/health 300
```

---

## Тестирование

Полный набор тестов API (К1), валидации схемы, многопоточной очереди, сторожа времени и нагрузочных тестов:
```bash
pytest -v
```

---

## Данные для жюри (M6 / Сдача)

- **Публичный HTTPS URL:** `https://usgs-transcripts-tactics-scott.trycloudflare.com`
- **X-API-Key жюри:** `jury_forte26_7f9c2d1b8e`
- **Swagger / OpenAPI схема:** `https://usgs-transcripts-tactics-scott.trycloudflare.com/docs`
- **Healthcheck:** `https://usgs-transcripts-tactics-scott.trycloudflare.com/health`
