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

Опциональная локальная VLM для поиска пропущенных типов и воспроизводимый эксперимент: [VLM и качество типов](docs/vlm-type-quality.md). Подготовка независимого видео другого ракурса: [angle holdout](docs/angle-holdout.md).

## Переменные окружения (`.env`)

Создайте файл `.env` в корне проекта на основе `.env.example`:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `API_KEYS` | *обязательная* | Список разрешённых ключей авторизации через запятую |
| `DATA_DIR` | `./var` | Директория сохранения заданий, загруженных видео и артефактов |
| `WORKERS` | `2` | Количество параллельных воркеров обработки очереди |
| `PIPELINE` | `mock` | Режим работы: `mock` (быстрое тестирование контракта) или `real` (ML) |
| `PIPELINE_TIMEOUT_SEC` | `200` | Мягкий бюджет времени ML (по К2 пайплайн возвращает лучший результат) |
| `JOB_TIMEOUT_SEC` | `270` | Жёсткий сторож времени от `created_at` (при превышении ставится `failed`) |
| `FRONTEND_DIST` | Автопоиск `frontend/dist` | Директория статики UI (при отсутствии отдаётся встроенная витрина на `/`) |
| `MAX_UPLOAD_MB` | `200` | Максимальный размер загружаемого видеофайла в мегабайтах |

---

## Быстрый старт (локально)

### 1. Установка окружения и запуск
```bash
# Системная зависимость для извлечения кадров-доказательств:
brew install ffmpeg  # macOS (или sudo apt install -y ffmpeg на Linux)

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # задать свои ключи
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
  -H "X-API-Key: change-me-jury-key" \
  -F "file=@data/open/videos/X-1.mov"
# Ответ: {"job_id":"a1b2c3d4e5f6","status":"processing"}
```

**Получение результатов анализа:**
```bash
curl -s http://localhost:8000/api/v1/jobs/a1b2c3d4e5f6 \
  -H "X-API-Key: change-me-jury-key"
```

Пример успешного ответа (`status: "done"`):
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Возврат банкнот клиенту после проверки",
      "timecode_sec": 22.0,
      "description": "Кассир извлекает банкноты из машины и возвращает клиенту.",
      "rule": {
        "document": "Постановление Правления НБРК от 29.11.2019 № 231 (рег. № 19680)",
        "clause": "п. 59–61",
        "quote": "Банкноты и монеты, имеющие признаки подделки, клиенту не возвращаются, сумма принятых банкнот и монет на банковский счет клиента не зачисляется.",
        "url": "https://adilet.zan.kz/rus/docs/V1900019680"
      },
      "evidence_frames": [
        {
          "timestamp_sec": 22.0,
          "image_base64": "/9j/4AAQSkZJRgABAQ..."
        }
      ]
    }
  ]
}
```

Как ведёт себя mock:
- если в имени файла есть `clean`, вердикт будет `no_violation`;
- любой другой файл даёт два демо-нарушения с настоящими кадрами из загруженного видео.

---

## Запуск в Docker (включая офлайн-режим)

### Стандартный запуск через Compose:
```bash
docker compose up --build
```
После запуска:
- **Веб-интерфейс (UI)** доступен в браузере по адресу `http://localhost:8000/`. В `Dockerfile` настроена multi-stage сборка — фронтенд компилируется автоматически и подключается к бэкенду с защитой по `X-API-Key`.
- **REST API** доступен по адресу `http://localhost:8000/api/v1/`.
- **Healthcheck:** `http://localhost:8000/health`.
- **Документация Swagger:** `http://localhost:8000/docs`.

### Запуск в полностью изолированной сети (без интернета):
Контейнер сконфигурирован с `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1` (согласно ТЗ п. 7, полная переносимость). Для проверки автономности создаётся внутренняя сеть Docker без шлюза в интернет:
```bash
# 1. Создать изолированную сеть без внешнего шлюза в интернет
docker network create --internal offline_net

# 2. Запустить контейнер API в изолированной сети
docker run -d --name datathon-api \
  --network offline_net \
  --env-file .env \
  -e DATA_DIR=/data \
  -v $(pwd)/var:/data \
  datathon26-api:latest

# 3. Проверить доступность API клиентом из этой же изолированной сети
docker run --rm --network offline_net curlimages/curl:latest -s http://datathon-api:8000/health
```

---

## Публичный HTTPS-туннель и мониторинг (`scripts/keepalive.sh`)

### Runbook запуска туннеля (Cloudflare Tunnel):
Для предоставления внешнего HTTPS URL жюри запускается туннель к локальному uvicorn:
```bash
# Запуск quick tunnel
cloudflared tunnel --url http://127.0.0.1:8000
# Для named tunnel (со стабильным доменом) используется:
# cloudflared tunnel run <TUNNEL_NAME>
```

### Предотвращение засыпания и мониторинг:
Скрипт `scripts/keepalive.sh` предотвращает переход macOS в режим сна (через `caffeinate`) и выполняет сквозной мониторинг публичного Cloudflare туннеля и локального сервиса `/health` каждые 5 минут:
```bash
./scripts/keepalive.sh https://usgs-transcripts-tactics-scott.trycloudflare.com/health 300 var/keepalive_health.log
```

---

## Тестирование

Реальный пайплайн, бюджет сцены, полный прогон датасета и известные ограничения описаны в [docs/scene-runtime.md](docs/scene-runtime.md).

Полный набор тестов API (К1), JPEG/base64, очереди, сторожа времени и ML-пайплайна:
```bash
pytest -v
```

---

## Данные для жюри (M6 / Сдача)

- **Публичный HTTPS URL:** `https://usgs-transcripts-tactics-scott.trycloudflare.com`
- **X-API-Key жюри:** передаётся жюри конфиденциально перед сдачей (в репозитории не хранится)
- **Swagger / OpenAPI схема:** `https://usgs-transcripts-tactics-scott.trycloudflare.com/docs`
- **Healthcheck:** `https://usgs-transcripts-tactics-scott.trycloudflare.com/health`

---

## Правила

- Ветка `main` всегда должна деплоиться. Работаем в своих ветках, в `main` попадаем через PR.
- Договорённости из [docs/contracts.md](docs/contracts.md) меняем только через Агыбая.
- Модели берём только с лицензиями Apache-2.0, MIT или BSD. Ultralytics (AGPL) и веса с NC-лицензией не используем. Каждую модель записываем в `docs/licenses.md`.
- Видео и кадры не загружаем ни в какие облачные сервисы.
