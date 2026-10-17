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

## Быстрый старт (бэкенд на mock)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
cp .env.example .env            # поменять ключи
set -a && source .env && set +a
uvicorn --factory backend.app.main:app_from_env --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/v1/analyze -H "X-API-Key: change-me-jury-key" -F "file=@data/open/videos/X-1.mov"
curl -s localhost:8000/api/v1/jobs/<job_id> -H "X-API-Key: change-me-jury-key"
```

Как ведёт себя mock:
- если в имени файла есть `clean`, вердикт будет `no_violation`;
- любой другой файл даёт два демо-нарушения с настоящими кадрами из загруженного видео.

## Тесты

```bash
pytest --cov=backend --cov=ml
```

## Docker

```bash
docker compose up --build
```

## Правила

- Ветка `main` всегда должна деплоиться. Работаем в своих ветках, в `main` попадаем через PR.
- Договорённости из [docs/contracts.md](docs/contracts.md) меняем только через Агыбая.
- Модели берём только с лицензиями Apache-2.0, MIT или BSD. Ultralytics (AGPL) и веса с NC-лицензией не используем. Каждую модель записываем в `docs/licenses.md`.
- Видео и кадры не загружаем ни в какие облачные сервисы.

## Фронтенд Рамазана

Готовый React/Vite-интерфейс в стилистике Forte. Запуск и подключение к mock/real API: [frontend/README.md](frontend/README.md).

```bash
cd frontend
npm ci
npm run dev
```

Материалы выступления: [слайды](docs/pitch.html), [текст питча](docs/pitch-script.md), [сценарий демо](docs/demo-script.md).
