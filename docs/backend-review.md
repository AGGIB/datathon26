# API Наримана — анализ командного репозитория

Источник: https://github.com/AGGIB/datathon26
Дата: 17.10.2026.
Просмотренный main: 5a7db52640821124c549ae35e837be6ab5938153.
Ветка Наримана: nariman/mock-api-tunnel, commit 33fd08776d35606037523e970fd26676b057def1.

Копия для анализа загружена отдельно в /private/tmp/datathon26-api-review, без checkout и запуска кода. Текущий фронтенд не изменён.

## Найденный API

Реализация: backend/app/main.py (FastAPI). Бэкенд уже находится в main. README прямо указывает владельца backend/: Нариман.

- GET /health без ключа, результат status и pipeline.
- POST /api/v1/analyze, multipart file, X-API-Key. Возвращает job_id и status=processing.
- GET /api/v1/jobs/{job_id}, X-API-Key. Статусы processing, done, failed.
- Стандартная документация FastAPI: /docs и /openapi.json (пути не переопределены).
- Принимаются MOV и MP4, до 200 МиБ. Проверяется заголовок контейнера.
- CORS разрешает любой origin, GET/POST, X-API-Key и Content-Type.

В README, документации, настройках и ветке Наримана публичный URL туннеля не найден. Документирован локальный адрес http://localhost:8000. Это адрес на машине, где запущен backend; localhost Рамазана не указывает на ноутбук Наримана.

## Запуск и окружение

Из корня отдельного клона команды, по README:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
cp .env.example .env
# Задать собственные API_KEYS; не использовать примерные значения для публикации.
set -a
source .env
set +a
uvicorn --factory backend.app.main:app_from_env --host 0.0.0.0 --port 8000
```

API_KEYS — список ключей через запятую. PIPELINE=mock, WORKERS=2, DATA_DIR=./var. Это параметры бэкенда, не VITE-переменные.

Ветка Наримана добавляет автозагрузку .env; в просмотренном main её нет. Для main нужно экспортировать переменные, как в README. Не выполнять source для непроверенных чужих env-файлов.

## Адаптация фронтенда выполнена в ветке ramazan

1. `violation` / `no_violation` преобразуются в точные русские вердикты.
2. `{timestamp_sec,image_base64}` валидируются и отображаются как JPEG data URL. HTML/SVG не разрешены, нормативные URL по-прежнему принимают только HTTP(S).
3. Этапы decode/detect/rules/verify переведены; progress отображается в процентах.
4. Production без переменных использует API на том же origin; явное демо задаётся VITE_DEMO_MODE=true. Dev без настроек остаётся локальной демонстрацией.
5. MOV и MP4 разрешены в UI. Отдельный endpoint перекодированного превью по-прежнему требуется согласовать.

Подробности запуска: frontend/README.md. Контрактные проверки: npm test из frontend/.

## Готовность серверной части

PIPELINE=mock — заглушка. Кадры извлекаются ffmpeg, при сбое используется placeholder. Описания и нормативные основания помечены как демо. ml/pipeline.py в main выбрасывает NotImplementedError: реальный ML ещё не подключён в просмотренной версии.

Замечание для команды: обещанный clean-сценарий mock проверяет имя пути, однако загрузчик сохраняет каждый файл как input.mov/input.mp4. Исходное имя с clean до этой проверки не доходит; этот сценарий может не давать no_violation через HTTP API. По коду это несоответствие, реальный серверный прогон не выполнялся.

Живой API и /health не проверялись: адрес развёртывания не найден. Реальные ключи не извлекались и в отчёт не включены.

## Локальная проверка после адаптации

Временное Python-окружение с зависимостями backend/requirements.txt и httpx 0.28.1. Через FastAPI TestClient проверены /health, 401 без ключа, 415 для неподдерживаемого файла, POST analyze → GET job → done. Настоящий JSON API пропущен через frontend/src/api-contract.ts в Node: две карточки и шесть JPEG data URL сформированы успешно. Проверены также статическая главная страница, логотип и локальный шрифт через FRONTEND_DIST.

Это проверка локального mock-контракта: синтетический контейнер MOV, кадры-заглушки (ffmpeg отсутствует). Реальные видео, публичный сервер и ML этим прогоном не проверялись. Дополнительно прошли семь контрактных тестов фронтенда, TypeScript, Vite build и проверка Prettier.
