# Локальная VLM-проверка CV-кандидатов

## Назначение и ограничения

`ml.vlm` — дополнительная проверка видимых действий, **не юридическое заключение**.
Все результаты требуют просмотра человеком (`advisory_only=true`,
`human_review_required=true`). Идентификация людей/лиц, биометрия, выводы о личности,
чувствительных признаках, намерениях и виновности не запрашиваются. Аудио не читается
и не передаётся модели.

**Только собственные физические локальные CPU/GPU.** Никаких cloud GPU, Vast,
арендованных серверов, облачных API или загрузки видео/кадров/описаний в облачный ИИ.
Веса и зависимости заранее готовят пользователь/родительский агент на собственном
оборудовании. Этот компонент ничего не устанавливает и не скачивает. В ходе данной
работы реальные веса Qwen не загружались, качество модели на видео не проверялось.

## Интеграция с Агыбаем (Agybay)

Агыбай передаёт **весь список кандидатов одного видео одним вызовом**, вместе с
заранее проверенным юридическим каталогом. Здесь нет зависимости от ещё не
определённого формата CV-модуля: адаптер Агыбая должен явно отобразить поля в контракт
ниже. Не следует вызывать функцию отдельно для каждого кандидата: бюджет общий для
всего вызова/видео, а не на кандидата. Повторный самостоятельный вызов начинает новый
бюджет; интегратор не должен повторять один видео-job ради обхода ограничения.

```python
from dataclasses import asdict
from ml.vlm import verify_candidates

report = verify_candidates(
    video_path="C:/local-data/video.mp4",
    candidates=cv_candidates,
    rules=verified_rule_catalog,
    model_path="C:/local-models/qwen2.5-vl",  # существующий каталог, не HF repo ID/URL
    device="cpu",                           # либо собственная локальная cuda:0
    max_calls=6,
    timeout_s=60,
    video_timeout_s=300,
    enable_open_checklist=True,
)
review_queue = report.retained_candidates  # confirmed + uncertain; uncertain НЕ rejected
payload = asdict(report)
payload["retained_candidates"] = review_queue
```

На Windows запуск из скрипта, создающего worker-процесс, должен находиться под
`if __name__ == "__main__":`. CLI ниже уже соблюдает это требование.
`model_path=None`, отсутствующие зависимости/веса и ошибки модели — рабочий
fail-closed сценарий: исходные CV-кандидаты возвращаются как `uncertain`, без
выдуманной инференции. При отсутствии проверенного правила приоритетная причина —
`no_verified_rule`, независимо от наличия модели.

Для кеширования модели между **разными** видео можно явно создать
`LocalTransformersBackend(local_directory, device="cpu")`, передавать `backend=...`
и в `finally` вызывать `backend.close()`. Backend не предназначен для параллельных
вызовов. При `model_path=...` функция сама закрывает принадлежащий ей worker.

## Входные контракты

Экспортированы `Candidate`, `Rule` (`TypedDict`), `Backend`, `FrameSource` (`Protocol`)
и dataclass-контракты кадров/запроса/результата. На API передаются списки словарей.
Ниже **искусственные примеры**, не настоящий закон и не наблюдение реального видео.

```json
{
  "candidate_id": "cv-1",
  "violation_type": "fixture_action",
  "start_s": 0.0,
  "end_s": 1.0,
  "rule_id": "r1",
  "cv_metadata": {"score": 0.81}
}
```

- `candidate_id`: непустая строка, уникальная внутри видео. Дубликаты неоднозначны →
  `uncertain`; идентификаторы не обрезаются/не исправляются автоматически.
- `violation_type`: тип из проверенного каталога.
- `start_s`, `end_s`: конечные числа, не bool; `0 <= start_s <= end_s <= duration_s`.
  Это секунды относительно начала контейнера (его `format.start_time`), а не первого видеокадра и не `номер/FPS`.
- `rule_id`: строка из каталога; отсутствие/null/неизвестный ID не даёт юридического
  основания. Несовпадение типа кандидата и типа правила тоже блокирует подтверждение.
- Дополнительные CV-поля сохраняются в исходном кандидате, но **не входят в prompt**.

```json
{
  "rule_id": "r1",
  "violation_type": "fixture_action",
  "description": "Artificial fixture action only",
  "citation": "FIXTURE §1",
  "verified": true
}
```

`verified` должен быть именно JSON boolean `true`; `rule_id`, `violation_type`,
`description`, `citation` — непустые строки. Любой дубликат `rule_id` исключает это
правило, даже если записи совпадают. Флаг означает, что **владелец каталога** заранее
проверил применимость/актуальность нормы: модуль сам это не устанавливает. Никогда
не помечать автоматически сгенерированный каталог как проверенный.

Модель получает только IDs, типы и описания, но не `citation`. Юридическая ссылка
в результате берётся исключительно из входного проверенного каталога, после
проверки ответа. Без такого правила статус `confirmed` невозможен.

## Модельный JSON и выход

Для обычного кандидата модель обязана вернуть ровно:

```json
{
  "candidate_id": "cv-1",
  "rule_id": "r1",
  "violation": true,
  "confidence": 0.9,
  "insufficient_evidence": false,
  "observations": [{"timestamp_s": 0.5, "text": "Fixture visible action"}]
}
```

Это только пример тестового ответа, **не результат реальной модели**.

Парсер не чинит JSON, не извлекает его из markdown. Дубликаты ключей, неизвестные
поля, выдуманные IDs, `citation` от модели, строковые boolean/числа, bool вместо
confidence, NaN/Infinity и confidence вне `[0,1]` → `uncertain`. Требуются 1–12
наблюдений с непустым текстом не длиннее 1000 символов. Timestamp каждого наблюдения
должен принадлежать реально переданному кадру (допуск 1e-6 s) и лежать в интервале
кандидата/видео. Есть консервативный лексический запрет юридических ссылок в тексте
наблюдения; это **не семантическая гарантия отсутствия галлюцинаций**. Ложное
срабатывание этого фильтра также означает `uncertain`, а не отклонение.

`VerificationResult` содержит:

| Поле | Значение |
|---|---|
| `candidate` | исходный словарь CV-кандидата, без потери дополнительных полей |
| `status` | `confirmed`, `rejected` или `uncertain` |
| `reason` | безопасный машинный код, без текста исключения/пути/кадров |
| `confidence` | проверенное число либо null; не калиброванная вероятность |
| `observations` | проверенные по схеме описания/таймкоды, либо пустой список |
| `rule_id`, `citation` | проверенный ID; ссылка только для confirmed и только из каталога |
| `additional` | true только для нового кандидата open checklist |
| `advisory_only` | всегда true |

- `confirmed`: `violation=true`, достаточные данные, confidence >= 0.7, проверенное
  соответствующее правило. Это подтверждение VLM-гипотезы для ручной проверки,
  **не окончательное установление нарушения**.
- `rejected`: только валидный ответ `violation=false`, достаточные данные и
  confidence >= 0.7. Запись остаётся в `results` для аудита; только эта категория
  исключается из `retained_candidates`.
- `uncertain`: исходный CV-кандидат обязательно остаётся в `results` и
  `retained_candidates`. Нехватка видимости, низкая уверенность, битый ответ,
  отсутствие модели/правила/кадров, timeout и исчерпание бюджета — не отрицание.

Коды причин: `visible_evidence`, `insufficient_evidence`, `invalid_response`,
`invalid_candidate`, `no_verified_rule`, `model_unavailable`, `frames_unavailable`,
`insufficient_frames`, `timeout`, `video_timeout`, `backend_error`, `budget_exhausted`.
Ошибка уже запущенного runtime также может обозначаться `backend_error`; никакие
сырые исключения не включаются. Значение `no_violation` компонент не генерирует.

`VerificationReport`: `results`, `calls_used`, `human_review_required`,
`checklist_status`; свойство `retained_candidates` вычисляется, а не хранится отдельно.
`checklist_status` — диагностика прохода (`disabled`, `completed` либо код причины),
не заключение об отсутствии нарушений. Даже пустой `findings` не означает, что
видео полностью проверено.

## Кадры, бюджет и таймауты

`ffprobe` читает `best_effort_timestamp_time` выбранного видеопотока. Для каждого
кадра сохраняются исходный `pts_s` и нормализованный `timestamp_s`. `ffmpeg` выбирает
соответствующий индекс декодированного кадра (`select=eq(n,index)`), не предполагая
постоянный FPS. Среди кадров интервала выбирается до трёх распределённых по
последовательности; для checklist — до шести по всему видео. JPEG/PNG из внешних
сервисов не используются; временные PNG удаляются после запроса. `-an -sn -dn`
исключает аудио/субтитры/данные. Протоколы ffmpeg ограничены `file,pipe`, URL/UNC
на входе источника кадров не принимаются.

Общий `max_calls` — целое 0..6, **включая** open checklist, ошибки/таймауты
инференса. Сначала проверяются исходные кандидаты; отдельной скрытой квоты для
checklist нет. Остальные кандидаты при исчерпании квоты остаются uncertain.
Проверка каталога/ffprobe не являются вызовами модели. Timeout по умолчанию:
60 s на шаг/инференс, суммарный дедлайн видеопрохода 300 s. Учитывается время
загрузки модели; оставшееся время ограничивает последующие subprocess-запуски.

Реальная модель живёт в отдельном локальном spawn-процессе. Зависший процесс
завершается/убивается при timeout; веса кешируются между запросами одного backend.
Инъецированный тестовый backend ограничен daemon-thread timeout: поток Python
принудительно убить нельзя. Это тестовый/доверенный seam, не замена production
process isolation. Сам `FrameSource` обязан соблюдать переданный timeout;
production `FFmpegFrames` делает это через `subprocess.run(timeout=...)`.

Open checklist разрешает **только типы из preverified каталога** и один модельный
вызов с `{"findings": [...]}` (до 6 findings). У каждого finding схема выше без
`candidate_id`. ID нового кандидата `vlm-open-N` назначает код, тип и citation —
каталог, временной интервал — проверенные timestamps наблюдений. Поддержанная,
но неуверенная положительная гипотеза сохраняется как дополнительный uncertain.
Выдуманное правило/ссылка/таймкод делает проход invalid_response без изменения
исходных CV-результатов. Нельзя расширять юридический каталог ответом модели.

## Runtime и CLI

Импорт `ml.vlm` не импортирует torch/PIL/transformers. Требования перечислены в
`requirements-ml-vlm.txt`; ffmpeg/ffprobe должны быть на локальном PATH. Модуль
принудительно выставляет `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, отключает
HF telemetry. Для всех `from_pretrained`: `local_files_only=True`,
`trust_remote_code=False`; веса только safetensors. Qwen2.5-VL загружается через
`Qwen2_5_VLForConditionalGeneration`; совместимые локальные chat/image-to-text
архитектуры — через `AutoModelForImageTextToText`. Нестандартный remote Python-код
не запускается. Модели с несовместимым processor/chat template возвращают uncertain.
Загрузчик запрашивает `output_loading_info=True` и отклоняет missing/mismatched/error diagnostics:
случайно инициализированные недостающие параметры не считаются рабочими весами.
Неполный каталог проверяется до запуска тяжёлого worker. Неверные `weight_map` дают
явную недоступность, сохраняя CV-кандидатов как uncertain.

```bash
py -3.10 -B -m ml.vlm preflight --model "C:/local-models/qwen2.5-vl"
py -3.10 -B -m ml.vlm preflight --model "C:/local-models/qwen2.5-vl" --load --timeout 120
py -3.10 -B -m ml.vlm verify --video "C:/local-data/video.mp4" --model "C:/local-models/qwen2.5-vl" --candidates "C:/local-data/candidates.json" --rules "C:/local-data/rules.json" --output "C:/local-data/vlm-result.json" --open-checklist
```

`candidates.json`, `rules.json` содержат JSON-массивы. CLI-парсер принимает строгий
JSON размером до 64 KiB на файл. Результат записывается **только в явно заданный
локальный output**, stdout содержит лишь счётчики/статус. Нет логирования видео,
кадров, prompt или описаний; stdout/stderr дочерней модели подавлены. Файл результата
содержит evidence, поэтому его нельзя отправлять во внешние системы/логи.
Код выхода preflight: 0 при успешных проверках, 2 при недоступности. `ready=true`
без `--load` означает лишь структурную проверку файлов/зависимостей, не успешную
загрузку и тем более не точность. `runtime_loaded=true` возможен только после
реального `--load`. Verify: 0 — отчёт сформирован (в том числе весь uncertain),
2 — неверный вход/не удалось записать output. Output не может совпадать с входами.

## Реальный CPU smoke (не benchmark)

На собственной физической машине отдельно загружены и проверены SHA256 веса `HuggingFaceTB/SmolVLM-256M-Instruct` (Apache-2.0; точные реквизиты в `licenses-ml.md`). Пакеты для этого smoke находятся в отдельной локальной среде, с Transformers 4.57.6. Preflight с `--load` вернул `ready=true`, `runtime_loaded=true`, `model_type=idefics3`. На синтетическом изображении реально выполнена генерация, без подмены ответа.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  artifacts/runtime-vlm/Scripts/python.exe -m ml.vlm preflight \
  --model models/smolvlm-256m-instruct --load --timeout 120
```

Исходный SmolVLM processor увеличивает longest edge до 2048 и разбивает изображение на patches: на нескольких кадрах это резко повышает расход CPU RAM и приводило к потере worker на данной машине. Для **локального CPU smoke-профиля** явно установлен `size.longest_edge=512` в локальном `preprocessor_config.json`; исходная конфигурация сохранена отдельно, веса не менялись. Это ограничение качества/памяти, а не безусловно рекомендуемый production-профиль.

Проверка одного реального локального фрагмента с тремя кадрами прошла через настоящий backend и генерацию. Внешние socket connect/send/DNS запрещались, HF работал offline. Ответ модели не прошёл строгий JSON-контракт: получено `uncertain/invalid_response`, исходный кандидат сохранён. Это **не успешная классификация** и не доказательство снижения ложных срабатываний. Таймкод кандидата был взят из разметки исключительно для smoke, поэтому результат исключён из любых метрик качества. Нужны калибровка модели/промпта и проверка по независимым кандидатам реального CV-пайплайна.

## Проверка и реальные RED/GREEN evidence

Команда без установки пакетов:

```bash
python -B -m unittest discover -s tests -p test_vlm.py -v
py -3.10 -B -m unittest discover -s tests -p test_vlm.py -q
```

На данной машине выполнено **21 тест, все прошли** на Python 3.14 и 3.10.
Это проверки кода и искусственных fixtures, не оценка VLM.
Вертикальные slices выполнялись по одному: сначала запуск с ожидаемым FAIL,
затем минимальная реализация и повторный GREEN всего `test_vlm.py`.

| Slice | Наблюдавшийся RED | GREEN после реализации |
|---|---|---|
| Сохранение CV при недоступности | verification API отсутствует | 1 тест |
| Подтверждение с каталогом/кадрами | uncertain != confirmed | 2 |
| Строгий JSON | 18 неверных подтверждений, 5 исключений парсинга | 3 |
| false/нехватка evidence/низкая confidence | confirmed вместо rejected/uncertain | 4 |
| Проверенный уникальный каталог | неподходящий/дублированный rule подтверждается | 5 |
| Общий budget | 8 calls вместо 6 | 6 |
| Open checklist | 1 call вместо 2 | 7 |
| Сбой backend | исключение вместо сохранённого uncertain | 8 |
| Неверные входы/нет кадров | 7 fail-closed нарушений | 9 |
| Реальный ffmpeg + искусственный VFR | локального frame source нет | 10 |
| Preflight | отсутствует local preflight | 11 |
| Transformers adapter | нет local backend | 12 |
| Killable timeout | нет закрываемого worker | 13 |
| Дедлайн | confirmed вместо uncertain при зависании | 14 |
| Offline default + socket poison | unavailable вместо явного model_unavailable | 15 |
| Ошибка extraction/приватность prompt | необработанное исключение | 16 |
| CLI | отсутствует ml.vlm.__main__ | 17 |
| Реальная runtime validation ошибки | нет validate | 18 |
| Дубликат IDs/битый rule_id/огромное число | подтверждение/TypeError/OverflowError | 19 |
| Неуверенная open-гипотеза | исчезает вместо uncertain | 20 |
| Юридический текст от модели | confirmed вместо uncertain | 21 |

В тестах scripted backend определён **только в tests/test_vlm.py**, явным DI.
Socket/create_connection/getaddrinfo отравляются исключениями в offline-тесте.
Контракт настоящего adapter проверяется искусственными зависимостями; это не
подмена отсутствующих весов в production. Создано и обработано ffmpeg-видео из
искусственного чёрного источника с VFR PTS **0, 0.5, 1.5, 2.5**; проверены реальные
PNG и очистка временных файлов, без визуального просмотра и без реальных видео.
Также реально запущены spawn-процесс timeout, runtime-ошибка недоступной модели и CLI.

Фактический CLI preflight для отсутствующего локального каталога вернул:
`ready=false`, `runtime_loaded=false`, ошибки `model_directory_required`,
`dependency_missing:torch`, `dependency_missing:PIL`, `dependency_missing:transformers`.
Никакой инференс реальными весами, полнота обнаружения и точность Qwen **не подтверждены**.

Изменённые/созданные owned paths: `ml/vlm/`, `tests/test_vlm.py`,
`docs/ml-vlm.md`, `requirements-ml-vlm.txt`. Git-операций не выполнялось;
чужие файлы и `IDEA.md` не изменялись.
