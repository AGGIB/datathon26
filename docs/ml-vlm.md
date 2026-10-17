# Локальная VLM-проверка CV-кандидатов

## Границы и production-контракт

`ml.vlm` — advisory-проверка видимых действий для человека, не юридическое
заключение и не замена CV. Production **сохраняет все исходные Candidate**, включая
`rejected`, ошибки, отсутствие модели/нормы/кадров и timeout. Объекты не
пересоздаются: identity, порядок, повторы, observation, t_peak, evidence_times,
highlight_boxes и CV confidence не меняются. Калибровка, разрешающая фильтрацию
или снижение confidence, не выполнена.

Только собственная физическая локальная машина CPU/GPU. Никаких cloud GPU,
арендованных серверов, внешних inference API или отправки видео, кадров, prompt,
ответов модели и описаний сцен внешнему ИИ. Веса/зависимости готовятся заранее;
модуль ничего не скачивает. Аудио, идентификация людей/лиц, биометрия,
чувствительные признаки, намерения и виновность не запрашиваются.

## Startup отдельно от остатка К2

```python
from ml import vlm
from ml.kb import catalog

# Один раз при запуске сервиса, ДО обработки заданий/бюджета К2.
# Windows: создание worker-процессов под if __name__ == "__main__".
ready = vlm.configure(
    model_path="C:/local-models/qwen2.5-vl",  # локальный каталог, не HF ID/URL
    catalog=catalog(),                       # проверенный загрузчик владельца
    device="cpu",                           # либо собственная cuda:0
    startup_timeout_s=120,
    max_new_tokens=128,                      # строго целое 1..128
)

# Один вызов на ПОЛНЫЙ список кандидатов видео; остаток считает runner.
assembly_candidates = vlm.verify(cv_candidates, video_path, remaining_k2_sec)
assert isinstance(assembly_candidates, list)
assert all(a is b for a, b in zip(assembly_candidates, cv_candidates))
```

`configure(...) -> bool` проверяет/загружает processor/model один раз. `False` —
недоступность runtime, не отсутствие нарушений. После startup failure backend
закрыт и `verify` не повторяет загрузку. Без аргумента catalog startup лениво
вызывает `ml.kb.catalog()`; отсутствие владельческого модуля даёт пустой каталог
и fail-closed. Сырой YAML не заменяет проверенный каталог.

`verify(candidates, video_path, budget_sec) -> list[исходный Candidate]` не
загружает модель и не проверяет файлы весов. Без конфигурации возвращает оригиналы.
`budget_sec` обязателен: конечное положительное число, не bool. Отсутствующий,
нулевой/отрицательный, NaN/Infinity бюджет отвергается. Это **остаток общего К2**,
не новый лимит 300 s. Повторными вызовами нельзя обходить бюджет одного job.

Абсолютный monotonic deadline создаётся до materialization/adaptation кандидатов
и каталога и передаётся через runtime → adapter → verifier. Импорты и подготовка
verifier тоже расходуют исходный остаток: новый относительный budget после них
не выдаётся. Если preparation уже исчерпала лимит, probe/extract/infer не
начинаются, все оригиналы сохраняются. Это не позволяет прервать зависшее
пользовательское property/iterator: синхронная адаптация сама не killable.

Backend — один cached model в spawn-процессе. Nonblocking lock защищает целиком
validate/infer/close: конкурентный вызов сразу получает
`ModelUnavailable("runtime_busy")`, не ждёт чужое задание, не пишет/читает Pipe
и не закрывает активный worker. Production переводит busy в advisory
backend_error и сохраняет все Candidate без изменения confidence. Очереди и
неограниченного ожидания нет; configure/reconfigure выполняют при отсутствии
активных заданий. После
смерти/timeout/close worker не перезагружается даже повторным `validate`: нужна
явная новая startup-конфигурация/новый экземпляр. `vlm.configure(catalog={})`
закрывает принадлежащий модулю worker. Успешно инъецированный backend принадлежит
вызывающему коду и закрывается им.

Ошибка ОС в Process.start у model/frame worker закрывает оба конца Pipe и
дескриптор созданного, но не запущенного Process, без join unstarted process.
Cleanup идемпотентен; configure возвращает False при ошибке старта backend,
verify сохраняет оригиналы, повторного запуска нет.

## Кадры: CV index/FPS не равен PTS

Production `verify` по умолчанию создаёт per-video `OwnerFrames`. Он лениво
импортирует `ml.video`/OpenCV **только внутри отдельного frame worker**, вызывает
владельческие `probe` и `frame_at(path, t_sec)`, сохраняет их семантику:
`round(t * fps)` с ограничением границ и время `index / fps`, как CV `sample_frames`.
Это **не истинный container PTS**, особенно при VFR/смещении видеопотока. Поле
legacy-контракта `pts_s` здесь — та же координата index/FPS, не обещание PTS.
VLM probe кешируется на весь проход. Полного `ffprobe -show_frames` в production
CV-пути нет. Сам owner `frame_at` внутри повторяет metadata probe и декодирует
последовательно; владельческий код не изменён и не считается быстрым seek.

Owner `frame_at` не принимает timeout, поэтому одного передачи числа недостаточно.
Родительский frame worker ограничен оставшимся дедлайном и terminate/kill завершает
декодирование до очистки временных кадров. Между кадрами проверяется абсолютный
deadline: после истечения не начинаются следующий frame_at, сохранение изображения
или inference. Ошибка/timeout делает источник недоступным до конца прохода, без
reload. Создание worker, imports, probe, загрузка кадров и inference входят в
остаток К2. Завершение процессов/очистка требуют небольшой задержки ОС; это не
доказательство жёсткого real-time SLA.

Legacy `FFmpegFrames` остаётся PTS-источником: ffprobe читает реальные
`best_effort_timestamp_time` относительно container `format.start_time`, ffmpeg
выбирает индекс декодированного кадра, без предположения CFR. Он **не совместим
с CV index/FPS автоматически**. Для него и injected источников сохранён явный
`timebase_verified=True` после независимой проверки/перевода таймкодов. По
умолчанию модель не вызывается: `uncertain/unverified_timebase`. Предохранитель
не снят и не заменён молчаливым переименованием времён.

## Адаптер CV/К3 и низкоуровневый advisory API

Поддерживаются оба владельческих RuleText: legacy `basis` и новый
`document/number_date/clause` (pipeline `215b51a`, без `basis`). Только при
отсутствии legacy-поля цитата собирается из непустых строк проверенного
источника; явный пустой `basis` не получает fallback. Модель не пишет цитату,
а статус verified по-прежнему должен прийти из владельческого catalog loader.

Проверка на неизменённых Candidate/KB/assemble/video этого snapshot:
17 типов / 16 правил, все 17 исходных объектов сохранены после 6 scripted
advisory rejects; OwnerFrames действительно декодировал 3 кадра синтетического
MP4. Pending-норма осталась в валидном К1 без rule. Только Box type alias stub;
это совместимость и декодирование, не pretrained inference/детектор/качество.

Новый `configure` отдельно проверен с настоящим локальным SmolVLM-256M, CPU2,
3 синтетическими 96×96 кадрами и 128 tokens: startup 13.4844 s, warm generation
8.4577 / 8.2003 s, один worker. Strict JSON object не означает прохождение
violation-schema или end-to-end SLA; отчёт — в `ml-verification.md`.

```python
from ml.vlm import verify_pipeline_candidates

review = verify_pipeline_candidates(
    video_path, cv_candidates, validated_catalog,
    backend=already_validated_backend,
    frame_source=pts_frame_source,
    timebase_verified=False,  # True только после проверки timeline
    budget_sec=remaining_k2_sec,
    timeout_s=60,             # cap шага, не новый бюджет видео
    video_timeout_s=30,       # optional legacy cap: только сокращает остаток
)
assembly_candidates = review.retained_candidates  # tuple исходных объектов
advisory_report = review.verification
```

`allow_rejection` принят только для совместимости: строго bool и **никогда не
разрешает удаление**. Оба уровня retained_candidates сохраняют rejected.
Верхний tuple — объекты владельца; вложенный `VerificationReport` — advisory
словари, не вход сборщика К1. `timebase_verified` также строго bool.

Адаптер назначает уникальные cv-N в порядке входа, отображает type_id на
violation_type и t_start/t_end на start_s/end_s. Rule_id — literal type_id,
ключ маршрутизации, не придуманный номер нормы. Используются только совпадающие
key/id, status="verified", непустые document/summary/RuleText.basis из
проверенного ml.kb. Draft Т04/unknown/непроверенная норма не расходуют inference.
CV observation/boxes/confidence и citation не входят в prompt. Open checklist
в production/адаптере выключен: findings не превращаются в выдуманные CV-объекты.

Низкоуровневый `verify_candidates(video_path, candidates, rules, *, backend=...,
frame_source=..., budget_sec=...)` принимает словари, возвращает VerificationReport
и **никогда не создаёт/загружает backend**. Старые model_path/device инертны.
Передавать подготовленный backend явно:

```python
from ml.vlm import LocalTransformersBackend, verify_candidates

backend = LocalTransformersBackend(local_directory, max_new_tokens=128)
try:
    backend.validate(timeout_s=120)  # startup отдельно
    report = verify_candidates(
        video_path, dictionary_candidates, verified_dictionary_rules,
        backend=backend, frame_source=pts_frame_source,
        budget_sec=remaining_k2_sec,
    )
finally:
    backend.close()
```

Вход кандидата: candidate_id (уникальная непустая строка), violation_type,
rule_id (строка/null), start_s/end_s (конечные числа, не bool,
0 <= start_s <= end_s <= duration_s в timeline выбранного источника).
Дополнительные поля сохраняются, но не отправляются модели. IDs не обрезаются и
не нормализуются. Rule-словарь: непустые rule_id, violation_type, description,
citation и строго verified=True; дубликаты rule_id исключаются. Цитаты берутся
только из входного проверенного каталога, никогда из ответа модели.

## JSON, budget и fail-closed

Искусственный пример ответа, **не результат реальной генерации**:

```json
{"candidate_id":"cv-1","rule_id":"fixture_action","violation":false,"confidence":0.9,"insufficient_evidence":false,"observations":[{"timestamp_s":0.5,"text":"Synthetic visible action"}]}
```

Только строгий JSON с ровно указанными полями. Markdown, дубликаты ключей,
неизвестные поля/IDs, citation от модели, NaN/Infinity, bool вместо confidence,
неправильные типы и наблюдения вне реально переданных кадров дают uncertain.
Нехватка видимости или confidence < 0.7 — uncertain. Валидный violation=false
при достаточных данных может дать advisory rejected, **но оригинал сохраняется**.
Confidence не калибрована. Лексический запрет юридического текста в observations
не является семантической гарантией отсутствия галлюцинаций.

VerificationReport: results, calls_used, human_review_required=True,
checklist_status. VerificationResult: исходный словарь, status/reason, advisory
confidence/observations, rule_id/citation, additional, advisory_only=True.
Причины: no_verified_rule, model_unavailable, unverified_timebase,
invalid_candidate, frames_unavailable, insufficient_frames, invalid_response,
insufficient_evidence, timeout, video_timeout, backend_error, budget_exhausted.
Сырые исключения, пути и сцены не включаются в диагностику.

Общий max_calls — целое 0..6, включая ошибки inference и optional open checklist.
До трёх кадров на кандидата; checklist — до шести кадров и один вызов. Probe
кеширован для всех кандидатов/checklist. Шаг ограничен min(timeout_s, остаток),
cap по умолчанию 60 s; **никакого свежего 300 s бюджета нет**. video_timeout_s
может только сократить явно переданный budget_sec; без budget_sec вызов отвергается.
Инъецированные доверенные fixtures ограничиваются daemon-thread ожиданием даже
если игнорируют timeout; такие потоки нельзя принудительно убить. Это тестовый
seam, не замена production process cancellation.

## Offline runtime и CLI

Импорт ml.vlm не импортирует torch/PIL/transformers/OpenCV/NumPy/ml.video/ml.kb.
Модель получает локальные пути, offline-переменные, local_files_only=True,
trust_remote_code=False, safetensors. Проверяются конфигурация и все шарды;
missing/mismatched/error loading diagnostics отвергаются. Worker stdout/stderr
сторонних библиотек подавлены. Offline-настройки — не OS network sandbox.

```bash
.venv/Scripts/python.exe -m ml.vlm preflight --model "C:/local-models/qwen2.5-vl"
.venv/Scripts/python.exe -m ml.vlm preflight --model "C:/local-models/qwen2.5-vl" --load --timeout 120
.venv/Scripts/python.exe -m ml.vlm verify --video "C:/local-data/video.mp4" --model "C:/local-models/qwen2.5-vl" --candidates "C:/local-data/candidates.json" --rules "C:/local-data/rules.json" --output "C:/local-data/vlm-result.json" --budget-sec 20 --startup-timeout 120
```

CLI использует низкоуровневый PTS-контракт, не CV-кандидаты напрямую. --budget-sec
обязателен; --video-timeout — optional cap, без default 300. Модель загружается
по --startup-timeout до verification budget. Startup failure даёт uncertain-отчёт,
не скрытый reload на кандидате. JSON-входы — массивы до 64 KiB каждый. Output
явно задан, не совпадает с входами, содержит локальные evidence и не предназначен
для внешних логов. Stdout — только счётчики/статус. Verify exit 0 — отчёт записан
(возможен весь uncertain), 2 — ошибка входа/выхода. Preflight ready=true без --load
означает структурную проверку; runtime_loaded=true — загрузку, не качество/SLA.

## Проверки текущего исправления

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_vlm.py tests/test_vlm_budget.py tests/test_vlm_pipeline_adapter.py --tb=short
.venv/Scripts/python.exe -m ruff check ml/vlm tests/test_vlm.py tests/test_vlm_budget.py tests/test_vlm_pipeline_adapter.py
```

Синтетические RED/GREEN-регрессии: warm worker без per-request artifact checks,
failed/dead/timed-out worker без reload, обязательный К2 budget, probe-cache,
caller deadline при игнорируемом timeout, уничтожение медленного frame worker
без отложенной записи, lazy owner frame_at, подавление third-party diagnostics,
retention rejected/frozen Candidate, production startup/verify seam и CLI startup
вне verification budget.

Независимые review-регрессии проверены RED → GREEN отдельно по поведению:
5 deadline-cases (catalog/candidate adaptation в runtime/adapter и preparation
низкоуровневого verifier; 0.08 s задержки при budget 0.03 s), 2 Process.start
OSError-cases с настоящими незапущенными Process/Pipe и 4 busy-cases
(infer/validate/close/production verify) на синтетическом OS worker. Owner
сохраняет собственный ответ; конкурент не трогает Pipe. Целевые VLM-тесты:
98 passed; полный финальный repository snapshot: 675 passed, 2 skipped. Ruff и git diff --check
прошли. Совместимость нового RuleText без basis и запрет fallback при явном
пустом legacy basis сохранены.

Дополнительно проверены неизменённые Candidate из CV commit
269f55606d14c8c30f93e43dc354cda80e40ac12 и ml.kb из pipeline commit
3ae9caa4b9e1da8a88124eb837f8dd497e505a68: git show, исполнены только эти контракты
с синтетическим каталогом/кандидатом/backend. Box заменён **type-only** заглушкой;
детектор не импортировался. Проверены identity, frozen и исходная confidence
после валидного rejected. Это не forward, модельное качество или детекция.

Implementation/review агенты работали только с code/synthetic fixtures без
приватных inputs. Parent отдельно повторил настоящий offline CPU warm-worker
runtime после deadline/startup/busy исправлений: один cold start 15.4585 s,
две warm generation 9.6981 / 9.2974 s, strict JSON objects. Application-schema
и качество этим не подтверждены. Финальное независимое VLM v2 ревью — passed,
10 дополнительных synthetic in-memory checks, code hashes не изменились.

Parent также исполнил неизменённые owner Candidate/KB/assemble/video из более
нового pipeline `215b51ad608e884a2e035d1f459a04ec3dd3950a`: 17 типов/16 правил,
17 исходных objects сохранены, 3 декодированных synthetic кадра, 6 scripted rejects,
pending norm сохранена в валидном К1; Box только type alias stub. Это интеграция
контрактов, не detector forward. Владельческие модули не объединялись с ML-веткой
и не менялись. Калибровка, detector quality и end-to-end SLA не проверены.
