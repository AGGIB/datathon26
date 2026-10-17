# Проверенная поставка ML #2

## Исправления по ревью PR #30

- Восстановлены все строки защиты `.gitignore` из PR #29, включая вложенные
  `data/`, регистронезависимые видео/кадры и ключи; наши ML-исключения только
  добавлены сверху по смыслу, без ослабления командных правил. Объединены все
  testpaths и маркер `slow` из PR #31. Проверка синтетических имён/конфига:
  RED 15 failed / 12 passed → GREEN **27 passed**. Rebase не выполнялся:
  на момент проверки #29 и #31 ещё открыты, `main` — `2f9dd5c`.
- `docs/licenses-ml.md` и `docs/checkpoints-ml.json` закрепляют шесть разрешённых
  Apache-2.0 checkpoints, точные ревизии и SHA256 всех 28 safetensors-файлов.
  Метаданные получены из публичного HF API по каждой закреплённой ревизии;
  это не утверждение, что все веса скачаны или проверены inference.
- Новый реальный локальный synthetic runtime замер SmolVLM-256M:
  загрузка/готовность **46.0185 с**, два вызова в том же worker — **9.1938 с**
  и **8.7786 с**. CPU, threads=2, три искусственных изображения 96×96,
  `max_new_tokens=128`, прежний 512-pixel processor profile, Python 3.10.0,
  Torch 2.13.0, Transformers 4.57.6. Веса сверены по SHA256; оба коротких ответа
  прошли только synthetic JSON-object check. Качество, прикладной violation
  contract и SLA полного pipeline здесь не измерялись. Сеть блокировал Python
  audit hook, в том числе в spawned worker; приватные кадры не использовались.
  Локальный артефакт: `artifacts/review/vlm-latency/timings.json`.

- До дополнительного независимого ревью: regression **578 passed, 2 skipped**, Ruff по ML/tests и
  `git diff --check` чистые. Два пропуска — `sendmsg` на Windows.
- Baseline/extractor/split/final-review tests: **191 passed**. Обязательные SHA256,
  трёхсторонний split/model schema v2, canonical joins и дедупликация train с
  объединением альтернативных якорей. На локальных 42 extraction caches проверены
  canonical joins и совпадение всех 42 hashes с реальным audit; fitting не запускался.
- Evaluator/labels/split/final-review tests: **188 passed**. На реальных файлах audit
  даёт 42/42, 40 содержимых и две группы дублей. Командный annotations CSV с actual
  hashes проходит score с пустыми predictions: 42 diagnostics / 40 groups. Это
  только проверка joins, alternative-anchor semantics и знаменателей, не качество.
- Runner: **21 passed** — атомарный текущий checkpoint после каждой попытки,
  сохранение завершённого префикса при interrupt, allowlist без evidence/base64.
  Resume не заявлен; подробности — `ml-runner.md`.
- Независимое code-only ревью baseline/runner/root/licensing первого snapshot
  прошло (`passed=true`). Дополнительные findings evaluator/VLM исправлены
  через отдельные RED/GREEN-проверки: порядок событий, malformed Windows ID,
  ранний local-path guard, known-hash integrity неполного audit, абсолютный
  deadline, cleanup нестартавших процессов и неблокирующий backend busy-guard.
  Parent после них: **642 passed, 2 skipped**, Ruff/diff clean. VLM отдельно:
  **98 passed**; финальное независимое ревью VLM v2 — `passed=true`, без
  security/logic/suggestions, дополнительно 10 synthetic in-memory проверок.
  Reviewed VLM code hashes не изменились. Evaluator теперь отклоняет symlink/reparse
  root/ancestor/recursive entries до target I/O и сохраняет известные multi-dot
  canonical hash keys с fail-closed ambiguity/collision guards. Parent полный
  pytest после этих исправлений: **675 passed, 2 skipped**, scoped Ruff/diff clean.
  Offline actual audit/cache validation повторена: 42/42 joins/hashes, 40 contents,
  2 duplicate groups, 0 predictions и 0 scored; обучения/качества нет.
  Итоговое независимое evaluator v3 ревью — `passed=true`: все 1979 строк snapshot
  прочитаны, security/logic/suggestions пусты. Ревьюер выполнил stdlib synthetic
  harness; parent отдельно выполнил полный pytest выше. Перед публикацией проверены
  hashes reviewer snapshots (единственная разница tests — удалённая пустая строка
  в EOF), allowlist 38 публичных staged файлов и отсутствие credentials/бинарных
  приватных artifacts. `IDEA.md` исключён. Это локальное ревью, не GitHub approval;
  detector quality, calibration/holdout и issues #17–#20 остаются открытыми.
- Повторный настоящий offline CPU runtime после нового startup API:
  SmolVLM-256M, 3 синтетических кадра 96×96, threads=2, max_new_tokens=128.
  `configure` cold start **13.4844 s**, два warm generation **8.4577 / 8.2003 s**,
  один PID; оба вернули strict JSON object. Не проверялся violation-schema,
  качество либо end-to-end SLA; ранний замер выше сохранён как отдельный прогон.
- После fresh fetch pipeline force-update до `215b51a` обнаружен новый RuleText
  без `basis`: отдельный RED AttributeError → source-only citation fallback,
  VLM **87 passed**. Offline synthetic integration на неизменённых owner
  Candidate/KB/assemble/video: **17 типов / 16 правил**, все 17 объектов сохранены,
  6 scripted advisory calls, 3 кадра действительно декодированы через OwnerFrames;
  pending-норма сохранена в валидном К1. Только Box type alias stub, без детектора
  и pretrained inference. Полный production pass **0.8584 s / 15 s** относится
  лишь к этому synthetic/scripted тесту.
- После VLM fixcycle1 повторены оба локальных smoke: настоящий cold start
  **15.4585 s**, warm generation **9.6981 / 9.2974 s** в одном worker, оба strict
  JSON objects; owner-source integration снова сохранила 17/17 кандидатов,
  декодировала 3 synthetic кадра и дала валидный К1 pending-нормы, scripted pass
  **0.9599 s / 15 s**. Это по-прежнему не application quality или end-to-end SLA.

Результаты ниже относятся к предыдущей поставке и не заменяют проверку текущих
исправлений выше.

## Что действительно запускалось

- Полный локальный pytest после пакетного extractor и CV/VLM-адаптера: **349 passed, 2 skipped**. Два пропуска — отсутствующий `sendmsg` на Windows. Отдельно 31 тест адаптера + 21 существующий VLM-тест: 52 passed. Ruff и проверка whitespace прошли.
- Независимое финальное ревью пакетного extractor прошло без замечаний после исправления гонки output ownership; конкурентный доступ и освобождение блокировки покрыты synthetic-регрессиями. Это локальное агентское ревью, не GitHub approval PR #30. Продолжение оставлено локально, без нового commit/push; `IDEA.md` и чужие компоненты не изменялись.
- Независимое ревью CV/VLM-адаптера также прошло: `passed=true`, security/logic/suggestions пусты. Ревьюер отдельно выполнил 31 synthetic case адаптера с тестовыми frame/backend-зависимостями; parent выполнил полный набор и интеграцию с командными контрактами. Реальный inference и качество модели этим ревью не оценивались. Это локальное агентское ревью, не approval PR в GitHub.
- Аудит исходных 42 видео: все имена сопоставлены, метаданные прочитаны, исходные якоря в пределах длительности.
- Импорт предоставленного human-analysis bundle: SHA256 подтверждает все файлы; 17 типов, 22 положительных первичных типа и 20 отрицательных строк. Найдены 40 уникальных содержимых и две группы дублей.
- Предыдущий `ml.eval run` действительно вызвал `ml.pipeline:analyze` нашей ветки по всем 42 входам: **42 `NotImplementedError`**, coverage=0. Это диагностика заглушки, не качество модели. При свежей проверке отдельная командная `feat/pipeline-assembly` обновилась до `40774a2` с настоящей сборкой `analyze()`, но `find_candidates()` всё ещё безусловно возвращает `[]`, как и в свежих scene/events-ветках; называть её полноценным детектором и измерять качество как готового решения нельзя. Наша ветка и чужие компоненты не перезаписывались.
- Настоящий SigLIP encoder локально на CPU: два кадра настоящего `.mov`, два конечных 768-мерных вектора с PTS 0.0 и 1.0. Официальные веса проверены SHA256, кадры не отправлялись внешнему AI.
- Пакетный SigLIP preprocessing завершён на всех **42 видео / 40 уникальных содержимых**: **1464 × 768** конечных числовых признаков в 42 кэшах (1319 кадров при подсчёте уникальных содержимых), без ошибок и frame cap. Это фиксированные pretrained-признаки, не predictions детектора и не обучение baseline. Индекс: `artifacts/baseline-full-cpu1/index.json`; независимая проверка: `validation.json` в той же папке. Сверены все literal IDs с 42 исходными метками, SHA256 каждого видео/snapshot, равенство кэшей дублей и все выбранные PTS с независимым ffprobe; расхождение PTS — 0.0 с.
- Рабочий полный профиль: CPU, 1 fps, batch=1, threads=2, Python 3.10, Transformers 5.17.0; Python socket audit guard запрещал внешнюю сеть. Из-за Windows commit limit исходный полный checkpoint и затем batch=4 не завершили подготовку; эти неуспешные артефакты не считать полным набором. Для успешного прогона создан отдельный локальный vision-only snapshot: все 208 `vision_model.*` тензоров проверены по dtype/shape/SHA256 как точные копии оригинала; config/processor не менялись. Оригинальный snapshot сохранён. Веса и признаки остаются вне Git.
- Настоящий CPU VLM runtime: SmolVLM-256M загрузился и выполнил генерацию. На ограниченном реальном фрагменте ответ не прошёл строгий JSON-контракт, сохранён `uncertain/invalid_response`; кандидат не исчез. Этот oracle-timed smoke исключён из benchmark.
- Отдельный `verify_pipeline_candidates` проверен синтетически с точными Candidate/KB/assemble из PR #31 (`40774a2`) и каталогом PR #32 (`8f2017c`), без модели/видео: 27 типов, 26 проверенных цитат, Т04 остаётся uncertain и проходит сборщик в валидный К1 без `rule`. Заменён только type alias Box; detector не импортировался. Scripted fixture дал 6 advisory rejected + 21 uncertain; shadow сохранил все 27 исходных объектов. Это проверка контрактов, не снижение FP. В отдельном свежем процессе `import ml.vlm` не загрузил torch/PIL/Transformers/OpenCV/YAML/основной pipeline. Локальные результаты — `artifacts/review/vlm-adapter-integration.json`.
- Обнаружено ограничение времени интеграции: CV из PR #31 вычисляет index/FPS, VLM использует реальные container PTS. Адаптер по умолчанию блокирует модель через `timebase_verified=False`; по умолчанию `allow_rejection=False` также запрещает удаление CV-кандидатов даже по валидному advisory rejected. Основной pipeline не изменён, VLM в него автоматически не включён.

## Как повторить проверку инструмента

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check ml/eval.py ml/evaluation_labels.py ml/eval_runner.py ml/import_analysis.py ml/baseline ml/vlm tests

python -m ml.eval audit --labels '<open_labels.xlsx>' --videos-dir '<videos>' --output artifacts/audit
python -m ml.eval run --labels '<open_labels.xlsx>' --videos-dir '<videos>' --annotations '<annotations.csv>' --output artifacts/current
```

Для полного численного/ML-набора тестов дополнительно нужны зависимости из `requirements-ml-baseline.txt`; runtime VLM задаётся отдельно через `requirements-ml-vlm.txt`. Без optional ML-зависимостей соответствующие проверки пропускаются — это не равно полному набору результатов выше. Выбирать torch wheel под собственную физическую CPU/GPU-машину. Веса подготовить до офлайн-запуска.

Подробные интерфейсы: `ml-evaluation.md`, `ml-runner.md`, `ml-baseline.md`, `ml-vlm.md`, `analysis-import.md`, `licenses-ml.md`.

## Что ещё НЕ подтверждено

- accuracy/recall/F1 настоящего CV-пайплайна на датасете;
- устойчивость по ракурсам: `angle_id` отсутствует у всех 42 строк;
- реальные обученные веса baseline и независимые holdout-метрики;
- снижение false positives VLM без потери recall;
- production SLA и нагрузка API — это зона интеграции с Агыбаем/Нариманом;
- независимая правовая экспертиза. Для экспорта правил использованы только явно отмеченные командой verified-записи с `verified_by`; источник доверия — каталог команды.

Не использовать test-ракурс для подбора порога. Побайтовые дубли должны оставаться на одной стороне split. Ошибки/таймауты не заменяются `no_violation`, а отсутствие модели не подменяется синтетическими ответами.

Все вычисления выполнены на собственном физическом компьютере. Ни облачных GPU, ни облачных AI-API при обработке данных не использовалось. В коммит ML #2 не добавляются веса, видео, кадры, исходные наблюдения, настоящие predictions/отчёты, окружения и `.env.vast`. Существующие командные файлы `kb/` эта поставка не перезаписывает.
