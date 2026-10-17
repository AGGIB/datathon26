# Проверка компонентов ML #2

## Текущее состояние

Основная поставка вошла в `main` через PR #30. Для текущего CV-пайплайна
результаты команды описаны отдельно в
[scene-runtime-validation.md](scene-runtime-validation.md); прежние запуски
заглушек `analyze()` не характеризуют текущий pipeline.

Разделять synthetic-тесты контрактов, фактическую загрузку весов, pretrained
forward/generation, пригодность прикладного JSON, независимое качество и
production SLA. Успех одного уровня не доказывает другой.

## Воспроизводимые проверки кода

Из корня репозитория, в выбранном локальном окружении:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q --tb=short
python -m ruff check ml/eval.py ml/evaluation_labels.py ml/eval_runner.py ml/import_analysis.py ml/baseline ml/vlm tests
git diff --check
```

Для численного набора дополнительно нужны `requirements-ml-baseline.txt`;
для VLM — `requirements-ml-vlm.txt`; для владельческих CV-модулей —
`ml/requirements.txt`. Torch wheel подбирается под собственную физическую
машину. Веса подготовить заранее, не скачивать во время обработки.

В локальном Windows-окружении 2026-10-17 на follow-up к merged PR #30:
**1282 passed, 9 skipped**. Семь пропусков — optional integration-тесты,
требующие локального `data/open`; ещё два — отсутствие `sendmsg` на Windows.
Это не утверждение о native macOS/Linux и не результат обработки датасета.

При одном повторе зафиксирован Windows `PermissionError [WinError 5]` в
`backend/app/store.py:110` (`os.replace`) под concurrent upload-тестом.
Backend не отличается от main и в этой поставке не менялся. Изолированный
повтор теста и следующий полный прогон прошли; единичный сбой не скрывается
и не считается исправленным этим ML-дополнением.

Проверяются identity исходных frozen Candidate, неизменность confidence,
fail-open при ошибках адаптера/кадров/cleanup, безопасные warnings,
один `frames_at` на extraction, завершение worker по дедлайну, отсутствие
скрытого reload и двухполевой JSON-контракт pair screening.

## Audit, runner и оценка

```bash
python -m ml.eval audit --labels '<open_labels.xlsx>' --videos-dir '<videos>' --output artifacts/audit
python -m ml.eval run --labels '<open_labels.xlsx>' --videos-dir '<videos>' --annotations kb/annotations.csv --output artifacts/current
```

Контракты: [ml-evaluation.md](ml-evaluation.md), [ml-runner.md](ml-runner.md).
Runner сохраняет machine `type_id`/confidence, но не evidence/base64/наблюдения;
checkpoint атомарный после каждой попытки. Автоматического resume нет.
Невалидные ответы и ошибки не подменяются `no_violation`; exact-content дубли
группируются по SHA256, не по имени.

## Фактический локальный ML runtime

- SigLIP: на всех 42 видео / 40 уникальных содержимых подготовлены 42 кэша,
  **1464 × 768** конечных pretrained-признаков, 1319 кадров по уникальным
  содержимым. Ошибок и frame cap нет; literal IDs, hashes и PTS сверены.
  Это preprocessing, не обученный baseline и не detector predictions.
- CPU-профиль SigLIP: batch=1, threads=2, Python 3.10, Transformers 5.17.0.
  Из-за Windows commit limit использован отдельный vision-only snapshot:
  все 208 `vision_model.*` тензоров — точные копии исходных dtype/shape/bytes;
  config/processor не менялись. Приватный индекс:
  `artifacts/baseline-full-cpu1/index.json`.
- SmolVLM-256M: настоящий офлайн-прогон **пяти пар / десяти видео**,
  30 реальных кадров. JSON-контракт не прошли все десять ответов:
  **0/5 правильных пар, покрытие 0%**. Загрузка 30.625 с; warm inference
  2.328–7.390 с на видео, медиана 2.9695 с. Методика, отдельные времена
  извлечения и ограничения: [ml-vlm-pairs.md](ml-vlm-pairs.md).

Advisory runtime из этого документа не вызывается в `ml.pipeline`: пользы
исследованного SmolVLM-профиля не показано. Отдельная opt-in ветвь type discovery
из [PR #42](https://github.com/AGGIB/datathon26/pull/42) по умолчанию выключена;
её измерения не заменяют результаты advisory-эксперимента выше.
Кадры, prompts, ответы и полные отчёты остаются локально, вне Git.
Лицензии и revision/hash pins: [licenses-ml.md](licenses-ml.md).

## Что ещё не подтверждено

- обученный baseline и независимые validation/test-метрики;
- обобщение на новый ракурс без проверенного angle/group split;
- снижение false positives или повышение recall от VLM;
- сравнение с Qwen3-VL-4B и CPU-латентность Qwen2.5-VL-7B/32B;
- production К2 SLA и API-нагрузка;
- независимая юридическая экспертиза: нормативные ссылки берутся только
  из проверенного командного каталога, не из генерации модели.

Не использовать test для подбора порога. Exact-content дубли держать на
одной стороне train/validation/test. Ошибки, timeout и отсутствие модели
сохранять в знаменателях. Acceptance issues закрывать только при фактическом
выполнении критериев, а не по числу synthetic-тестов.
