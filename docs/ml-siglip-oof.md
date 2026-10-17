# SigLIP: CPU content-group OOF diagnostic (этап A)

**Решение: пользы первого mean-pooled binary сигнала не показано. В production не включать.**

Задание: `docs/team-ml2-next-task.md` из командной ветки, revision `583e1ba7ad451c41f656d5c206520afb916180c7`. Это отдельная поставка ML #2, не изменение PR #43, Qwen/MLX или других командных компонентов.

## Проверенные входы

- 42/42 кэша, 0 extraction errors/caps; 40 уникальных содержимых: 21 positive / 19 negative. Точные дубли имеют один вес.
- 1464 cached frame vectors, 1319 после content deduplication; размерность 768. Все embeddings конечны, реальные PTS возрастают.
- Свежие SHA256 исходных файлов совпали с extraction index, каждым cache и manifest; optional content hash в label table при наличии обязан совпадать с той же цепочкой. Encoder/config/processor/checkpoint fingerprint сверён с настоящими локальными байтами.
- У дублей проверено точное равенство числовых кэшей. Альтернативные human anchors сохранены локально, не усредняются и не входят в признаки.
- Известных source groups во входном manifest — 0; камеры неизвестны. Связанные записи при появлении explicit metadata объединяются транзитивно. Hash-only разделение не исключает общую съёмку, re-encoding или камеру.

## Замороженный протокол

- Все cached frames клипа усредняются (float64) по фиксированному label-free правилу. Нет отбора по меткам, type ID, human timestamp, имени, CV-вердикту или event frames.
- Seed 0; 5 outer content-group folds. В каждом outer train отдельная group validation (около четверти); обе train/validation стороны содержат оба класса. Все копии и связанные source components остаются вместе.
- L2 logistic: 1000 epochs, learning rate 0.1, L2 `[0.0001, 0.001, 0.01]`. Mean/std и balanced loss обучаются только на inner train; после выбора нет refit на validation.
- Threshold grid `0.05..0.95`, шаг 0.05. Выбор только по inner validation balanced accuracy; ties — threshold ближе к 0.5, далее порядок L2; для threshold tie далее меньшее значение.
- Cache audit, input manifest, folds и policy записаны до fitting/outer predictions. AND/OR с CV объявлены заранее, но не оценены без actual frozen responses.
- Ровно одно внешнее предсказание на content. Нет подбора лучшего seed или изменения протокола по pooled OOF; четыре запуска, включая повторы после review-исправлений входного аудита, дали те же folds, scores и decisions.
- Estimator существует только в памяти. Strict `train_classifier` / `save_model` не вызываются и не изменены; production schema/weights не созданы.

## Реальный результат

Обе доступные системы измерены на одинаковых 40/40 content units, ошибок fit/predict — 0. Train-prior берётся из inner train каждого fold; все majority decisions здесь положительные. Pooled AUC относится к объединённым внешним scores разных fold-estimators, а не к единой калиброванной модели.

| Метрика | SigLIP OOF | Train-prior / majority |
|---|---:|---:|
| TP / FP / FN / TN | 7 / 12 / 14 / 7 | 21 / 19 / 0 / 0 |
| Recall | 7/21 = 0.3333 | 21/21 = 1.0000 |
| Specificity | 7/19 = 0.3684 | 0/19 = 0.0000 |
| Precision | 7/19 = 0.3684 | 21/40 = 0.5250 |
| Balanced accuracy | 0.3509 | 0.5000 |
| Accuracy | 14/40 = 0.3500 | 21/40 = 0.5250 |
| F1 | 0.3500 | 0.6885 |
| MCC | -0.2982 | не определено |
| ROC-AUC | 0.3383 | 0.4599 |
| PR average precision (не trapezoidal AUC) | 0.4744 | 0.5060 |

### Fold sensitivity

Counts в таблице: negative / positive. Нумерация fold — 0-based.

| Fold | Train | Validation | Test | L2 | Threshold | Test BA |
|---|---|---|---|---:|---:|---:|
| 0 | 11 / 13 | 4 / 4 | 4 / 4 | 0.01 | 0.80 | 0.2500 |
| 1 | 11 / 13 | 4 / 4 | 4 / 4 | 0.01 | 0.30 | 0.2500 |
| 2 | 11 / 13 | 4 / 4 | 4 / 4 | 0.001 | 0.05 | 0.6250 |
| 3 | 11 / 13 | 4 / 4 | 4 / 4 | 0.0001 | 0.50 | 0.3750 |
| 4 | 12 / 12 | 4 / 4 | 3 / 5 | 0.0001 | 0.50 | 0.2667 |

Fold BA: macro 0.3533, range 0.2500–0.6250, defined 5/5. Малый development-набор не даёт надёжного camera-generalization вывода или узкого confidence interval.

## Что заблокировано

- **Matched CV / AND / OR comparison: blocked.** Настоящие frozen per-video/content ответы команды на этой машине отсутствуют. Найденный старый `artifacts/pipeline-current/predictions.json` содержит 42 failed outputs, а не пригодные вердикты. Он не принят как актуальный comparator.
- CV coverage в diagnostic — 0/40 (missing), common intersection — 0/40. Метрики CV/AND/OR и lost-CV-TP/new-FP не интерпретируются. Опубликованные ранее агрегаты CV не превращались обратно в предсказания.
- Для завершения comparison нужен локальный numeric export настоящего замороженного CV: producer revision, actual content hashes и verdict/status для всех исходных записей. Он может не содержать кадров/описаний. Существующие CV development-метрики не были слепой holdout-оценкой.
- **Этап B / независимые камеры: blocked.** Нужны human `angle_id` / `source_group`, проверка достаточного числа непересекающихся реальных групп и новый законно доступный набор с human truth. Это зависимость data-команды, не задача выдумать камеры.
- Clip head не предсказывает тип нарушения или таймкод. Type/timing/joint type+time, service SLA и качество на новой съёмке здесь не измерены. Никакого 17-way head или production deployment.

## Время и воспроизведение

Windows, CPU, Python 3.10.0, NumPy 1.26.4; OMP/OPENBLAS/MKL threads = 1. Повтор r4 после review-исправлений: audit 0.5957 с; 15 fits (3 configs × 5 folds) 0.2754 с wall / 0.2813 с process CPU; validation predict 0.000715 с wall; outer predict 0.000176 с wall; полный numeric job 0.8973 с. CPU timer для крошечных predict-вызовов может округлиться в 0; это не нулевая стоимость. Extraction не включена, model calls/downloads — 0. Это не latency/SLA полного сервиса.

Запуск из корня репозитория; `$VIDEOS`, `$LABELS`, `$MANIFEST` — настоящие локальные пути. Parent output-директория должна существовать, сама output-директория — новая. В shell заранее выставить thread limits.

```bash
python -m ml.baseline.development \
  --index-path artifacts/baseline-full-cpu1/index.json \
  --videos-dir "$VIDEOS" --labels-path "$LABELS" --manifest-path "$MANIFEST" \
  --weights models/siglip-base-patch16-224-vision-only \
  --output artifacts/siglip-oof-new --seed 0 --max-folds 5
```

CLI устанавливает offline network guard до NumPy; guard — не OS sandbox. Exit 0 означает complete OOF + complete matched comparison; exit 2 — incomplete OOF/comparison (в этом прогоне только missing CV); exit 1 — ошибка входа/запуска. После успешного резервирования output при ошибке аудита остаётся class-only `failure.json`, не отрицательный вердикт. Существующие output, включая dangling symlinks/reparse points, отвергаются до input audit и любых записей. No-follow проверки — не TOCTOU-free OS sandbox.

Для frozen CV: `--frozen-cv-path path/to/local.json`. Envelope: `schema_version=1`, полный lowercase 40-hex `producer_revision`, явный boolean `development_tuned`, `records` в существующем evaluator-формате (`video`, `status`, `verdict`, `violations`) с actual `content_sha256` у каждой записи. Missing/invalid/error duplicate нельзя заменить успешной копией; несколько successful verdicts группы агрегируются OR. Boolean verdict не превращается в выдуманный continuous AUC score.

Локальные ignored артефакты: `artifacts/siglip-oof-cpu-r4/{cache-audit,input-manifest,folds,oof-predictions,report}.json`; предыдущие неизменённые протоколы/результаты сохранены в `artifacts/siglip-oof-cpu-r1/`, `artifacts/siglip-oof-cpu-r2/` и `artifacts/siglip-oof-cpu-r3/`. В Git — только код, synthetic tests и эти агрегаты; raw video, frames, embeddings, per-content scores/hashes и приватные отчёты не публикуются.

## Проверки

- Новый diagnostic/audit/CLI: 101 synthetic tests passed, в том числе leakage, duplicate weighting, related groups, plan tampering, thresholds, failures, strict unknown-angle rejection и отсутствие production calls/exports. Review-регрессии проверяют реальные dangling symlinks, Windows reparse metadata, неверные label hashes, не-object extraction metadata и повреждённые CSV/XLSX/config/cache с настоящими загрузчиками; исправления сделаны RED → GREEN. Loader guards локальны: control-flow exceptions и ошибки вне parser boundary не маскируются.
- С отдельными reviewer probes и AUC/AP oracle: **105 passed**. Полный pytest этой отдельной ветки: **1333 passed, 9 skipped**. Семь skips: отсутствующий `data/open` link; два: Windows без `sendmsg`. Scoped Ruff clean.
- Граница независимого review: public code/docs и synthetic tests; качество на приватных входах reviewer не проверяет. Совпадение реальных r1/r2/r3/r4 и code hashes текущего запуска проверены отдельно локально.
