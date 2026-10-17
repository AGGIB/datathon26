# Нурхан / ML #2: измерить пользу обученного SigLIP baseline

**Задача для человека в команде, в два этапа:** сейчас измерить CPU content-group OOF diagnostic на уже извлечённых SigLIP-признаках; после появления нужной разметки и новых данных — отдельно проверить независимые камеры. Первый результат — числовые out-of-fold predictions и честное решение о полезности резервного бинарного сигнала, без deployable-модели. Работы по Qwen/MLX, prompt, JSON decoding и новым VLM выполняет другой трек; их здесь не повторяем. В production pipeline ничего не подключать.

## От чего продолжаем

Первоисточник — [PR #43: feat(ml): offline VLM pair screening and advisory guards](https://github.com/AGGIB/datathon26/pull/43), head `568d090592ee2d441f6c7deabf7f6e2f8c58b603`, base `af24b8b`. На момент проверки через GitHub connector: open, не draft, mergeable; 16 изменённых файлов. Legacy commit-status API вернул пустой список, поэтому успешный CI этой проверкой не подтверждён.

PR #43 добавляет `ml/vlm/pair_eval.py`, пакетное извлечение в `OwnerFrames`, advisory failure guards, тесты и документацию. Это пять заранее заданных пар: одинаковое условие внутри пары, три равномерных кадра, заранее известный целевой тип. Реальный результат автора SmolVLM-256M — 10 generation, 10 невалидных JSON, 0/5 правильных пар; это не обучение SigLIP и не оценка открытого обнаружения типов. Не переделывать этот эксперимент в рамках новой задачи.

Encoder/extractor, безопасные кэши, NumPy binary classifier и оценочные инструменты уже пришли через PR #30: `ml/baseline/{features,classifier,core}.py`, `ml/eval.py`, `ml/eval_runner.py`. Повторно извлекать 42 видео, скачивать другую модель или строить новый evaluator не нужно. Доступные 42 extraction caches — вход для обучения, а не результат качества.

## 1. Зафиксировать кэши и границы двух экспериментов

1. Сверить полный extraction index: 42 ожидаемых/успешных кэша, ноль ошибок и frame cap; конечные embeddings и timestamps; единые feature specification, encoder/processor/weights fingerprints. Сопоставить каждый cache с настоящим SHA256 исходника и manifest. Перенос кэшей в Git запрещён.
2. Оценочная единица — **40 уникальных содержимых**, 21 positive / 19 negative. X-4/X-47 и X-5/X-48 — точные дубли. Копии не увеличивают вес обучения или число примеров; alternative human anchors одного содержимого сохраняются, не усредняются.
3. В этапе A группы split определяются настоящим content SHA256; известные связанные записи также объединяются. Все кадры ролика остаются вместе. Зафиксировать manifest/folds до просмотра результатов. Точные хеши не выявляют связанные перезаписи или общую камеру: при неизвестных `source_group` / `angle_id` это явное ограничение, а не доказанная изоляция.
4. Human camera/source metadata — отдельная зависимость от data-команды. Человек локально заполняет реальные `angle_id` и `source_group`; можно использовать `var/manual-angle-review-2026-10-17/index.html`. Сейчас ракурсы не заполнены: число камер неизвестно. Только после этой разметки можно строить компоненты по **содержимому + связанной съёмке + камере** и заявлять camera-isolated оценку.

Существующий `train_classifier` требует явные непустые train/validation/holdout angle-группы. **Не подставлять hashes/video IDs вместо камер, не ослаблять этот контракт.** Этап A не вызывает этот strict fitting API: это отдельный явно diagnostic runner с временным числовым estimator в памяти, без `save_model`, production model schema и экспортируемых weights. В каждом его отчёте: `development_only=true`, `camera_isolation_verified=false`, `new_angle_holdout_validated=false`, `deployable_model_created=false`. Неизвестные камеры не блокируют этот ограниченный content-group diagnostic, но запрещают вывод о независимых камерах.

## 2. Этап A сейчас: небольшой content-group OOF diagnostic

- Использовать числовые кэши на CPU: для первого опыта фиксированный label-free mean pooling всех доступных frame embeddings в один вектор клипа и небольшой NumPy L2 logistic estimator. Переиспользовать безопасный cache loader и математику baseline там, где это не требует обхода strict fitting API. Отдельный diagnostic код не должен принимать фиктивные angle IDs или создавать deployable schema. Encoder фиксирован; новые pretrained forward, downloads и end-to-end дообучение не нужны.
- План outer content-group CV зависит от фактического числа групп/классов. K=5 допустим лишь при достаточном числе независимых class-bearing групп; иначе заранее выбрать меньший выполнимый K. Это content OOF внутри development, не angle holdout. Если fold невозможно обучить с обоими train-классами, явно показать его невозможность; не удалять неудобные группы из отчёта.
- Внутри каждого outer train выделить отдельную group validation. Mean/std, class balancing, weights и любые PCA/feature selection обучаются только на train этого fold. Outer test не участвует ни в одном выборе.
- Для первого прогона зафиксировать seed и небольшой набор конфигураций заранее, например существующие defaults + три значения L2. Learning rate/epochs, positive radius, window aggregation и правило выбора threshold записать в manifest. Выбирать конфигурацию/threshold только по внутренней validation; не выбирать лучший seed или threshold по объединённым OOF-ответам.
- Mean pooling не использует human timestamps, выбранные event frames или CV-предсказания. Бинарные labels участвуют только в fit/validation соответствующей стороны. Test anchors и названия видео не являются признаками, не выбирают входные кадры и не задают threshold. Сохранить существующий label-free sampling кэшей.
- Получить ровно одно внешнее OOF-предсказание на каждое проверяемое уникальное содержимое. Сохранить provenance train/validation/test, реальные hashes, конфигурацию, seed, OOF scores, coverage и ошибки; estimator остаётся в памяти. Не сохранять production model/weights и не имитировать fallback, если fitting не удался.

## 3. Показать пользу на общем числовом протоколе

Основная задача бинарная. На одинаковых OOF-единицах показать TP/FP/FN/TN и числитель/знаменатель recall `TP/(TP+FN)`, specificity `TN/(TN+FP)`, precision `TP/(TP+FP)`; дополнительно balanced accuracy, F1/MCC и ROC-AUC/PR-AUC там, где определены. Дать фактические evaluated/planned counts, coverage, fold sizes/class counts и чувствительность к группам. Missing/error не считать чистым negative. Не публиковать per-angle строки, пока камерная разметка отсутствует. Малое число компонентов не позволяет выдавать узкий доверительный интервал как сильное доказательство.

Сравнить с константным train-prior/majority baseline и замороженными CV-ответами на том же наборе. CV ранее настраивали на этих development-видео: сравнение с ним полезно инженерно, но не является симметричным слепым holdout двух систем. Если проверяются offline комбинации `CV AND SigLIP` и `CV OR SigLIP`, объявить их до outer test, threshold выбрать только внутри fold; показать и потерянные CV true positives, и новые false positives. В pipeline такие комбинации пока не переносить.

**Типы:** 21 уникальное положительное видео содержит 17 первичных типов, у 13 типов всего один пример. Обычный стратифицированный type K-fold здесь невозможен для всех классов. Не обучать 17-way head с обещанием достоверной per-type оценки и не удалять singleton-классы ради красивого macro score. Бинарный SigLIP-кандидат не получает предполагаемый type_id, правовую норму или человеческую метку на inference. Влияние фильтра на уже найденные CV-типы можно посчитать отдельно как retention/loss; это не распознавание новых типов.

**Время:** mean-pooled clip head этапа A не локализует события; timing quality не измеряется, peak не выдумывается и из labels не копируется. Время CPU fit/predict — другая величина. На этапе B, если используется существующий window predictor, его peak берётся из предсказанных вероятностей и настоящих timestamps кэша без test-anchor input. Primary-anchor hit в ±2 с тогда показывается отдельно с conditional-on-detection и общим positive-знаменателем; это не full event recall/precision. Нельзя заявлять joint type+time для безтипового binary head.

Все 40 содержимых — development, уже использованный при CV-разработке. Даже camera-group CV на нём не доказывает качество на новой внешней съёмке. Новый ракурс потребует отдельного законно доступного набора с human truth.

## Этап B после data-зависимостей

После human camera/source labels проверить фактическую выполнимость трёх непересекающихся train/validation/holdout групп, наличие обоих train-классов и изоляцию связанных записей. Только тогда использовать существующий strict `train_classifier` / безопасный model export. Если реальных групп недостаточно, честно зафиксировать blocked camera evaluation и недостающие данные. Новая съёмка или иной новый законно доступный набор с human truth необходимы для вывода о ранее не виденном ракурсе; сейчас такого набора нет. Не заменять это доказательство content OOF из этапа A.

## Что сдать

1. Новый PR в собственной ветке: компактный **development diagnostic runner**, осмысленные synthetic leakage/duplicate/threshold/failure regressions и короткий публичный числовой отчёт. Отдельный regression подтверждает, что strict production fitting всё ещё отклоняет unknown/fake angle groups. PR #43 не переписывать и историю не прятать.
2. Локальные ignored артефакты этапа A: cache audit, зафиксированные manifests/folds, OOF predictions, числовое сравнение и время CPU fit/predict отдельно от extraction. Deployable model/weights не создавать. Raw video, frames, embeddings и приватные отчёты не публиковать.
3. Короткий вывод: diagnostic обучен и измерен / конкретная ошибка кэшей или невозможный split / пользы не показано. При отрицательном результате задача всё равно выполнена, если эксперимент честный и воспроизводимый. Отдельной строкой — blocked независимые камеры и data-team dependency. По одному удачному ролику пользу не заявлять.

Первый ответ Нурхана: подтверждение состояния 42 кэшей, плана content-group OOF и конкретного CPU-опыта этапа A. Следующая поставка — реальные OOF scores/ошибки резервного binary сигнала. Human camera/source metadata и новые данные запросить как отдельную зависимость этапа B, не выдавая их отсутствие за независимый результат этапа A.
