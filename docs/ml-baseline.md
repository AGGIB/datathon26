# Локальный визуальный baseline: SigLIP + NumPy logistic regression

Это **резервный сигнал для поиска кандидатов**, не самостоятельный юридический классификатор. Он не определяет категорию нарушения и не пишет визуальные описания. `candidate` требует ручной проверки; `no_candidate` означает только отсутствие кандидата в обработанных окнах, **не** доказанное `no_violation`. `final_legal_label` всегда `null`.

## Зависимости и приватность

- Python 3.10+; для текущей локальной машины запускайте `py -3.10` (в обычном `python` 3.14 нет NumPy/torch).
- `requirements-ml-baseline.txt`: NumPy, Pillow, torch, transformers, safetensors. sklearn не нужен. `ffmpeg` должен быть в PATH либо передан через `--ffmpeg`.
- CLI help, split/time/path/metadata-тесты используют только stdlib. Тяжёлые импорты ленивые.
- Только собственная физическая локальная машина. Нет облачного compute, Vast, внешнего API, загрузки видео/кадров/описаний, телеметрии либо скачивания весов во время работы.
- Веса `google/siglip-base-patch16-224` владелец проекта подготавливает **отдельно** в `models/siglip-base-patch16-224/`: `config.json`, `preprocessor_config.json`, `model.safetensors` либо полный sharded safetensors snapshot. CLI принимает только существующий каталог, не Hub ID/URL.
- Перед импортом transformers устанавливаются `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`. Оба `from_pretrained` получают `local_files_only=True`, `trust_remote_code=False`; модель также `use_safetensors=True`. Missing/mismatched vision weights — ошибка, а не случайная инициализация.
- Используются `SiglipVisionModel` + PIL-процессор `SiglipImageProcessorPil` (fallback `SiglipImageProcessor` для Transformers 4.x), `pooler_output`, `eval()` и `inference_mode()`. `AutoImageProcessor` в Transformers 5.x требует отсутствующий torchvision даже при `use_fast=False`, поэтому не используется. Токенизатор и облачный VLM не нужны.

## Извлечение: bounded CPU smoke на одном настоящем видео

Из корня проекта (bash/MSYS; замените `<ТОЧНОЕ_ИМЯ>.MOV` буквальным именем существующего файла):

```bash
python -m ml.baseline extract \
  --video 'data/open/videos/<ТОЧНОЕ_ИМЯ>.MOV' \
  --weights models/siglip-base-patch16-224 \
  --output artifacts/baseline/smoke.npz \
  --fps 1 --max-frames 2 --batch-size 1 --cpu-threads 2 --device cpu
```

Это **только smoke**, не полнота покрытия видео. Для рабочего кэша увеличьте `--max-frames`, сохранив ограниченные `--batch-size` и `--cpu-threads`. Значения по умолчанию: 1 fps, 3600 кадров, batch 8, CPU threads 2. `--decode-timeout 600` — общий wall-clock лимит декодера, включая ожидание батч-инференса; для длинных CPU-задач увеличьте его явно. CUDA не выбирается автоматически; недоступная `--device cuda` приводит к ошибке.

В каждом батче держится не более `batch-size` RGB224 кадров. На диске кадры не сохраняются; в логах нет изображений/эмбеддингов/декодерного frame dump. Память числового кэша ограничена `max-frames × embedding_dim`; пик при склейке может быть примерно вдвое больше. Штатные признаки SigLIP base имеют 768 измерений.

`ffmpeg` явно включает autorotate до resize RGB224 и допускает только протоколы `file,pipe`. Выборка оставляет первый кадр и последующие реальные кадры не чаще заданного интервала: у VFR интервалы могут быть больше `1/fps`. Таймкоды берутся из `showinfo` выбранного кадра, **не** из индекса/fps и не из синтетической временной сетки. `-copyts -start_at_zero` задаёт PTS относительно начала контейнера; это таймлайн воспроизведения, не wall-clock камеры. Невалидные/повторяющиеся PTS, повреждённые кадры, несоответствие PTS/RGB, таймаут и ненулевой exit ffmpeg — явные ошибки, в том числе после последнего кадра. Поворот из MOV display matrix обрабатывает ffmpeg.

`video_id` по умолчанию — полное имя файла с расширением; можно передать `--video-id` явно. Кириллица/латиница и регистр сохраняются буквально, без транслитерации, нормализации и догадок по имени. Для одинаковых имён в разных папках задавайте уникальные явные ID.

## Формат безопасного кэша

### Пакетная подготовка без обучения

```bash
.venv/Scripts/python.exe -m ml.baseline extract-dataset \
  --videos-dir '<local-videos>' --weights models/siglip-base-patch16-224 \
  --output-dir artifacts/baseline-full \
  --fps 1 --max-frames 3600 --batch-size 1 --cpu-threads 2 --device cpu
```

Обрабатываются все `.mov`/`.MOV` записи непосредственно в указанной папке, без рекурсии. Одна модель загружается на весь набор. Побайтовые дубли используют уже извлечённые признаки, но каждый буквальный `video_id` получает отдельный числовой кэш; имя кэша — SHA256 UTF-8 имени, а не нормализованный ID. В каждом кэше сохраняются SHA256 исходного содержимого и fingerprint конфигурации/процессора/всех используемых safetensors. Это идентификация байтов, не независимая проверка их происхождения.

`index.json` дополняется после каждого входа и содержит полное ожидаемое число, успехи, ошибки и достижение frame cap. CLI печатает только агрегаты и возвращает код 2 при ошибках либо достижении cap хотя бы одного видео. Успех набора требует совпадения `expected_videos`, числа записей и `successful_videos`, а также нулевых `failed_videos` и `capped_videos`. Уже непустой output-каталог не перезаписывается; для нового прогона нужен другой каталог. Атомарный каталог `.extract.lock` резервирует output до загрузки модели: второй writer отклоняется сразу и не удаляет чужую блокировку. После обычного завершения/ошибки lock снимается владельцем; после принудительного убийства процесса может остаться — автоматически его не перехватывать, использовать новый output. Неполный индекс после прерывания не является завершённым набором.

Это label-free inference фиксированного pretrained encoder. Он не читает метки, не обучает модель, не выбирает ракурсы/пороги и не выдаёт predictions детектора. Подготовить признаки до появления `angle_id` допустимо, но обучать стандартизацию/классификатор и калибровать пороги можно только после фиксации train/validation/test с изоляцией одинакового содержимого. `quality_metrics_measured` остаётся `false`.

На Windows ошибка 1455 при загрузке может означать исчерпание **commit limit**, даже если физическая RAM свободна. Не менять pagefile/процессы пользователя автоматически. Для локальной подготовки допустим отдельный snapshot только с неизменёнными `vision_model.*` тензорами официального SigLIP: сохраняются исходные config/processor, SHA256 оригинала и provenance преобразования; равенство каждого тензора проверяется по dtype, shape и SHA256 байтов. Text tower baseline не использует. Такой snapshot не заменяет проверку loading diagnostics и не должен попадать в Git.

NPZ с `allow_pickle=False`: `embeddings` float32 `[N,D]`, `timestamps_s` float64 `[N]`, Unicode scalar `video_id`, Unicode scalar `metadata_json`. JSON содержит схему, feature specification, параметры extraction, размер/mtime исходника, `autorotate`, `cap_reached`. Последнее означает достижение лимита; хвост видео мог остаться необработанным, но не утверждает, что точно был усечён. Кэш не содержит кадров или labels.

CLI разрешает запись кэша/модели/JSON только внутри `<project>/artifacts/`, включая проверку symlink/path escape. Для отдельного локального рабочего корня используйте **до подкоманды** `--artifact-root 'C:/local/project'`; разрешённая папка станет `C:/local/project/artifacts/`. Артефакты и веса не должны попадать в git; `.gitignore` ведёт родительский агент. Кэш не переиспользуется автоматически по имени: после изменения файла пересоздайте его. Все низкоуровневые Python API записи требуют от интегратора той же проверки `artifact_path`; CLI делает её сам.

## Явная разметка и split

Manifest — UTF-8 JSON **массив**, не XLSX и не JSONL. Пример ниже — **схема с условными ID**, не фактическая разметка датасета:

```json
[
  {"video_id":"точное_имя_1.MOV","angle_id":"явно_размеченный_ракурс_A","label":1,"positive_times_s":[12.4],"content_sha256":"<actual lowercase SHA256 of source 1>"},
  {"video_id":"точное_имя_2.MOV","angle_id":"явно_размеченный_ракурс_A","label":0,"content_sha256":"<actual lowercase SHA256 of source 2>"},
  {"video_id":"точное_имя_3.MOV","angle_id":"явно_размеченный_ракурс_VALIDATION","label":0,"content_sha256":"<actual lowercase SHA256 of source 3>"},
  {"video_id":"точное_имя_4.MOV","angle_id":"явно_размеченный_ракурс_B","label":1,"positive_times_s":[8.2],"content_sha256":"<actual lowercase SHA256 of source 4>"}
]
```

- `label`: только целое `0` или `1`; положительный видеоролик требует непустого списка секунд `positive_times_s`.
- `angle_id` предоставляет человек/внешняя разметка. Код не угадывает ракурс по имени/видео. При неизвестных ракурсах обучение/holdout-отчёт заблокированы, пока не задана явная разметка.
- Для fitting обязательны три стороны: повторяемые `--train-angle`, `--validation-angle`, `--holdout-angle`; множества непустые и попарно непересекающиеся. Каждая строка manifest должна попасть ровно в одну сторону; каждый указанный ракурс должен иметь видео. Двухсторонний Python helper `split_videos` оставлен только для диагностики, он не разрешает обучение/калибровку без validation.
- `content_sha256` **обязателен у каждой строки**, ровно 64 lowercase hex, полученных из реальных исходных байтов. Placeholders выше нужно заменить. Один hash не может пересекать ни одну пару train/validation/holdout, даже при разных angle IDs. Конфликтующие binary labels одного содержимого запрещены. В joins manifest/cache используется `ml.labels.normalize_video_name`; literal ID в provenance сохраняется, коллизии canonical ID внутри источника запрещены.
- SHA manifest сравнивается с SHA в extraction-cache metadata при его наличии; отсутствие последнего явно записывается `unavailable`, а не считается независимым подтверждением. Метаданные не заменяют повторное хэширование исходных файлов. Кэши с одинаковым hash и разными timestamps/features отклоняются.
- Все кадры видео остаются на одной стороне. Holdout не участвует в mean/std, оптимизации или подборе порога.
- В positive-видео supervised positive — только кадры в `±positive-radius` секунд от размеченного события. Остальные кадры этого видео **unknown и исключаются из loss**, не отрицательные. В negative-видео все sampled кадры имеют label 0. Если возле positive timecode не оказалось sampled кадра в train, обучение останавливается с ошибкой. Для train нужны оба класса.
- Стандартизация mean/std считается **только на supervised train-кадрах**; константный признак получает scale 1. Побайтовые дубли считаются один раз, с объединением alternative positive anchors. Full-batch logistic loss балансируется по классам; seed фиксирует инициализацию. Schema v2 сохраняет train/validation/holdout video IDs, angle IDs, hashes и provenance проверки кэшей. Старые модели без validation-партиции не загружаются. Это не zero-shot и не эвристический random predictor.

## Train → predict → evaluate

Сначала извлеките отдельный кэш каждого manifest-видео. Все имена/ракурсы ниже замените фактическими размеченными значениями; число `--cache` должно точно соответствовать manifest:

```bash
py -3.10 -m ml.baseline train \
  --manifest artifacts/baseline/manifest.json \
  --train-angle 'явно_размеченный_ракурс_A' --holdout-angle 'явно_размеченный_ракурс_B' \
  --validation-angle 'явно_размеченный_ракурс_VALIDATION' \
  --cache artifacts/baseline/video1.npz --cache artifacts/baseline/video2.npz \
  --cache artifacts/baseline/video3.npz --cache artifacts/baseline/video4.npz \
  --positive-radius 2 --epochs 1000 --learning-rate 0.1 --l2 0.001 --seed 0 \
  --threshold 0.5 --window-seconds 2 \
  --model artifacts/baseline/model.npz --report artifacts/baseline/holdout.json

py -3.10 -m ml.baseline predict \
  --cache artifacts/baseline/smoke.npz --model artifacts/baseline/model.npz \
  --output artifacts/baseline/smoke-candidate.json

py -3.10 -m ml.baseline evaluate \
  --manifest artifacts/baseline/manifest.json \
  --train-angle 'явно_размеченный_ракурс_A' --holdout-angle 'явно_размеченный_ракурс_B' \
  --validation-angle 'явно_размеченный_ракурс_VALIDATION' \
  --cache artifacts/baseline/video4.npz \
  --model artifacts/baseline/model.npz --report artifacts/baseline/repeated-holdout.json
```

`train` получает кэши **всех** видео и сохраняет diagnostic holdout JSON. `evaluate` получает **только** holdout-кэши и требует повторить оригинальный split модели целиком; пропуск предсказания не заменяется отрицательным. Отчёт содержит полный список candidate predictions с reference label и angle, но не публикует выдуманные benchmark-метрики: официальный evaluator — `ml/eval.py`, которым владеет другой исполнитель.

Predict требует настоящую обученную модель; fallback при её отсутствии отсутствует. Модель — безопасный NPZ (mean, scale, weights, intercept + metadata JSON), **без pickle/joblib**. Входные размерности/числа проверяются. `predict --threshold ... --window-seconds ...` может явно менять параметры для новой инференс-задачи, но это не повод подбирать их по holdout.

Для неперекрывающихся окон `[k·window, (k+1)·window)` берётся mean вероятностей кадров; video score — max этих средних. При превышении `>= threshold` выдаётся `candidate`, реальный PTS максимального кадра лучшего окна (при равенстве самый ранний), timecode `HH:MM:SS.mmm`, описание ровно **«Кандидат baseline; требуется проверка»**. Окна без sampled кадров не заполняются фиктивными значениями. Вывод содержит coverage и признак `seen_in_training`; нет guessed legal category. Пустой/битый кэш, NaN, missing model/prediction и неявный split завершают CLI с ненулевым кодом.

## Интеграция

```python
from ml.baseline.features import load_cache, extract_video
from ml.baseline.classifier import load_model, predict_cache, train_classifier, evaluate_holdout

candidate = predict_cache(load_model("artifacts/baseline/model.npz"),
                          load_cache("artifacts/baseline/smoke.npz"))
```

Схема предсказаний baseline отдельная, но canonical joins используют общий `ml.labels.normalize_video_name`. При подключении к официальному evaluator нужен **явный адаптер** candidate → его входной формат, не подмена `candidate` на окончательное нарушение. Внешний pipeline должен сохранять ошибки/пропуски как ошибки. Вызовы изображений/кадров внешним API запрещены.

## Проверка и TDD evidence

Выполнены последовательные RED → GREEN циклы: реальные PTS sampling → explicit grouped split → segment labels → artifact/UTF-8 contract → windows → safe cache → numpy training → safe model → holdout reporting → ffmpeg VFR → offline guard → bounded batches → CLI → loader guard → extract wiring. Первоначальный stdlib RED: `Missing baseline behavior: core`; исправление реального sparse Google config отдельно дало RED `safetensors missing ... does not match Expected Google SigLIP...`, затем GREEN. Для ffmpeg 9 `-autorotate` — boolean flag **без** аргумента `1`; это проверено реальным запуском, вариант с `1` отвергнут ffmpeg и исправлен до GREEN.

```bash
python -m unittest discover -s tests -p test_baseline.py -v
py -3.10 -m unittest discover -s tests -p test_baseline.py -v
```

На момент записи: Python 3.14 — **18 tests, OK (skipped=7)**; Python 3.10 — **18 tests, OK**. Числовые train/predict/CLI проверки исполняют настоящий NumPy на небольшом **явно synthetic TEST FIXTURE**, не результаты датасета. Decoder tests используют настоящий локальный ffmpeg на сгенерированном lavfi VFR-ролике без визуального просмотра. Спай/mock применяется лишь на тяжёлой границе загрузчика pretrained и для wiring extraction; это **не** подтверждение реального SigLIP inference. Реальная rotation-проверка на отдельном локальном lavfi fixture также выполнена: MOV display matrix `rotation=90`, decoder output совпал с явным `transpose=cclock` (150528 RGB bytes, PTS 0.0), а не с неразвёрнутым кадром. При создании fixture ffmpeg 9 потребовался `-display_rotation 90` перед `-i`; старый `-metadata:s:v:0 rotate=90` не создал display matrix, что проверено ffprobe. Реальные импорты `SiglipVisionModel`, `AutoImageProcessor`, torch 2.13.0+cpu проверены в Python 3.10. Полный pretrained forward затем выполнен на собственной CPU-машине: официальный checkpoint проверен SHA256, из настоящего локального `X-1.mov` получены 2 конечных вектора по 768 компонент с PTS 0.0 и 1.0. Использовались `HF_HUB_OFFLINE=1`, локальные safetensors, batch=1, CPU threads=2; кадры не сохранялись и не передавались внешним сервисам. Кэш: `artifacts/baseline/smoke.npz`. `cap_reached=true`: это только smoke, не анализ всего видео и не измерение качества обнаружения. Никаких фактических метрик качества на 42 видео этим модулем не заявлено.

По ревью PR #30 `content_sha256` обязателен без opt-in. Baseline/split/extractor и соседние regression tests после исправлений: **191 passed**, Ruff clean. Набор проверяет также canonical collisions, все пары трёхстороннего split, отсутствие повторного веса дублей, объединение якорей, нетронутые validation/holdout features, model schema и CLI. Это synthetic correctness, не обучение на датасете. Импорт предоставленного набора сохраняет SHA256 в draft manifest (см. `analysis-import.md`).
