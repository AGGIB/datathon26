# Лицензии ML-компонентов Нурхана

Это дополнение для общей таблицы `docs/licenses.md`, которую ведёт команда. Лицензия каждого нового набора весов проверяется отдельно, не переносится с названия семейства.

## SigLIP baseline

- Модель: `google/siglip-base-patch16-224`.
- Источник: https://huggingface.co/google/siglip-base-patch16-224
- Model card: https://huggingface.co/google/siglip-base-patch16-224/blob/main/README.md
- В метаданных репозитория указано `apache-2.0`.
- Проверенная ревизия: `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.
- Файл весов: `model.safetensors`, 812672320 байт по данным API модели.
- SHA256 исходных весов: `2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`.
- Используются только визуальные признаки; идентификация лиц не выполняется.
- Загрузка весов — отдельный подготовительный этап. Во время анализа допустим только локальный путь и `local_files_only=True`.

## VLM

Адаптер поддерживает локальные веса. Конкретная production-модель ещё не утверждена: успешная загрузка не подтверждает качество проверки нарушений.

Для CPU smoke подготовлен `HuggingFaceTB/SmolVLM-256M-Instruct`:

- Model card: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- Лицензия checkpoint: Apache-2.0 (проверено в model card и metadata).
- Ревизия: `7e3e67edbbed1bf9888184d9df282b700a323964`.
- `model.safetensors`: 513028808 байт, SHA256 `74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e`, проверен после скачивания.
- Локальный CPU-профиль меняет только `preprocessor_config.json`: `size.longest_edge` с 2048 на 512, чтобы ограничить число image patches и память. Исходные параметры сохранены отдельно; нейронные веса не изменены.
- Модель английская и малая; качество русскоязычного чек-листа и корректность структурированного ответа не подтверждены. В smoke непрошедший JSON был сохранён как `uncertain`, кандидат не удалён.

### Разрешённые checkpoints (решение владельца PR #30)

Публичные метаданные сверены 2026-10-17 по **точным ревизиям**, не только по
названию семейства. Каждый перечисленный checkpoint имеет `apache-2.0` в HF API.
Лицензионное разрешение не означает, что модель скачана, помещается на этой машине
или проходит прикладную проверку качества.

| Checkpoint | Закреплённая ревизия | Файлов safetensors |
|---|---|---:|
| `google/siglip-base-patch16-224` | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | 1 |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | `7e3e67edbbed1bf9888184d9df282b700a323964` | 1 |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb59b408707209984f360a5f4ad7e47` | 1 |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `482adb537c021c86670beed01cd58990d01e72e4` | 2 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 5 |
| `Qwen/Qwen2.5-VL-32B-Instruct` | `7cfb30d71a1f4f49a57592323337a4a4727301da` | 18 |

[checkpoints-ml.json](checkpoints-ml.json) содержит для **каждого** файла весов
точное имя, размер, SHA256 из LFS metadata и ссылки на API/model card выбранной
ревизии. Поле `weight_manifest_sha256` — SHA256 UTF-8 JSON списка `weights`,
отсортированного по `path`, сериализованного с `sort_keys=True,
separators=(',', ':')`. Это проверка целостности списка, не замена проверки
SHA256 каждого фактически скачанного файла. Веса и приватные данные в этом
публичном манифесте отсутствуют; сетевые запросы выполнялись только за публичными
метаданными на подготовительном этапе.

**Запрещены:** `Qwen/Qwen2.5-VL-3B-Instruct` (research license) и
`Qwen/Qwen2.5-VL-72B-Instruct` (Qwen license, не Apache-2.0). Не выбирать их и
не переносить разрешение 7B/32B на другие размеры, квантования или производные
checkpoints. Любой иной checkpoint требует отдельного согласования лицензии.

## Qwen3-VL: отдельный локальный эксперимент

По новому запросу на улучшение распознавания **типов** выбран
`mlx-community/Qwen3-VL-4B-Instruct-4bit`. Это кандидат для эксперимента на Apple
M4 с 16 GB общей памяти; результат качества и production-пригодность ещё не
подтверждены. Первичный аудит не обнаружил этих весов в HF cache или каталогах
Hackathon. Метаданные проверены 2026-10-17; они сами по себе не подтверждают
последующую загрузку, проверку локального файла или успешный инференс.

| Набор весов | Ревизия | Лицензия по HF API / model card | Размер safetensors |
|---|---|---|---:|
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md), исходник | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Apache-2.0 | 8 875 719 344 байт, два файла |
| [mlx-community/Qwen3-VL-4B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/blob/2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b/README.md), производный checkpoint | `2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b` | Apache-2.0 | 3 093 767 283 байт, один файл |

У производного checkpoint отдельно сверены лицензия и [конфигурация](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/blob/2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b/config.json):
`qwen3_vl`, affine quantization, 4 бита, group size 64. Его model card указывает
конвертацию из Qwen3-VL-4B-Instruct через mlx-vlm 0.3.4; точная историческая
ревизия исходника при той конвертации не указана. Ревизия исходника выше —
проверенный текущий источник лицензии и архитектуры, а не выдуманная гарантия
побитовой воспроизводимости конвертации. SHA256 файла производных весов из LFS:
`90eeb02604181dbcccd0a30a1f550a4a8928ca7dcbee4aee1449239306cfdfca`.
Все три файла весов и канонические хеши списков записаны в
[checkpoints-ml.json](checkpoints-ml.json). Исходные 8.88 GB не требуются для
запуска уже квантованного checkpoint.

Эксперимент использует отдельное окружение; основной Transformers 4.51.3 не
обновляется и не поддерживает `qwen3_vl`. Проверенные метаданные
[mlx-vlm 0.7.8](https://pypi.org/project/mlx-vlm/0.7.8/) требуют Python >=3.10,
MLX >=0.32.3 и Transformers >=5.14.0. У [MLX-VLM](https://github.com/Blaizzy/mlx-vlm/blob/main/LICENSE),
[MLX](https://github.com/ml-explore/mlx/blob/main/LICENSE) и пакета
[mlx-metal 0.32.4](https://pypi.org/project/mlx-metal/0.32.4/) указана MIT.
Это проверка перечисленных компонентов, не лицензий всех транзитивных пакетов;
фактически установленные 79 пакетов фиксируются в
[requirements-ml-vlm-mlx-lock.txt](../requirements-ml-vlm-mlx-lock.txt).
Подготовка `AutoProcessor` требует `torch==2.7.0` и `torchvision==0.22.0`
даже при последующем inference через MLX; они установлены только в отдельное
окружение `var/venvs/vlm-mlx` и закреплены в
[requirements-ml-vlm-mlx.txt](../requirements-ml-vlm-mlx.txt).
Наличие этих пакетов ещё не подтверждает загрузку полной модели, совместимость
процессора и структурированного декодирования или качество распознавания.

Загрузка выбранного snapshot — отдельный согласованный подготовительный шаг.
Во время эксперимента используются локальный каталог, офлайн-режим, штатная
реализация архитектуры без `trust_remote_code` и один процесс с моделью на GPU.
Видео, кадры, промпты и ответы не отправляются во внешние inference API. Выбор
промпта и параметров выполняется только на train/validation; новый ракурс
оставляется для замороженной проверки. Совместимость runtime, расход памяти,
латентность и точность проверяются фактическим локальным запуском.

## Qwen3.5-4B: дополнительный эксперимент

Дополнительный кандидат —
[mlx-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit/blob/32f3e8ecf65426fc3306969496342d504bfa13f3/README.md),
ревизия `32f3e8ecf65426fc3306969496342d504bfa13f3`. Карточка производных весов
указывает Apache-2.0; у исходного
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE)
также проверена Apache-2.0. Ревизия исходника
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` фиксирует проверку лицензии,
а точная историческая ревизия конвертации остаётся неизвестной.

В [публичном LFS pointer](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit/commit/32f3e8ecf65426fc3306969496342d504bfa13f3)
для единственного `model.safetensors` указаны **3 034 300 695 байт** и SHA256
`5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db`.
Параметры: `qwen3_5`, affine 4 bit, group size 64. Идентификаторы и канонический
хеш списка весов добавлены в [checkpoints-ml.json](checkpoints-ml.json).
Кандидат с похожим именем `mlx-community/Qwen3.5-4B-4bit` отдельно не разрешён:
у него при проверке отсутствовала карточка с явным указанием лицензии.

Для отдельного согласованного запуска используется каталог
`models/Qwen3.5-4B-MLX-4bit`, явно переданный через `--model` экспериментального
runner или `ML_VLM_MODEL` при включённом экспериментальном runtime.
Существующий каталог Qwen3-VL и настройки по умолчанию не заменяются.
Перед запуском проверяются локальные веса и все ссылки `weight_map` в index;
наличие одного файла не позволяет игнорировать отсутствующие shards.

Установленный `mlx-vlm==0.7.8` содержит штатную реализацию `qwen3_5` и
регистрацию `Qwen3VLProcessor`; конфигурация кандидата использует
`Qwen2VLImageProcessorFast`, patch size 16 и merge size 2. Проверка исходного
кода не потребовала обновления окружения или внешней реализации attention.
MLX-подготовка и worker разрешают только `qwen3_vl` и `qwen3_5`.
Для обоих явно передаётся `enable_thinking=False` в `apply_chat_template`:
у Qwen3.5 [thinking включён по умолчанию](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md#instruct-or-non-thinking-mode).
Лимиты кадров, пикселей, JSON и времени сохраняются.

Это разрешение кандидата для подготовки и локального измерения, а не результат
качества. Fake API tests не доказывают успешный запуск модели, расход памяти
на M4 16 GB, качество на train/validation или перенос на новый ракурс.
Приватные видео, кадры, промпты и ответы остаются локальными.

## Qwen3.5-9B: отдельный локальный кандидат

Подготовлен один snapshot
[mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit/blob/8b2b98c00a6b4d291155e4890773ca8f769aee53/README.md),
ревизия `8b2b98c00a6b4d291155e4890773ca8f769aee53`, с явной Apache-2.0.
В карточке есть расхождение происхождения: поле `base_model` называет
`Qwen/Qwen3.5-9B-Base`, а текст конвертации — `Qwen/Qwen3.5-9B` и mlx-vlm 0.3.12.
Лицензии обоих названных исходников проверены отдельно:
[9B, ревизия c202236](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE) и
[9B-Base, ревизия 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/LICENSE)
содержат Apache-2.0. Эти ревизии фиксируют источники лицензий; исторический
вход конвертации не установлен, `conversion_input_revision` остаётся `null`.

Закреплённые [HF metadata](https://huggingface.co/api/models/mlx-community/Qwen3.5-9B-4bit/revision/8b2b98c00a6b4d291155e4890773ca8f769aee53?blobs=true)
указывают два файла весов:

| Файл | Байты | SHA256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 5 349 771 222 | `a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd` |
| `model-00002-of-00002.safetensors` | 600 449 850 | `b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d` |

Веса занимают **5 950 221 072 байта**, все 13 файлов snapshot —
5 977 073 303 байта. Канонический хеш списка весов записан в
[checkpoints-ml.json](checkpoints-ml.json). Локальный каталог —
`models/Qwen3.5-9B-4bit`, передаваемый явно через `--model` или `ML_VLM_MODEL`.
Подготовка проверяет потоковый SHA256 каждого shard; независимый аудит сверяет
receipt с закреплёнными LFS SHA, фактическими размерами и локальными metadata.
Все 1 260 записей `weight_map` ссылаются на два ожидаемых shard; лицензия
исходника сохраняется локально. Отдельное повторное чтение 5.95 GB аудитором
для вычисления второго SHA во время параллельного timed-прогона не выполнялось.

[Конфигурация](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit/blob/8b2b98c00a6b4d291155e4890773ca8f769aee53/config.json)
указывает штатный `qwen3_5`, affine 4 bit, group size 64. [Metadata процессора](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit/blob/8b2b98c00a6b4d291155e4890773ca8f769aee53/processor_config.json)
указывают `Qwen3VLProcessor`, patch size 16, merge size 2 и temporal patch size 2.
Используется отдельное закреплённое окружение mlx-vlm 0.7.8 с
`enable_thinking=False`, офлайн-режимом и `trust_remote_code=False`.
Проверка структуры выполнена без импорта ML, загрузки модели или inference.
В рамках этого аудита расход unified memory на M4 16 GB, латентность и качество 9B не измерялись.
Следующий шаг — отдельный локальный smoke; приватные данные остаются локальными.

## Зависимости

Лицензии библиотек не заменяют проверку лицензий весов. Итоговый образ должен содержать необходимые тексты лицензий и уведомления поставщиков. Данные и веса в Git не входят.
