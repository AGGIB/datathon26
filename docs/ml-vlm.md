# Локальная advisory VLM

## Статус и границы

**Этот advisory runtime не вызывается из `ml.pipeline`.** Отдельный офлайн-эксперимент на пяти
парах с реальными кадрами не показал пользы исследованного SmolVLM-256M:
0/5 пар, все десять ответов не прошли JSON-контракт. Методика и настоящие
времена: [ml-vlm-pairs.md](ml-vlm-pairs.md). Qwen3-VL-4B здесь не запускалась.

Отдельная opt-in ветвь type discovery из [PR #42](https://github.com/AGGIB/datathon26/pull/42)
использует собственный runtime и по умолчанию выключена. Её экспериментальные
добавления не меняют advisory-контракт ниже и не удаляют исходные CV-события.

`ml.vlm` — инструмент проверки видимых действий, не юридическое заключение
и не замена CV. Advisory API сохраняет **все исходные Candidate**, включая
rejected, ошибки и timeout. Identity, порядок, повторы, observation, t_peak,
evidence_times, highlight_boxes и confidence не меняются. Калибровка для
фильтрации/изменения confidence не выполнена.

Только собственная физическая локальная машина CPU/GPU. Никаких cloud GPU,
арендованных серверов, внешних inference API или отправки видео, кадров,
prompts, ответов и описаний сцен внешнему ИИ. Зависимости/веса готовятся
заранее; runtime ничего не скачивает. Аудио, идентификация людей/лиц,
биометрия, чувствительные признаки, намерения и виновность не запрашиваются.

## Advisory API: startup отдельно от остатка К2

Это пример явного использования инструмента, **не инструкция включить его
в production**. При Windows запуск worker — под `if __name__ == "__main__"`.

```python
from ml import vlm
from ml.kb import catalog

# Один раз при старте, до обработки заданий и до бюджета К2.
ready = vlm.configure(
    model_path="models/smolvlm-256m-instruct",  # локальные веса, не HF ID/URL
    catalog=catalog(),
    device="cpu",
    startup_timeout_s=120,
    max_new_tokens=128,
)

# Только явный advisory-проход, если он действительно нужен вызывающему коду.
assembly_candidates = vlm.verify(cv_candidates, video_path, remaining_k2_sec)
assert len(assembly_candidates) == len(cv_candidates)
assert all(a is b for a, b in zip(assembly_candidates, cv_candidates))
```

`configure(...) -> bool` готовит model/processor один раз. False означает
недоступный runtime, не отсутствие нарушений. Без catalog лениво вызывается
`ml.kb.catalog()`; ошибка загрузки/снимка каталога возвращает False, оставляет
runtime неконфигурированным и пишет warning только с классом исключения.
Пустой «здоровый» каталог вместо ошибки не создаётся. Сырой YAML не заменяет
проверенный нормативный каталог.

`verify(candidates, video_path, budget_sec) -> list[ml.candidates.Candidate]`
не загружает модель и не проверяет заново файлы весов. Без конфигурации
возвращает оригиналы. Ошибки адаптера/каталога/кадров/cleanup дают безопасный
warning и оригиналы (**fail-open** для необязательного advisory-этапа).
KeyboardInterrupt/SystemExit не поглощаются. Неверный budget по-прежнему
отвергается: требуется конечное положительное число, не bool.

Budget — **оставшийся общий К2**, не новый лимит 300 с. Абсолютный monotonic
deadline начинается до materialization/adaptation и проходит через runtime →
adapter → verifier. Preparation, imports, probe, extract и inference расходуют
его; истёкший budget не обновляется. Пользовательский синхронный property/
iterator не становится прерываемым от передачи дедлайна.

Backend — один cached spawn-процесс. Nonblocking lease защищает validate/
infer/close; конкурент получает `ModelUnavailable("runtime_busy")` без чужого
Pipe I/O/ответа/cleanup. После смерти, timeout или close скрытого reload нет:
нужен новый backend и явный startup. Configure/reconfigure выполняются без
активных заданий. Внутренний backend закрывается при переконфигурации;
успешно инъецированный backend закрывает его владелец.

## Кадры и временная координата

`OwnerFrames` лениво импортирует `ml.video` и OpenCV только в отдельном
killable worker. Probe кешируется на проход видео. На каждый extraction —
**один** `ml.video.frames_at(path, times)`, не три `frame_at` с начала.
Результаты возвращаются в исходном выбранном порядке; отсутствие кадра — ошибка.
Перед извлечением, сохранением PNG и ответом проверяется дедлайн.

Сохраняется координата владельческого CV: `round(t * fps)` с ограничением
границ, время `index / fps`. Это **не container PTS**, особенно при VFR или
смещении потока. Legacy-поле `pts_s` здесь содержит ту же index/FPS координату.
Владельческий декодер не принимает timeout: родитель ограничивает worker
оставшимся дедлайном и завершает его terminate/kill до cleanup кадров.
Завершение процесса имеет накладные расходы ОС; hard real-time SLA не заявлен.

`FFmpegFrames` остаётся отдельным PTS-источником: ffprobe читает реальные
best_effort_timestamp_time относительно container start_time. Он не совместим
с CV timeline автоматически. Для legacy/injected источника требуется явный
`timebase_verified=True` после независимой проверки/преобразования;
по умолчанию inference блокируется как unverified_timebase.

## Адаптер и низкоуровневый контракт

`verify_pipeline_candidates` принимает исходные `ml.candidates.Candidate`
и проверенный `ml.kb.catalog()`, возвращает `PipelineVerificationReport`.
Retained tuple содержит исходные объекты; вложенный `VerificationReport` —
низкоуровневые advisory-словари, не вход сборщика К1. `allow_rejection` принят
только для совместимости: строго bool и никогда не разрешает удаление.

Адаптер отображает type_id, t_start/t_end в violation_type, start_s/end_s,
назначает уникальные cv-N в порядке входа. Rule_id — literal type_id, не
выдуманный номер нормы. Используются только verified источники владельца.
Поддерживаются legacy RuleText.basis и document/number_date/clause; fallback
разрешён лишь при отсутствии basis, не при явном пустом значении. Модель не
генерирует нормативные цитаты. CV observation/boxes/confidence и цитаты не
попадают в prompt. Draft/unknown нормы не расходуют inference. Open checklist
в production-адаптере выключен и не создаёт новые CV-кандидаты.

Основное имя низкоуровневого TypedDict — **`VerifierCandidate`**, чтобы не путать
его с владельческим frozen dataclass `ml.candidates.Candidate`. Старое имя
`ml.vlm.Candidate` сохранено как совместимый alias. API
`verify_candidates(video_path, candidates, rules, *, backend, frame_source,
budget_sec)` не создаёт/не загружает backend. Передавать явно подготовленный
`LocalTransformersBackend`, затем закрывать его в finally.

Кандидат-словарь: unique непустой candidate_id, violation_type, rule_id
(строка/null), конечные start_s/end_s, 0 ≤ start ≤ end ≤ duration.
Дополнительные поля сохраняются, но модели не передаются. IDs не обрезаются.
Rule-словарь: непустые rule_id, violation_type, description, citation и строго
verified=True; дубликаты исключаются.

## Ответы, лимиты и runtime

Низкоуровневая candidate-schema содержит candidate_id, rule_id, violation,
confidence, insufficient_evidence и observations с timestamp_s/text.
Строгий JSON: неизвестные поля/IDs, Markdown, дубли ключей, NaN/Infinity,
bool вместо confidence и наблюдения вне переданных кадров дают uncertain.
Низкая confidence/нехватка данных — uncertain. Violation=false может дать
advisory rejected, но исходный кандидат остаётся. Confidence не калибрована;
лексический запрет юридического текста не гарантирует отсутствие галлюцинаций.
Двухполевой pair-screening JSON — **другой** экспериментальный контракт.

Общий max_calls — целое 0..6, включая ошибки и optional checklist.
До трёх кадров на кандидата; probe кешируется. Cap шага по умолчанию 60 с,
но фактически min(cap, остаток). Optional video_timeout_s только сокращает
budget, не заменяет обязательный остаток К2. Injected тестовые зависимости
могут ограничиваться daemon-thread ожиданием; поток не killable и не является
production cancellation.

Импорт `ml.vlm` не загружает torch/PIL/transformers/OpenCV/NumPy/ml.video/ml.kb.
Loader использует локальные пути, offline-переменные, local_files_only=True,
trust_remote_code=False и safetensors, проверяет конфигурацию/все шарды и
отвергает missing/mismatched/error loading diagnostics. Worker stdout/stderr
подавлены. Offline-настройки сами по себе не OS network sandbox.

```bash
python -m ml.vlm preflight --model models/smolvlm-256m-instruct
python -m ml.vlm preflight --model models/smolvlm-256m-instruct --load --timeout 120
python -m pytest -q tests/test_vlm.py tests/test_vlm_budget.py tests/test_vlm_pipeline_adapter.py tests/test_vlm_pair_eval.py
python -m ruff check ml/vlm tests/test_vlm*.py
```

Preflight ready без --load подтверждает структуру; runtime_loaded — загрузку,
не пригодность JSON, качество или SLA. Низкоуровневый CLI verify использует
PTS-контракт, не CV-кандидаты напрямую; требует --budget-sec и отдельно
--startup-timeout. Его локальные отчёты не предназначены для внешних логов.
Разрешённые точные размеры/revisions: [licenses-ml.md](licenses-ml.md).
