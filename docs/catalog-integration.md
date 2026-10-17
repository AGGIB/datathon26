# Совместимость каталога с К3

Каталог сохраняет существующие 17 английских `id` и исторические значения `examples`; `codex_code` остаётся стабильным исходным кодом. Четырнадцать отображаемых названий смягчены до наблюдаемых признаков и синхронизированы с таблицей ID и аналитическим сопоставлением. `summary` — короткая нейтральная формулировка. Разметка и исследовательские материалы не изменены, новые частные примеры не добавлены. Общие критерии и нормы обновлены согласно решению тимлида; добавлены десять активных дополнительных типов и пять записей исключений.

Корневые `version: 1` и `violations` сохранены. `id` — стабильный runtime ID, `codex_code` — исходный код Т/Д. Машиночитаемая таблица — [type-ids.json](../kb/type-ids.json); аналитический [type-mapping.json](../kb/legal/type-mapping.json) сохраняет кириллические ID и ссылается на runtime ID через `teamlead_policy.catalog_id`.

## Статусы и проверяющий

`policy_status: approved_indicator` и `enabled: true` обозначают принятую индикаторную политику для всех 27 активных записей. `implementation_status: not_implemented` означает отсутствие подтверждённой реализации детектора в этой работе. Исключённые пять типов находятся только в `excluded_types`, имеют `enabled: false` и не передаются загрузчику активных типов.

Для совместимости с загрузчиком `status: verified` у 26 записей означает только сверку цитаты и источника, явно обозначенную `citation_status: citation_verified`. У каждой основной и дополнительной ссылки указаны `verified_by: Codex (агент Олжаса), сверка официального текста`, `verified_on: 2026-10-17` и `verification_scope: citation_and_source_integrity`. Это не личная аттестация Олжаса, не проверка факта нарушения и не визуальная валидация. Срез официальных текстов — 14.10.2026, дата обновления портала — 16.10.2026.

Т04 (`computer_left_unlocked`) имеет `status: draft`, `citation_status: pending`, `normative_basis: pending`, остаётся `enabled: true` и `policy_status: approved_indicator`. Ключ `rule` отсутствует. Состояние цитаты не должно отключать индикатор или менять вердикт.

## Контракт загрузчика и шаблонов

К3 описан в [contracts.md](contracts.md). Актуальный для этой правки загрузчик — `origin/pr-31:ml/kb.py`, коммит `215b51ad608e884a2e035d1f459a04ec3dd3950a`. Он читает `description_template`; прежнее описание поведения `9c287e5`, где использовался только `summary`, к этому контракту не относится.

- Загрузчик требует snake_case `id`, `name` и `status` из `draft` / `ready` / `verified`; `summary` сохраняется как дополнительное поле. Проверенное правило возвращается только при `status: verified`, наличии `document` / `verified_by` и допустимом `citation_status`. Статус цитаты управляет ссылкой, а не активностью индикатора.
- Все 27 шаблонов используют `{t}` и `{observation}`. Разрешены только пять полей: `t`, `observation`, `rule_document`, `rule_clause`, `rule_number_date`; преобразования и format specifications запрещены. Наличие `{observation}` включает строгую проверку; отсутствие переводит старый шаблон в общий fallback. Согласованные шаблоны должны загружаться без такого замещения.
- У 26 типов с правилом окончание строгое: `Основание: {rule_document}, {rule_clause}.` У Т04 — `Нормативное основание уточняется.`; ссылочных placeholders и ключа `rule` нет. `status: draft` Т04 не должен исключать принятый индикатор или автоматически давать `no_violation`.
- `document` — официальное название; `number_date` — отдельные реквизиты акта; `clause` содержит только номер пункта / подпункта, пояснение редакции — в `clause_note`. `RuleText.as_ref()` возвращает прежние четыре API-поля `document`, `clause`, `quote`, `url`; наличие `number_date` в KB не расширяет К1.
- Для Т06 `criterion_branches[].template_values.observation` заполняет одно наблюдение выбранной ветви (`one_matched_branch`). Условия OR и порог только первой ветви сохраняются. Выбор ветви и подтверждение применимости остаются обязанностью локального анализа; загрузка корректного шаблона их не реализует.

В независимой ветке `origin/feat/nurkhan-ml` (`8aa2680`) кандидат без проверенного правила получает внутренние состояния `uncertain` / `no_verified_rule` с сохранением кандидата. При интеграции это нельзя превращать в удаление Т04 или автоматическое `no_violation`. Вердикты К1 остаются только `no_violation` / `violation` и определяются итоговыми событиями.

## Источники и объём PR

В Git остаются полные текстовые снимки официальных актов, стабильные URL, SHA256 и происхождение. Исходные HTML и инфокарточки сохранены локально в игнорируемом `data/legal-source-backup/`. Проверка текста и исторического индекса якорей не является новой DOM-проверкой или подтверждением живой страницы. Формат и воспроизводимые команды — в [README](../kb/legal/README.md).

PR32 охватывает нормативные источники, каталог, таблицу ID, документацию и проверки этой базы. `tools/audit_dataset.py`, `tools/check_audit_dataset.py` и `docs/data-intake.md` сохранены отдельно в ветке `codex/dataset-intake-tools`. Они не являются зависимостью нормативного валидатора.

Backend, ML и `docs/contracts.md` здесь не изменяются. Совместимость проверяется синтетически через реальный `parse_catalog` указанной версии загрузчика: 27 активных типов, 26 правил, Т04 без правила, пять исключённых вне активного списка; все шаблоны должны рендериться без fallback. Проверки целостности выполняются командами `python -B tools/validate_legal_kb.py` и `python -B tools/check_legal_catalog.py` в отдельном окружении с [requirements](../tools/requirements-legal.txt). Текущий прогон валидатора прошёл: 8 источников / 14 кандидатов / 32 аналитические записи / каталог 27/5. В изолированной копии отклонены все 20 отрицательных сценариев. Неизменённый реальный загрузчик `215b51a` получил 27 типов и 26 правил; Т04 остался активным без правила, все 27 шаблонов прошли синтетический рендер без fallback, пять исключённых типов не попали в активные. Это проверка структуры и рендера, не наблюдений видео. Коммит проверяемой версии фиксируется при передаче; прежние прогоны `9c287e5` не подменяют проверку текущего контракта.

Эти проверки не являются боевым QA, оценкой модели или анализом видео. Реальные детекторы требуют отдельной реализации и локальной проверки. Issues остаются открытыми; нормативная часть относится к #21, #22, #24, документация QA — только к части #25. Банковская выборка #23 ведётся отдельно; #25 ожидает готовый API.

## Соответствие кодов

| Исходный код | Runtime ID | В каталоге |
| --- | --- | --- |
| Т01 | `banknote_returned_after_check` | активен |
| Т02 | `no_visible_client_id_check` | активен |
| Т03 | `previous_cash_left_on_desk` | активен |
| Т04 | `computer_left_unlocked` | активен |
| Т05 | `cash_from_bag_stored_without_recount` | активен |
| Т06 | `door_left_open` | активен |
| Т07 | `bag_accepted_without_check` | активен |
| Т08 | `second_client_served_alongside` | активен |
| Т09 | `safe_left_open` | активен |
| Т10 | `key_left_in_safe_lock` | активен |
| Т11 | `stamps_and_book_left_on_desk` | активен |
| Т12 | `cash_left_on_desk_when_leaving` | активен |
| Т13 | `person_entered_cash_area` | активен |
| Т14 | `recount_after_client_left` | активен |
| Т15 | `personal_items_on_desk` | активен |
| Т16 | `no_banknote_detector_used` | активен |
| Т17 | `food_or_drink_at_workplace` | активен |
| Д01 | `cashier_left_before_service_completed` | активен |
| Д02 | `loose_banknotes_issued_without_recount` | активен |
| Д03 | `mixed_receipts_across_documents` | активен |
| Д04 | `receipt_not_visibly_issued` | активен |
| Д06 | `damaged_bag_without_special_procedure` | активен |
| Д08 | `cash_processed_without_required_controller` | активен |
| Д09 | `incorrect_sorting_or_final_packaging` | активен |
| Д10 | `cassette_transferred_unsealed` | активен |
| Д11 | `vault_access_without_required_escort` | активен |
| Д13 | `personal_camera_phone_or_cash_in_cash_unit` | активен |
| Д05 | `cash_amount_document_mismatch` | исключён |
| Д07 | `cash_transfer_without_accounting` | исключён |
| Д12 | `protected_information_disclosed` | исключён |
| Д14 | `cash_day_documents_not_preserved` | исключён |
| Д15 | `system_checks_or_archive_noncompliance` | исключён |
