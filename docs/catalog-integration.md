# Совместимость каталога с К3

Основа интеграции — `main` на `5a7db52640821124c549ae35e837be6ab5938153`. Существующие 17 английских `id`, названия и значения `examples` сохранены. Разметка и исследовательские материалы из `main` не изменены, новые частные примеры не добавлены. Общие критерии и нормы обновлены согласно решению тимлида; добавлены десять активных дополнительных типов и пять записей исключений.

Корневые `version: 1` и `violations` сохранены. `id` — стабильный runtime ID, `codex_code` — исходный код Т/Д. Машиночитаемая таблица — [type-ids.json](../kb/type-ids.json); аналитический [type-mapping.json](../kb/legal/type-mapping.json) сохраняет кириллические ID и ссылается на runtime ID через `teamlead_policy.catalog_id`.

## Статусы и проверяющий

`policy_status: approved_indicator` и `enabled: true` обозначают принятую индикаторную политику для всех 27 активных записей. `implementation_status: not_implemented` означает отсутствие подтверждённой реализации детектора в этой работе. Исключённые пять типов находятся только в `excluded_types`, имеют `enabled: false` и не передаются текущему загрузчику активных типов.

Для совместимости с существующим загрузчиком `status: verified` у 26 записей означает только сверку цитаты и источника, явно обозначенную `citation_status: citation_verified`. У каждой основной и дополнительной ссылки указаны `verified_by: Codex (агент Олжаса), сверка официального текста`, `verified_on: 2026-10-17` и `verification_scope: citation_and_source_integrity`. Это не личная аттестация Олжаса, не проверка факта нарушения и не визуальная валидация. Срез официальных текстов — 14.10.2026, дата обновления портала — 16.10.2026.

Т04 (`computer_left_unlocked`) имеет `status: draft`, `citation_status: pending`, `normative_basis: pending`, остаётся `enabled: true` и `policy_status: approved_indicator`. Ключ `rule` отсутствует. Состояние цитаты не должно отключать индикатор или менять вердикт.

## Проверенные потребители и пределы интеграции

В `main` К3 описан в [contracts.md](contracts.md); действующего загрузчика каталога там ещё нет. В `origin/feat/pipeline-assembly` (`9c287e5`) проверены `ml/kb.py` и `ml/assemble.py`:

- Загрузчик требует snake_case ID, `name`, `summary` и `status` из `draft` / `ready` / `verified`. Он сохраняет все известные типы; статус цитаты управляет только наличием `rule`.
- Сборщик фильтрует по известному ID, confidence и наличию кадров, а не по `draft`. Вердикт формируется по итоговому списку событий. Синтетическая проверка с новым каталогом подтверждает: Т04 может давать `violation` без `rule`, с текстом «нормативное основание уточняется».
- Сборщик пока читает `summary`, а `description_template`, условия, дополнительные ссылки и выбор ветви Т06 не использует. `summary` обновлены на осторожные общие критерии без утверждений о скрытых сведениях. Подключение согласованных шаблонов и условий остаётся задачей ML; этот PR не меняет renderer и не заявляет реализацию детекторов.
- По независимой проверке `origin/feat/nurkhan-ml` (`8aa2680`) кандидат без проверенного правила получает `uncertain` / `no_verified_rule` с сохранением кандидата. При последующей интеграции это состояние нельзя превращать в удаление Т04 или автоматическое `no_violation`. Это внутреннее состояние, не третий вердикт К1.

Backend, ML и `docs/contracts.md` здесь не изменяются. Проверка совместимости использует синтетические данные и не является боевым QA, оценкой модели или анализом видео. Реальные детекторы требуют отдельной реализации и локальной проверки. Issues остаются открытыми; нормативная часть относится к #21, #22, #24, документация QA — только к части #25. Собственная съёмка #23 этим изменением не выполнена.

## Результаты проверки перед передачей

- `python -B -m pytest -q` на текущем `main` с нашей частью: 39 тестов прошли; одно предупреждение зависимости Starlette о deprecated alias AnyIO.
- Неизменённые `ml/kb.py`, `ml/assemble.py`, `ml/candidates.py` и существующие тесты из `9c287e5` проверены в отдельной временной папке с новым каталогом: 37 тестов прошли, включая дополнительный случай Т04. Зависимость `Box` заменена только на соответствующий type alias; детекция, видео и реальные кадры не запускались.
- `tools/validate_legal_kb.py`: 8 источников, 14 кандидатов, 32 аналитические записи, 27 активных и 5 исключённых типов.
- `tools/check_legal_catalog.py`: изолированная база и 14 отрицательных сценариев прошли; `tools/check_audit_dataset.py` — синтетические проверки прошли.
- `ruff check tools` и `pip check` прошли. Использовано отдельное временное окружение с `requirements-dev.txt` и `tools/requirements-legal.txt`; глобальное окружение не менялось.

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
