# Реестр внешних источников #23

Текущий scope — только банковские сцены Казахстана с подтверждённой свободной лицензией. Ограниченный поиск завершён без подходящих источников и остановлен по решению пользователя. Это не утверждение, что таких материалов не существует. Активных источников, клипов, меток и gold — **0**.

Дополнение от 18.10.2026: [проверка кандидатов из исследовательского архива](research-datasets-review.md) отделяет AI-подсказки от содержания и условий первоисточников. Датасеты из архива в активную выборку не импортировались.

Ниже — история происхождения полученных ранее материалов, полностью исключённых из активного импорта, review, экспорта и метрик. Байты локальных медиа сохранены; каталоги обратимо перемещены в excluded quarantine без чтения содержимого, активные media dirs пусты; metadata-only inventory находится в игнорируемом `data/external23/excluded/pr34-kz-rights-scope/`. Исторические ID не переписываются и не считаются допустимыми новыми ID.

| Публичная страница | Заявленная лицензия / статус прав | Атрибуция источника | Получено / способ | Причина исключения |
| --- | --- | --- | --- | --- |
| [commons-01](https://commons.wikimedia.org/wiki/File:Money_of_Russia_2025_exchange.webm) | `CC-BY-4.0` | Zeliotvankaizer | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-02](https://commons.wikimedia.org/wiki/File:Money_of_Russia_2025_exchange_rubli.webm) | `CC-BY-4.0` | Zeliotvankaizer | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-03](https://commons.wikimedia.org/wiki/File:Money_of_Russia_2025_changing_rubli.webm) | `CC-BY-4.0` | Zeliotvankaizer | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-04](https://commons.wikimedia.org/wiki/File:Money_of_box_case_Russia_2025_Aerodizain_%D0%90%D1%8D%D1%80%D0%BE%D0%B4%D0%B8%D0%B7%D0%B0%D0%B9%D0%BD_lefortovo_work.webm) | `CC-BY-4.0` | Zeliotvankaizer | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-05](https://commons.wikimedia.org/wiki/File:Flick_Cash_with_fingers.webm) | `LicenseRef-Commons-AttributionOnly` | User:Hoben7599 | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-06](https://commons.wikimedia.org/wiki/File:New_money_market_in_Dhaka_on_the_occasion_of_Eid_in_Ramadan_(1).webm) | `CC-BY-SA-4.0` | Delwar Hossain | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-07](https://commons.wikimedia.org/wiki/File:New_money_market_in_Dhaka_on_the_occasion_of_Eid_in_Ramadan_(4).webm) | `CC-BY-SA-4.0` | Delwar Hossain | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-08](https://commons.wikimedia.org/wiki/File:New_money_market_in_Dhaka_on_the_occasion_of_Eid_in_Ramadan_(12).webm) | `CC-BY-SA-4.0` | Delwar Hossain | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [commons-09](https://commons.wikimedia.org/wiki/File:201905_Coin_counter_working.webm) | `CC-BY-SA-4.0` | Jonashtand | 2026-10-17; публичный оригинал Wikimedia Commons; метаданные API | `bank_context_not_confirmed` |
| [careeronestop-tellers](https://cloudfront.careeronestop.org/Toolkit/Careers/Occupations/occupation-profile.aspx?keyword=Tellers&lang=en&location=07457&onetcode=43307100) | `PublicDomain` | CareerOneStop, U.S. Department of Labor | 2026-10-17; публичный CDN проекта US DoL | `out_of_geographic_scope` |
| [bank-nara12089](https://catalog.archives.gov/id/12089) | `LicenseRef-NARA-Undetermined` | National Archives and Records Administration, NAID 12089; FBI bank teller training; local identifier 65.15. Official record: https://catalog.archives.gov/id/12089 | 2026-10-17; прямая публичная WMV-ссылка NARA | `rights_unconfirmed` |
| [rbs-tcr-training](https://productionattic.com/portfolio/rbstcr) | `LicenseRef-RBS-RightsUnconfirmed` | Production Attic | 2026-10-17; временная progressive-ссылка публичного плеера портфолио; signed URL не публикуется | `rights_unconfirmed` |

CareerOneStop: официальная [FAQ о Public Domain](https://cloudfront.careeronestop.org/Videos/FAQs-about-videos.aspx) сохранена локально как подтверждение происхождения; источник исключён по географическому scope. NARA `Undetermined` и RBS `RightsUnconfirmed` не являются разрешением на использование; `private_evaluation_only` не заменяет лицензию. Для них исключены 2 полных исходника-записи и 20 планов. Commons не подтверждают требуемый банковский контекст и остаются только историческими метаданными.

Whitelist текущего инструментария: `CC0-1.0`, `CC-BY-4.0`, `PDM-1.0`, `PD-US-GOV` с точными страницами из `tools/external23_policy.py`. Это техническая проверка декларации и доказательств происхождения; фактическое право и контекст устанавливает человек. CC BY-SA и отдельные AttributionOnly / RightsUnconfirmed автоматически к whitelist не приравниваются.

Исходный issue #23 описывает свою съёмку. Позднее прямое указание пользователя ограничило текущую подготовку внешними банковскими сценами Казахстана; поиск остановлен, задача не закрыта. Постановочная съёмка этой веткой не предписывается. Инструкция — [external23-intake.md](external23-intake.md); неподтверждённая история опытов — [external23-history.md](external23-history.md).
