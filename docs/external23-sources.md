# Источники банковской выборки #23

Сохранены три исходных банковских видео. Отдельные планы и будущие клипы сохраняют группу исходного фильма и не считаются независимыми источниками. Все материалы предназначены только для `external_holdout`; обучение на них в этой задаче не выполняется. Пока нет проверенных человеком событий и gold-меток.

| Стабильная страница источника | Автор / публикатор | Права по первичным сведениям | Группа |
| --- | --- | --- | --- |
| [CareerOneStop — Tellers](https://cloudfront.careeronestop.org/Toolkit/Careers/Occupations/occupation-profile.aspx?keyword=Tellers&lang=en&location=07457&onetcode=43307100) | CareerOneStop, проект U.S. Department of Labor | Public Domain по [официальному FAQ](https://cloudfront.careeronestop.org/Videos/FAQs-about-videos.aspx); сохранено текстовое подтверждение web retrieval | `careeronestop-tellers` |
| [NARA — VICTIM? OR WITNESS, NAID 12089](https://catalog.archives.gov/id/12089) | National Archives; банковский учебный материал, связанный с FBI по карточке | Access: Unrestricted; use restriction: Undetermined. `LicenseRef-NARA-Undetermined`, `private_evaluation_only`; статус Public Domain не заявляется | `nara12089` |
| [Production Attic — RBS TCR training](https://productionattic.com/portfolio/rbstcr) | Production Attic; банковский заказчик RBS | Права не подтверждены: `LicenseRef-RBS-RightsUnconfirmed`, только приватная оценка. Карточка подтверждает съёмку учебного материала в выведенном из эксплуатации банковском отделении | `rbs-tcr-training` |

Исходные SHA256, технические параметры, подтверждения происхождения / прав и временные границы хранятся в приватном `data/external23/`. Временные signed progressive URL RBS находятся только там. В публичный Git не включаются видео, кадры, модельные ответы, интервалы событий и тексты частной разметки.

Девять ранее импортированных Commons-кандидатов сохранены в резерве с причиной `bank_context_not_confirmed`; в банковский набор не включаются. Retail MEVA также исключён по уточнению пользователя. Источники Internet Archive *Using the Bank* и *Banks And Credits* пока недоступны из локальной сети после ограниченных попыток TLS; их файлы не получены и в набор не засчитываются.

Алгоритмические границы планов используются только для локального screening. Они не подтверждают содержательную пригодность или наличие индикатора. Типы Т/Д, видимые и неподтверждённые условия могут сохраняться только как `local_model / pending_review`, с `legal_gold: null`; окончательное число пригодных клипов не доводится до 10–15 искусственно. Порядок работы и ограничения — в [инструкции](external23-intake.md).
