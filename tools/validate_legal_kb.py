#!/usr/bin/env python3
"""Офлайн-проверка целостности базы, без оценки юридической применимости."""

import argparse
import hashlib
import json
import re
import sys
import string
from types import SimpleNamespace
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


OFFICIAL_HOSTS = {
    "adilet.zan.kz", "old.adilet.zan.kz", "nationalbank.kz", "www.nationalbank.kz",
}
ARTICLE_MARKER = "<!-- BEGIN OFFICIAL ARTICLE -->"
BLOCK_TAGS = {"p", "br", "table", "tr", "td", "th", "h1", "h2", "h3", "h4"}
EXCLUDED_IDS = {"Д05", "Д07", "Д12", "Д14", "Д15"}
ALL_TYPE_IDS = {f"Т{i:02}" for i in range(1, 18)} | {f"Д{i:02}" for i in range(1, 16)}
ACTIVE_IDS = ALL_TYPE_IDS - EXCLUDED_IDS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(text):
    """Меняем только whitespace, включая NBSP; буквы и пунктуацию не меняем."""
    return " ".join(text.replace("\u00a0", " ").split())


class SourceText(HTMLParser):
    """Извлекаем текст сохранённой разметки таблиц; это не проверка live DOM."""

    def __init__(self, article_only=True):
        super().__init__(convert_charrefs=True)
        self.article_only = article_only
        self.inside_article = False
        self.article_count = 0
        self.parts = []
        self.anchors = set()

    @property
    def active(self):
        return self.inside_article or not self.article_only

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.inside_article = True
            self.article_count += 1
        if self.active:
            attrs = dict(attrs)
            if attrs.get("id"):
                self.anchors.add(attrs["id"])
            if tag == "a" and attrs.get("name"):
                self.anchors.add(attrs["name"])
            if tag in BLOCK_TAGS:
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.active and tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "article":
            self.inside_article = False

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    @property
    def text(self):
        return normalize("".join(self.parts))


def parse_html(text, article_only=True):
    parser = SourceText(article_only)
    parser.feed(text)
    parser.close()
    if article_only:
        require(parser.article_count == 1, "Ожидается ровно один article в source.html")
    return parser


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Повторяющийся ключ JSON: {key}")
        result[key] = value
    return result


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except ValueError as error:
        raise ValueError(f"{path}: некорректный JSON: {error}") from error


def load_yaml(path):
    """PyYAML SafeLoader, с отказом от повторных ключей (не собственный парсер)."""
    try:
        import yaml
    except ImportError as error:
        raise ValueError("Для каталога нужен PyYAML 6.0.3; установите tools/requirements-legal.txt в отдельное окружение") from error

    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            require(isinstance(key, str), f"YAML: ключ должен быть строкой, строка {key_node.start_mark.line + 1}")
            require(key not in result, f"YAML: повторяющийся ключ {key!r}, строка {key_node.start_mark.line + 1}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: некорректный или небезопасный YAML: {error}") from error


def string_field(entry, name, label):
    value = entry.get(name)
    require(isinstance(value, str) and value.strip(), f"{label}: нужно непустое поле {name}")
    return value


def string_list(entry, name, label):
    value = entry.get(name)
    require(isinstance(value, list) and value, f"{label}: нужен непустой список {name}")
    require(all(isinstance(item, str) and item.strip() for item in value),
            f"{label}: элементы {name} должны быть непустыми строками")
    return value


def official_url(value, label):
    parsed = urlsplit(value)
    require(parsed.scheme == "https" and parsed.hostname in OFFICIAL_HOSTS
            and not parsed.username and not parsed.password and parsed.port in (None, 443),
            f"{label}: URL должен использовать HTTPS и разрешённый официальный домен")
    return parsed


def local_file(base, relative, label):
    path = (base / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(base),
            f"{label}: локальный путь должен находиться внутри {base}")
    require(path.is_file(), f"{label}: файл не найден: {path}")
    return path


def check_anchors(anchors, parsed, label):
    for anchor in anchors:
        require(anchor in parsed.anchors, f"{label}: якорь #{anchor} отсутствует в историческом индексе snapshot_anchor_ids")


def check_reference(entry, indexed, label):
    sid = string_field(entry, "source_id", label)
    require(sid in indexed, f"{label}: неизвестный source_id {sid}")
    source, parsed, body = indexed[sid]
    string_field(entry, "clause", label)
    url = string_field(entry, "url", label)
    parsed_url = official_url(url, label)
    require(url.split("#", 1)[0] == source["canonical_url"] and parsed_url.fragment,
            f"{label}: URL должен ссылаться на canonical_url источника с якорем")
    check_anchors([parsed_url.fragment], parsed, label)
    quote = normalize(string_field(entry, "quote", label))
    require(quote in parsed.text,
            f"{label}: цитата не найдена в текстовом snapshot (нормализация только whitespace/NBSP)")
    require(quote in body, f"{label}: цитата не найдена в локальном тексте {source['text_file']}")
    introductions = {
        ("NBRK-231", "z444"): "19. Кассовый работник банка осуществляет следующие действия:",
        ("NBRK-231", "z445"): "19. Кассовый работник банка осуществляет следующие действия:",
        ("NBRK-231", "z454"): "31. Кассовый работник банка осуществляет следующие действия:",
        ("NBRK-14", "z121"): "16. Устройство помещения кассы (касс) соответствуют следующим параметрам:",
    }
    intro = introductions.get((sid, parsed_url.fragment))
    if intro:
        require(quote.startswith(intro), f"{label}: нужна полная вводная фраза нормы")
    return parsed_url.fragment


def load_type_ids(path):
    data = load_json(path)
    require(isinstance(data, dict) and data.get('schema_version') == 1
            and isinstance(data.get('types'), list), 'type-ids.json: неверная структура')
    result, runtime_ids = {}, set()
    for entry in data['types']:
        require(isinstance(entry, dict) and set(entry) == {'codex_code', 'id', 'name', 'catalog_section'},
                'type-ids.json: неверные поля')
        code = string_field(entry, 'codex_code', 'type-ids.json')
        runtime_id = string_field(entry, 'id', code)
        require(code in ALL_TYPE_IDS and code not in result, f'{code}: неверный / повторный исходный код')
        require(re.fullmatch(r'[a-z][a-z0-9_]*', runtime_id) and runtime_id not in runtime_ids,
                f'{code}: runtime ID должен быть уникальным snake_case')
        string_field(entry, 'name', code)
        require(entry['catalog_section'] == ('excluded_types' if code in EXCLUDED_IDS else 'violations'),
                f'{code}: неверный раздел каталога')
        result[code] = entry
        runtime_ids.add(runtime_id)
    require(set(result) == ALL_TYPE_IDS, 'type-ids.json: нужны все 32 исходных кода')
    return result


def validate_mapping(base, indexed):
    type_ids = load_type_ids(base.parent / 'type-ids.json')
    mapping = load_json(base / "type-mapping.json")
    require(isinstance(mapping, dict) and set(mapping) == {"metadata", "types"},
            "type-mapping.json: ожидаются metadata и types")
    metadata = mapping["metadata"]
    require(isinstance(metadata, dict), "mapping metadata: ожидается объект")
    for field in ("checked_on", "provenance", "documents_as_of", "portal_updated_on", "scope"):
        string_field(metadata, field, "mapping metadata")
    require(metadata.get("visual_validation") == "not_performed"
            and metadata.get("owner_acceptance") == "indicator_policy_accepted",
            "mapping metadata: принята только индикаторная политика; визуальная проверка не выполнена")
    policy = metadata.get("teamlead_policy")
    require(isinstance(policy, dict) and policy.get("source") == "teamlead_decision"
            and policy.get("active_indicators") == 27 and policy.get("excluded_types") == 5
            and policy.get("legal_assessment_counts_are_not_exclusion_policy") is True,
            "mapping metadata: индикаторная политика должна быть отделена от правовых оценок")
    require(metadata.get("supported_detectors") == 0, "mapping metadata: поддержанные детекторы ещё не подтверждены")
    temporal = metadata.get("temporal_refs")
    require(isinstance(temporal, list) and temporal, "mapping metadata: нужны temporal_refs")
    for number, reference in enumerate(temporal, start=1):
        require(isinstance(reference, dict) and set(reference) == {"source_id", "clause", "url", "quote"},
                f"temporal_refs[{number}]: неверная структура")
        check_reference(reference, indexed, f"temporal_refs[{number}]")
    entries = mapping["types"]
    require(isinstance(entries, list), "mapping types: ожидается список")
    expected = ALL_TYPE_IDS
    fields = {"id", "title", "group", "legal_assessment", "refs", "required_context",
              "observable_checks", "limitations", "pipeline_status", "teamlead_policy"}
    assessments = {"conditional", "no_direct_basis_as_named", "requires_nonvisual_evidence"}
    ids, counts = set(), Counter()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == fields,
                "mapping: неверные поля; допустимы только поля общего типа, без per-video данных")
        tid = string_field(entry, "id", "mapping")
        require(tid in expected and tid not in ids, f"mapping {tid}: неизвестный или повторяющийся ID")
        ids.add(tid)
        string_field(entry, "title", tid)
        require(entry["group"] == ("primary" if tid.startswith("Т") else "additional"), f"{tid}: неверная group")
        require(isinstance(entry["legal_assessment"], str) and entry["legal_assessment"] in assessments,
                f"{tid}: неизвестная legal_assessment")
        counts[entry["legal_assessment"]] += 1
        require(entry["pipeline_status"] == "not_integrated", f"{tid}: ожидается pipeline_status=not_integrated")
        policy = entry["teamlead_policy"]
        require(isinstance(policy, dict) and policy.get("source") == "teamlead_decision"
                and policy.get("status") == ("excluded" if tid in EXCLUDED_IDS else "approved_indicator")
                and policy.get("catalog_section") == ("excluded_types" if tid in EXCLUDED_IDS else "violations")
                and policy.get("catalog") == "../violations.yaml" and policy.get("catalog_id") == type_ids[tid]['id'],
                f"{tid}: неверная связь аналитического слоя с индикаторной политикой")
        for field in ("required_context", "observable_checks", "limitations"):
            string_list(entry, field, tid)
        require(isinstance(entry["refs"], list) and entry["refs"], f"{tid}: нужен непустой список refs")
        for number, reference in enumerate(entry["refs"], start=1):
            require(isinstance(reference, dict) and set(reference) == {"source_id", "clause", "url", "quote"},
                    f"{tid} refs[{number}]: неверная структура")
            check_reference(reference, indexed, f"{tid} refs[{number}]")
        if tid == "Т13":
            require(all(ref["source_id"] == "NBRK-14" and ref["url"].endswith("#z49") for ref in entry["refs"]),
                    "Т13 mapping: для обычной банковской кассы основание только №14 п.8")
    require(ids == expected, f"mapping: ожидаются ровно 32 ID Т01–Т17 / Д01–Д15; отсутствуют {sorted(expected - ids)}")
    require(metadata.get("assessment_counts") == dict(counts), "mapping: assessment_counts не совпадает с записями")
    return mapping


def rule_key(rule):
    return rule["source_id"], rule["url"]


def validate_catalog(path, indexed, mapping):
    catalog = load_yaml(path)
    type_ids = load_type_ids(path.parent / 'type-ids.json')
    require(isinstance(catalog, dict) and set(catalog) == {"version", "metadata", "violations", "excluded_types"}
            and catalog["version"] == 1, "Каталог: ожидается version=1 и metadata/violations/excluded_types")
    metadata = catalog["metadata"]
    require(isinstance(metadata, dict), "Каталог: metadata должен быть объектом")
    require(metadata.get("owner_acceptance") == "indicator_policy_accepted"
            and metadata.get("policy_source") == "teamlead_decision"
            and metadata.get("legal_assessment_counts_are_not_exclusion_policy") is True,
            "Каталог: принятие политики не должно подменять правовой анализ")
    require(metadata.get("implemented_detectors") == 0 and metadata.get("visual_validation") == "not_performed",
            "Каталог: реализованные детекторы 0, визуальная проверка not_performed")
    require(metadata.get("verdict_contract") == ["no_violation", "violation"]
            and metadata.get("missing_rule_changes_verdict") is False,
            "Каталог: только два verdict; отсутствие rule не меняет verdict автоматически")
    require(metadata.get('id_mapping') == 'type-ids.json'
            and metadata.get('legacy_status_scope') == 'citation_verification_only',
            'Каталог: status относится к цитате, исходные коды связаны со стабильными runtime ID')
    active, excluded = catalog["violations"], catalog["excluded_types"]
    require(isinstance(active, list) and len(active) == 27, "Каталог: требуется ровно 27 активных записей")
    require(isinstance(excluded, list) and len(excluded) == 5, "Каталог: требуется ровно 5 исключённых записей")
    analytical = {row["id"]: row for row in mapping["types"]}
    required = {"id", "codex_code", "name", "analysis_name", "criterion", "applicability", "description_template", "status",
                "implementation_status", "normative_basis", "limitations", "legal_analysis_id", "policy_status",
                "citation_status", "enabled", "summary", "visible_signs", "legal_caveat", "conclusion"}
    optional = {"rule", "related_rules", "indicator_label", "criterion_branches", "team_heuristics", "note", "context_policy", "template_selection", "examples"}
    ids, with_rules = set(), 0
    forbidden = re.compile(r"доказан|хищен|недостач|поддел|разглаш|посторонн|неуполномоч|проводк|сверк|"
                           r"разблокирован|содержим(?:ое|ого) экрана|содержани[ея] экрана|рабочее содержимое|"
                           r"не заперт\w* на ключ|подлинность не проверена|отсутствие аутентификации|"
                           r"без аутентификации|недостаточно данных", re.I)
    for entry in active:
        require(isinstance(entry, dict) and required <= set(entry) <= required | optional,
                "Каталог: неверные поля активного типа")
        tid = string_field(entry, "codex_code", "Каталог")
        require(tid in ACTIVE_IDS and tid not in ids, f"{tid}: неизвестный, исключённый или повторяющийся активный ID")
        ids.add(tid)
        require(entry['id'] == type_ids[tid]['id'] and entry['name'] == type_ids[tid]['name'],
                f'{tid}: изменён runtime ID / название из таблицы соответствия')
        require(entry["analysis_name"] == analytical[tid]["title"] and entry["legal_analysis_id"] == tid,
                f"{tid}: исходное имя / ссылка на аналитический слой изменены")
        for field in ("criterion", "description_template", "summary", "legal_caveat", "conclusion"):
            string_field(entry, field, tid)
        for field in ("applicability", "limitations", "visible_signs"):
            string_list(entry, field, tid)
        require(entry['enabled'] is True and entry['policy_status'] == 'approved_indicator',
                f'{tid}: согласованный индикатор должен оставаться включённым')
        require(entry["status"] == ("draft" if tid == 'Т04' else 'verified')
                and entry['citation_status'] == ('pending' if tid == 'Т04' else 'citation_verified')
                and entry["implementation_status"] == "not_implemented",
                f"{tid}: запись каталога не должна утверждать реализацию детектора")
        template = entry["description_template"]
        require("это может указывать" in template.casefold(), f"{tid}: нужен условный description_template")
        require(not forbidden.search(template), f"{tid}: категоричное / невидимое утверждение в description_template")
        try:
            placeholders = list(string.Formatter().parse(template))
        except ValueError as error:
            raise ValueError(f"{tid}: некорректный description_template: {error}") from error
        names = set()
        allowed = {"t", "observation", "rule_document", "rule_clause", "rule_number_date"}
        for literal, name, spec, conversion in placeholders:
            if name is not None:
                require(name in allowed and not spec and not conversion,
                        f"{tid}: недопустимый placeholder / формат в description_template")
                names.add(name)
        require({"t", "observation"} <= names, f"{tid}: description_template требует {{t}} и {{observation}}")
        suffix = "Нормативное основание уточняется." if tid == "Т04" else "Основание: {rule_document}, {rule_clause}."
        require(template.endswith(suffix), f"{tid}: неверное окончание description_template")
        if tid == "Т04":
            require(not names & {"rule_document", "rule_clause", "rule_number_date"},
                    "Т04: description_template не может ссылаться на отсутствующее rule")
            require("rule" not in entry and "related_rules" not in entry and entry["normative_basis"] == "pending",
                    "Т04: rule и related_rules должны отсутствовать; нормативное основание pending")
            require(entry.get("note") == "нормативное основание уточняется" and entry["note"] in template.casefold(),
                    "Т04: нужна фраза «нормативное основание уточняется»")
        else:
            require("rule" in entry and entry["normative_basis"] == "contextual_reference", f"{tid}: нужен rule с контекстом")
            with_rules += 1
            related = entry.get("related_rules", [])
            require(isinstance(related, list), f"{tid}: related_rules должен быть списком")
            known_refs = {(r["source_id"], r["url"], normalize(r["quote"])) for r in analytical[tid]["refs"]}
            for number, reference in enumerate([entry["rule"], *related]):
                fields = {"document", "number_date", "source_id", "clause", "url", "quote", "verified_by", "verified_on",
                          "citation_status", "verification_scope"} | ({"usage"} if number else set())
                require(isinstance(reference, dict) and fields <= set(reference) <= fields | {"clause_note"},
                        f"{tid}: неверные поля rule/related_rules; обязательно number_date")
                source = indexed.get(reference.get("source_id"), ({},))[0]
                require(string_field(reference, "document", tid) == source.get("title")
                        and string_field(reference, "number_date", tid) == source.get("number_date"),
                        f"{tid}: нужны официальное название документа и number_date из manifest")
                require(re.fullmatch(r"(?:п\.[0-9–-]+(?: подп\.[0-9–-]+)?|ст\.[0-9–-]+(?: п\.[0-9–-]+)?)", reference.get("clause", "")),
                        f"{tid}: clause содержит только номер; оговорки перенесите в clause_note")
                if 'clause_note' in reference:
                    string_field(reference, 'clause_note', tid)
                require(reference['verified_by'] == 'Codex (агент Олжаса), сверка официального текста'
                        and reference['verified_on'] == '2026-10-17'
                        and reference['citation_status'] == 'citation_verified'
                        and reference['verification_scope'] == 'citation_and_source_integrity',
                        f'{tid}: verified_by должен честно обозначать сверку источника агентом Codex')
                if number:
                    require(reference["usage"] in ("supporting", "exception", "reference_only"), f"{tid}: неверный usage")
                check_reference(reference, indexed, f"{tid} catalog rule[{number}]")
                require((reference["source_id"], reference["url"], normalize(reference["quote"])) in known_refs,
                        f"{tid}: цитата не совпадает с проверенной привязкой; условия и исключения нельзя обрезать")
            expected_refs = {rule_key(r) for r in analytical[tid]["refs"]}
            overrides = {"Т05": {("NBRK-231", "z445"), ("NBRK-231", "z123")}, "Т08": {("NBRK-231", "z111"), ("NBRK-231", "z113")},
                         "Т09": {("NBRK-231", "z111")}, "Т10": {("NBRK-231", "z226")},
                         "Т11": {("NBRK-231", "z226")}, "Т16": {("NBRK-231", "z445"), ("NBRK-231", "z162")}}
            if tid in overrides:
                expected_refs = {(sid, indexed[sid][0]["canonical_url"] + "#" + anchor) for sid, anchor in overrides[tid]}
            actual_refs = [rule_key(r) for r in [entry["rule"], *related]]
            require(len(actual_refs) == len(set(actual_refs)) and set(actual_refs) == expected_refs,
                    f"{tid}: состав нормативных ссылок / исключений не соответствует согласованному каталогу")

        heuristic = entry.get("team_heuristics")
        if heuristic is not None:
            require(tid in {"Т06", "Т09", "Т11", "Т16", "Д13"} and isinstance(heuristic, dict)
                    and heuristic.get("source") == "teamlead_heuristic" and heuristic.get("is_normative") is False,
                    f"{tid}: порог / уверенность должны быть отдельной командной эвристикой")
        if tid in {"Т06", "Т09"}:
            timed, untimed = ("open_while_working", "leave_without_visible_locking") if tid == "Т06" else ("open_while_unused", "open_on_departure")
            branches = entry.get("criterion_branches")
            require(isinstance(branches, list) and len(branches) == 2 and all(isinstance(b, dict) for b in branches),
                    f"{tid}: нужны две отдельные ветви критерия")
            by_branch = {b.get("id"): b for b in branches}
            untimed_fields = {"id", "observation"} | ({"template_values"} if tid == "Т06" else set())
            require(set(by_branch) == {timed, untimed} and by_branch[timed].get("threshold_ref") == timed
                    and set(by_branch[untimed]) == untimed_fields,
                    f"{tid}: порог нельзя переносить на ветвь ухода")
            require(isinstance(heuristic, dict) and heuristic.get("thresholds") == [
                {"branch_id": timed, "field": "visible_duration_seconds", "operator": ">=", "value": 5, "unit": "seconds"}],
                f"{tid}: командный порог >=5 секунд нужен только для ветви {timed}")
        if tid in {"Т11", "Т16", "Д13"}:
            require(isinstance(heuristic, dict) and heuristic.get("confidence") == ("medium" if tid == "Т11" else "low")
                    and heuristic.get("confidence_kind") == "qualitative_team_rating"
                    and heuristic.get("measured_probability") is False,
                    f"{tid}: confidence — качественная оценка команды, не измеренная вероятность")
        primary = {"Т01": ("NBRK-231", "z204"), "Т05": ("NBRK-231", "z445"), "Т06": ("NBRK-14", "z49"), "Т08": ("NBRK-231", "z111"), "Т09": ("NBRK-231", "z111"),
                   "Т10": ("NBRK-231", "z226"), "Т11": ("NBRK-231", "z226"), "Т13": ("NBRK-14", "z49"),
                   "Т16": ("NBRK-231", "z445"), "Т17": ("NBRK-231", "z81")}
        if tid in primary:
            sid, anchor = primary[tid]
            require(rule_key(entry["rule"]) == (sid, indexed[sid][0]["canonical_url"] + "#" + anchor),
                    f"{tid}: неверное основное основание индикатора")
        if tid == "Т06":
            require(any(r["source_id"] == "NBRK-14" and r["url"].endswith("#z121") for r in entry.get("related_rules", [])),
                    "Т06: нужен дополнительный временный п.16 подп.1 для двери отдельной кассы")
            require(entry.get("template_selection") == "one_matched_branch"
                    and by_branch["open_while_working"].get("template_values") == {
                        "observation": "Дверь кассового помещения остаётся открытой"}
                    and by_branch["leave_without_visible_locking"].get("template_values") == {
                        "observation": "Запирание двери кассового помещения после ухода кассира не видно"},
                    "Т06: шаблон должен выбирать только наблюдение одной сработавшей ветви")
        if tid == "Т13":
            require(not entry.get("related_rules") and entry.get("context_policy") == {
                "value": "ordinary_bank_cashdesk", "source": "teamlead_decision", "verified_by_our_video_analysis": False},
                "Т13: обычная банковская касса — решение тимлида; №49/п.75 не являются основаниями")
        if tid == "Т16":
            related = entry.get("related_rules", [])
            require(len(related) == 1 and related[0]["source_id"] == "NBRK-231"
                    and related[0]["url"].endswith("#z162") and related[0]["usage"] == "reference_only",
                    "Т16: п.41 допускается только как reference_only")
    require(ids == ACTIVE_IDS and with_rules == 26, "Каталог: ожидаются 27 согласованных ID, из них 26 с rule")
    ids = set()
    for entry in excluded:
        require(isinstance(entry, dict) and set(entry) == {"id", "codex_code", "name", "status", "enabled", "implementation_status", "reason",
                                                         "required_nonvisual_evidence", "legal_analysis_id"},
                "Каталог: неверные поля excluded_types")
        tid = string_field(entry, "codex_code", "excluded_types")
        require(tid in EXCLUDED_IDS and tid not in ids, f"{tid}: неверный / повторный исключённый ID")
        ids.add(tid)
        require(entry['id'] == type_ids[tid]['id'] and entry['enabled'] is False,
                f'{tid}: исключённый тип не должен включаться в runtime')
        require(entry["name"] == analytical[tid]["title"] and entry["legal_analysis_id"] == tid,
                f"{tid}: изменено имя исключённого типа")
        require(entry["status"] == "excluded_requires_nonvisual_evidence" and entry["implementation_status"] == "not_implemented",
                f"{tid}: неверный статус исключённого типа")
        string_field(entry, "reason", tid)
        string_list(entry, "required_nonvisual_evidence", tid)
    require(ids == EXCLUDED_IDS, "Каталог: состав excluded_types должен быть Д05/Д07/Д12/Д14/Д15")
    return len(active), len(excluded)


def validate(base):
    base = base.resolve()
    manifest = load_json(base / "manifest.json")
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
            "manifest.json: ожидается объект с schema_version=1")
    sources = manifest.get("sources")
    require(isinstance(sources, list) and sources, "manifest.json: нужен непустой список sources")
    indexed = {}
    for source in sources:
        require(isinstance(source, dict), "manifest.json: источник должен быть объектом")
        sid = string_field(source, "id", "источник")
        require(sid not in indexed, f"Повторяющийся source_id: {sid}")
        for key in ("title", "act_date", "checked_on", "status", "status_basis", "revision"):
            string_field(source, key, sid)
        canonical = official_url(string_field(source, "canonical_url", sid), sid)
        require(not canonical.fragment, f"{sid}: canonical_url не должен содержать якорь")
        official_url(string_field(source, "download_url", sid), sid)
        text_path = local_file(base, string_field(source, "text_file", sid), sid)
        historical_hash = string_field(source, "retrieved_html_sha256", sid)
        require(re.fullmatch(r"[0-9a-f]{64}", historical_hash), f"{sid}: некорректный исторический SHA256 HTML")
        expected = string_field(source, "text_sha256", sid)
        require(re.fullmatch(r"[0-9a-f]{64}", expected), f"{sid}: некорректный text_sha256")
        raw = text_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f"{sid}: text_sha256 не совпадает")
        require(source.get("snapshot_format") == "official_article_text_with_preserved_tables"
                and source.get("anchor_verification") == "historical_html_extraction_not_live_dom_validation",
                f"{sid}: источник должен честно описывать текстовый snapshot и исторический индекс якорей")
        if "retrieved_metadata_sha256" in source:
            require(re.fullmatch(r"[0-9a-f]{64}", string_field(source, "retrieved_metadata_sha256", sid)),
                    f"{sid}: некорректный исторический SHA256 инфокарточки")
            official_url(string_field(source, "metadata_url", sid), sid)
        for field in ("documents_as_of", "portal_updated_on"):
            require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", string_field(source, field, sid)),
                    f"{sid}: неверная историческая дата {field}")
        text = raw.decode("utf-8")
        require(text.count(ARTICLE_MARKER) == 1, f"{sid}: отсутствует или повторён маркер полного текста")
        body = parse_html(text.split(ARTICLE_MARKER, 1)[1], article_only=False).text
        anchors = string_list(source, "snapshot_anchor_ids", sid)
        require(len(anchors) == len(set(anchors)), f"{sid}: повторён ID исторического якоря")
        parsed = SimpleNamespace(text=body, anchors=set(anchors))
        check_anchors(string_list(source, "reference_anchors", sid), parsed, sid)
        indexed[sid] = (source, parsed, body)

    candidates = load_json(base / "candidates.json")
    require(isinstance(candidates, list), "candidates.json: ожидается список")
    ids = set()
    for entry in candidates:
        require(isinstance(entry, dict), "Кандидат должен быть объектом")
        cid = string_field(entry, "id", "кандидат")
        require(cid not in ids, f"Повторяющийся id кандидата: {cid}")
        ids.add(cid)
        sid = string_field(entry, "source_id", cid)
        require(sid in indexed, f"{cid}: неизвестный source_id {sid}")
        source, parsed, body = indexed[sid]
        for key in ("clause", "subject"):
            string_field(entry, key, cid)
        for key in ("conditions", "observable_facts", "cannot_infer"):
            string_list(entry, key, cid)
        require(entry.get("status") == "candidate", f"{cid}: status должен быть candidate")
        require(entry.get("mapping_status") == "draft_received_pending_review",
                f"{cid}: mapping_status должен быть draft_received_pending_review")
        require("type_id" not in entry, f"{cid}: type_id нельзя назначать до получения таксономии")
        url = string_field(entry, "url", cid)
        parsed_url = official_url(url, cid)
        require(url.split("#", 1)[0] == source["canonical_url"] and parsed_url.fragment,
                f"{cid}: URL должен ссылаться на canonical_url источника с якорем")
        anchors = string_list(entry, "anchors", cid)
        require(parsed_url.fragment == anchors[0], f"{cid}: URL должен указывать на первый якорь anchors")
        check_anchors(anchors, parsed, cid)
        quote = normalize(string_field(entry, "quote", cid))
        require(quote in parsed.text,
                f"{cid}: цитата не найдена в текстовом snapshot (нормализация только whitespace/NBSP)")
        require(quote in body, f"{cid}: цитата не найдена в локальном тексте {source['text_file']}")
    mapping = validate_mapping(base, indexed)
    active, excluded = validate_catalog(base.parent / "violations.yaml", indexed, mapping)
    return len(sources), len(candidates), len(mapping["types"]), active, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "kb" / "legal",
                        help="Каталог legal/; violations.yaml должен находиться рядом с legal/ (по умолчанию kb/legal)")
    args = parser.parse_args()
    try:
        sources, candidates, mappings, active, excluded = validate(args.kb_dir.resolve())
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        return 1
    print(f"OK: источников {sources}, кандидатов {candidates}, аналитических типов {mappings}; индикаторов {active}, исключённых {excluded}. YAML, цитаты, исторический индекс якорей, SHA256 текста, условия каталога и статусы проверены; live DOM/актуальность не проверялись.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
