#!/usr/bin/env python3
"""Validate a clean isolated KB and meaningful invalid catalog variants. No data/ reads."""

import argparse
import copy
import importlib.util
import logging
import sys
from pathlib import Path
import shutil
import tempfile

import yaml

from validate_legal_kb import load_yaml, validate


def check_ml_loader(loader_path, catalog):
    """Run the pinned real reader on synthetic observations, never media/examples."""
    spec = importlib.util.spec_from_file_location("datathon26_pinned_ml_kb", loader_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    handler = Capture()
    module.log.addHandler(handler)
    try:
        parsed = module.parse_catalog(catalog)
    finally:
        module.log.removeHandler(handler)
    assert len(parsed) == 27 and sum(x.rule is not None for x in parsed.values()) == 26
    raw_by_id = {row['id']: row for row in catalog['violations']}
    draft = next(row for row in catalog['violations'] if row['codex_code'] == 'Т04')
    assert parsed[draft['id']].rule is None and parsed[draft['id']].enabled
    assert not records, "real loader emitted a warning / fallback"
    assert not ({row['id'] for row in catalog['excluded_types']} & set(parsed))
    for key, item in parsed.items():
        template, fallback = module._parse_template(key, raw_by_id[key]['description_template'], item.rule)
        assert not fallback and template == item.template
        values = {'t': '12.5', 'observation': 'Синтетическое наблюдение'}
        if item.rule:
            values.update(rule_document=item.rule.document, rule_clause=item.rule.clause,
                          rule_number_date=item.rule.number_date)
        rendered = item.template.format_map(values)
        assert 'Синтетическое наблюдение' in rendered and '12.5' in rendered
        assert '{' not in rendered and '}' not in rendered
    print("OK: pinned ML loader: 27 types / 26 rules / T04 enabled without rule / 0 fallbacks / 27 renders / 5 excluded inactive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ml-loader', type=Path, help='Optional path to the real pinned ml/kb.py, read-only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="datathon26-catalog-check-") as temp:
        base = Path(temp) / "kb"
        shutil.copytree(root / "kb/legal", base / 'legal')
        for name in ['violations.yaml', 'type-ids.json']:
            shutil.copy2(root / 'kb' / name, base / name)
        path = base / "violations.yaml"
        raw = path.read_text(encoding="utf-8")
        clean = load_yaml(path)
        assert validate(base / "legal") == (8, 14, 32, 27, 5)
        assert not (Path(temp) / "data").exists()
        assert not list(base.rglob('*.html')), 'isolated text KB must work without HTML mirrors'
        door = next(row for row in clean["violations"] if row["codex_code"] == "Т06")
        rendered = [door["description_template"].format_map({**branch["template_values"], 't': '12.5',
                     'rule_document': door['rule']['document'], 'rule_clause': door['rule']['clause']})
                    for branch in door["criterion_branches"]]
        assert "остаётся открытой" in rendered[0] and "после ухода" not in rendered[0]
        assert "после ухода" in rendered[1] and "остаётся открытой" not in rendered[1]
        print("OK: isolated kb/, 8 sources / 14 candidates / 32 analytical / 27 active / 5 excluded")
        if args.ml_loader:
            check_ml_loader(args.ml_loader.resolve(), clean)
        cases = [
            ("altered_quote", "цитата не найдена"),
            ("excluded_type_enabled", "исключённый"),
            ("fake_t04_rule", "Т04: rule"),
            ("wrong_t13_rule", "Т13"),
            ("threshold_on_departure", "порог нельзя переносить"),
            ("combined_door_observations", "одной сработавшей ветви"),
            ("normative_threshold", "командной эвристикой"),
            ("missing_packaging_exception", "исключений"),
            ("categorical_template", "description_template"),
            ("duplicate_yaml_key", "повторяющийся ключ"),
            ("unsafe_yaml_tag", "небезопасный YAML"),
            ("disabled_t04", "оставаться включённым"),
            ("renamed_runtime_id", "runtime ID"),
            ("unverified_rule", "verified_by"),
            ("unknown_placeholder", "placeholder"),
            ("bad_template_suffix", "окончание description_template"),
            ("missing_number_date", "number_date"),
            ("editorial_clause", "clause содержит только номер"),
            ("missing_intro", "вводная фраза"),
            ("changed_source_text", "text_sha256"),
        ]
        for case, message in cases:
            catalog = copy.deepcopy(clean)
            entries = {row["codex_code"]: row for row in catalog["violations"]}
            if case == "altered_quote":
                entries["Т08"]["rule"]["quote"] += " ВСТАВЛЕННАЯ НЕСУЩЕСТВУЮЩАЯ ФРАЗА."
            elif case == "excluded_type_enabled":
                # Keep 27 rows, so the check exercises policy rather than count alone.
                entries["Д13"]["codex_code"] = "Д05"
                entries["Д13"]["name"] = next(row["name"] for row in catalog["excluded_types"] if row["codex_code"] == "Д05")
            elif case == "fake_t04_rule":
                # A real source/quote is still invalid as the missing immediate-lock rule.
                entries["Т04"]["rule"] = copy.deepcopy(entries["Т06"]["rule"])
            elif case == "wrong_t13_rule":
                # Genuine vault text must not become the ordinary cashdesk's basis.
                entries["Т13"]["rule"] = copy.deepcopy(entries["Д11"]["rule"])
            elif case == "threshold_on_departure":
                entries["Т06"]["criterion_branches"][1]["threshold_ref"] = "open_while_working"
            elif case == "combined_door_observations":
                entries["Т06"]["criterion_branches"][0]["template_values"]["observation"] = "Дверь открыта / запирание после ухода не видно"
            elif case == "normative_threshold":
                entries["Т09"]["team_heuristics"]["source"] = "legal_requirement"
            elif case == "missing_packaging_exception":
                entries["Д02"].pop("related_rules")
            elif case == "categorical_template":
                entries["Т03"]["description_template"] += " Доказано хищение."
            elif case == 'disabled_t04':
                entries['Т04']['enabled'] = False
            elif case == 'renamed_runtime_id':
                entries['Т01']['id'] = 'renamed_id'
            elif case == 'unverified_rule':
                entries['Т01']['rule']['verified_by'] = 'Олжас'
            elif case == 'unknown_placeholder':
                entries['Т03']['description_template'] = entries['Т03']['description_template'].replace('{observation}', '{screen_content}')
            elif case == 'bad_template_suffix':
                entries['Т03']['description_template'] += ' Дополнение.'
            elif case == 'missing_number_date':
                entries['Т03']['rule'].pop('number_date')
            elif case == 'editorial_clause':
                entries['Т03']['rule']['clause'] += ', оба абзаца'
            elif case == 'missing_intro':
                entries['Т02']['rule']['quote'] = '1)' + entries['Т02']['rule']['quote'].split('1)', 1)[1]
            source_path = base / 'legal/NBRK-231/2026-10-17/text.md'
            source_raw = source_path.read_bytes()
            if case == 'changed_source_text':
                source_path.write_bytes(source_raw + b'\nCHANGED\n')
            if case == "duplicate_yaml_key":
                text = raw + "\nversion: 1\n"
            elif case == "unsafe_yaml_tag":
                text = "!!python/object/apply:builtins.len [test]\n"
            else:
                text = yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False)
            path.write_text(text, encoding="utf-8")
            try:
                validate(base / "legal")
            except ValueError as error:
                assert message in str(error), (case, str(error))
            else:
                raise AssertionError(f"invalid catalog accepted: {case}")
            source_path.write_bytes(source_raw)
            print(f"REJECTED: {case}")
        path.write_text(raw, encoding="utf-8")
        assert validate(base / "legal") == (8, 14, 32, 27, 5)
    print(f"OK: {len(cases)} negative checks; tracked text snapshots unchanged; temporary copy removed")


if __name__ == "__main__":
    main()
