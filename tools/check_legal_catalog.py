#!/usr/bin/env python3
"""Validate a clean isolated KB and meaningful invalid catalog variants. No data/ reads."""

import copy
from pathlib import Path
import shutil
import tempfile

import yaml

from validate_legal_kb import load_yaml, validate


def main():
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
        door = next(row for row in clean["violations"] if row["codex_code"] == "Т06")
        rendered = [door["description_template"].format_map(branch["template_values"]) for branch in door["criterion_branches"]]
        assert "остаётся открытой" in rendered[0] and "после ухода" not in rendered[0]
        assert "после ухода" in rendered[1] and "остаётся открытой" not in rendered[1]
        print("OK: isolated kb/, 8 sources / 14 candidates / 32 analytical / 27 active / 5 excluded")
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
                entries["Т06"]["description_template"] = "Дверь остаётся открытой / запирание не видно; это может указывать на несоблюдение требования об изолированном входе."
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
            print(f"REJECTED: {case}")
        path.write_text(raw, encoding="utf-8")
        assert validate(base / "legal") == (8, 14, 32, 27, 5)
    print("OK: 14 negative checks; source files unchanged; temporary copy removed")


if __name__ == "__main__":
    main()
