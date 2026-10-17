import subprocess
from pathlib import Path

import pytest
import yaml

from ml.kb import (
    DRAFT_ENDING,
    FALLBACK_TEMPLATES,
    VERIFIED_ENDING,
    KBError,
    catalog,
    load_catalog,
    parse_catalog,
)

REPO = Path(__file__).resolve().parents[2]
PR32_REF = "origin/codex/olzhas-normative-kb:kb/violations.yaml"

VERIFIED_RULE = {
    "document": "Правила осуществления кассовых операций",
    "number_date": "Постановление Правления НБРК от 29.11.2019 № 231",
    "clause": "п. 20",
    "quote": "Наличные деньги принимаются и пересчитываются кассовым работником.",
    "url": "https://adilet.zan.kz/rus/docs/V1900019680",
    "verified_by": "Олжас",
}
EMPTY_RULE = dict.fromkeys(VERIFIED_RULE)
CONTRACT_VERIFIED = "На {t} с {observation}. Это может указывать на нарушение порядка хранения денег. " + VERIFIED_ENDING
CONTRACT_DRAFT = "На {t} с {observation}. Это может указывать на нарушение порядка. " + DRAFT_ENDING
LEGACY = "На {t} с <что видно>. Это может указывать на нарушение. Основание: {rule_document}, {rule_clause}."


def entry(type_id="cash_left_on_desk", status="verified", rule=None, template=CONTRACT_VERIFIED, **overrides):
    base = {
        "id": type_id,
        "status": status,
        "name": "Деньги оставлены на столе",
        "summary": "Пачка оставлена на столе при уходе кассира",
        "description_template": template,
        "rule": VERIFIED_RULE if rule is None else rule,
    }
    return {**base, **overrides}


def doc(*entries):
    return {"version": 1, "violations": list(entries)}


def one(raw_entry):
    return next(iter(parse_catalog(doc(raw_entry)).values()))


def pr32_raw():
    proc = subprocess.run(["git", "show", PR32_REF], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"{PR32_REF} is not available locally")
    return yaml.safe_load(proc.stdout)


# --- the real catalogs --------------------------------------------------------------------------


def test_real_catalog_loads_and_is_cached():
    first = catalog()
    assert first is catalog()
    assert len(first) >= 1
    assert all(vt.id == key for key, vt in first.items())


def test_real_catalog_draft_types_carry_no_rule():
    for vt in catalog().values():
        if vt.status != "verified":
            assert vt.rule is None


def test_pr32_catalog_loads_with_its_extra_fields():
    types = parse_catalog(pr32_raw())
    assert len(types) >= 17
    draft = [vt for vt in types.values() if vt.rule is None]
    assert [vt.id for vt in draft] == ["computer_left_unlocked"]
    for vt in types.values():
        assert vt.template.endswith(VERIFIED_ENDING if vt.rule else DRAFT_ENDING)


# --- rules ----------------------------------------------------------------------------------------


def test_verified_type_exposes_rule():
    vt = one(entry())
    assert vt.name == "Деньги оставлены на столе"
    assert vt.summary == "Пачка оставлена на столе при уходе кассира"
    assert vt.rule is not None
    assert vt.rule.as_ref() == {
        "document": VERIFIED_RULE["document"],
        "clause": "п. 20",
        "quote": VERIFIED_RULE["quote"],
        "url": VERIFIED_RULE["url"],
    }


def test_draft_type_has_no_rule_even_if_fields_present():
    vt = one(entry(status="draft", template=CONTRACT_DRAFT))
    assert vt.status == "draft"
    assert vt.rule is None


def test_verified_status_without_verifier_is_not_trusted():
    vt = one(entry(rule={**VERIFIED_RULE, "verified_by": None}, template=CONTRACT_DRAFT))
    assert vt.rule is None


def test_pending_citation_is_not_trusted():
    vt = one(entry(rule={**VERIFIED_RULE, "citation_status": "pending"}, template=CONTRACT_DRAFT))
    assert vt.rule is None


def test_empty_rule_block_gives_no_rule():
    vt = one(entry(status="draft", rule=EMPTY_RULE, template=CONTRACT_DRAFT))
    assert vt.rule is None


def test_disabled_type_is_marked():
    assert one(entry(enabled=False)).enabled is False
    assert one(entry()).enabled is True


# --- templates ------------------------------------------------------------------------------------


def test_contract_template_is_kept():
    assert one(entry()).template == CONTRACT_VERIFIED


@pytest.mark.parametrize("template", [LEGACY, None, "", "Пачка остаётся на столе; это может указывать на нарушение."])
def test_legacy_or_missing_template_falls_back(template):
    assert one(entry(template=template)).template == FALLBACK_TEMPLATES["verified"]


def test_draft_fallback_does_not_cite_anything():
    vt = one(entry(status="draft", rule=EMPTY_RULE, template=LEGACY))
    assert vt.template == FALLBACK_TEMPLATES["draft"]
    assert "{rule_" not in vt.template


def test_unknown_placeholder_fails_fast():
    with pytest.raises(KBError, match="door_observation"):
        parse_catalog(doc(entry(template="На {t} с {observation}; {door_observation}. " + VERIFIED_ENDING)))


def test_format_spec_in_placeholder_fails_fast():
    with pytest.raises(KBError, match="placeholder"):
        parse_catalog(doc(entry(template="На {t:.1f} с {observation}. " + VERIFIED_ENDING)))


def test_verified_template_must_end_with_basis():
    with pytest.raises(KBError, match="Основание"):
        parse_catalog(doc(entry(template="На {t} с {observation}. Это может указывать на нарушение.")))


def test_draft_template_must_say_basis_is_being_clarified():
    with pytest.raises(KBError, match="уточняется"):
        parse_catalog(doc(entry(status="draft", rule=EMPTY_RULE, template=CONTRACT_VERIFIED)))


def test_draft_template_cannot_reference_rule_fields():
    template = "На {t} с {observation} ({rule_document}). " + DRAFT_ENDING
    with pytest.raises(KBError, match="rule"):
        parse_catalog(doc(entry(status="draft", rule=EMPTY_RULE, template=template)))


def test_number_date_placeholder_requires_number_date():
    template = "На {t} с {observation} ({rule_number_date}). " + VERIFIED_ENDING
    assert one(entry(template=template)).template == template
    with pytest.raises(KBError, match="number_date"):
        parse_catalog(doc(entry(template=template, rule={**VERIFIED_RULE, "number_date": None})))


def test_missing_clause_drops_it_from_the_basis():
    vt = one(entry(rule={**VERIFIED_RULE, "clause": None}))
    assert vt.template.endswith("Основание: {rule_document}.")


# --- structure ------------------------------------------------------------------------------------


def test_duplicate_ids_are_rejected():
    with pytest.raises(KBError, match="duplicate"):
        parse_catalog(doc(entry(), entry()))


@pytest.mark.parametrize("missing", ["id", "name", "status"])
def test_missing_required_field_is_rejected(missing):
    bad = {k: v for k, v in entry().items() if k != missing}
    with pytest.raises(KBError, match=missing):
        parse_catalog(doc(bad))


def test_unknown_status_is_rejected():
    with pytest.raises(KBError, match="status"):
        parse_catalog(doc(entry(status="approved")))


def test_non_snake_case_id_is_rejected():
    with pytest.raises(KBError, match="id"):
        parse_catalog(doc(entry(type_id="Cash-Left")))


def test_missing_violations_list_is_rejected():
    with pytest.raises(KBError):
        parse_catalog({"version": 1})


def test_catalog_is_read_only():
    types = parse_catalog(doc(entry()))
    with pytest.raises(TypeError):
        types["x"] = types["cash_left_on_desk"]  # type: ignore[index]


def test_load_catalog_reads_yaml_file(tmp_path):
    path = tmp_path / "violations.yaml"
    path.write_text(
        "version: 1\nviolations:\n  - id: door_left_open\n    status: draft\n    name: Дверь открыта\n",
        encoding="utf-8",
    )
    types = load_catalog(path)
    assert list(types) == ["door_left_open"]
    assert types["door_left_open"].rule is None
    assert types["door_left_open"].template == FALLBACK_TEMPLATES["draft"]
