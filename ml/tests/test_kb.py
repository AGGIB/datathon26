import logging
from pathlib import Path

import pytest

from ml.kb import (
    DRAFT_ENDING,
    FALLBACK_TEMPLATES,
    VERIFIED_ENDING,
    VERIFIED_ENDING_NUMBER_DATE,
    VERIFIED_ENDINGS,
    KBError,
    catalog,
    load_catalog,
    parse_catalog,
)

# verbatim copy of main:kb/violations.yaml @2b5d7f9 (PR #32), the real catalog once branches meet
MAIN_KB_FIXTURE = Path(__file__).parent / "fixtures" / "kb_main_violations.yaml"

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


def errors(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "ml.kb" and r.levelno >= logging.ERROR]


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


def test_main_catalog_loads_with_its_own_templates(caplog):
    with caplog.at_level(logging.WARNING, logger="ml.kb"):
        types = load_catalog(MAIN_KB_FIXTURE)
    assert len(types) == 27
    assert not errors(caplog)
    assert [vt.id for vt in types.values() if vt.rule is None] == ["computer_left_unlocked"]
    assert all("{observation}" in vt.template and vt.template not in FALLBACK_TEMPLATES.values()
               for vt in types.values())
    for vt in types.values():
        assert vt.template.endswith(VERIFIED_ENDINGS) if vt.rule else vt.template.endswith(DRAFT_ENDING)


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


def test_empty_rule_block_gives_no_rule():
    assert one(entry(status="draft", rule=EMPTY_RULE, template=CONTRACT_DRAFT)).rule is None


def test_disabled_type_is_marked():
    assert one(entry(enabled=False)).enabled is False
    assert one(entry()).enabled is True


# --- templates: kb wording -----------------------------------------------------------------------


def test_contract_template_is_kept():
    assert one(entry()).template == CONTRACT_VERIFIED


def test_draft_ending_is_case_insensitive():
    template = "На {t} с: {observation}; нормативное основание уточняется."
    assert one(entry(status="draft", rule=EMPTY_RULE, template=template)).template == template


@pytest.mark.parametrize("template", [LEGACY, None, "", "Пачка остаётся на столе; это может указывать на нарушение."])
def test_legacy_or_missing_template_falls_back(template):
    assert one(entry(template=template)).template == FALLBACK_TEMPLATES["verified"]


def test_generic_wording_keeps_the_caveats():
    assert "если оно применимо" in FALLBACK_TEMPLATES["verified"]
    assert "применимого порядка, если он установлен" in FALLBACK_TEMPLATES["draft"]
    assert FALLBACK_TEMPLATES["draft"].endswith(DRAFT_ENDING)
    assert "{rule_" not in FALLBACK_TEMPLATES["draft"]


def test_missing_clause_drops_it_from_the_basis():
    assert one(entry(rule={**VERIFIED_RULE, "clause": None})).template.endswith("Основание: {rule_document}.")


# --- templates: a wording problem of one type never stops the service -----------------------------


@pytest.mark.parametrize(
    ("raw_entry", "expected"),
    [
        # verified wording, but the norm's verification was withdrawn / is pending / the type is draft
        (entry(rule={**VERIFIED_RULE, "verified_by": None}), "draft"),
        (entry(rule={**VERIFIED_RULE, "citation_status": "pending"}), "draft"),
        (entry(status="draft"), "draft"),
        # draft wording, but the norm got verified
        (entry(template=CONTRACT_DRAFT), "verified"),
        # wrong ending for a verified type
        (entry(template="На {t} с {observation}. Это может указывать на нарушение."), "verified"),
        # draft template citing rule fields
        (entry(status="draft", rule=EMPTY_RULE, template="На {t} с {observation} ({rule_document}). " + DRAFT_ENDING),
         "draft"),
        # number-and-date ending, but verification was withdrawn
        (entry(template="На {t} с {observation}. " + VERIFIED_ENDING_NUMBER_DATE,
               rule={**VERIFIED_RULE, "verified_by": None}), "draft"),
    ],
)
def test_template_that_does_not_match_the_norm_state_falls_back_with_error(raw_entry, expected, caplog):
    with caplog.at_level(logging.ERROR, logger="ml.kb"):
        vt = one(raw_entry)
    assert vt.template == FALLBACK_TEMPLATES[expected]
    assert (vt.rule is None) == (expected == "draft")
    assert [m for m in errors(caplog) if raw_entry["id"] in m]


def test_typo_in_observation_placeholder_is_reported_as_error(caplog):
    with caplog.at_level(logging.ERROR, logger="ml.kb"):
        vt = one(entry(template="На {t} с {observaton}. " + VERIFIED_ENDING))
    assert vt.template == FALLBACK_TEMPLATES["verified"]
    assert any("observaton" in m for m in errors(caplog))


def test_number_date_placeholder_works_when_present():
    template = "На {t} с {observation} ({rule_number_date}). " + VERIFIED_ENDING
    assert one(entry(template=template)).template == template


def test_number_and_date_ending_is_accepted_for_verified_types():
    template = "На {t} с: {observation}. Это может указывать на нарушение. " + VERIFIED_ENDING_NUMBER_DATE
    assert one(entry(template=template)).template == template


def test_number_and_date_ending_drops_a_missing_clause():
    template = "На {t} с: {observation}. " + VERIFIED_ENDING_NUMBER_DATE
    vt = one(entry(template=template, rule={**VERIFIED_RULE, "clause": None}))
    assert vt.template.endswith("Основание: {rule_number_date}.")


@pytest.mark.parametrize("ending", [VERIFIED_ENDING_NUMBER_DATE, VERIFIED_ENDING])
def test_number_date_placeholder_without_number_date_fails_fast(ending):
    template = "На {t} с {observation}. " + ending.replace("{rule_document}", "{rule_document} ({rule_number_date})")
    with pytest.raises(KBError, match="number_date"):
        parse_catalog(doc(entry(template=template, rule={**VERIFIED_RULE, "number_date": None})))


# --- structural errors fail fast -----------------------------------------------------------------------


def test_unknown_placeholder_in_contract_template_fails_fast():
    with pytest.raises(KBError, match="door_observation"):
        parse_catalog(doc(entry(template="На {t} с {observation}; {door_observation}. " + VERIFIED_ENDING)))


def test_format_spec_in_placeholder_fails_fast():
    with pytest.raises(KBError, match="placeholder"):
        parse_catalog(doc(entry(template="На {t:.1f} с {observation}. " + VERIFIED_ENDING)))


def test_broken_braces_in_contract_template_fail_fast():
    with pytest.raises(KBError, match="valid template"):
        parse_catalog(doc(entry(template="На {t с {observation}. " + VERIFIED_ENDING)))


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
