import pytest

from ml.kb import KBError, catalog, load_catalog, parse_catalog

VERIFIED_RULE = {
    "document": "Правила осуществления кассовых операций",
    "number_date": "Постановление Правления НБРК от 29.11.2019 № 231",
    "clause": "п. 20",
    "quote": "Наличные деньги принимаются и пересчитываются кассовым работником.",
    "url": "https://adilet.zan.kz/rus/docs/V1900019680",
    "verified_by": "Олжас",
}
EMPTY_RULE = dict.fromkeys(VERIFIED_RULE)


def entry(type_id="cash_left_on_desk", status="verified", rule=None, **overrides):
    base = {
        "id": type_id,
        "status": status,
        "name": "Деньги оставлены на столе",
        "summary": "Пачка оставлена на столе при уходе кассира",
        "rule": VERIFIED_RULE if rule is None else rule,
    }
    return {**base, **overrides}


def doc(*entries):
    return {"version": 1, "violations": list(entries)}


def test_real_catalog_loads_and_is_cached():
    first = catalog()
    assert first is catalog()
    assert len(first) >= 1
    assert all(vt.id == key for key, vt in first.items())


def test_real_catalog_draft_types_carry_no_rule():
    for vt in catalog().values():
        if vt.status != "verified":
            assert vt.rule is None


def test_verified_type_exposes_rule_and_basis():
    vt = parse_catalog(doc(entry()))["cash_left_on_desk"]
    assert vt.name == "Деньги оставлены на столе"
    assert vt.rule is not None
    assert vt.rule.basis == "Постановление Правления НБРК от 29.11.2019 № 231, п. 20"
    assert vt.rule.as_ref() == {
        "document": VERIFIED_RULE["document"],
        "clause": "п. 20",
        "quote": VERIFIED_RULE["quote"],
        "url": VERIFIED_RULE["url"],
    }


def test_basis_without_clause_or_number_falls_back_to_document():
    rule = {**VERIFIED_RULE, "number_date": None, "clause": None}
    vt = parse_catalog(doc(entry(rule=rule)))["cash_left_on_desk"]
    assert vt.rule.basis == VERIFIED_RULE["document"]


def test_draft_type_has_no_rule_even_if_fields_present():
    vt = parse_catalog(doc(entry(status="draft")))["cash_left_on_desk"]
    assert vt.status == "draft"
    assert vt.rule is None


def test_verified_status_without_verifier_is_not_trusted():
    rule = {**VERIFIED_RULE, "verified_by": None}
    vt = parse_catalog(doc(entry(rule=rule)))["cash_left_on_desk"]
    assert vt.rule is None


def test_empty_rule_block_gives_no_rule():
    vt = parse_catalog(doc(entry(status="draft", rule=EMPTY_RULE)))["cash_left_on_desk"]
    assert vt.rule is None


def test_duplicate_ids_are_rejected():
    with pytest.raises(KBError, match="duplicate"):
        parse_catalog(doc(entry(), entry()))


@pytest.mark.parametrize("missing", ["id", "name", "status", "summary"])
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
        "version: 1\nviolations:\n  - id: door_left_open\n    status: draft\n"
        "    name: Дверь открыта\n    summary: Открытая дверь\n",
        encoding="utf-8",
    )
    types = load_catalog(path)
    assert list(types) == ["door_left_open"]
    assert types["door_left_open"].rule is None
