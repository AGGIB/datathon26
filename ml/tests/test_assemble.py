import logging
from pathlib import Path

import pytest

from backend.app.schemas import AnalysisResult
from ml.assemble import MAX_VIOLATIONS, MIN_CONFIDENCE, assemble, clamp_time, describe, select
from ml.candidates import Candidate
from ml.kb import DRAFT_ENDING, FALLBACK_TEMPLATES, RuleText, ViolationType, catalog, load_catalog

PR32_FIXTURE = Path(__file__).parent / "fixtures" / "kb_pr32_violations.yaml"
RULE = RuleText(
    document="Правила осуществления кассовых операций",
    number_date="Постановление Правления НБРК от 29.11.2019 № 231",
    clause="п. 20",
    quote="Наличные деньги принимаются и пересчитываются кассовым работником.",
    url="https://adilet.zan.kz/rus/docs/V1900019680",
)


def vtype(type_id, status="verified", rule=RULE, template=None, enabled=True):
    template = template or FALLBACK_TEMPLATES["verified" if rule else "draft"]
    return ViolationType(id=type_id, name=f"Тип {type_id}", summary="сводка, которую нельзя вставлять в текст",
                         status=status, rule=rule, template=template, enabled=enabled)


TYPES = {
    "cash_left_on_desk": vtype("cash_left_on_desk"),
    "door_left_open": vtype("door_left_open"),
    "computer_left_unlocked": vtype("computer_left_unlocked", status="draft", rule=None),
    "switched_off": vtype("switched_off", enabled=False),
}
FRAME = {"timestamp_sec": 1.0, "image_base64": "/9j/AAAA"}


def cand(type_id="cash_left_on_desk", t0=10.0, t1=14.0, peak=None, conf=0.9, obs="кассир выходит, пачка на столе"):
    peak = t0 if peak is None else peak
    return Candidate(
        type_id=type_id, t_start=t0, t_end=t1, t_peak=peak, confidence=conf,
        observation=obs, evidence_times=(peak,),
    )


def frames_for(candidates):
    return {c: [{"timestamp_sec": round(t, 1), "image_base64": "/9j/AAAA"} for t in c.evidence_times]
            for c in candidates}


def run(candidates, render=frames_for, types=TYPES, duration=None):
    result = assemble(candidates, render, types=types, duration=duration)
    AnalysisResult.model_validate(result)  # every result must satisfy К1
    return result


# --- selection ------------------------------------------------------------------------------------


def test_no_candidates_is_no_violation():
    assert run([]) == {"verdict": "no_violation", "violations": []}


def test_render_is_not_called_without_candidates():
    def boom(_):
        raise AssertionError("must not decode the video for nothing")

    assert run([cand(conf=0.1)], render=boom)["verdict"] == "no_violation"


def test_candidates_below_threshold_are_dropped():
    assert run([cand(conf=MIN_CONFIDENCE - 0.01)]) == {"verdict": "no_violation", "violations": []}


def test_candidate_at_threshold_is_kept():
    result = run([cand(conf=MIN_CONFIDENCE)])
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 1


def test_unknown_type_is_dropped_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="ml.assemble"):
        result = run([cand(type_id="made_up_type")])
    assert result["verdict"] == "no_violation"
    assert "made_up_type" in caplog.text


def test_disabled_type_is_dropped():
    assert run([cand(type_id="switched_off")])["verdict"] == "no_violation"


def test_render_gets_all_selected_candidates_in_one_call():
    calls = []

    def render(candidates):
        calls.append(list(candidates))
        return frames_for(candidates)

    run([cand(t0=10.0, t1=11.0), cand(type_id="door_left_open", t0=30.0, t1=31.0)], render=render)
    assert len(calls) == 1 and len(calls[0]) == 2


def test_overlapping_same_type_candidates_merge_into_the_stronger_one():
    weak = cand(t0=10.0, t1=15.0, peak=12.0, conf=0.6, obs="слабое")
    strong = cand(t0=13.0, t1=20.0, peak=16.0, conf=0.9, obs="сильное")
    [v] = run([weak, strong])["violations"]
    assert (v["t_start"], v["t_end"]) == (10.0, 20.0)
    assert v["timecode_sec"] == 16.0
    assert v["confidence"] == 0.9
    assert "сильное" in v["description"]


def test_chain_of_overlaps_merges_into_one():
    chain = [cand(t0=0.0, t1=5.0, conf=0.7), cand(t0=4.0, t1=9.0, conf=0.8), cand(t0=8.5, t1=12.0, conf=0.6)]
    [v] = run(chain)["violations"]
    assert (v["t_start"], v["t_end"]) == (0.0, 12.0)
    assert v["confidence"] == 0.8


def test_distant_same_type_candidates_stay_separate():
    assert len(run([cand(t0=10.0, t1=12.0), cand(t0=40.0, t1=42.0)])["violations"]) == 2


def test_overlapping_different_types_stay_separate():
    result = run([cand(type_id="cash_left_on_desk"), cand(type_id="door_left_open")])
    assert {v["type_id"] for v in result["violations"]} == {"cash_left_on_desk", "door_left_open"}


def test_violations_are_sorted_by_time():
    result = run([cand(t0=50.0, t1=51.0), cand(type_id="door_left_open", t0=5.0, t1=6.0), cand(t0=20.0, t1=21.0)])
    times = [v["timecode_sec"] for v in result["violations"]]
    assert times == sorted(times)


def test_list_is_capped_keeping_the_most_confident():
    many = [cand(t0=10.0 * i, t1=10.0 * i + 1, conf=0.5 + 0.04 * i) for i in range(MAX_VIOLATIONS + 3)]
    result = run(many)
    assert len(result["violations"]) == MAX_VIOLATIONS
    kept = sorted(v["confidence"] for v in result["violations"])
    assert kept == pytest.approx(sorted(round(c.confidence, 3) for c in many)[-MAX_VIOLATIONS:])


def test_select_is_pure_and_returns_new_list():
    original = [cand(t0=10.0, t1=15.0, conf=0.6), cand(t0=12.0, t1=20.0, conf=0.9)]
    snapshot = list(original)
    picked = select(original, TYPES)
    assert original == snapshot
    assert picked[0] is not original[0]


# --- fields and text ------------------------------------------------------------------------------


def test_violation_fields_follow_k1():
    [v] = run([cand(t0=20.0, t1=31.04, peak=22.44, conf=0.8712)])["violations"]
    assert v["type"] == TYPES["cash_left_on_desk"].name
    assert v["type_id"] == "cash_left_on_desk"
    assert v["timecode_sec"] == 22.4
    assert (v["t_start"], v["t_end"]) == (20.0, 31.0)
    assert v["confidence"] == 0.871
    assert v["evidence_frames"] == [{"timestamp_sec": 22.4, "image_base64": "/9j/AAAA"}]
    assert v["rule"] == RULE.as_ref()


def test_fallback_description_fills_time_observation_and_basis():
    text = describe(cand(peak=22.4, t0=22.4, t1=30.0, obs="Кассир выходит из кадра, пачка остаётся на столе."),
                    TYPES["cash_left_on_desk"])
    assert text == (
        "На 22,4 с кассир выходит из кадра, пачка остаётся на столе. "
        "Это может указывать на несоблюдение требования, указанного в основании, если оно применимо. "
        "Основание: Правила осуществления кассовых операций, п. 20."
    )


def test_summary_is_never_spliced_into_the_text():
    for type_id in ("cash_left_on_desk", "computer_left_unlocked"):
        assert "сводка" not in describe(cand(type_id=type_id), TYPES[type_id])


def test_description_keeps_leading_abbreviation_case():
    assert "с ПК остаётся включённым." in describe(cand(obs="ПК остаётся включённым"), TYPES["cash_left_on_desk"])


def test_kb_template_wording_is_used_and_observation_capitalised_at_sentence_start():
    template = "{observation}, на {t} с. Это может указывать на нарушение. Основание: {rule_document}, {rule_clause}."
    custom = vtype("cash_left_on_desk", template=template)
    text = describe(cand(peak=5.0, t0=5.0, t1=6.0, obs="пачка остаётся на столе"), custom)
    assert text.startswith("Пачка остаётся на столе, на 5,0 с.")


def test_draft_type_has_no_rule_and_says_basis_is_being_clarified():
    [v] = run([cand(type_id="computer_left_unlocked")])["violations"]
    assert "rule" not in v
    assert v["description"].endswith(DRAFT_ENDING)
    assert "Основание:" not in v["description"]


def test_times_are_clamped_to_the_analysed_duration():
    [v] = run([cand(t0=15.0, t1=17.067, peak=17.067)], duration=17.067)["violations"]
    assert v["timecode_sec"] <= 17.067 and v["t_end"] <= 17.067
    assert v["timecode_sec"] == 17.0
    assert v["description"].startswith("На 17,0 с")


def test_clamp_time_bounds():
    assert clamp_time(-0.3) == 0.0
    assert clamp_time(12.34) == 12.3
    assert clamp_time(19.96, duration=19.97) == 19.9
    assert clamp_time(25.0, duration=19.97) == 19.9


# --- evidence ------------------------------------------------------------------------------------


def test_candidate_without_evidence_frames_is_dropped_with_error(caplog):
    with caplog.at_level(logging.ERROR, logger="ml.assemble"):
        result = run([cand()], render=lambda cs: {})
    assert result == {"verdict": "no_violation", "violations": []}
    assert "evidence" in caplog.text


def test_more_than_three_frames_are_trimmed():
    [v] = run([cand()], render=lambda cs: {c: [FRAME] * 5 for c in cs})["violations"]
    assert len(v["evidence_frames"]) == 3


# --- every type of both catalogs ---------------------------------------------------------------------


CATALOGS = {"main": dict(catalog()), "pr32": dict(load_catalog(PR32_FIXTURE))}
ALL_TYPES = [(name, type_id) for name, types in CATALOGS.items() for type_id in types]


@pytest.mark.parametrize(("catalog_name", "type_id"), ALL_TYPES)
def test_every_kb_type_assembles_into_valid_k1(catalog_name, type_id):
    types = CATALOGS[catalog_name]
    result = run([cand(type_id=type_id, t0=3.0, t1=6.0, peak=4.2)], types=types, duration=10.0)
    [v] = result["violations"]
    assert v["type"] == types[type_id].name
    assert "{" not in v["description"] and "<" not in v["description"]
    rule = types[type_id].rule
    if rule:
        assert v["rule"] == rule.as_ref()
        assert v["description"].endswith(f"Основание: {rule.document}, {rule.clause}.")
    else:
        assert "rule" not in v and v["description"].lower().endswith(DRAFT_ENDING.lower())


def test_both_catalogs_are_covered():
    assert len(CATALOGS["main"]) >= 17 and len(CATALOGS["pr32"]) == 27
