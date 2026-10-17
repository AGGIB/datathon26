import logging

import pytest

from backend.app.schemas import AnalysisResult
from ml.assemble import DRAFT_BASIS, MAX_VIOLATIONS, MIN_CONFIDENCE, assemble, describe, select
from ml.candidates import Candidate
from ml.kb import RuleText, ViolationType

RULE = RuleText(
    document="Правила осуществления кассовых операций",
    number_date="Постановление Правления НБРК от 29.11.2019 № 231",
    clause="п. 20",
    quote="Наличные деньги принимаются и пересчитываются кассовым работником.",
    url="https://adilet.zan.kz/rus/docs/V1900019680",
)
TYPES = {
    "cash_left_on_desk": ViolationType(
        id="cash_left_on_desk",
        name="Деньги оставлены на столе при уходе кассира",
        summary="Пачка оставлена на столе при уходе кассира",
        status="verified",
        rule=RULE,
    ),
    "door_left_open": ViolationType(
        id="door_left_open",
        name="Дверь кассы оставлена открытой",
        summary="Открытая дверь при уходе",
        status="verified",
        rule=RULE,
    ),
    "computer_left_unlocked": ViolationType(
        id="computer_left_unlocked",
        name="Компьютер оставлен открытым",
        summary="Открытый компьютер оставлен после ухода сотрудника",
        status="draft",
        rule=None,
    ),
}
FRAME = {"timestamp_sec": 1.0, "image_base64": "/9j/AAAA"}


def cand(type_id="cash_left_on_desk", t0=10.0, t1=14.0, peak=None, conf=0.9, obs="кассир выходит, пачка на столе"):
    peak = t0 if peak is None else peak
    return Candidate(
        type_id=type_id, t_start=t0, t_end=t1, t_peak=peak, confidence=conf,
        observation=obs, evidence_times=(peak,),
    )


def frames_for(c: Candidate) -> list[dict]:
    return [{"timestamp_sec": round(t, 1), "image_base64": "/9j/AAAA"} for t in c.evidence_times]


def run(candidates, render=frames_for):
    result = assemble(candidates, render, types=TYPES)
    AnalysisResult.model_validate(result)  # every result must satisfy К1
    return result


def test_no_candidates_is_no_violation():
    assert run([]) == {"verdict": "no_violation", "violations": []}


def test_candidates_below_threshold_are_dropped():
    result = run([cand(conf=MIN_CONFIDENCE - 0.01)])
    assert result == {"verdict": "no_violation", "violations": []}


def test_candidate_at_threshold_is_kept():
    result = run([cand(conf=MIN_CONFIDENCE)])
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 1


def test_unknown_type_is_dropped_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="ml.assemble"):
        result = run([cand(type_id="made_up_type")])
    assert result["verdict"] == "no_violation"
    assert "made_up_type" in caplog.text


def test_violation_fields_follow_k1():
    result = run([cand(t0=20.0, t1=31.04, peak=22.44, conf=0.8712)])
    v = result["violations"][0]
    assert result["verdict"] == "violation"
    assert v["type"] == TYPES["cash_left_on_desk"].name
    assert v["type_id"] == "cash_left_on_desk"
    assert v["timecode_sec"] == 22.4
    assert (v["t_start"], v["t_end"]) == (20.0, 31.0)
    assert v["confidence"] == 0.871
    assert v["evidence_frames"] == [{"timestamp_sec": 22.4, "image_base64": "/9j/AAAA"}]
    assert v["rule"] == RULE.as_ref()


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
    result = run([cand(t0=10.0, t1=12.0), cand(t0=40.0, t1=42.0)])
    assert len(result["violations"]) == 2


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
    expected = sorted(round(c.confidence, 3) for c in many)[-MAX_VIOLATIONS:]
    assert kept == pytest.approx(expected)


def test_description_has_observation_and_ends_with_basis():
    text = describe(cand(peak=22.4, t0=22.4, t1=30.0, obs="Кассир выходит из кадра, пачка остаётся на столе."),
                    TYPES["cash_left_on_desk"])
    assert text.startswith("На 22,4 с кассир выходит из кадра, пачка остаётся на столе.")
    assert "может указывать на нарушение" in text
    assert text.endswith(f"Основание: {RULE.basis}.")


def test_description_keeps_leading_abbreviation_case():
    text = describe(cand(obs="ПК остаётся включённым"), TYPES["cash_left_on_desk"])
    assert "с ПК остаётся включённым." in text


def test_draft_type_has_no_rule_and_says_basis_is_being_clarified():
    [v] = run([cand(type_id="computer_left_unlocked")])["violations"]
    assert "rule" not in v
    assert v["description"].endswith(f"Основание: {DRAFT_BASIS}.")
    assert "adilet" not in v["description"]


def test_empty_observation_falls_back_to_kb_summary():
    text = describe(cand(obs="  "), TYPES["cash_left_on_desk"])
    assert text.startswith("На 10,0 с пачка оставлена на столе при уходе кассира.")


def test_candidate_without_evidence_frames_is_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger="ml.assemble"):
        result = run([cand()], render=lambda c: [])
    assert result == {"verdict": "no_violation", "violations": []}
    assert "evidence" in caplog.text


def test_more_than_three_frames_are_trimmed():
    [v] = run([cand()], render=lambda c: [FRAME] * 5)["violations"]
    assert len(v["evidence_frames"]) == 3


def test_negative_start_is_clamped_to_zero():
    [v] = run([cand(t0=-0.3, t1=2.0, peak=0.0)])["violations"]
    assert v["t_start"] == 0.0


def test_select_is_pure_and_returns_new_list():
    original = [cand(t0=10.0, t1=15.0, conf=0.6), cand(t0=12.0, t1=20.0, conf=0.9)]
    snapshot = list(original)
    picked = select(original, TYPES)
    assert original == snapshot
    assert picked[0] is not original[0]
