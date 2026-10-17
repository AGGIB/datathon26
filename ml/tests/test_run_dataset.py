from types import SimpleNamespace
import pickle

import pytest

from ml.tools import run_dataset


def test_capture_restores_rules_after_a_failed_run():
    from ml import pipeline

    original = pipeline.rules.find_candidates
    captured = {}
    with pytest.raises(RuntimeError), run_dataset._capturing_rules(captured):
        raise RuntimeError("interrupted")
    assert pipeline.rules.find_candidates is original


def test_resume_keeps_successes_and_retries_failures(tmp_path, monkeypatch):
    from ml import pipeline

    out = tmp_path / "results.json"
    run_dataset._save(out, b'{"X-1":{"verdict":"no_violation"},"X-2":{"error":"interrupted"}}')
    monkeypatch.setattr(run_dataset, "_videos", lambda only: [tmp_path / "X-1.mov", tmp_path / "X-2.mov"])
    calls = []

    def analyze(path, **kwargs):
        calls.append(path)
        return {"verdict": "no_violation", "violations": []}

    monkeypatch.setattr(pipeline, "analyze", analyze)
    results = run_dataset.run(out, set(), 200, None, resume=True)
    assert calls == [str(tmp_path / "X-2.mov")]
    assert results["X-2"]["verdict"] == "no_violation"
    assert not out.with_name(out.name + ".part").exists()


def test_failed_positive_stays_in_quality_denominators(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb/annotations.csv").write_text("video,type_id\nX-1,example\nX-2,example\n")
    monkeypatch.setattr(run_dataset, "load_labels", lambda path: [
        SimpleNamespace(video=name, has_violation=True, t_sec=1.0) for name in ("X-1", "X-2")
    ])
    run_dataset.score({
        "X-1": {"verdict": "violation", "violations": [{"type_id": "example", "t": 1.0}], "sec": 1.0},
        "X-2": {"error": "TimeoutError", "sec": 200.0},
    })
    output = capsys.readouterr().out
    assert "completed=1 errors=1" in output
    assert "recall=0.50" in output
    assert "accuracy=0.50" in output


def test_rules_only_uses_the_same_candidate_selection_as_the_api(tmp_path, monkeypatch):
    from ml import rules
    from ml.candidates import Candidate

    perception = SimpleNamespace(info=SimpleNamespace(duration=10.0))
    (tmp_path / "X-1.pkl").write_bytes(pickle.dumps((perception, None)))
    known = "safe_left_open"
    candidates = [
        Candidate(known, 1, 4, 2, 0.8, "дверца открыта", (2,)),
        Candidate(known, 3, 7, 5, 0.7, "дверца открыта", (5,)),
        Candidate("unknown_type", 1, 4, 2, 0.99, "неизвестный тип", (2,)),
    ]
    monkeypatch.setattr(rules, "find_candidates", lambda *args, **kwargs: candidates)
    result = run_dataset.rules_only(tmp_path)["X-1"]
    assert result["verdict"] == "violation"
    assert result["violations"] == [{"type_id": known, "t": 2.0, "conf": 0.8}]
