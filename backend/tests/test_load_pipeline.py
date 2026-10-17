import pytest

import ml.pipeline
from backend.app.config import Settings
from backend.app.main import load_pipeline
from ml.kb import KBError


def settings(tmp_path, pipeline: str) -> Settings:
    return Settings(api_keys=frozenset({"k"}), data_dir=tmp_path, pipeline=pipeline)


def test_real_pipeline_loads_the_kb_at_startup(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ml.pipeline, "catalog", lambda: calls.append(1) or {})
    analyze = load_pipeline(settings(tmp_path, "real"))
    assert analyze is ml.pipeline.analyze
    assert calls == [1]


def test_broken_kb_fails_the_boot_not_every_job(tmp_path, monkeypatch):
    def broken():
        raise KBError("unknown placeholder {door_observation}")

    monkeypatch.setattr(ml.pipeline, "catalog", broken)
    with pytest.raises(KBError):
        load_pipeline(settings(tmp_path, "real"))


def test_mock_pipeline_does_not_touch_the_kb(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("mock mode must not load the kb")

    monkeypatch.setattr(ml.pipeline, "catalog", boom)
    assert callable(load_pipeline(settings(tmp_path, "mock")))
