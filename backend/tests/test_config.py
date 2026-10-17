import pytest

from backend.app.config import Settings


def test_from_env_parses_keys_and_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEYS", " jury , ui ,")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PIPELINE", raising=False)
    settings = Settings.from_env()
    assert settings.api_keys == frozenset({"jury", "ui"})
    assert settings.pipeline == "mock"
    assert settings.data_dir == tmp_path.resolve()


def test_from_env_requires_keys(monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    with pytest.raises(RuntimeError):
        Settings.from_env()


@pytest.mark.parametrize("field, value", [("pipeline", "gpt"), ("workers", 0), ("api_keys", frozenset())])
def test_invalid_settings_rejected(tmp_path, field, value):
    kwargs = {"api_keys": frozenset({"k"}), "data_dir": tmp_path, field: value}
    with pytest.raises(ValueError):
        Settings(**kwargs)
