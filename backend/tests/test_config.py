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


def test_from_env_allows_no_keys_by_default(monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    monkeypatch.delenv("REQUIRE_API_KEY", raising=False)
    assert Settings.from_env().require_api_key is False


def test_api_key_auth_requires_keys_when_enabled(tmp_path):
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings(api_keys=frozenset(), data_dir=tmp_path, require_api_key=True)


@pytest.mark.parametrize("field, value", [("pipeline", "gpt"), ("workers", 0)])
def test_invalid_settings_rejected(tmp_path, field, value):
    kwargs = {"api_keys": frozenset({"k"}), "data_dir": tmp_path, field: value}
    with pytest.raises(ValueError):
        Settings(**kwargs)
