import os
import pytest

from backend.app.config import Settings


def test_from_env_parses_keys_and_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEYS", " jury , ui ,")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PIPELINE", raising=False)
    # Pass a non-existent env_path so it relies purely on the monkeypatched environment
    settings = Settings.from_env(env_path=tmp_path / "nonexistent.env")
    assert settings.api_keys == frozenset({"jury", "ui"})
    assert settings.pipeline == "mock"
    assert settings.data_dir == tmp_path.resolve()
    assert settings.pipeline_timeout_sec == 200.0
    assert settings.job_timeout_sec == 270.0


def test_from_env_parses_inline_comments_and_exports(monkeypatch, tmp_path):
    # Clear out any environment variables that would override the file
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.delenv("PIPELINE", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "API_KEYS=\"key#1,key2\" # inline comment\n"
        "export WORKERS=3 # three workers\n"
        "export PIPELINE=mock # test mock\n"
        "# comment line\n"
        f"DATA_DIR={tmp_path}\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(env_path=env_file)
    assert settings.api_keys == frozenset({"key#1", "key2"})
    assert settings.workers == 3
    assert settings.pipeline == "mock"
    # Ensure os.environ was NOT polluted
    assert "WORKERS" not in os.environ


def test_from_env_requires_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("API_KEYS", "")
    with pytest.raises(RuntimeError):
        Settings.from_env(env_path=tmp_path / "nonexistent.env")


@pytest.mark.parametrize("field, value", [("pipeline", "gpt"), ("workers", 0), ("api_keys", frozenset())])
def test_invalid_settings_rejected(tmp_path, field, value):
    kwargs = {"api_keys": frozenset({"k"}), "data_dir": tmp_path, field: value}
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_placeholder_keys_rejected(tmp_path):
    for bad_key in ["change-me-jury-key", "change-me-ui-key", "change-me-foo"]:
        with pytest.raises(ValueError, match="Placeholder API key"):
            Settings(api_keys=frozenset({bad_key}), data_dir=tmp_path)


def test_min_budget_sec_setting(tmp_path):
    s = Settings(api_keys=frozenset({"test"}), data_dir=tmp_path, min_budget_sec=15.0)
    assert s.min_budget_sec == 15.0

    for invalid in [0.0, -1.0]:
        with pytest.raises(ValueError, match="MIN_BUDGET_SEC"):
            Settings(api_keys=frozenset({"test"}), data_dir=tmp_path, min_budget_sec=invalid)

    with pytest.raises(ValueError, match="PIPELINE_TIMEOUT_SEC"):
        Settings(api_keys=frozenset({"test"}), data_dir=tmp_path, pipeline_timeout_sec=0.0)

    with pytest.raises(ValueError, match="JOB_TIMEOUT_SEC"):
        Settings(api_keys=frozenset({"test"}), data_dir=tmp_path, job_timeout_sec=0.0)



