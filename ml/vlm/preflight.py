"""Local artifact checks. These do NOT prove model loadability or accuracy."""
import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflightResult:
    ready: bool
    errors: tuple[str, ...]
    model_type: str | None = None
    runtime_loaded: bool = False


class ModelUnavailable(RuntimeError):
    pass


def enforce_offline() -> None:
    # Assignment, not setdefault: a caller's online configuration cannot override this.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_directory(model_path: str | Path) -> Path:
    raw = str(model_path)
    if not raw or "://" in raw or raw.replace("\\", "/").startswith("//"):
        raise ModelUnavailable("model_directory_required")
    directory = Path(model_path).resolve()
    if str(directory).replace("\\", "/").startswith("//"):
        raise ModelUnavailable("model_directory_required")
    if not directory.is_dir():
        raise ModelUnavailable("model_directory_required")
    return directory


def _artifact_status(directory: Path) -> tuple[list[str], str | None]:
    """Shared structural guard; never imports model libraries or starts workers."""
    errors = []
    model_type = None
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        model_type = config["model_type"]
        if not isinstance(model_type, str) or not model_type:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        errors.append("config_invalid")
    if not (directory / "preprocessor_config.json").is_file():
        errors.append("processor_missing")
    if not (directory / "tokenizer_config.json").is_file() or not any(
        (directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    ):
        errors.append("tokenizer_missing")
    weights = directory / "model.safetensors"
    index = directory / "model.safetensors.index.json"
    if not weights.is_file() and not index.is_file():
        errors.append("weights_missing")
    elif index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            weight_map = data["weight_map"]
            if (not isinstance(weight_map, dict) or not weight_map or any(
                not isinstance(key, str) or not key.strip()
                or not isinstance(shard, str) or not shard.strip()
                for key, shard in weight_map.items()
            )):
                raise ValueError
            for shard in set(weight_map.values()):
                relative = shard.replace("\\", "/")
                if ":" in relative or relative.startswith("/") or ".." in relative.split("/"):
                    raise ValueError
                target = (directory / shard).resolve()
                if not target.is_relative_to(directory) or target.suffix != ".safetensors" or not target.is_file():
                    raise ValueError
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("weight_shards_invalid")
    return errors, model_type


def _validated_model_directory(model_path: str | Path) -> Path:
    directory = model_directory(model_path)
    errors, _ = _artifact_status(directory)
    if errors:
        raise ModelUnavailable("model_artifacts_invalid")
    return directory


def preflight(model_path: str | Path) -> PreflightResult:
    enforce_offline()
    errors = []
    model_type = None
    try:
        directory = model_directory(model_path)
    except (OSError, ModelUnavailable):
        errors.append("model_directory_required")
    else:
        errors, model_type = _artifact_status(directory)
    for module in ("torch", "PIL", "transformers"):
        try:
            found = importlib.util.find_spec(module)
        except (ValueError, ImportError):
            found = None
        if found is None:
            errors.append(f"dependency_missing:{module}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            errors.append(f"executable_missing:{executable}")
    return PreflightResult(not errors, tuple(errors), model_type)
