"""Dataset extraction uses artificial bytes/embeddings, never private model benchmarks."""
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest


@pytest.fixture
def dataset(tmp_path):
    pytest.importorskip("numpy")
    source = tmp_path / "videos"
    source.mkdir()
    # Literal Latin/Cyrillic names remain distinct; two files share exact contents.
    for name, content in [("X-1.mov", b"ARTIFICIAL ONE"), ("Х-1.mov", b"ARTIFICIAL TWO"),
                          ("Х-2.MOV", b"ARTIFICIAL ONE")]:
        (source / name).write_bytes(content)
    (source / "unrelated.txt").write_text("NOT A VIDEO", encoding="utf-8")
    return tmp_path, source


def test_dataset_extracts_every_literal_id_with_one_encoder(dataset, monkeypatch):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    features = importlib.import_module("ml.baseline.features")
    np = pytest.importorskip("numpy")
    loader = Mock(return_value=object())
    monkeypatch.setattr(features, "SiglipEncoder", loader)
    monkeypatch.setattr(module, "snapshot_fingerprint", lambda weights: "a" * 64)
    extracted = []

    def extract(video, weights, output, **kwargs):
        # Unavoidable pretrained boundary fixture, not real runtime evidence.
        extracted.append((video.name, kwargs["encoder"]))
        features.save_cache(output, video.name, [0.0, 1.0], np.ones((2, 768)),
                            {"extraction": {"cap_reached": False}})
        return {"n_frames": 2, "embedding_dim": 768, "cap_reached": False}

    monkeypatch.setattr(features, "extract_video", extract)
    report = module.extract_dataset(source, "fixture_weights", root / "artifacts" / "full",
                                    artifact_root=root)
    assert report["expected_videos"] == 3
    assert report["successful_videos"] == 3
    assert report["failed_videos"] == 0
    assert report["unique_contents"] == 2
    assert report["encoded_contents"] == 2
    assert len(extracted) == 2
    loader.assert_called_once()
    assert len({id(encoder) for _, encoder in extracted}) == 1
    assert {r["video_id"] for r in report["videos"]} == {"X-1.mov", "Х-1.mov", "Х-2.MOV"}
    for row in report["videos"]:
        cache = features.load_cache(root / "artifacts" / "full" / row["cache_file"])
        assert cache["video_id"] == row["video_id"]
        assert cache["metadata"]["content_sha256"] == row["content_sha256"]
        assert cache["metadata"]["snapshot_sha256"] == "a" * 64
    assert module.read_json(root / "artifacts" / "full" / "index.json") == report
    assert report["quality_metrics_measured"] is False


def test_single_video_accepts_a_shared_encoder_without_reloading(dataset, monkeypatch):
    root, source = dataset
    features = importlib.import_module("ml.baseline.features")
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(features, "SiglipEncoder", Mock(side_effect=AssertionError("must reuse encoder")))
    monkeypatch.setattr(features, "ffmpeg_command", lambda *a, **kw: [])
    monkeypatch.setattr(features, "decode_frames", lambda *a, **kw: (f for f in [(0.0, b"FIXTURE")]))
    batches = []
    def encode(batch):
        batches.append(list(batch))
        return np.ones((len(batch), 768))
    path = root / "artifacts" / "shared.npz"
    result = features.extract_video(source / "X-1.mov", "unused", path, encoder=encode, artifact_root=root)
    assert result["n_frames"] == 1
    assert batches == [[b"FIXTURE"]]
    assert features.load_cache(path)["video_id"] == "X-1.mov"


@pytest.mark.parametrize("failures,capped,exit_code", [(0, 0, 0), (1, 0, 2), (0, 1, 2)])
def test_dataset_cli_does_not_report_partial_extraction_as_success(monkeypatch, capsys, failures, capped, exit_code):
    cli = importlib.import_module("ml.baseline.__main__")
    module = importlib.import_module("ml.baseline.dataset")
    monkeypatch.setattr(module, "extract_dataset", lambda *a, **kw: {
        "expected_videos": 3, "successful_videos": 3 - failures, "failed_videos": failures,
        "capped_videos": capped, "unique_contents": 2, "encoded_contents": 2})
    assert cli.main(["extract-dataset", "--videos-dir", "fixture", "--weights", "fixture",
                     "--output-dir", "artifacts/fixture"]) == exit_code
    assert '"expected_videos": 3' in capsys.readouterr().out


@pytest.mark.parametrize("configuration", [{"fps": 0}, {"fps": float("nan")}, {"max_frames": 0},
    {"batch_size": 0}, {"timeout_s": -1}, {"cpu_threads": 0}, {"device": "cloud"}])
def test_dataset_rejects_invalid_configuration_before_loading(dataset, monkeypatch, configuration):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    monkeypatch.setattr(module, "snapshot_fingerprint", Mock(side_effect=AssertionError("preflight first")))
    with pytest.raises(ValueError):
        module.extract_dataset(source, "unused", root / "artifacts" / "invalid", artifact_root=root,
                               **configuration)


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_failed_extractions_stay_failed_and_every_input_is_accounted_for(dataset, monkeypatch, error_type):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    features = importlib.import_module("ml.baseline.features")
    monkeypatch.setattr(module, "snapshot_fingerprint", lambda weights: "a" * 64)
    monkeypatch.setattr(features, "SiglipEncoder", Mock(return_value=object()))

    def fail(video, weights, output, **kwargs):
        output.write_bytes(b"PARTIAL ARTIFICIAL CACHE")
        raise error_type("PRIVATE DIAGNOSTIC MUST NOT BE IN REPORT")

    monkeypatch.setattr(features, "extract_video", fail)
    output = root / "artifacts" / "failed"
    report = module.extract_dataset(source, "unused", output, artifact_root=root)
    assert report["expected_videos"] == len(report["videos"]) == report["failed_videos"] == 3
    assert report["successful_videos"] == 0
    assert all(row["status"] == "failed" and row["error_class"] == error_type.__name__ for row in report["videos"])
    assert "PRIVATE DIAGNOSTIC" not in (output / "index.json").read_text(encoding="utf-8")
    assert list(output.glob("*.npz")) == []


def test_dataset_refuses_to_overwrite_old_results(dataset, monkeypatch):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    output = root / "artifacts" / "old"
    output.mkdir(parents=True)
    old = output / "index.json"
    old.write_bytes(b"PRESERVE EXACTLY")
    monkeypatch.setattr(module, "snapshot_fingerprint", Mock(side_effect=AssertionError("must not load")))
    with pytest.raises(ValueError, match="never overwritten"):
        module.extract_dataset(source, "unused", output, artifact_root=root)
    assert old.read_bytes() == b"PRESERVE EXACTLY"


@pytest.mark.parametrize("phase", ["fingerprint", "encoder", "extract", "index"])
@pytest.mark.parametrize("existing_empty", [False, True])
def test_dataset_rejects_concurrent_writers_without_releasing_owner(dataset, monkeypatch, phase, existing_empty):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    features = importlib.import_module("ml.baseline.features")
    np = pytest.importorskip("numpy")
    output = root / "artifacts" / "concurrent"
    if existing_empty:
        output.mkdir(parents=True)
    entered = threading.Event()
    release = threading.Event()
    owner_ident = None

    def pause(boundary):
        if (boundary == phase and threading.get_ident() == owner_ident
                and not entered.is_set()):
            entered.set()
            assert release.wait(10), "owner was not released"

    def fingerprint(weights):
        pause("fingerprint")
        return "a" * 64

    def encoder(*args, **kwargs):
        pause("encoder")
        return object()

    def extract(video, weights, cache, **kwargs):
        pause("extract")
        features.save_cache(cache, video.name, [0.0], np.ones((1, 768)),
                            {"extraction": {"cap_reached": False}})

    original_write = module.write_json

    def write_index(path, report):
        pause("index")
        original_write(path, report)

    fingerprint_mock = Mock(side_effect=fingerprint)
    encoder_mock = Mock(side_effect=encoder)
    monkeypatch.setattr(module, "snapshot_fingerprint", fingerprint_mock)
    monkeypatch.setattr(features, "SiglipEncoder", encoder_mock)
    monkeypatch.setattr(features, "extract_video", extract)
    monkeypatch.setattr(module, "write_json", write_index)

    def run_owner():
        nonlocal owner_ident
        owner_ident = threading.get_ident()
        return module.extract_dataset(source, "unused", output, artifact_root=root)

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(run_owner)
        try:
            assert entered.wait(5), "owner did not reach the controlled boundary"
            before = set(output.iterdir()) if output.exists() else set()
            # A rejected contender must not remove the active owner's lock: retry it.
            for _ in range(2):
                contender = pool.submit(module.extract_dataset, source, "unused", output,
                                        artifact_root=root)
                with pytest.raises(ValueError, match="already.*extraction"):
                    contender.result(timeout=2)
                assert set(output.iterdir()) == before
            fingerprint_mock.assert_called_once()
            assert encoder_mock.call_count == (0 if phase == "fingerprint" else 1)
        finally:
            release.set()
            report = owner.result(timeout=5)

    assert report["successful_videos"] == 3
    assert report["failed_videos"] == 0
    assert module.read_json(output / "index.json") == report
    assert {path.name for path in output.iterdir()} == {
        "index.json", *(row["cache_file"] for row in report["videos"])}


@pytest.mark.parametrize("phase", ["fingerprint", "encoder"])
def test_dataset_releases_output_ownership_after_initialization_failure(dataset, monkeypatch, phase):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    features = importlib.import_module("ml.baseline.features")
    output = root / "artifacts" / "retry"
    output.mkdir(parents=True)
    fingerprint = Mock(return_value="a" * 64)
    encoder = Mock(return_value=object())
    (fingerprint if phase == "fingerprint" else encoder).side_effect = RuntimeError("synthetic failure")
    monkeypatch.setattr(module, "snapshot_fingerprint", fingerprint)
    monkeypatch.setattr(features, "SiglipEncoder", encoder)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        module.extract_dataset(source, "unused", output, artifact_root=root)
    assert list(output.iterdir()) == []

    fingerprint.side_effect = None
    encoder.side_effect = None
    monkeypatch.setattr(features, "extract_video", Mock(side_effect=MemoryError("synthetic failure")))
    report = module.extract_dataset(source, "unused", output, artifact_root=root)
    assert report["failed_videos"] == 3
    assert {path.name for path in output.iterdir()} == {"index.json"}


def test_snapshot_fingerprint_tracks_processor_and_checkpoint_bytes(tmp_path):
    module = importlib.import_module("ml.baseline.dataset")
    (tmp_path / "config.json").write_text('{"model_type":"siglip"}', encoding="utf-8")
    processor = tmp_path / "preprocessor_config.json"
    processor.write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"ARTIFICIAL WEIGHTS, NOT A MODEL")
    first = module.snapshot_fingerprint(tmp_path)
    assert first == module.snapshot_fingerprint(tmp_path)
    processor.write_text('{"size":224}', encoding="utf-8")
    assert first != module.snapshot_fingerprint(tmp_path)


@pytest.mark.parametrize("raw", ["//fixture.invalid/share/videos", r"\\fixture.invalid\share\videos"])
def test_dataset_rejects_network_input_without_probing_it(tmp_path, monkeypatch, raw):
    module = importlib.import_module("ml.baseline.dataset")
    from pathlib import Path
    monkeypatch.setattr(Path, "resolve", Mock(side_effect=AssertionError("network probe forbidden")))
    with pytest.raises(ValueError, match="network shares"):
        module.extract_dataset(raw, "unused", tmp_path / "artifacts" / "forbidden", artifact_root=tmp_path)


def test_dataset_checks_each_local_path_before_following_links(dataset, monkeypatch):
    root, source = dataset
    module = importlib.import_module("ml.baseline.dataset")
    features = importlib.import_module("ml.baseline.features")
    from pathlib import Path
    original = features._local_path
    def guard(value):
        if Path(value).suffix.lower() == ".mov":
            raise ValueError("Artificial unsafe link")
        return original(value)
    monkeypatch.setattr(features, "_local_path", guard)
    monkeypatch.setattr(Path, "is_file", Mock(side_effect=AssertionError("must reject before probing")))
    monkeypatch.setattr(module, "snapshot_fingerprint", lambda weights: "a" * 64)
    monkeypatch.setattr(features, "SiglipEncoder", Mock(return_value=object()))
    report = module.extract_dataset(source, "unused", root / "artifacts" / "links", artifact_root=root)
    assert report["failed_videos"] == 3
    assert all(row["error_class"] == "ValueError" for row in report["videos"])
