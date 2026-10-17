"""Final reviewer regressions using synthetic metadata and structural files only."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ml.baseline import features
from ml.baseline.core import split_videos
from ml.eval import score


@pytest.mark.parametrize("sentinel", ["unknown", "неизвестно", "unassigned", "UnAssigned"])
@pytest.mark.parametrize("side", ["train", "test"])
@pytest.mark.parametrize("layout", ["rows", "test", "holdout"])
def test_evaluation_rejects_unknown_camera_ids_on_either_split(sentinel, side, layout):
    labels = [{"video": name, "positive": False, "timestamp": None}
              for name in ("synthetic-train", "synthetic-test")]
    rows = [{"video": "synthetic-train", "subset": "train", "angle_id": "camera-a"},
            {"video": "synthetic-test", "subset": "test", "angle_id": "camera-b"}]
    rows[0 if side == "train" else 1]["angle_id"] = sentinel
    split = rows if layout == "rows" else {"train": [rows[0]], layout: [rows[1]]}
    with pytest.raises(ValueError, match="known angle_id"):
        score(labels, [], split=split)


@pytest.mark.parametrize("sentinel", ["unknown", "неизвестно", "unassigned", "UnAssigned",
                                     " unassigned ", " UNKNOWN ", " неизвестно "])
@pytest.mark.parametrize("side", ["train", "holdout"])
def test_baseline_rejects_unknown_camera_ids_on_either_split(sentinel, side):
    angles = {"train": "camera-a", "holdout": "camera-b"}
    angles[side] = sentinel
    rows = [{"video_id": "synthetic-" + subset, "angle_id": angle, "label": 0}
            for subset, angle in angles.items()]
    with pytest.raises(ValueError, match="Unknown/empty angle IDs"):
        split_videos(rows, [angles["train"]], [angles["holdout"]])


@pytest.fixture
def siglip_snapshot(tmp_path):
    """Structural placeholders only: never pass these bytes to a model loader."""
    root = tmp_path / "siglip"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"siglip"}', encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return root


def write_index(root, payload):
    (root / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_baseline_accepts_complete_local_sharded_checkpoint_fixture(siglip_snapshot):
    root = siglip_snapshot
    (root / "nested").mkdir()
    for name in ("shard-1.safetensors", "nested/shard-2.safetensors"):
        (root / name).write_bytes(b"STRUCTURAL FIXTURE ONLY; NOT MODEL WEIGHTS")
    write_index(root, {"metadata": {}, "weight_map": {
        "vision_model.first.weight": "shard-1.safetensors",
        "vision_model.second.weight": "nested/shard-2.safetensors",
        "vision_model.second.bias": "nested/shard-2.safetensors"}})
    assert features.local_weights(root) == root.resolve()


@pytest.mark.parametrize("payload", [
    [], None, {}, {"weight_map": []}, {"weight_map": None}, {"weight_map": {}},
    {"weight_map": "shard.safetensors"},
    {"weight_map": {"": "shard.safetensors"}},
    {"weight_map": {" ": "shard.safetensors"}},
    *[{"weight_map": {"vision_model.synthetic.weight": value}}
      for value in (None, [], {}, 17, "", " ", "shard.bin", "missing.safetensors")],
])
def test_baseline_malformed_shard_indexes_raise_valueerror(siglip_snapshot, payload):
    (siglip_snapshot / "shard.safetensors").write_bytes(b"STRUCTURAL SHARD FIXTURE")
    write_index(siglip_snapshot, payload)
    # AttributeError/TypeError must not escape for malformed JSON shapes or map values.
    with pytest.raises(ValueError):
        features.local_weights(siglip_snapshot)


@pytest.mark.parametrize("reference", [
    "//fixture.invalid/share/weights.safetensors",
    r"\\fixture.invalid\share\weights.safetensors",
    r"\\?\UNC\fixture.invalid\share\weights.safetensors",
    "/outside.safetensors", r"\outside.safetensors",
    "C:/outside.safetensors", r"C:\outside.safetensors", "C:outside.safetensors",
    "https://fixture.invalid/weights.safetensors", "smb://fixture.invalid/weights.safetensors",
    "file:/outside.safetensors", "../outside.safetensors", r"..\outside.safetensors",
    "nested/../../outside.safetensors", "nested/../shard.safetensors",
    "invalid\x00.safetensors",
])
def test_baseline_unsafe_shards_rejected_before_resolution_or_probe(siglip_snapshot, reference):
    root = siglip_snapshot
    write_index(root, {"metadata": {}, "weight_map": {"vision_model.synthetic.weight": reference}})
    original_resolve, original_is_file = Path.resolve, Path.is_file
    allowed_probes = {root / name for name in ("config.json", "preprocessor_config.json",
                                              "model.safetensors", "model.safetensors.index.json")}

    def guarded_resolve(path, *args, **kwargs):
        assert path == root, "Unsafe shard must be rejected before path resolution"
        return original_resolve(path, *args, **kwargs)

    def guarded_is_file(path):
        assert path in allowed_probes, "Unsafe shard must be rejected before file probe"
        return original_is_file(path)

    with patch.object(Path, "resolve", autospec=True, side_effect=guarded_resolve), \
         patch.object(Path, "is_file", autospec=True, side_effect=guarded_is_file):
        with pytest.raises(ValueError):
            features.local_weights(root)


@pytest.mark.parametrize("network_target", [False, True])
def test_baseline_resolved_shard_escape_rejected_before_file_probe(siglip_snapshot, network_target):
    root = siglip_snapshot
    link = root / "linked.safetensors"
    outside = (Path("//fixture.invalid/share/weights.safetensors") if network_target
               else root.parent / "outside.safetensors")
    write_index(root, {"weight_map": {"vision_model.synthetic.weight": link.name}})
    original_resolve, original_is_file = Path.resolve, Path.is_file

    def resolved_link(path, *args, **kwargs):
        # Deterministic symlink-resolution boundary, including a UNC target never accessed.
        return outside if path == link else original_resolve(path, *args, **kwargs)

    def guarded_is_file(path):
        assert path not in (link, outside), "Escaped symlink target must never be probed"
        return original_is_file(path)

    with patch.object(Path, "resolve", autospec=True, side_effect=resolved_link), \
         patch.object(Path, "is_file", autospec=True, side_effect=guarded_is_file):
        with pytest.raises(ValueError):
            features.local_weights(root)


def test_baseline_real_local_symlink_escape_rejected_before_file_probe(siglip_snapshot):
    root = siglip_snapshot
    outside = root.parent / "outside.safetensors"
    outside.write_bytes(b"STRUCTURAL OUTSIDE FIXTURE; MUST NOT BE LOADED")
    link = root / "linked.safetensors"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("OS does not permit local symlink fixtures: " + str(exc))
    write_index(root, {"weight_map": {"vision_model.synthetic.weight": link.name}})
    original_is_file = Path.is_file

    def guarded_is_file(path):
        assert path not in (link, outside), "Escaped symlink target must never be probed"
        return original_is_file(path)

    with patch.object(Path, "is_file", autospec=True, side_effect=guarded_is_file):
        with pytest.raises(ValueError):
            features.local_weights(root)


def test_baseline_preserves_local_single_file_checkpoint_fixture(siglip_snapshot):
    (siglip_snapshot / "model.safetensors").write_bytes(b"STRUCTURAL SINGLE-FILE FIXTURE ONLY")
    assert features.local_weights(siglip_snapshot) == siglip_snapshot.resolve()
