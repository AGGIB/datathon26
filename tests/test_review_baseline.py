"""PR30 regressions: synthetic identities/features only, never private dataset input."""
import copy
import hashlib

import pytest

from ml.baseline import classifier
from ml.baseline.core import FEATURE_SPEC, split_videos
from ml.labels import normalize_video_name


def manifest_row(name, angle, label=0, times=None, digest=None):
    row = {"video_id": name, "angle_id": angle, "label": label,
           "content_sha256": digest or hashlib.sha256(name.encode("utf-8")).hexdigest()}
    if times is not None:
        row["positive_times_s"] = times
    return row


def diagnostic_rows():
    return [manifest_row("synthetic-one.mov", "train"),
            manifest_row("synthetic-two.mov", "test")]


@pytest.mark.parametrize("missing", ["all", "partial"])
def test_every_manifest_row_requires_content_hash_even_without_opt_in(missing):
    rows = diagnostic_rows()
    for row in rows if missing == "all" else rows[:1]:
        row.pop("content_sha256")
    with pytest.raises(ValueError, match="SHA256"):
        split_videos(rows, ["train"], ["test"])


@pytest.mark.parametrize("invalid", [None, "", "a" * 63, "A" * 64, "a" * 64 + "\n", 17, []])
def test_manifest_content_hash_must_be_literal_lowercase_sha256(invalid):
    rows = copy.deepcopy(diagnostic_rows())
    rows[0]["content_sha256"] = invalid
    with pytest.raises(ValueError, match="SHA256"):
        split_videos(rows, ["train"], ["test"])


@pytest.mark.parametrize("alias", ["Х-1.MOV", "х-1.mov", "x-1", "X-1.mp4"])
def test_manifest_canonical_alias_collisions_are_rejected(alias):
    rows = [manifest_row("X-1.mov", "train"), manifest_row(alias, "test")]
    assert normalize_video_name(rows[0]["video_id"]) == normalize_video_name(alias)
    with pytest.raises(ValueError, match="canonical"):
        split_videos(rows, ["train"], ["test"])


@pytest.mark.parametrize("token", [" X-1.mov", "X-1.mov ", "../X-1.mov", "folder/X-1.mov",
                                   r"folder\X-1.mov", "C:X-1.mov", "X-1.mov\n",
                                   "X-1\x00.mov", "X-1 .mov", "X -1.mov", "X?1.mov", ".", "..", "",
                                   "fixture.", "fixture.mov.", "CON", "CON.mov", "aux.txt",
                                   "COM1.mp4", "LPT9", "COM¹.mov"])
def test_malformed_source_id_is_rejected_before_canonicalization(token):
    rows = diagnostic_rows()
    rows[0]["video_id"] = token
    with pytest.raises(ValueError, match="video_id"):
        split_videos(rows, ["train"], ["test"])


def test_manifest_split_preserves_literal_source_identifiers():
    rows = [manifest_row("х-1.mov", "train"), manifest_row("X-2.MOV", "test")]
    result = split_videos(rows, ["train"], ["test"])
    assert result["train"][0]["video_id"] == "х-1.mov"
    assert rows[0]["video_id"] == "х-1.mov"


def three_way_rows():
    return [manifest_row("train-pos.mov", "train", 1, [1.0]),
            manifest_row("train-neg.mov", "train"),
            manifest_row("valid.mov", "valid"),
            manifest_row("test.mov", "test")]


def test_explicit_three_way_angle_split_keeps_all_source_rows():
    rows = three_way_rows()
    original = copy.deepcopy(rows)
    split = split_videos(rows, ["train"], ["test"], validation_angles=["valid"])
    assert set(split) == {"train", "validation", "holdout"}
    assert split["train"] == rows[:2]
    assert split["validation"] == rows[2:3]
    assert split["holdout"] == rows[3:]
    assert rows == original


@pytest.mark.parametrize("first,second", [(0, 2), (0, 3), (2, 3)])
def test_duplicate_content_cannot_cross_any_of_three_partitions(first, second):
    rows = three_way_rows()
    rows[second]["content_sha256"] = rows[first]["content_sha256"]
    # Equal labels isolate the identity leak from label-conflict rejection.
    rows[first]["label"] = rows[second]["label"] = 0
    rows[first].pop("positive_times_s", None)
    with pytest.raises(ValueError, match="content cannot cross"):
        split_videos(rows, ["train"], ["test"], validation_angles=["valid"])


@pytest.mark.parametrize("validation", [[], ["train"], ["test"], ["absent"],
                                       ["unknown"], [" valid"], ["valid "]])
def test_validation_angles_must_be_known_nonempty_disjoint_and_complete(validation):
    with pytest.raises(ValueError):
        split_videos(three_way_rows(), ["train"], ["test"], validation_angles=validation)


def test_identical_content_with_conflicting_binary_labels_is_rejected_within_train():
    rows = three_way_rows()
    rows[1]["content_sha256"] = rows[0]["content_sha256"]
    with pytest.raises(ValueError, match="conflicting binary labels"):
        split_videos(rows, ["train"], ["test"], validation_angles=["valid"])


def synthetic_caches(rows):
    np = pytest.importorskip("numpy")
    caches = []
    for i, row in enumerate(rows):
        values = [[1e6, 1], [8, 1]] if row["label"] else [[-2 - i, 1], [-3 - i, 1]]
        caches.append({"video_id": row["video_id"], "timestamps_s": np.array([0.0, 1.0]),
                       "embeddings": np.array(values, dtype=np.float32),
                       "metadata": {"schema_version": 1, "feature_spec": dict(FEATURE_SPEC),
                                    "content_sha256": row["content_sha256"]}})
    return caches


def fit(rows=None, caches=None, **kwargs):
    rows = three_way_rows() if rows is None else rows
    caches = synthetic_caches(rows) if caches is None else caches
    return classifier.train_classifier(caches, rows, ["train"], ["test"],
                                       validation_angles=["valid"], positive_radius_s=0.01,
                                       epochs=5, **kwargs)


def test_two_way_diagnostic_split_cannot_enable_model_fitting():
    rows = [r for r in three_way_rows() if r["angle_id"] != "valid"]
    assert set(split_videos(rows, ["train"], ["test"])) == {"train", "holdout"}
    with pytest.raises(ValueError, match="validation"):
        classifier.train_classifier(synthetic_caches(rows), rows, ["train"], ["test"], epochs=5)


def test_fitting_records_explicit_three_way_partition_and_leaves_other_groups_out_of_fit():
    np = pytest.importorskip("numpy")
    rows = three_way_rows()
    caches = synthetic_caches(rows)
    before = copy.deepcopy(caches)
    model = fit(rows, caches)
    meta = model["metadata"]
    assert meta["validation_video_ids"] == [rows[2]["video_id"]]
    assert meta["validation_angles"] == ["valid"]
    assert meta["holdout_video_ids"] == [rows[3]["video_id"]]
    assert meta["train_video_ids"] == [r["video_id"] for r in rows[:2]]
    assert meta["n_supervised_train_frames"] == 3
    for cache, original in zip(caches, before):
        np.testing.assert_array_equal(cache["embeddings"], original["embeddings"])
    for cache in caches[2:]:
        cache["embeddings"] *= 10000
    again = fit(rows, caches)
    for key in ("mean", "scale", "weights", "intercept"):
        np.testing.assert_array_equal(model[key], again[key])
    assert classifier.validate_model(model)["metadata"] == meta


def test_legacy_model_without_validation_partition_is_not_loadable():
    np = pytest.importorskip("numpy")
    legacy = {"mean": np.zeros(2), "scale": np.ones(2), "weights": np.zeros(2),
              "intercept": np.asarray(0.0),
              "metadata": {"schema_version": 1, "feature_spec": dict(FEATURE_SPEC),
                           "train_video_ids": ["train.mov"], "holdout_video_ids": ["test.mov"],
                           "train_angles": ["train"], "holdout_angles": ["test"],
                           "threshold": 0.5, "window_s": 2.0}}
    with pytest.raises(ValueError, match="split|validation|schema"):
        classifier.validate_model(legacy)


def test_canonical_manifest_cache_join_preserves_both_literal_source_ids():
    rows = three_way_rows()
    rows[0]["video_id"] = "Х-1.MOV"
    caches = synthetic_caches(rows)
    caches[0]["video_id"] = "x-1.mov"
    model = fit(rows, caches)
    assert model["metadata"]["train_video_ids"][0] == "Х-1.MOV"
    assert model["metadata"]["cache_video_ids"]["X-1"] == "x-1.mov"
    assert classifier.predict_cache(model, caches[0])["seen_in_training"] is True
    assert caches[0]["video_id"] == "x-1.mov"


def test_cache_alias_collisions_are_rejected_even_if_literal_ids_differ():
    rows = three_way_rows()
    caches = synthetic_caches(rows)
    caches.append(dict(caches[0], video_id="TRAIN-POS.MOV"))
    with pytest.raises(ValueError, match="Duplicate.*canonical"):
        fit(rows, caches)


@pytest.mark.parametrize("index", [0, 2, 3])
@pytest.mark.parametrize("digest", [None, "bad", "A" * 64, "a" * 64])
def test_manifest_hash_must_match_each_available_extraction_cache_hash(index, digest):
    rows = three_way_rows()
    caches = synthetic_caches(rows)
    caches[index]["metadata"]["content_sha256"] = digest
    with pytest.raises(ValueError, match="SHA256|content hash"):
        fit(rows, caches)


def test_missing_cache_hash_is_explicitly_unverified_not_asserted_from_manifest():
    rows = three_way_rows()
    caches = synthetic_caches(rows)
    caches[0]["metadata"].pop("content_sha256")
    model = fit(rows, caches)
    status = model["metadata"]["cache_content_hash_verification"]
    assert status["TRAIN-POS"] == "unavailable"
    assert status["TRAIN-NEG"] == "matched_extraction_metadata"
    assert model["metadata"]["content_sha256_by_video"]["TRAIN-POS"] == rows[0]["content_sha256"]


def test_training_counts_exact_content_once_with_union_of_alternative_anchors():
    np = pytest.importorskip("numpy")
    rows = three_way_rows()
    duplicate = dict(rows[0], video_id="duplicate.mov", positive_times_s=[0.0])
    rows.append(duplicate)
    caches = synthetic_caches(rows[:4])
    caches.append(copy.deepcopy(caches[0]))
    caches[-1]["video_id"] = duplicate["video_id"]
    model = fit(rows, caches)
    assert model["metadata"]["n_supervised_train_frames"] == 4
    assert model["metadata"]["n_unique_train_contents"] == 2
    merged = copy.deepcopy(rows[:4])
    merged[0]["positive_times_s"] = [0.0, 1.0]
    reference = fit(merged, caches[:4])
    for key in ("mean", "scale", "weights", "intercept"):
        np.testing.assert_array_equal(model[key], reference[key])


def test_identical_content_cannot_choose_arbitrary_disagreeing_feature_caches():
    rows = three_way_rows()
    rows.append(dict(rows[0], video_id="duplicate.mov"))
    caches = synthetic_caches(rows[:4])
    caches.append(copy.deepcopy(caches[0]))
    caches[-1]["video_id"] = "duplicate.mov"
    caches[-1]["embeddings"][0, 0] += 10
    with pytest.raises(ValueError, match="Identical content.*cache"):
        fit(rows, caches)


def test_duplicate_same_anchor_does_not_double_training_weight():
    np = pytest.importorskip("numpy")
    rows = three_way_rows()
    caches = synthetic_caches(rows)
    reference = fit(rows, caches)
    rows.append(dict(rows[0], video_id="duplicate.mov"))
    caches.append(copy.deepcopy(caches[0]))
    caches[-1]["video_id"] = "duplicate.mov"
    model = fit(rows, caches)
    assert model["metadata"]["n_supervised_train_frames"] == 3
    for key in ("mean", "scale", "weights", "intercept"):
        np.testing.assert_array_equal(model[key], reference[key])


def test_evaluation_repeats_three_way_partition_and_checks_content_identity():
    rows = three_way_rows()
    caches = synthetic_caches(rows)
    model = fit(rows, caches)
    report = classifier.evaluate_holdout(model, caches[3:], rows, ["train"], ["test"],
                                         validation_angles=["valid"])
    assert report["n_videos"] == 1
    assert report["validation_angles"] == ["valid"]
    changed = copy.deepcopy(rows)
    changed[2]["content_sha256"] = hashlib.sha256(b'changed validation bytes').hexdigest()
    with pytest.raises(ValueError, match="content|partition"):
        classifier.evaluate_holdout(model, caches[3:], changed, ["train"], ["test"],
                                     validation_angles=["valid"])
    caches[3]["metadata"]["content_sha256"] = hashlib.sha256(b'changed holdout bytes').hexdigest()
    with pytest.raises(ValueError, match="content hash"):
        classifier.evaluate_holdout(model, caches[3:], rows, ["train"], ["test"],
                                     validation_angles=["valid"])


def test_loaded_model_rejects_hash_leakage_and_canonical_id_collisions():
    model = fit()
    changed = copy.deepcopy(model)
    hashes = changed["metadata"]["content_sha256_by_video"]
    hashes["TEST"] = hashes["TRAIN-POS"]
    with pytest.raises(ValueError, match="content|split"):
        classifier.validate_model(changed)
    changed = copy.deepcopy(model)
    changed["metadata"]["holdout_video_ids"] = ["TRAIN-POS.MOV"]
    with pytest.raises(ValueError, match="canonical|split"):
        classifier.validate_model(changed)


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_train_evaluate_cli_requires_explicit_validation_angle(command):
    from ml.baseline.__main__ import argument_parser
    args = [command, "--manifest", "synthetic.json", "--train-angle", "train", "--holdout-angle", "test",
            "--cache", "synthetic.npz", "--model", "synthetic-model.npz", "--report", "synthetic-report.json"]
    with pytest.raises(SystemExit):
        argument_parser().parse_args(args)
    assert argument_parser().parse_args(args + ["--validation-angle", "valid"]).validation_angle == ["valid"]
