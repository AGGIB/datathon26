"""Baseline contracts. Embeddings generated here are TEST FIXTURES, not model results."""
import importlib
import importlib.util
import math
import json
import os
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

SCRATCH = Path(os.environ.get("TMPDIR", "C:/Users/esenb/AppData/Local/hermes/cache/scratch"))


def baseline_module(name):
    try:
        return importlib.import_module("ml.baseline." + name)
    except ModuleNotFoundError as exc:
        raise AssertionError("Missing baseline behavior: " + name) from exc


class SamplingTests(unittest.TestCase):
    def test_sampling_selects_actual_pts_with_a_hard_frame_bound(self):
        core = baseline_module("core")
        actual_pts = [0.03, 0.12, 0.49, 0.57, 0.96, 1.12, 1.71]
        self.assertEqual(core.sample_timestamps(actual_pts, fps=2, max_frames=3),
                         [0.03, 0.57, 1.12])
        for fps, limit in [(0, 3), (math.nan, 3), (1, 0)]:
            with self.assertRaises(ValueError):
                core.sample_timestamps(actual_pts, fps=fps, max_frames=limit)
        with self.assertRaises(ValueError):
            core.sample_timestamps([0.2, 0.1], fps=10, max_frames=10)


class SplitTests(unittest.TestCase):
    def test_explicit_angles_keep_whole_videos_and_literal_ids(self):
        core = baseline_module("core")
        rows = [
            {"video_id": "камера_A.MOV", "angle_id": "ракурс_1", "label": 1,
             "positive_times_s": [1.25]},
            {"video_id": "камера_A_2.MOV", "angle_id": "ракурс_1", "label": 0},
            {"video_id": "camera_A.MOV", "angle_id": "angle_2", "label": 0},
        ]
        split = core.split_videos(rows, ["ракурс_1"], ["angle_2"])
        self.assertEqual([r["video_id"] for r in split["train"]],
                         ["камера_A.MOV", "камера_A_2.MOV"])
        self.assertEqual([r["video_id"] for r in split["holdout"]], ["camera_A.MOV"])
        for train, holdout in [([], []), (["ракурс_1"], []),
                               (["ракурс_1"], ["ракурс_1"]),
                               (["ракурс_1"], ["angle_other"])]:
            with self.assertRaises(ValueError):
                core.split_videos(rows, train, holdout)
        for broken in [rows + [rows[0]],
                       [dict(rows[0], angle_id=None), rows[2]],
                       [dict(rows[0], positive_times_s=[]), rows[2]],
                       [dict(rows[0], label="1"), rows[2]],
                       [dict(rows[0], positive_times_s=[math.nan]), rows[2]]]:
            with self.assertRaises(ValueError):
                core.split_videos(broken, ["ракурс_1"], ["angle_2"])


class SegmentTests(unittest.TestCase):
    def test_positive_video_only_supervises_known_event_neighborhood(self):
        core = baseline_module("core")
        self.assertTrue(hasattr(core, "segment_labels"), "segment labels are missing")
        row = {"video_id": "камера.MOV", "label": 1, "positive_times_s": [2.2]}
        self.assertEqual(core.segment_labels(row, [0, 1.9, 2.2, 2.6, 8], radius_s=0.4),
                         [None, 1, 1, 1, None])
        self.assertEqual(core.segment_labels({"label": 0}, [0, 2, 8], radius_s=1), [0, 0, 0])
        with self.assertRaises(ValueError):
            core.segment_labels(row, [8, 9], radius_s=0.1)
        for radius in [-1, math.nan]:
            with self.assertRaises(ValueError):
                core.segment_labels(row, [2.2], radius_s=radius)


class MetadataTests(unittest.TestCase):
    def test_outputs_are_confined_to_artifacts_and_json_preserves_literal_unicode(self):
        core = baseline_module("core")
        self.assertTrue(hasattr(core, "artifact_path"), "artifact path contract is missing")
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            target = core.artifact_path(root / "artifacts" / "камера_A.json", root=root)
            payload = {"video_id": "камера_A.MOV", "description": "требуется проверка"}
            core.write_json(target, payload)
            self.assertEqual(core.read_json(target), payload)
            self.assertIn("камера_A.MOV", target.read_text(encoding="utf-8"))
            for bad in [root / "cache.npz", root / "artifacts" / ".." / "leak.npz"]:
                with self.assertRaises(ValueError):
                    core.artifact_path(bad, root=root)


class PredictionTests(unittest.TestCase):
    def test_window_verdict_has_real_peak_time_and_is_only_a_candidate(self):
        core = baseline_module("core")
        self.assertTrue(hasattr(core, "aggregate_predictions"), "window aggregation is missing")
        result = core.aggregate_predictions("камера_A.MOV", [0.03, 0.7, 2.12, 2.8, 4.1],
                                            [0.1, 0.3, 0.9, 0.9, 0.4],
                                            threshold=0.8, window_s=2)
        self.assertEqual(result["verdict"], "candidate")
        self.assertEqual(result["candidate_time_s"], 2.12)
        self.assertEqual(result["candidate_timecode"], "00:00:02.120")
        self.assertEqual(result["description"], "Кандидат baseline; требуется проверка")
        self.assertIsNone(result["final_legal_label"])
        self.assertEqual(len(result["windows"]), 3)
        negative = core.aggregate_predictions("x", [0.03], [0.2], threshold=0.8)
        self.assertEqual(negative["verdict"], "no_candidate")
        self.assertIsNone(negative["candidate_time_s"])
        for times, probs in [([], []), ([0.1], []), ([0.1], [math.nan]),
                              ([0.1], [2]), ([0.2, 0.1], [0.1, 0.2])]:
            with self.assertRaises(ValueError):
                core.aggregate_predictions("x", times, probs)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional; use py -3.10 for numeric tests")
class CacheTests(unittest.TestCase):
    def test_safe_numeric_cache_roundtrip_rejects_corrupt_or_empty_features(self):
        features = baseline_module("features")
        np = importlib.import_module("numpy")
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            path = Path(tmp) / "artifacts" / "камера.npz"
            embeddings = np.array([[0.2, 0.5], [0.1, 0.9]], dtype=np.float32)  # TEST FIXTURE
            features.save_cache(path, "камера_A.MOV", [0.03, 0.57], embeddings)
            cached = features.load_cache(path)
            self.assertEqual(cached["video_id"], "камера_A.MOV")
            self.assertEqual(cached["metadata"]["feature_spec"], baseline_module("core").FEATURE_SPEC)
            np.testing.assert_allclose(cached["embeddings"], embeddings)
            np.testing.assert_allclose(cached["timestamps_s"], [0.03, 0.57])
            with np.load(path, allow_pickle=False) as safe:
                self.assertTrue(all(safe[key].dtype.kind != "O" for key in safe.files))
            for times, values in [([], np.zeros((0, 2))), ([0.1], [[math.nan, 1]]),
                                   ([0.1], [["text", "bad"]]), ([0.1, 0.1], embeddings),
                                   ([0.1], embeddings)]:
                with self.assertRaises(ValueError):
                    features.save_cache(path, "x", times, values)
            np.savez(path, embeddings=np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                features.load_cache(path)


def training_fixture():
    """Tiny SYNTHETIC TEST FIXTURE; never a reported model benchmark."""
    np = importlib.import_module("numpy")
    core = baseline_module("core")
    rows = [
        {"video_id": "train_нарушение.MOV", "angle_id": "train_view", "label": 1,
         "positive_times_s": [2.12]},
        {"video_id": "train_negative.MOV", "angle_id": "train_view", "label": 0},
        {"video_id": "hold_нарушение.MOV", "angle_id": "hold_view", "label": 1,
         "positive_times_s": [2.12]},
        {"video_id": "hold_negative.MOV", "angle_id": "hold_view", "label": 0},
    ]
    values = [[[1e6, 1], [8, 1]], [[-2, 1], [-3, 1]],
              [[1000, 1], [2000, 1]], [[-1000, 1], [-2000, 1]]]
    caches = [{"video_id": row["video_id"], "timestamps_s": np.array([0.03, 2.12]),
               "embeddings": np.array(value, dtype=np.float32),
               "metadata": {"schema_version": 1, "feature_spec": dict(core.FEATURE_SPEC)}}
              for row, value in zip(rows, values)]
    return rows, caches


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional; use py -3.10 for numeric tests")
class ClassifierTests(unittest.TestCase):
    def test_holdout_report_requires_original_explicit_split_and_complete_predictions(self):
        classifier = baseline_module("classifier")
        self.assertTrue(hasattr(classifier, "evaluate_holdout"), "grouped holdout reporting is missing")
        rows, caches = training_fixture()
        model = classifier.train_classifier(caches, rows, ["train_view"], ["hold_view"], epochs=60)
        report = classifier.evaluate_holdout(model, caches[2:], rows, ["train_view"], ["hold_view"])
        self.assertEqual(report["n_videos"], 2)
        self.assertEqual({p["video_id"] for p in report["predictions"]},
                         {r["video_id"] for r in rows[2:]})
        for predictions in [caches[2:3], caches, caches[2:] + [caches[2]]]:
            with self.assertRaises(ValueError):
                classifier.evaluate_holdout(model, predictions, rows, ["train_view"], ["hold_view"])
        with self.assertRaises(ValueError):
            classifier.evaluate_holdout(model, caches[2:], rows, [], [])
        with self.assertRaises(ValueError):
            classifier.evaluate_holdout(model, caches[:2], rows, ["hold_view"], ["train_view"])
        result = classifier.predict_cache(model, caches[2])
        self.assertEqual(result["video_id"], rows[2]["video_id"])
        self.assertIsNone(result["final_legal_label"])

    def test_model_roundtrip_is_safe_npz_and_rejects_invalid_scaling(self):
        classifier = baseline_module("classifier")
        self.assertTrue(hasattr(classifier, "save_model"), "safe model persistence is missing")
        np = importlib.import_module("numpy")
        rows, caches = training_fixture()
        model = classifier.train_classifier(caches, rows, ["train_view"], ["hold_view"], epochs=50)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            path = Path(tmp) / "model.npz"
            classifier.save_model(path, model)
            loaded = classifier.load_model(path)
            np.testing.assert_array_equal(classifier.predict_probabilities(loaded, [[3, 1]]),
                                          classifier.predict_probabilities(model, [[3, 1]]))
            self.assertEqual(loaded["metadata"], model["metadata"])
            with np.load(path, allow_pickle=False) as safe:
                self.assertTrue(all(safe[key].dtype.kind != "O" for key in safe.files))
            model["scale"][0] = 0
            with self.assertRaises(ValueError):
                classifier.save_model(path, model)
            np.savez(path, weights=np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                classifier.load_model(path)

    def test_training_uses_only_train_supervised_segments_and_explicit_video_groups(self):
        classifier = baseline_module("classifier")
        np = importlib.import_module("numpy")
        rows, caches = training_fixture()
        model = classifier.train_classifier(caches, rows, ["train_view"], ["hold_view"],
                                            positive_radius_s=0.1, seed=17, epochs=600)
        np.testing.assert_allclose(model["mean"], [1, 1])
        self.assertEqual(model["scale"][1], 1)
        self.assertEqual(set(model["metadata"]["train_video_ids"]), {r["video_id"] for r in rows[:2]})
        scores = classifier.predict_probabilities(model, [[8, 1], [-3, 1]])
        self.assertGreater(scores[0], 0.9)
        self.assertLess(scores[1], 0.1)
        caches[2]["embeddings"] *= -999  # held-out data must not affect fitting
        second = classifier.train_classifier(caches, rows, ["train_view"], ["hold_view"],
                                             positive_radius_s=0.1, seed=17, epochs=600)
        for key in ("mean", "scale", "weights", "intercept"):
            np.testing.assert_array_equal(model[key], second[key])
        for bad_caches in [caches[:-1], caches + [caches[0]]]:
            with self.assertRaises(ValueError):
                classifier.train_classifier(bad_caches, rows, ["train_view"], ["hold_view"])
        with self.assertRaises(ValueError):
            classifier.train_classifier(caches, rows, [], [])
        with self.assertRaises(ValueError):
            classifier.predict_probabilities(model, [[math.nan, 1]])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not on PATH")
class DecoderTests(unittest.TestCase):
    def test_real_vfr_decoding_uses_selected_frame_pts_and_hard_bound(self):
        features = baseline_module("features")
        self.assertTrue(hasattr(features, "decode_frames"), "bounded local decoder is missing")
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            video = Path(tmp) / "синтетический_fixture.MOV"
            # Generated lavfi media, not dataset imagery; no visual inspection.
            create = [shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                      "-f", "lavfi", "-i", "color=c=black:s=48x32:r=10:d=2",
                      "-vf", "select='eq(n,0)+eq(n,2)+eq(n,7)+eq(n,9)+eq(n,12)+eq(n,18)'",
                      "-fps_mode", "vfr", "-c:v", "mpeg4", str(video)]
            subprocess.run(create, check=True, capture_output=True, timeout=20)
            frames = list(features.decode_frames(video, fps=2, max_frames=3, timeout_s=20))
            self.assertEqual([round(t, 6) for t, _ in frames], [0.0, 0.7, 1.2])
            self.assertTrue(all(len(raw) == 224 * 224 * 3 for _, raw in frames))
            command = features.ffmpeg_command(video, fps=2, max_frames=3)
            self.assertIn("-autorotate", command)
            self.assertLess(command.index("-autorotate"), command.index("-i"))
            self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file,pipe")
            with self.assertRaises(ValueError):
                list(features.decode_frames(video, fps=0))
            with self.assertRaises(ValueError):
                list(features.decode_frames(video, max_frames=0))
            broken = Path(tmp) / "broken.MOV"
            broken.write_bytes(b"not a video")
            with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                list(features.decode_frames(broken, timeout_s=20))
            with self.assertRaises(FileNotFoundError):
                list(features.decode_frames(Path(tmp) / "absent.MOV"))


class OfflineTests(unittest.TestCase):
    def test_official_sparse_siglip_config_accepts_base_defaults_without_guessing_weights(self):
        features = baseline_module("features")
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            # Official Google config omits these default vision dimensions.
            Path(tmp, "config.json").write_text(json.dumps({"model_type": "siglip", "vision_config":
                                                 {"model_type": "siglip_vision_model", "patch_size": 16}}), encoding="utf-8")
            Path(tmp, "preprocessor_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safetensors missing"):
                features.local_weights(tmp)

    def test_loader_boundary_never_downloads_executes_remote_code_or_accepts_random_missing_weights(self):
        features = baseline_module("features")
        self.assertTrue(hasattr(features, "SiglipEncoder"), "SigLIP vision encoder is missing")
        # Unavoidable heavyweight-loader boundary spies, not a fake model benchmark.
        torch = mock.MagicMock()
        processor_factory, model_factory, image_factory = mock.Mock(), mock.Mock(), mock.Mock()
        vision = mock.Mock()
        model_factory.from_pretrained.return_value = (vision, {"missing_keys": [], "mismatched_keys": [], "error_msgs": []})
        pooled = mock.Mock()
        pooled.shape = (1, 768)
        pooled.detach.return_value.to.return_value.numpy.return_value = "MOCK_BOUNDARY_FIXTURE_ONLY"
        vision.return_value = SimpleNamespace(pooler_output=pooled)
        processor_factory.from_pretrained.return_value.return_value = {"pixel_values": mock.Mock()}
        modules = {"torch": torch, "transformers": SimpleNamespace(SiglipImageProcessor=processor_factory,
                    SiglipVisionModel=model_factory), "PIL": SimpleNamespace(Image=image_factory)}
        with mock.patch.object(features, "local_weights", return_value=Path("local_weights_fixture")):
            with mock.patch.dict(sys.modules, modules):
                encoder = features.SiglipEncoder("local_weights_fixture", device="cpu", seed=7)
                self.assertEqual(encoder([bytes(224 * 224 * 3)]), "MOCK_BOUNDARY_FIXTURE_ONLY")
                for factory in (processor_factory, model_factory):
                    kwargs = factory.from_pretrained.call_args.kwargs
                    self.assertIs(kwargs["local_files_only"], True)
                    self.assertIs(kwargs["trust_remote_code"], False)
                self.assertIs(model_factory.from_pretrained.call_args.kwargs["use_safetensors"], True)
                vision.eval.assert_called_once()
                torch.manual_seed.assert_called_with(7)
                model_factory.from_pretrained.return_value = (vision, {"missing_keys": ["vision_model.encoder.weight"]})
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    features.SiglipEncoder("local_weights_fixture")

    def test_weight_resolution_forces_offline_and_refuses_missing_local_weights(self):
        features = baseline_module("features")
        self.assertTrue(hasattr(features, "local_weights"), "offline-only weight guard is missing")
        keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
                with self.assertRaisesRegex(ValueError, "local"):
                    features.local_weights("google/siglip-base-patch16-224")
                with self.assertRaisesRegex(ValueError, "config.json"):
                    features.local_weights(tmp)
                Path(tmp, "config.json").write_text('{"model_type":"not_siglip"}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SigLIP"):
                    features.local_weights(tmp)
                self.assertTrue(all(os.environ.get(key) == "1" for key in keys))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional; use py -3.10 for numeric tests")
class EmbeddingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
    def test_extract_wires_real_decoder_to_bounded_batches_and_numeric_cache(self):
        features = baseline_module("features")
        self.assertTrue(hasattr(features, "extract_video"), "extract command pipeline is missing")
        np = importlib.import_module("numpy")
        batches = []

        def fixture_encoder(raw):
            """Only a test boundary fixture; no replacement in production or reported metrics."""
            batches.append(len(raw))
            return np.full((len(raw), 768), 0.25, dtype=np.float32)

        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            video, output = root / "камера_A.MOV", root / "artifacts" / "features.npz"
            subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-f", "lavfi", "-i", "color=c=black:s=48x32:r=10:d=2", "-c:v", "mpeg4", str(video)],
                           check=True, capture_output=True, timeout=20)
            with mock.patch.object(features, "SiglipEncoder", return_value=fixture_encoder):
                result = features.extract_video(video, "unused_fixture_weights", output, fps=2,
                                                max_frames=3, batch_size=2, artifact_root=root)
            cache = features.load_cache(output)
            self.assertEqual(result["n_frames"], 3)
            self.assertEqual(batches, [2, 1])
            np.testing.assert_allclose(cache["timestamps_s"], [0, 0.5, 1])
            self.assertEqual(cache["video_id"], "камера_A.MOV")
            self.assertTrue(cache["metadata"]["extraction"]["cap_reached"])
            self.assertTrue(cache["metadata"]["extraction"]["autorotate"])
            with self.assertRaises(ValueError):
                features.extract_video(video, "absent", output, batch_size=0, artifact_root=root)
            # A decoder failing AFTER its last yield must not produce a successful cache.
            def failed_decoder(*args, **kwargs):
                yield 0.0, bytes(224 * 224 * 3)
                raise RuntimeError("ffmpeg late failure fixture")
            failed_output = root / "artifacts" / "must_not_exist.npz"
            with mock.patch.object(features, "SiglipEncoder", return_value=fixture_encoder):
                with mock.patch.object(features, "decode_frames", side_effect=failed_decoder):
                    with self.assertRaisesRegex(RuntimeError, "late failure"):
                        features.extract_video(video, "unused", failed_output, max_frames=1, artifact_root=root)
            self.assertFalse(failed_output.exists())

    def test_embedding_pipeline_preserves_pts_and_never_exceeds_batch_or_frame_limits(self):
        features = baseline_module("features")
        self.assertTrue(hasattr(features, "embed_frames"), "bounded embedding pipeline is missing")
        np = importlib.import_module("numpy")
        calls = []

        def fixture_encoder(raw_batch):
            """Numeric TEST FIXTURE encoder, NOT SigLIP and NOT model-quality evidence."""
            calls.append(len(raw_batch))
            return np.array([[raw[0], 1.0] for raw in raw_batch], dtype=np.float32)

        frames = [(t, bytes([i])) for i, t in enumerate([0.03, 0.7, 1.3, 1.9, 2.8])]
        times, values = features.embed_frames(iter(frames), fixture_encoder, batch_size=2, max_frames=4)
        self.assertEqual(calls, [2, 2])
        np.testing.assert_allclose(times, [0.03, 0.7, 1.3, 1.9])
        np.testing.assert_allclose(values[:, 0], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            features.embed_frames(iter(frames), fixture_encoder, batch_size=0)
        with self.assertRaises(ValueError):
            features.embed_frames(iter([]), fixture_encoder, batch_size=2)
        with self.assertRaises(ValueError):
            features.embed_frames(iter(frames), lambda batch: np.zeros((1, 2)), batch_size=2)


class CliTests(unittest.TestCase):
    def test_extract_cli_executes_pipeline_and_errors_are_not_silent_success(self):
        cli, features = baseline_module("__main__"), baseline_module("features")
        with mock.patch.object(features, "extract_video", return_value={"status": "ok", "n_frames": 2}) as extract:
            with mock.patch("builtins.print"):
                self.assertEqual(cli.main(["extract", "--video", "камера_A.MOV", "--weights", "local",
                                           "--output", "artifacts/smoke.npz", "--fps", "0.5",
                                           "--max-frames", "2", "--batch-size", "1"]), 0)
            extract.assert_called_once()
            kwargs = extract.call_args.kwargs
            self.assertEqual(kwargs["max_frames"], 2)
            self.assertEqual(kwargs["batch_size"], 1)
            self.assertEqual(kwargs["fps"], 0.5)
        result = subprocess.run([sys.executable, "-m", "ml.baseline", "extract", "--video", "absent.MOV",
                                 "--weights", "absent_weights", "--output", "artifacts/absent.npz"],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("baseline error", result.stderr)
        self.assertIn("Local video does not exist", result.stderr)

    def test_help_and_missing_explicit_split_work_without_heavy_imports(self):
        command = [sys.executable, "-m", "ml.baseline"]
        result = subprocess.run(command + ["--help"], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("extract", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("predict", result.stdout)
        missing_split = subprocess.run(command + ["train", "--manifest", "not-read.json"],
                                       capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(missing_split.returncode, 0)
        self.assertIn("--train-angle", missing_split.stderr)
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ml.baseline.__main__; "
                                "assert not any(n in sys.modules for n in ['torch','transformers','numpy','PIL'])"],
                               capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(probe.returncode, 0, probe.stderr)


    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_cli_trains_predicts_and_reports_only_explicit_holdout(self):
        features, core = baseline_module("features"), baseline_module("core")
        rows, caches = training_fixture()
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            manifest = root / "fixture_manifest.json"
            core.write_json(manifest, rows)
            cache_args = []
            for i, cache in enumerate(caches):
                path = artifacts / f"fixture_{i}.npz"
                features.save_cache(path, cache["video_id"], cache["timestamps_s"], cache["embeddings"])
                cache_args += ["--cache", str(path)]
            model, report = artifacts / "model.npz", artifacts / "report.json"
            base = [sys.executable, "-m", "ml.baseline", "--artifact-root", str(root)]
            split = ["--manifest", str(manifest), "--train-angle", "train_view", "--holdout-angle", "hold_view"]
            train = subprocess.run(base + ["train"] + split + ["--model", str(model), "--report", str(report),
                                         "--positive-radius", "0.1", "--epochs", "60"] + cache_args,
                                   capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(train.returncode, 0, train.stderr)
            self.assertTrue(model.is_file(), "train must actually save a model")
            self.assertEqual(core.read_json(report)["n_videos"], 2)
            target = artifacts / "prediction.json"
            predict_args = ["predict", "--model", str(model), "--cache", cache_args[5], "--output"]
            predict = subprocess.run(base + predict_args + [str(target)],
                                     capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(predict.returncode, 0, predict.stderr)
            self.assertEqual(core.read_json(target)["video_id"], rows[2]["video_id"])
            evaluate = subprocess.run(base + ["evaluate"] + split + ["--model", str(model), "--report", str(report)]
                                      + cache_args[4:], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(evaluate.returncode, 0, evaluate.stderr)
            refused = subprocess.run(base + predict_args + [str(root / "outside.json")],
                                     capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse((root / "outside.json").exists())


if __name__ == "__main__":
    unittest.main()
