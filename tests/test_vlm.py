"""Artificial fixtures only; no competition videos or model downloads."""
import importlib
import unittest
import json
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import sys
import types
import time
from contextlib import nullcontext
from unittest.mock import patch


CANDIDATE = {"candidate_id": "cv-1", "violation_type": "fixture_action",
             "start_s": 0.0, "end_s": 1.0, "rule_id": "r1"}
RULE = {"rule_id": "r1", "violation_type": "fixture_action",
        "description": "Artificial fixture action only", "citation": "FIXTURE §1",
        "verified": True}


def response(**changes):
    value = {"candidate_id": "cv-1", "rule_id": "r1", "violation": True,
             "confidence": 0.9, "insufficient_evidence": False,
             "observations": [{"timestamp_s": 0.5, "text": "Fixture visible action"}]}
    value.update(changes)
    return json.dumps(value)


class ScriptedFixtureBackend:
    """Test-only injection; never production inference."""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def infer(self, request, timeout_s):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FixtureFrames:
    def probe(self, video_path, timeout_s):
        api = importlib.import_module("ml.vlm")
        return api.VideoInfo(2.0, tuple(api.FrameTimestamp(i, t, t)
                                       for i, t in enumerate((0.0, 0.5, 1.0, 1.5))))

    def extract(self, video_path, selected, directory, timeout_s):
        api = importlib.import_module("ml.vlm")
        return tuple(api.Frame(t.timestamp_s, Path(directory) / f"{t.index}.png", t.pts_s)
                     for t in selected)


def sleeping_worker_fixture(connection, model_path, device, max_new_tokens):
    """Explicit timeout fixture executed by a real spawned OS process."""
    time.sleep(3)


def structural_model_fixture(root):
    """Placeholder artifacts for mocked loaders/workers; not model weights."""
    (root / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"ARTIFICIAL STRUCTURAL FIXTURE ONLY")


class VLMTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module("ml.vlm")
        except ModuleNotFoundError:
            self.fail("ml.vlm verification API has not been implemented")

    def test_missing_local_model_preserves_original_candidate_as_uncertain(self):
        api = self.api()
        candidate = {"candidate_id": "cv-1", "violation_type": "fixture_action",
                     "start_s": 0.0, "end_s": 1.0, "rule_id": "r1",
                     "cv_metadata": {"score": 0.81}}
        report = api.verify_candidates("absent-fixture.mp4", [candidate], [])
        self.assertEqual(report.results[0].status, "uncertain")
        self.assertEqual(report.results[0].candidate, candidate)
        self.assertEqual(report.retained_candidates, [candidate])
        self.assertTrue(report.human_review_required)
        self.assertEqual(report.calls_used, 0)
    def test_confirmation_uses_only_verified_catalog_citation_and_visible_frames(self):
        api = self.api()
        backend = ScriptedFixtureBackend(response())
        report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                                       backend=backend, frame_source=FixtureFrames())
        result = report.results[0]
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(result.citation, RULE["citation"])
        self.assertEqual(result.rule_id, "r1")
        self.assertTrue(result.advisory_only)
        self.assertEqual(report.calls_used, 1)
        self.assertEqual([f.timestamp_s for f in backend.requests[0].frames], [0, 0.5, 1])
        self.assertIn("visible actions only", backend.requests[0].prompt)
        self.assertIn("Do not identify", backend.requests[0].prompt)
        self.assertIn("human review", backend.requests[0].prompt)
        self.assertIn("no audio", backend.requests[0].prompt)
    def test_untrusted_model_json_is_fail_closed_and_preserves_candidate(self):
        api = self.api()
        invalid = [response(violation="yes"), response(confidence=True),
                   response(confidence=-0.01), response(confidence=1.01),
                   response(confidence=float("nan")), response(confidence=float("inf")),
                   response(confidence="0.9"), response(candidate_id="other"),
                   response(rule_id="invented"), response(citation="INVENTED LAW"),
                   response(observations=[]), response(insufficient_evidence="false"),
                   response(observations=[{"timestamp_s": 1.5, "text": "out of candidate"}]),
                   response(observations=[{"timestamp_s": 0.25, "text": "not a frame"}]),
                   response(observations=[{"timestamp_s": 0.5, "text": ""}]),
                   response(observations=[{"timestamp_s": True, "text": "invalid timestamp"}]),
                   response(observations=[{"timestamp_s": 0.5, "text": "x", "citation": "law"}]),
                   "```json\n" + response() + "\n```", "garbled", "{}", "[]",
                   response()[:-1] + ', "confidence": 0.2}',
                   response() + " trailing"]
        for raw in invalid:
            with self.subTest(raw=raw):
                report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                    backend=ScriptedFixtureBackend(raw), frame_source=FixtureFrames())
                self.assertEqual(report.results[0].status, "uncertain")
                self.assertEqual(report.retained_candidates, [CANDIDATE])
                self.assertIsNone(report.results[0].citation)
    def test_only_sufficient_high_confidence_evidence_can_confirm_or_reject(self):
        api = self.api()
        cases = [(response(violation=False), "rejected"),
                 (response(violation=False, insufficient_evidence=True), "uncertain"),
                 (response(insufficient_evidence=True), "uncertain"),
                 (response(confidence=0.69), "uncertain"),
                 (response(violation=False, confidence=0.69), "uncertain")]
        for raw, expected in cases:
            with self.subTest(expected=expected, raw=raw):
                report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                    backend=ScriptedFixtureBackend(raw), frame_source=FixtureFrames())
                self.assertEqual(report.results[0].status, expected)
                self.assertEqual(report.results[0].candidate, CANDIDATE)
                if expected == "uncertain":
                    self.assertEqual(report.retained_candidates, [CANDIDATE])
                    self.assertIsNone(report.results[0].citation)
                else:
                    self.assertEqual(report.retained_candidates, [])
    def test_only_unique_verified_matching_catalog_rules_enable_inference(self):
        api = self.api()
        invalid_catalogs = [[], [{**RULE, "verified": False}], [{**RULE, "verified": "true"}],
                            [{**RULE, "citation": ""}], [RULE, RULE],
                            [{**RULE, "violation_type": "other"}],
                            [{**RULE, "rule_id": " r1"}]]
        for catalog in invalid_catalogs:
            with self.subTest(catalog=catalog):
                backend = ScriptedFixtureBackend(response())
                report = api.verify_candidates("fixture.mp4", [CANDIDATE], catalog,
                    backend=backend, frame_source=FixtureFrames())
                self.assertEqual(report.results[0].status, "uncertain")
                self.assertEqual(report.results[0].reason, "no_verified_rule")
                self.assertEqual(report.calls_used, 0)
                self.assertIsNone(report.results[0].citation)
    def test_six_call_video_budget_preserves_unprocessed_candidates(self):
        api = self.api()
        candidates = [{**CANDIDATE, "candidate_id": f"cv-{i}"} for i in range(8)]
        backend = ScriptedFixtureBackend(*(response(candidate_id=c["candidate_id"]) for c in candidates))
        report = api.verify_candidates("fixture.mp4", candidates, [RULE], backend=backend,
                                       frame_source=FixtureFrames(), enable_open_checklist=True)
        self.assertEqual(report.calls_used, 6)
        self.assertEqual(len(backend.requests), 6)
        self.assertEqual(len(report.results), 8)
        self.assertEqual([r.reason for r in report.results[-2:]], ["budget_exhausted"] * 2)
        self.assertEqual(report.retained_candidates, candidates)
    def test_open_checklist_adds_only_catalog_grounded_candidate_with_shared_budget(self):
        api = self.api()
        finding = json.loads(response())
        finding.pop("candidate_id")
        backend = ScriptedFixtureBackend(response(), json.dumps({"findings": [finding]}))
        report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE], backend=backend,
            frame_source=FixtureFrames(), enable_open_checklist=True, max_calls=2)
        self.assertEqual(report.calls_used, 2)
        self.assertEqual(report.checklist_status, "completed")
        self.assertEqual(len(report.results), 2)
        added = report.results[1]
        self.assertTrue(added.additional)
        self.assertEqual(added.status, "confirmed")
        self.assertEqual(added.citation, RULE["citation"])
        self.assertEqual(added.candidate["violation_type"], RULE["violation_type"])
        self.assertNotEqual(added.candidate["candidate_id"], CANDIDATE["candidate_id"])
        self.assertEqual(backend.requests[1].mode, "checklist")
        self.assertEqual(backend.requests[1].allowed_rule_ids, ("r1",))
    def test_backend_failures_are_explicit_uncertainty_not_dropped_candidates(self):
        api = self.api()
        for error in (RuntimeError("private fixture description"), TimeoutError("fixture")):
            with self.subTest(error=type(error).__name__):
                try:
                    report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                        backend=ScriptedFixtureBackend(error), frame_source=FixtureFrames())
                except Exception:
                    self.fail("Backend failure escaped instead of preserving uncertain candidate")
                self.assertEqual(report.results[0].status, "uncertain")
                self.assertIn(report.results[0].reason, {"backend_error", "timeout"})
                self.assertEqual(report.calls_used, 1)
                self.assertEqual(report.retained_candidates, [CANDIDATE])
                self.assertNotIn("private", repr(report))
    def test_frame_and_input_failures_preserve_candidates_without_inference(self):
        api = self.api()
        class BrokenFrames(FixtureFrames):
            def probe(self, video_path, timeout_s):
                raise RuntimeError("private fixture path")
        cases = [(CANDIDATE, BrokenFrames(), "frames_unavailable"),
                 ({**CANDIDATE, "start_s": -1}, FixtureFrames(), "invalid_candidate"),
                 ({**CANDIDATE, "end_s": 5}, FixtureFrames(), "invalid_candidate"),
                 ({**CANDIDATE, "start_s": float("nan")}, FixtureFrames(), "invalid_candidate"),
                 ({**CANDIDATE, "start_s": True}, FixtureFrames(), "invalid_candidate"),
                 ({**CANDIDATE, "candidate_id": ""}, FixtureFrames(), "invalid_candidate"),
                 ({**CANDIDATE, "start_s": 0.1, "end_s": 0.2}, FixtureFrames(), "insufficient_frames")]
        for candidate, frames, reason in cases:
            with self.subTest(reason=reason, candidate=candidate):
                try:
                    report = api.verify_candidates("fixture.mp4", [candidate], [RULE],
                        backend=ScriptedFixtureBackend(response()), frame_source=frames)
                except Exception:
                    self.fail("Input/frame failure must preserve candidate")
                self.assertEqual(report.results[0].status, "uncertain")
                self.assertEqual(report.results[0].reason, reason)
                self.assertEqual(report.calls_used, 0)
                self.assertEqual(report.retained_candidates, [candidate])
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_real_ffmpeg_extracts_artificial_vfr_frames_at_actual_pts(self):
        api = self.api()
        self.assertTrue(hasattr(api, "FFmpegFrames"), "timestamp-accurate local frame source missing")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            video = root / "artificial-vfr.mkv"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=black:s=32x32:r=2", "-frames:v", "4", "-vf",
                r"setpts=if(lt(N\,2)\,N\,2*N-1)/(2*TB)", "-fps_mode", "vfr",
                "-c:v", "ffv1", str(video)], check=True, capture_output=True, timeout=15)
            source = api.FFmpegFrames()
            info = source.probe(video, 10)
            self.assertEqual([f.timestamp_s for f in info.frames], [0, 0.5, 1.5, 2.5])
            selected = (info.frames[1], info.frames[3])
            extracted = source.extract(video, selected, root, 10)
            self.assertEqual([f.timestamp_s for f in extracted], [0.5, 2.5])
            self.assertEqual([f.pts_s for f in extracted], [0.5, 2.5])
            self.assertTrue(all(f.path.is_file() and f.path.stat().st_size > 0 for f in extracted))
            backend = ScriptedFixtureBackend(response(observations=[
                {"timestamp_s": 1.5, "text": "Artificial fixture only"}]))
            report = api.verify_candidates(video, [{**CANDIDATE, "end_s": 2.5}], [RULE],
                backend=backend, frame_source=source)
            self.assertEqual(report.results[0].status, "confirmed")
            self.assertEqual([f.timestamp_s for f in backend.requests[0].frames], [0, 1.5, 2.5])
            self.assertTrue(all(not f.path.exists() for f in backend.requests[0].frames))
    def test_preflight_rejects_remote_or_incomplete_models_without_loading_libraries(self):
        api = self.api()
        # Other suites may already have imported these libraries. Preflight
        # must not introduce or replace them; it need not erase global state.
        before = {name: sys.modules.get(name) for name in ("torch", "transformers", "PIL")}
        self.assertTrue(hasattr(api, "preflight"), "real local runtime preflight missing")
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}):
            for model in ("https://example.invalid/model", "Qwen/Qwen2.5-VL-3B-Instruct", "absent-model"):
                result = api.preflight(model)
                self.assertFalse(result.ready)
                self.assertIn("model_directory_required", result.errors)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
            result = api.preflight(root)
            self.assertFalse(result.ready)
            self.assertIn("weights_missing", result.errors)
            self.assertIn("processor_missing", result.errors)
        self.assertEqual(before, {name: sys.modules.get(name) for name in before})
    def test_transformers_runtime_adapter_calls_local_load_and_real_generation_contract(self):
        api = self.api()
        self.assertTrue(hasattr(api, "LocalTransformersBackend"), "local inference backend missing")
        runtime_module = importlib.import_module("ml.vlm.transformers_backend")
        events = []
        class Batch(dict):
            def to(self, device):
                events.append(("device", device))
                return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                events.append(("messages", messages))
                return "fixture template"
            def __call__(self, **kwargs):
                events.append(("processor", kwargs))
                return Batch(input_ids=[[10, 11]])
            def batch_decode(self, tokens, **kwargs):
                self_test.assertEqual(tokens, [[12, 13]])
                return [response()]
        class Model:
            device = "cpu"
            def eval(self):
                return self
            def generate(self, **kwargs):
                events.append(("generate", kwargs))
                return [[10, 11, 12, 13]]
        def loader(kind, value):
            class Loader:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    events.append((kind, str(path), kwargs))
                    if kind != "load_processor" and kwargs.get("output_loading_info"):
                        return value, {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}
                    return value
            return Loader
        class ImageFixture:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def convert(self, mode): return self
            def copy(self): return self
            def close(self): pass
        self_test = self
        modules = {
            "torch": types.SimpleNamespace(inference_mode=nullcontext),
            "PIL": types.SimpleNamespace(Image=types.SimpleNamespace(open=lambda path: ImageFixture())),
            "transformers": types.SimpleNamespace(
                AutoProcessor=loader("load_processor", Processor()),
                Qwen2_5_VLForConditionalGeneration=loader("load_qwen", Model()),
                AutoModelForImageTextToText=loader("load_generic", Model()))}
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            request = api.InferenceRequest("visible actions only", (api.Frame(0.5, root / "fixture.png", 0.5),),
                                           "candidate", "cv-1", ("r1",))
            structural_model_fixture(root)
            with patch.dict(sys.modules, modules), patch("socket.socket", side_effect=AssertionError("network denied")):
                runtime = runtime_module._LocalRuntime(root, "cpu", 128)
                self.assertEqual(runtime.infer(request), response())
            for event in events:
                if event[0].startswith("load_"):
                    self.assertEqual(event[1], str(root))
                    self.assertIs(event[2]["local_files_only"], True)
                    self.assertIs(event[2]["trust_remote_code"], False)
                    if event[0] != "load_processor":
                        self.assertIs(event[2]["output_loading_info"], True)
            self.assertIn("load_qwen", [event[0] for event in events])
            generation = next(event[1] for event in events if event[0] == "generate")
            self.assertIs(generation["do_sample"], False)
            self.assertEqual(generation["max_new_tokens"], 128)
            messages = next(event[1] for event in events if event[0] == "messages")
            self.assertEqual(sum(c["type"] == "image" for m in messages for c in m["content"]), 1)
            self.assertFalse(any(c["type"] == "audio" for m in messages for c in m["content"]))
    def test_real_worker_timeout_terminates_process_including_model_load(self):
        api = self.api()
        self.assertTrue(hasattr(api.LocalTransformersBackend, "close"), "killable runtime worker missing")
        module = importlib.import_module("ml.vlm.transformers_backend")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            structural_model_fixture(Path(directory))
            backend = api.LocalTransformersBackend(directory)
            request = api.InferenceRequest("fixture", (), "candidate", "cv-1", ("r1",))
            start = time.monotonic()
            with patch.object(module, "_worker_main", sleeping_worker_fixture):
                with self.assertRaises(TimeoutError):
                    backend.infer(request, timeout_s=0.05)
            self.assertLess(time.monotonic() - start, 2)
            self.assertIsNone(backend._process)
            backend.close()
        backend = api.LocalTransformersBackend("absent-model")
        with self.assertRaises(api.ModelUnavailable):
            backend.infer(request, timeout_s=1)
        backend.close()
    def test_injected_hung_backend_is_bounded_and_deadline_preserves_every_candidate(self):
        api = self.api()
        class SlowFixtureBackend:
            def infer(self, request, timeout_s):
                time.sleep(0.2)
                return response(candidate_id=request.candidate_id)
        candidates = [CANDIDATE, {**CANDIDATE, "candidate_id": "cv-2"}]
        start = time.monotonic()
        report = api.verify_candidates("fixture.mp4", candidates, [RULE],
            backend=SlowFixtureBackend(), frame_source=FixtureFrames(), timeout_s=0.01,
            video_timeout_s=0.015, enable_open_checklist=True)
        self.assertEqual([r.status for r in report.results], ["uncertain", "uncertain"])
        self.assertTrue(all(r.reason in ("timeout", "video_timeout") for r in report.results))
        self.assertEqual(report.retained_candidates, candidates)
        self.assertLess(time.monotonic() - start, 0.15)
        self.assertIn(report.checklist_status, {"timeout", "video_timeout"})
    def test_default_backend_is_local_only_and_missing_weights_are_uncertain(self):
        api = self.api()
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
             patch("socket.create_connection", side_effect=AssertionError("network forbidden")), \
             patch("socket.getaddrinfo", side_effect=AssertionError("network forbidden")), \
             patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}):
            report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                model_path="absent-local-weights", frame_source=FixtureFrames(),
                enable_open_checklist=True)
            self.assertEqual(report.results[0].status, "uncertain")
            self.assertEqual(report.results[0].reason, "model_unavailable")
            self.assertEqual(report.retained_candidates, [CANDIDATE])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            scripted = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                backend=ScriptedFixtureBackend(response()), frame_source=FixtureFrames())
            self.assertEqual(scripted.results[0].status, "confirmed")
    def test_preprocessing_failure_keeps_original_in_both_modes_without_sensitive_prompts(self):
        api = self.api()
        class BrokenExtract(FixtureFrames):
            def extract(self, video_path, selected, directory, timeout_s):
                raise RuntimeError("private frame description")
        for frames in (BrokenExtract(),):
            try:
                report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                    backend=ScriptedFixtureBackend(response()), frame_source=frames,
                    enable_open_checklist=True)
            except Exception:
                self.fail("Extraction failure must not discard report or candidates")
            self.assertEqual(report.results[0].status, "uncertain")
            self.assertEqual(report.results[0].reason, "frames_unavailable")
            self.assertEqual(report.retained_candidates, [CANDIDATE])
            self.assertEqual(report.checklist_status, "frames_unavailable")
            self.assertEqual(report.calls_used, 0)
        backend = ScriptedFixtureBackend(response())
        original = {**CANDIDATE, "cv_metadata": {"private": "DO_NOT_FORWARD_FIXTURE"}}
        report = api.verify_candidates("fixture.mp4", [original], [RULE],
                                       backend=backend, frame_source=FixtureFrames())
        self.assertEqual(report.results[0].candidate, original)
        self.assertNotIn("DO_NOT_FORWARD_FIXTURE", backend.requests[0].prompt)
        self.assertNotIn(RULE["citation"], backend.requests[0].prompt)
    def test_cli_preflight_and_json_verification_execute_real_unavailable_paths(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            process = subprocess.run([sys.executable, "-B", "-m", "ml.vlm", "preflight",
                "--model", str(root / "missing-model"), "--load"], capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertTrue(process.stdout.lstrip().startswith("{"), "CLI must emit safe structured preflight")
            data = json.loads(process.stdout)
            self.assertFalse(data["ready"])
            self.assertIn("model_directory_required", data["errors"])
            self.assertFalse(data["runtime_loaded"])
            candidates = root / "candidates.json"
            rules = root / "rules.json"
            output = root / "result.json"
            candidates.write_text(json.dumps([CANDIDATE]), encoding="utf-8")
            rules.write_text(json.dumps([RULE]), encoding="utf-8")
            process = subprocess.run([sys.executable, "-B", "-m", "ml.vlm", "verify",
                "--video", str(root / "artificial-absent.mp4"), "--model", str(root / "missing-model"),
                "--candidates", str(candidates), "--rules", str(rules), "--output", str(output),
                "--open-checklist"], capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["results"][0]["status"], "uncertain")
            self.assertEqual(data["retained_candidates"], [CANDIDATE])
            self.assertTrue(data["human_review_required"])
            self.assertNotIn(str(root), process.stdout + process.stderr)
            self.assertNotIn(RULE["description"], process.stdout + process.stderr)
    def test_runtime_validation_exercises_real_missing_dependency_or_weight_error(self):
        api = self.api()
        self.assertTrue(hasattr(api.LocalTransformersBackend, "validate"), "runtime model validation missing")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            backend = api.LocalTransformersBackend(directory)
            try:
                with self.assertRaises(api.ModelUnavailable):
                    backend.validate(timeout_s=5)
                self.assertIsNone(backend._process)
            finally:
                backend.close()
    def test_ambiguous_candidate_ids_and_malformed_rule_ids_fail_closed(self):
        api = self.api()
        cases = [[CANDIDATE, CANDIDATE], [{**CANDIDATE, "rule_id": ["r1"]}],
                 [{**CANDIDATE, "candidate_id": 1}],
                 [{**CANDIDATE, "start_s": 10 ** 1000}]]
        for candidates in cases:
            with self.subTest(candidates=candidates):
                try:
                    report = api.verify_candidates("fixture.mp4", candidates, [RULE],
                        backend=ScriptedFixtureBackend(response(), response()), frame_source=FixtureFrames())
                except Exception:
                    self.fail("Malformed candidates must be explicitly uncertain")
                self.assertTrue(all(r.status == "uncertain" for r in report.results))
                self.assertEqual(report.retained_candidates, candidates)
                self.assertEqual(report.calls_used, 0)
    def test_open_checklist_uncertainty_is_retained_and_hallucinated_rules_fail_closed(self):
        api = self.api()
        finding = json.loads(response(confidence=0.5, insufficient_evidence=True))
        finding.pop("candidate_id")
        report = api.verify_candidates("fixture.mp4", [], [RULE],
            backend=ScriptedFixtureBackend(json.dumps({"findings": [finding]})),
            frame_source=FixtureFrames(), enable_open_checklist=True)
        self.assertEqual(len(report.results), 1, "Uncertain supported checklist suggestion must not vanish")
        self.assertEqual(report.results[0].status, "uncertain")
        self.assertTrue(report.results[0].additional)
        self.assertIsNone(report.results[0].citation)
        self.assertEqual(len(report.retained_candidates), 1)
        for change in ({"rule_id": "invented"}, {"citation": "invented"},
                       {"observations": [{"timestamp_s": 3, "text": "fixture"}]}):
            with self.subTest(change=change):
                report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                    backend=ScriptedFixtureBackend(response(), json.dumps({"findings": [{**finding, **change}]})),
                    frame_source=FixtureFrames(), enable_open_checklist=True)
                self.assertEqual(len(report.results), 1)
                self.assertEqual(report.checklist_status, "invalid_response")
                self.assertEqual(report.retained_candidates, [CANDIDATE])
    def test_observation_text_cannot_smuggle_model_authored_legal_citations(self):
        api = self.api()
        for text in ("Fixture action under Article 999", "Fixture §999", "Нарушение статьи 999 закона"):
            with self.subTest(text=text):
                report = api.verify_candidates("fixture.mp4", [CANDIDATE], [RULE],
                    backend=ScriptedFixtureBackend(response(observations=[{"timestamp_s": 0.5, "text": text}])),
                    frame_source=FixtureFrames())
                self.assertEqual(report.results[0].status, "uncertain")
                self.assertIsNone(report.results[0].citation)
                self.assertEqual(report.retained_candidates, [CANDIDATE])


if __name__ == "__main__":
    unittest.main()
