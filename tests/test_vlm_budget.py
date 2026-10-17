"""Synthetic-only regressions for the shared K2 deadline and warm VLM worker."""
import math

import pytest

from ml import vlm


@pytest.mark.parametrize("value", [None, 0, -1, True, math.nan, math.inf, "5"])
def test_low_level_requires_explicit_remaining_positive_finite_budget(value):
    kwargs = {} if value is None else {"budget_sec": value}
    with pytest.raises(ValueError, match="budget"):
        vlm.verify_candidates("synthetic.mp4", [], [], **kwargs)


def test_legacy_timeout_cannot_create_a_fresh_k2_budget():
    with pytest.raises(ValueError, match="budget"):
        vlm.verify_candidates("synthetic.mp4", [], [], video_timeout_s=300)


@pytest.mark.parametrize("entrypoint, preparation", [
    ("runtime", "catalog"), ("runtime", "candidate"),
    ("adapter", "catalog"), ("adapter", "candidate"), ("low_level", "setup"),
])
def test_original_deadline_includes_all_verification_preparation(monkeypatch, entrypoint, preparation):
    import importlib
    import time
    from test_vlm import CANDIDATE, RULE
    from test_vlm_pipeline_adapter import (
        FixtureBackend, FixtureCatalogType, FixtureFrames, candidate,
    )

    events = []
    class CountedFrames(FixtureFrames):
        def probe(self, *args):
            events.append("probe")
            return super().probe(*args)
        def extract(self, *args):
            events.append("extract")
            return super().extract(*args)

    class SlowCatalog:
        id = "fixture_action"
        status = "verified"
        rule = FixtureCatalogType().rule
        @property
        def summary(self):
            time.sleep(0.08)
            return "Artificial action"

    class SlowCandidate:
        type_id = "fixture_action"
        t_end = 1.0
        @property
        def t_start(self):
            time.sleep(0.08)
            return 0.0

    original = SlowCandidate() if preparation == "candidate" else candidate()
    catalog = {"fixture_action": SlowCatalog() if preparation == "catalog" else FixtureCatalogType()}
    backend = FixtureBackend(["reject"])
    source = CountedFrames()
    try:
        if entrypoint == "runtime":
            vlm.configure(backend=backend, catalog=catalog, frame_source=source, timebase_verified=True)
            retained = vlm.verify([original], "synthetic.mp4", 0.03)
            assert retained[0] is original
        elif entrypoint == "adapter":
            report = vlm.verify_pipeline_candidates(
                "synthetic.mp4", [original], catalog, backend=backend, frame_source=source,
                timebase_verified=True, budget_sec=0.03)
            assert report.retained_candidates[0] is original
        else:
            preflight = importlib.import_module("ml.vlm.preflight")
            monkeypatch.setattr(preflight, "enforce_offline", lambda: time.sleep(0.08))
            report = vlm.verify_candidates("synthetic.mp4", [CANDIDATE], [RULE],
                backend=backend, frame_source=source, budget_sec=0.03)
            assert report.retained_candidates == [CANDIDATE]
        assert events == []
        assert backend.requests == []
        if entrypoint != "runtime":
            verification = report if entrypoint == "low_level" else report.verification
            assert verification.calls_used == 0
            assert verification.results[0].reason == "video_timeout"
    finally:
        vlm.configure(catalog={})


@pytest.mark.parametrize("worker", ["owner_frames", "backend"])
def test_failed_process_start_closes_unstarted_process_and_both_pipe_ends(tmp_path, monkeypatch, worker):
    import multiprocessing
    import types
    from ml.vlm import owner_frames, transformers_backend
    from test_vlm import structural_model_fixture
    from test_vlm_pipeline_adapter import FixtureBackend, FixtureFrames, candidate, catalog_entry

    context = multiprocessing.get_context("spawn")
    connections, processes = [], []
    def pipe():
        endpoints = context.Pipe()
        connections.extend(endpoints)
        return endpoints
    def process(**kwargs):
        instance = context.Process(**kwargs)
        processes.append(instance)
        def fail_start():
            raise OSError("SYNTHETIC process creation failure")
        monkeypatch.setattr(instance, "start", fail_start)
        return instance
    monkeypatch.setattr(multiprocessing, "get_context", lambda *args: types.SimpleNamespace(
        Pipe=pipe, Process=process))
    original = candidate()
    catalog = {"fixture_action": catalog_entry()}
    try:
        if worker == "backend":
            structural_model_fixture(tmp_path)
            resource = transformers_backend.LocalTransformersBackend(tmp_path)
            assert vlm.configure(backend=resource, catalog=catalog, frame_source=FixtureFrames(),
                                 timebase_verified=True) is False
        else:
            video = tmp_path / "synthetic.mp4"
            video.write_bytes(b"SYNTHETIC FILE ONLY")
            resource = owner_frames.OwnerFrames()
            assert vlm.configure(backend=FixtureBackend([]), catalog=catalog,
                                 frame_source=resource, timebase_verified=True)
        for _ in range(2):
            retained = vlm.verify([original], tmp_path / "synthetic.mp4", 1)
            assert retained == [original]
            assert retained[0] is original
        assert len(processes) == 1
        assert resource._process is resource._connection is None
        assert resource._closed
        assert processes[0]._closed
        assert all(connection.closed for connection in connections)
        resource.close()  # Failed startup cleanup is idempotent.
    finally:
        vlm.configure(catalog={})
        for connection in connections:
            connection.close()
        for instance in processes:
            instance.close()


def test_low_level_model_path_never_preflights_or_constructs_backend(monkeypatch):
    from ml.vlm import transformers_backend
    import importlib
    from test_vlm import CANDIDATE, RULE, FixtureFrames

    def forbidden(*args, **kwargs):
        pytest.fail("candidate path must not prepare model artifacts")
    monkeypatch.setattr(importlib.import_module("ml.vlm.preflight"), "preflight", forbidden)
    monkeypatch.setattr(transformers_backend, "LocalTransformersBackend", forbidden)
    report = vlm.verify_candidates("synthetic.mp4", [CANDIDATE], [RULE], budget_sec=1,
                                   model_path="synthetic-model", frame_source=FixtureFrames())
    assert report.results[0].reason == "model_unavailable"


def test_probe_is_cached_once_for_all_candidates_and_checklist():
    from test_vlm import CANDIDATE, RULE, FixtureFrames, ScriptedFixtureBackend, response

    class CountedFrames(FixtureFrames):
        probes = 0

        def probe(self, video_path, timeout_s):
            self.probes += 1
            return super().probe(video_path, timeout_s)

    source = CountedFrames()
    candidates = [{**CANDIDATE, "candidate_id": f"cv-{i}"} for i in range(3)]
    backend = ScriptedFixtureBackend(*(response(candidate_id=c["candidate_id"]) for c in candidates),
                                     '{"findings": []}')
    report = vlm.verify_candidates("synthetic.mp4", candidates, [RULE], budget_sec=5,
                                   backend=backend, frame_source=source, enable_open_checklist=True)
    assert report.calls_used == 4
    assert source.probes == 1


def test_local_worker_defaults_to_at_most_128_output_tokens():
    backend = vlm.LocalTransformersBackend("synthetic-absent-model")
    try:
        assert backend.max_new_tokens == 128
    finally:
        backend.close()


@pytest.mark.parametrize("value", [0, -1, 129, 512, True, 1.5, math.inf])
def test_local_worker_rejects_invalid_output_token_limits(value):
    with pytest.raises(ValueError, match="max_new_tokens"):
        vlm.LocalTransformersBackend("synthetic-absent-model", max_new_tokens=value)


def test_infer_requires_startup_validation_without_touching_model_files(monkeypatch):
    from ml.vlm import transformers_backend as module

    def forbidden(*args):
        pytest.fail("candidate inference must never validate/load model files")

    monkeypatch.setattr(module, "_validated_model_directory", forbidden)
    backend = vlm.LocalTransformersBackend("synthetic-absent-model")
    request = vlm.InferenceRequest("synthetic", (), "candidate", "cv-1", ("r1",))
    try:
        with pytest.raises(vlm.ModelUnavailable, match="not_prepared"):
            backend.infer(request, 0.1)
    finally:
        backend.close()


@pytest.mark.parametrize("phase", ["probe", "extract"])
def test_frame_loading_deadline_is_enforced_even_when_source_ignores_timeout(phase):
    import time
    from test_vlm import CANDIDATE, RULE, FixtureFrames, ScriptedFixtureBackend, response

    class SlowFrames(FixtureFrames):
        calls = 0

        def probe(self, *args):
            self.calls += 1
            if phase == "probe":
                time.sleep(0.3)
            return super().probe(*args)

        def extract(self, *args):
            self.calls += 1
            if phase == "extract":
                time.sleep(0.3)
            return super().extract(*args)

    source = SlowFrames()
    backend = ScriptedFixtureBackend(response())
    candidates = [CANDIDATE, {**CANDIDATE, "candidate_id": "cv-2"}]
    start = time.monotonic()
    report = vlm.verify_candidates("synthetic.mp4", candidates, [RULE], budget_sec=0.04,
                                   backend=backend, frame_source=source, enable_open_checklist=True)
    assert time.monotonic() - start < 0.2
    assert report.retained_candidates == candidates
    assert report.calls_used == 0
    assert backend.requests == []
    assert source.calls == (1 if phase == "probe" else 2)
    assert report.results[0].reason == "timeout"
    assert report.results[1].reason == "video_timeout"
    assert report.checklist_status == "video_timeout"


def test_one_deadline_spans_successful_candidate_then_next_frame_load(monkeypatch):
    import types
    from ml.vlm import verifier
    from test_vlm import CANDIDATE, RULE, FixtureFrames, ScriptedFixtureBackend, response

    clock = [0.0]
    timeouts = []
    class CostlyFrames(FixtureFrames):
        extracts = 0
        def probe(self, *args):
            clock[0] += 0.1
            return super().probe(*args)
        def extract(self, *args):
            self.extracts += 1
            timeouts.append(args[-1])
            clock[0] += 0.4
            return super().extract(*args)
    class CostlyBackend(ScriptedFixtureBackend):
        def infer(self, request, timeout_s):
            clock[0] += 0.2
            return super().infer(request, timeout_s)
    monkeypatch.setattr(verifier, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    source = CostlyFrames()
    backend = CostlyBackend(response())
    candidates = [{**CANDIDATE, "candidate_id": f"cv-{i}"} for i in range(1, 4)]
    report = vlm.verify_candidates("synthetic.mp4", candidates, [RULE], budget_sec=1,
                                   backend=backend, frame_source=source, enable_open_checklist=True)
    assert [result.reason for result in report.results] == ["visible_evidence", "timeout", "video_timeout"]
    assert report.calls_used == len(backend.requests) == 1
    assert source.extracts == 2
    assert timeouts == pytest.approx([0.9, 0.3])
    assert report.checklist_status == "video_timeout"
    assert report.retained_candidates == candidates


def test_owner_frame_source_lazily_uses_cv_index_fps_contract(tmp_path, monkeypatch):
    import sys
    import types
    import time
    from ml.vlm import owner_frames as module

    calls = []
    images = {0.5: object(), 1.0: object()}
    saved = []
    def frames_at(path, times):
        calls.append((path, times))
        # Mapping order differs from selection order; repeated slots stay intact.
        return images
    owner = types.SimpleNamespace(
        probe=lambda path: types.SimpleNamespace(fps=2.0, frame_count=4, duration=2.0),
        frames_at=frames_at)
    def save(path, frame):
        from pathlib import Path
        saved.append((Path(path).name, frame))
        Path(path).write_bytes(b"SYNTHETIC IMAGE FIXTURE")
        return True
    before = {name: sys.modules.get(name) for name in ("ml.video", "cv2", "numpy")}
    source = module.OwnerFrames()
    assert {name: sys.modules.get(name) for name in before} == before
    assert source.timebase == "cv_index_fps"
    deadline = time.monotonic() + 5
    selected = (vlm.FrameTimestamp(2, 1.0, 1.0), vlm.FrameTimestamp(1, 0.5, 0.5),
                vlm.FrameTimestamp(2, 1.0, 1.0))
    commands = iter([("probe", ("synthetic.mp4",), deadline),
                     ("extract", ("synthetic.mp4", selected, tmp_path), deadline)])
    replies = []
    class Connection:
        def recv(self):
            try:
                return next(commands)
            except StopIteration:
                raise EOFError from None
        def send(self, value):
            replies.append(value)
        def close(self):
            pass
    monkeypatch.setattr(module, "_silence_worker", lambda: None)
    monkeypatch.setitem(sys.modules, "ml.video", owner)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imwrite=save))
    module._owner_video_worker(Connection())
    assert replies[0][0] == replies[1][0] == "ok"
    assert [f.timestamp_s for f in replies[0][1].frames] == [0, 0.5, 1, 1.5]
    assert replies[0][1].duration_s == 2
    assert calls == [("synthetic.mp4", [1.0, 0.5, 1.0])]
    assert [f.timestamp_s for f in replies[1][1]] == [1.0, 0.5, 1.0]
    assert [f.pts_s for f in replies[1][1]] == [1.0, 0.5, 1.0]
    assert saved == [("frame-0.png", images[1.0]), ("frame-1.png", images[0.5]),
                     ("frame-2.png", images[1.0])]
    source.close()


def _owner_extract_reply(tmp_path, monkeypatch, *, images, deadline_phase=None, save_ok=True):
    import types
    from ml.vlm import owner_frames as module

    clock = [0.0]
    events, replies = [], []
    selected = (vlm.FrameTimestamp(2, 1.0, 1.0), vlm.FrameTimestamp(1, 0.5, 0.5))
    def decode(path, times):
        events.append(("decode", times))
        if deadline_phase == "decode":
            clock[0] = 2.0
        return images
    def save(path, image):
        events.append(("save", image))
        if deadline_phase == "save":
            clock[0] = 2.0
        return save_ok
    owner = types.SimpleNamespace(frames_at=decode)
    def load(name):
        if deadline_phase == "imports":
            clock[0] = 2.0
        return owner if name == "ml.video" else types.SimpleNamespace(imwrite=save)
    class Connection:
        consumed = False
        closed = False
        def recv(self):
            if self.consumed:
                raise EOFError
            self.consumed = True
            return "extract", ("synthetic.mp4", selected, tmp_path), 1.0
        def send(self, reply):
            replies.append(reply)
        def close(self):
            self.closed = True
    connection = Connection()
    if deadline_phase == "receive":
        clock[0] = 2.0
    monkeypatch.setattr(module, "_silence_worker", lambda: None)
    monkeypatch.setattr(module, "importlib", types.SimpleNamespace(import_module=load))
    monkeypatch.setattr(module, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    module._owner_video_worker(connection)
    assert connection.closed
    return replies, events


@pytest.mark.parametrize("images, save_ok", [({0.5: object()}, True),
                                            ({1.0: None, 0.5: object()}, True),
                                            ({1.0: object(), 0.5: object()}, False)])
def test_owner_batch_missing_or_unwritable_frame_returns_no_partial_result(tmp_path, monkeypatch, images, save_ok):
    replies, events = _owner_extract_reply(tmp_path, monkeypatch, images=images, save_ok=save_ok)
    assert replies == [("error", None)]
    assert events[0] == ("decode", [1.0, 0.5])
    assert sum(event[0] == "decode" for event in events) == 1


@pytest.mark.parametrize("phase", ["receive", "imports", "decode", "save"])
def test_owner_batch_checks_deadline_before_decode_after_decode_and_after_save(tmp_path, monkeypatch, phase):
    replies, events = _owner_extract_reply(tmp_path, monkeypatch,
        images={1.0: object(), 0.5: object()}, deadline_phase=phase)
    assert replies == [("timeout", None)]
    assert [event[0] for event in events] == {
        "receive": [], "imports": [], "decode": ["decode"], "save": ["decode", "save"]}[phase]


def test_real_owner_batch_integration_preserves_selected_frame_identity(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from ml.vlm.owner_frames import OwnerFrames

    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 32))
    if not writer.isOpened():
        writer.release()
        pytest.skip("synthetic MP4 encoder unavailable")
    try:
        for level in (20, 70, 140, 220):
            writer.write(np.full((32, 32, 3), level, dtype=np.uint8))
    finally:
        writer.release()
    source = OwnerFrames()
    try:
        info = source.probe(video, 5)
        assert source.timebase == "cv_index_fps"
        assert info.duration_s == 2.0
        selected = tuple(info.frames[index] for index in (3, 0, 2))
        frames = source.extract(video, selected, tmp_path, 5)
        assert [frame.timestamp_s for frame in frames] == [1.5, 0.0, 1.0]
        assert [frame.pts_s for frame in frames] == [1.5, 0.0, 1.0]
        levels = [cv2.imread(str(frame.path)).mean() for frame in frames]
        assert levels == pytest.approx([220, 20, 140], abs=8)
    finally:
        source.close()


def synthetic_slow_owner_frames(connection):
    import time
    from pathlib import Path
    try:
        while True:
            operation, args, deadline = connection.recv()
            if operation == "probe":
                info = vlm.VideoInfo(1.0, (vlm.FrameTimestamp(0, 0.0, 0.0),))
                connection.send(("ok", info))
            else:
                time.sleep(0.3)
                (Path(args[2]) / "expired-work.txt").write_text("should not finish", encoding="utf-8")
                connection.send(("ok", ()))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def test_owner_sequential_frame_load_is_killed_at_deadline_without_reload(tmp_path, monkeypatch):
    import time
    from ml.vlm import owner_frames as module

    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"SYNTHETIC FILE ONLY")
    monkeypatch.setattr(module, "_owner_video_worker", synthetic_slow_owner_frames)
    source = module.OwnerFrames()
    try:
        info = source.probe(video, 5)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            source.extract(video, info.frames, tmp_path, 0.03)
        assert time.monotonic() - start < 0.25
        assert source._process is None
        with pytest.raises(Exception, match="unavailable"):
            source.probe(video, 1)
        time.sleep(0.35)
        assert not (tmp_path / "expired-work.txt").exists()
    finally:
        source.close()


def test_cli_prepares_once_before_spending_explicit_verification_budget(tmp_path, monkeypatch, capsys):
    import json
    import time
    from ml.vlm import __main__ as cli
    from test_vlm import CANDIDATE, RULE, FixtureFrames, response

    events = []
    class StartupFixture:
        def __init__(self, model_path, **kwargs):
            events.append("construct")
        def validate(self, timeout_s):
            events.append(("startup", timeout_s))
            time.sleep(0.08)
        def infer(self, request, timeout_s):
            events.append("infer")
            return response()
        def close(self):
            events.append("close")
    real_verify = cli.verify_candidates
    def verify(*args, **kwargs):
        assert kwargs["budget_sec"] == 0.03
        assert "model_path" not in kwargs
        assert events[-1] == ("startup", 2.0)
        return real_verify(*args, **kwargs, frame_source=FixtureFrames())
    monkeypatch.setattr(cli, "LocalTransformersBackend", StartupFixture)
    monkeypatch.setattr(cli, "verify_candidates", verify)
    candidates, rules, output = [tmp_path / name for name in ("candidates.json", "rules.json", "result.json")]
    candidates.write_text(json.dumps([CANDIDATE]), encoding="utf-8")
    rules.write_text(json.dumps([RULE]), encoding="utf-8")
    args = ["verify", "--video", "synthetic.mp4", "--model", "synthetic-model",
            "--candidates", str(candidates), "--rules", str(rules), "--output", str(output),
            "--budget-sec", "0.03", "--startup-timeout", "2"]
    assert cli.main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"][0]["status"] == "confirmed"
    assert events == ["construct", ("startup", 2.0), "infer", "close"]
    with pytest.raises(SystemExit) as error:
        cli.main(args[:-4])
    assert error.value.code == 2
    capsys.readouterr()


def synthetic_noisy_owner_frames(connection):
    import sys
    import types
    from ml.vlm.owner_frames import _owner_video_worker
    def probe(path):
        print("PRIVATE_SYNTHETIC_OWNER_DIAGNOSTIC", flush=True)
        return types.SimpleNamespace(fps=1.0, frame_count=1, duration=1.0)
    sys.modules["ml.video"] = types.SimpleNamespace(probe=probe)
    _owner_video_worker(connection)


def test_owner_worker_does_not_forward_third_party_diagnostics(tmp_path, monkeypatch, capfd):
    from ml.vlm import owner_frames as module
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"SYNTHETIC FILE ONLY")
    monkeypatch.setattr(module, "_owner_video_worker", synthetic_noisy_owner_frames)
    source = module.OwnerFrames()
    try:
        assert source.probe(video, 5).duration_s == 1.0
    finally:
        source.close()
    captured = capfd.readouterr()
    assert "PRIVATE_SYNTHETIC_OWNER_DIAGNOSTIC" not in captured.out + captured.err


def synthetic_warm_worker(connection, model_path, device, max_new_tokens):
    """OS-process fixture only; no model or scene inputs."""
    import time
    calls = 0
    try:
        while True:
            request = connection.recv()
            if request is not None:
                calls += 1
                if request.prompt == "sleep":
                    time.sleep(1)
            connection.send(("ok", str(calls)))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def test_prepared_worker_reuses_process_without_per_request_artifact_checks(tmp_path, monkeypatch):
    from ml.vlm import transformers_backend as module
    from test_vlm import structural_model_fixture

    structural_model_fixture(tmp_path)
    monkeypatch.setattr(module, "_worker_main", synthetic_warm_worker)
    backend = vlm.LocalTransformersBackend(tmp_path)
    try:
        backend.validate(5)
        pid = backend._process.pid
        def forbidden(*args):
            pytest.fail("warm requests must not revalidate/reload local artifacts")
        monkeypatch.setattr(module, "_validated_model_directory", forbidden)
        backend.validate(5)
        request = vlm.InferenceRequest("synthetic", (), "candidate", "cv-1", ("r1",))
        assert backend.infer(request, 1) == "1"
        assert backend.infer(request, 1) == "2"
        assert backend._process.pid == pid
    finally:
        backend.close()


@pytest.mark.parametrize("contender", ["infer", "validate", "close", "verify"])
def test_busy_shared_worker_fails_closed_without_waiting_or_interleaving_pipe(tmp_path, monkeypatch, contender):
    import threading
    import time
    from ml.vlm import transformers_backend as module
    from test_vlm import structural_model_fixture
    from test_vlm_pipeline_adapter import FixtureFrames, candidate, catalog_entry

    structural_model_fixture(tmp_path)
    monkeypatch.setattr(module, "_worker_main", synthetic_warm_worker)
    backend = vlm.LocalTransformersBackend(tmp_path)
    entered, release = threading.Event(), threading.Event()
    sent, outcomes = [], []
    request = vlm.InferenceRequest("synthetic", (), "candidate", "owner", ("r1",))
    def owner():
        try:
            outcomes.append(backend.infer(request, 3))
        except Exception as exc:
            outcomes.append(exc)
    thread = threading.Thread(target=owner)
    try:
        backend.validate(5)
        assert vlm.configure(backend=backend, catalog={"fixture_action": catalog_entry()},
                             frame_source=FixtureFrames(), timebase_verified=True)
        connection = backend._connection
        real_send, real_poll = connection.send, connection.poll
        def send(value):
            sent.append(value)
            return real_send(value)
        def poll(timeout):
            entered.set()
            return release.wait(timeout) and real_poll(timeout)
        monkeypatch.setattr(connection, "send", send)
        monkeypatch.setattr(connection, "poll", poll)
        thread.start()
        assert entered.wait(2)
        start = time.monotonic()
        if contender == "verify":
            originals = [candidate(), candidate()]
            retained = vlm.verify(originals, "synthetic.mp4", 0.03)
            assert len(retained) == len(originals)
            assert all(actual is original for actual, original in zip(retained, originals))
            assert all(original.confidence == 0.8 for original in retained)
        else:
            with pytest.raises(vlm.ModelUnavailable, match="busy"):
                if contender == "infer":
                    backend.infer(request, 0.03)
                elif contender == "validate":
                    backend.validate(0.03)
                else:
                    backend.close()
        assert time.monotonic() - start < 0.15
        assert sent == [request]
        assert backend._process.is_alive()
        assert not backend._closed
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)
        vlm.configure(catalog={})
        backend.close()
    assert not thread.is_alive()
    assert outcomes == ["1"]  # The busy caller cannot consume/close the owner's response.


def synthetic_failed_startup(connection, model_path, device, max_new_tokens):
    connection.recv()
    connection.send(("error", "model_unavailable"))
    connection.close()


@pytest.mark.parametrize("failure", ["startup", "timeout", "dead"])
def test_failed_workers_cannot_be_restarted_or_revalidated(tmp_path, monkeypatch, failure):
    from ml.vlm import transformers_backend as module
    from test_vlm import structural_model_fixture

    structural_model_fixture(tmp_path)
    worker = synthetic_failed_startup if failure == "startup" else synthetic_warm_worker
    monkeypatch.setattr(module, "_worker_main", worker)
    backend = vlm.LocalTransformersBackend(tmp_path)
    request = vlm.InferenceRequest("sleep", (), "candidate", "cv-1", ("r1",))
    try:
        if failure == "startup":
            with pytest.raises(vlm.ModelUnavailable):
                backend.validate(5)
        else:
            backend.validate(5)
            if failure == "timeout":
                with pytest.raises(TimeoutError):
                    backend.infer(request, 0.02)
            else:
                backend._process.terminate()
                backend._process.join(2)
        def forbidden(*args):
            pytest.fail("failed worker may only be replaced by explicit new startup configuration")
        monkeypatch.setattr(module, "_validated_model_directory", forbidden)
        with pytest.raises(vlm.ModelUnavailable):
            backend.validate(1)
        with pytest.raises(vlm.ModelUnavailable):
            backend.infer(request, 1)
        assert backend._process is None
    finally:
        backend.close()
