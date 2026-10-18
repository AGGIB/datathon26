"""Exact numeric metadata and tiny generated media only; no private sources."""
from fractions import Fraction
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import pytest

from tools import source_time_map as tool


def metadata(pts=(0, 40, 80), *, tb="1/1000", origin=0, stream=0):
    return {"streams": [{"index": stream, "codec_type": "video", "time_base": tb, "start_pts": origin}],
            "frames": [{"stream_index": stream, "pts": value} for value in pts]}


def mapped(document, times, **kwargs):
    return tool.map_requested_times(tool.parse_metadata(document, **kwargs), times)


def test_cfr_zero_pts_exact_integer_and_rational_mapping():
    report = mapped(metadata(), [0, .04, "0.08"])
    assert [row["chosen_metadata_frame_index"] for row in report["mappings"]] == [0, 1, 2]
    assert [row["pts"] for row in report["mappings"]] == [0, 40, 80]
    assert report["mappings"][1]["pts_seconds"] == {"numerator": 1, "denominator": 25}
    assert report["decoded_frame_identity_verified"] is False
    assert report["live_evidence_roundtrip_verified"] is False


def test_vfr_uses_actual_metadata_not_nominal_index_fps():
    report = mapped(metadata((0, 33, 91, 200)), ["0.075", "0.1455"])
    assert report["mappings"][0]["chosen_metadata_frame_index"] == 2
    tie = report["mappings"][1]
    assert tie["status"] == "ambiguous" and tie["reason"] == "equidistant_pts"
    assert tie["candidate_metadata_frame_indices"] == [2, 3]
    assert tie["chosen_metadata_frame_index"] is None


@pytest.mark.parametrize("origin,pts", [(9000, (9000, 9040, 9100)), (-100, (-100, -60, 0))])
def test_origin_is_stream_start_pts_not_container_or_first_seen_guess(origin, pts):
    document = metadata(pts, origin=origin)
    document["format"] = {"start_time": "999.5"}
    row = mapped(document, ["0.04"])["mappings"][0]
    assert row["pts"] == pts[1]
    assert row["video_stream_relative_time_seconds"] == {"numerator": 1, "denominator": 25}
    assert row["pts_seconds"] == {"numerator": Fraction(pts[1], 1000).numerator,
                                  "denominator": Fraction(pts[1], 1000).denominator}


def test_missing_origin_stays_unknown_despite_float_start_time_or_first_pts():
    document = metadata((9000, 9040), origin=None)
    document["streams"][0]["start_time"] = "9"
    row = mapped(document, [0])["mappings"][0]
    assert row["reason"] == "unknown_stream_start_pts" and row["pts"] is None


def test_missing_pts_is_not_filled_from_best_effort_or_nominal_rate():
    document = metadata((0, None, 80))
    document["frames"][1].update(best_effort_timestamp=40, best_effort_timestamp_time="0.04")
    report = mapped(document, [0, ".04"])
    assert report["missing_frame_pts_count"] == 1
    assert all(row["status"] == "unknown" and row["reason"] == "unknown_missing_frame_pts" for row in report["mappings"])
    assert all(row["chosen_metadata_frame_index"] is None for row in report["mappings"])


def test_repeated_pts_and_midpoint_preserve_all_ambiguity():
    rows = mapped(metadata((0, 40, 40, 80)), [".04", ".02"])["mappings"]
    assert rows[0]["candidate_metadata_frame_indices"] == [1, 2] and rows[0]["reason"] == "repeated_pts"
    assert rows[1]["candidate_metadata_frame_indices"] == [0, 1, 2] and rows[1]["reason"] == "equidistant_pts"
    assert all(row["pts"] is None for row in rows)


def test_duplicate_requests_do_not_invent_different_source_frames():
    rows = mapped(metadata(), [".04", ".04"])["mappings"]
    assert rows[0] == rows[1]


def test_far_and_leading_requests_are_unknown_not_clamped():
    rows = mapped(metadata((40, 80)), [0, 999])["mappings"]
    assert all(row["reason"] == "outside_observed_pts_range" and row["pts"] is None for row in rows)


def test_large_integer_pts_remains_exact_above_float_precision():
    origin = 2**53 + 1
    row = mapped(metadata((origin, origin + 9000), origin=origin, tb="1/90000"), [".1"])["mappings"][0]
    assert row["pts"] == origin + 9000
    assert row["video_stream_relative_time_seconds"] == {"numerator": 1, "denominator": 10}


def test_multiple_streams_require_selection_and_keep_probe_array_binding():
    document = metadata((0, 40))
    document["streams"].append({"index": 2, "codec_type": "video", "time_base": "1/1000", "start_pts": 100})
    document["frames"] = [document["frames"][0], {"stream_index": 2, "pts": 100},
                          document["frames"][1], {"stream_index": 2, "pts": 150}]
    assert mapped(document, [0])["timeline_status"] == "unknown_stream_selection"
    report = mapped(document, [".05"], stream_index=2)
    row = report["mappings"][0]
    assert row["chosen_metadata_frame_index"] == 1 and row["chosen_probe_frame_position"] == 3
    assert row["pts"] == 150 and report["stream_index"] == 2


def test_frame_without_stream_binding_is_unknown():
    document = metadata()
    del document["frames"][1]["stream_index"]
    assert mapped(document, [0])["timeline_status"] == "unknown_frame_stream_binding"


@pytest.mark.parametrize("pts,status", [((0, 80, 40), "unknown_nonmonotonic_pts"), ((), "unknown_frames")])
def test_incomplete_or_reordered_metadata_not_silently_sorted(pts, status):
    assert mapped(metadata(pts), [0])["timeline_status"] == status


def test_missing_time_base_remains_unknown():
    assert mapped(metadata(tb=None), [0])["timeline_status"] == "unknown_time_base"


@pytest.mark.parametrize("value", [True, 0.5, "1.5", 2**63, -(2**63)-1])
def test_noninteger_or_overflow_pts_rejected(value):
    with pytest.raises(ValueError):
        tool.parse_metadata(metadata((value,)))


@pytest.mark.parametrize("value", ["0/1", "1/0", "-1/1000", .001, "1/2147483648"])
def test_time_base_requires_positive_bounded_rational(value):
    with pytest.raises(ValueError):
        tool.parse_metadata(metadata(tb=value))


@pytest.mark.parametrize("value", [True, -1, math.nan, math.inf, "1e999999", "PRIVATE_SENTINEL"])
def test_bad_requested_time_rejected(value):
    with pytest.raises(ValueError):
        mapped(metadata(), [value])


def test_frame_query_and_ambiguity_resource_limits(monkeypatch):
    monkeypatch.setattr(tool, "MAX_FRAMES", 2)
    with pytest.raises(ValueError):
        tool.parse_metadata(metadata())
    monkeypatch.setattr(tool, "MAX_FRAMES", 100)
    document = metadata((0,) * 40)
    row = mapped(document, [0])["mappings"][0]
    assert row["candidate_count"] == 40 and len(row["candidate_metadata_frame_indices"]) == 32
    assert row["candidates_truncated"] is True and row["chosen_metadata_frame_index"] is None
    with pytest.raises(ValueError):
        mapped(metadata(), [0] * 65)


@pytest.mark.parametrize("data", [b'{"frames":[],"frames":[]}', b'{"value":NaN}'])
def test_strict_json_duplicate_and_nonfinite(data):
    with pytest.raises(ValueError):
        tool.read_metadata_json(data)


def test_cli_metadata_input_pins_no_private_fields_no_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    source = tmp_path / "source.json"
    document = metadata()
    document["private_text"] = "PRIVATE_SENTINEL"
    source.write_text(json.dumps(document))
    out = tmp_path / "var/mapping.json"
    args = ["--probe-json", str(source), "--times", "0", ".04", "--out", str(out)]
    assert tool.main(args) == 0
    report = json.loads(out.read_text())
    assert report["provenance"]["metadata_json_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["provenance"]["source_media_sha256"] is None
    assert "PRIVATE_SENTINEL" not in out.read_text()
    before = out.read_bytes()
    assert tool.main(args) == 2 and out.read_bytes() == before
    assert tool.main(["--PRIVATE_SENTINEL"]) == 2
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().out


def test_output_cannot_overwrite_input_or_escape_private_var(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    source = tmp_path / "var/source.json"
    source.parent.mkdir()
    source.write_text(json.dumps(metadata()))
    before = source.read_bytes()
    for output in (source, tmp_path / "public.json"):
        assert tool.main(["--probe-json", str(source), "--times", "0", "--out", str(output)]) == 2
    assert source.read_bytes() == before


def test_probe_runner_resource_limit_and_failed_child_stderr_are_not_echoed(monkeypatch, capsys):
    monkeypatch.setattr(tool, "MAX_BYTES", 32)
    with pytest.raises(ValueError):
        tool._run_probe([sys.executable, "-c", "print('x'*100)"], 2)
    with pytest.raises(ValueError):
        tool._run_probe([sys.executable, "-c", "import sys;sys.stderr.write('PRIVATE_SENTINEL');sys.exit(1)"], 2)
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().out


def test_probe_timeout_reaps_child(tmp_path):
    pid = tmp_path / "pid"
    code = f"import os,time;open({str(pid)!r},'w').write(str(os.getpid()));time.sleep(10)"
    with pytest.raises(TimeoutError):
        tool._run_probe([sys.executable, "-c", code], .2)
    if pid.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid.read_text()), 0)


def test_local_probe_rejects_remote_input_before_command():
    with pytest.raises(ValueError):
        tool.probe_video("https://example.invalid/private.mov")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="local ffmpeg tools unavailable")
def test_actual_ffprobe_on_tiny_generated_cfr_pins_source_but_not_evidence_identity(tmp_path):
    source = tmp_path / "synthetic.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=c=gray:s=32x32:r=5:d=1",
                    "-an", "-c:v", "mpeg4", str(source)], check=True, capture_output=True, timeout=10)
    document, pins = tool.probe_video(source, timeout_s=5)
    report = mapped(document, [0, ".2", ".8"])
    assert [row["chosen_metadata_frame_index"] for row in report["mappings"]] == [0, 1, 4]
    assert pins["source_media_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert pins["ffprobe_binary_sha256"] and pins["probe_metadata_sha256"]
    assert report["decoded_frame_identity_verified"] is False and report["live_evidence_roundtrip_verified"] is False


def test_import_does_not_load_model_or_media_libraries():
    code = ("import sys;import tools.source_time_map;"
            "assert not {'torch','mlx','transformers','cv2','PIL','numpy'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="local ffprobe unavailable")
def test_source_replacement_during_probe_rejects_stale_pin(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-placeholder.mp4"
    source.write_bytes(b"first synthetic source")
    def replace_source(command, timeout_s):
        source.write_bytes(b"changed synthetic source")
        return json.dumps(metadata()).encode()
    monkeypatch.setattr(tool, "_run_probe", replace_source)
    with pytest.raises(ValueError, match="input_changed"):
        tool.probe_video(source)


@pytest.mark.parametrize("suffix", ["var/../escaped.json", "var/not-created/../../escaped.json"])
def test_output_dotdot_escape_rejected_before_creating_directories(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="fresh_ignored_output_required"):
        tool._save(tmp_path / suffix, {"numeric": 1}, ())
    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path / "var").exists()


def test_direct_and_canonical_source_target_collisions_preserve_input(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    source = tmp_path / "var/source.json"
    source.parent.mkdir()
    source.write_text('{"source":1}')
    before = source.read_bytes()
    for output in (source, source.parent / "not-created/../source.json"):
        with pytest.raises(ValueError, match="fresh_ignored_output_required"):
            tool._save(output, {"replacement": 2}, (source,))
    assert source.read_bytes() == before
    assert not (source.parent / "not-created").exists()
