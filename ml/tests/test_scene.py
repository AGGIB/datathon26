from ml.events import Event
from ml.scene import Layout, SceneAnalysis, SceneState, pick_keyframes, sample_times
from ml.scene_items import DeskItem


def _state(t, items=(), keyframe=False, safe=0.0):
    return SceneState(t=t, safe_open=safe, door_open=0.0, laptop_screen_on=None, laptop_screen_work=None,
                      key_in_lock=None, desk_items=tuple(items), keyframe=keyframe)


def test_sample_times_cover_the_video_at_fixed_rate():
    assert sample_times(2.4, fps=2.0) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert sample_times(0.1, fps=2.0) == [0.0]


def test_pick_keyframes_puts_one_inside_each_absence_first():
    events = [
        Event("enter", 1.0, track_id=1),
        Event("exit", 9.0, track_id=1),
        Event("room_empty", 9.0, t_end=19.0),
    ]
    keys = pick_keyframes(events, duration=20.0, max_n=4)
    assert 10.5 in keys  # 1.5 s into the empty room
    assert keys == [3.0, 10.5, 18.4]  # the exit keyframe (10.0) is too close to 10.5


def test_pick_keyframes_last_exit_of_x25_gets_a_keyframe():
    exits = [Event("exit", t, track_id=i) for i, t in enumerate([5.7, 9.2, 13.7, 26.0, 29.8])]
    events = exits + [Event("room_empty", 29.8, t_end=31.7)]
    keys = pick_keyframes(events, duration=31.7, max_n=4)
    assert any(29.8 <= k <= 31.7 for k in keys)
    assert 27.0 in keys  # then exits, latest first


def test_pick_keyframes_cashier_left_span_of_x31_gets_a_keyframe():
    events = [Event("cashier_left", 4.5, t_end=11.2), Event("exit", 4.5, track_id=1)]
    keys = pick_keyframes(events, duration=23.8, max_n=4)
    assert any(4.5 < k < 11.2 for k in keys)


def test_pick_keyframes_without_events_uses_start_middle_end_and_respects_gap():
    assert pick_keyframes([], duration=10.0) == [1.5, 5.0, 9.2]
    assert pick_keyframes([], duration=1.0) == [0.2]


def test_scene_analysis_lookups():
    cup = DeskItem("cup", (0, 0, 5, 5), 0.5, "owlv2", "a cup")
    phone = DeskItem("phone", (0, 0, 5, 5), 0.5, "coco", "cell phone")
    states = (_state(0.0, [phone]), _state(0.5, safe=1.0), _state(1.0, [cup], keyframe=True))
    layout = Layout(464, 848, None, None, None, None, None)
    scene = SceneAnalysis(layout, states, keyframes=(1.0,), elapsed_sec=0.1)
    assert scene.state_at(0.6).safe_open == 1.0
    assert scene.items_near(0.0) == (cup,)  # keyframe within 3 s wins over COCO items at t
    assert scene.items_near(0.0, max_dt=0.5) == (phone,)
    assert scene.series("safe_open") == [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
    assert states[2].has_item("cup") and not states[2].has_item("cup", min_score=0.9)
    assert states[0].item_labels == ("phone",)


def test_key_signal_abstains_for_static_or_unreliable_lock():
    from ml.scene import key_trusted

    assert not key_trusted([20.0] * 20, lock_confidence=0.9)  # dark screw mistaken for a key
    assert not key_trusted([0.0] * 5 + [20.0] * 5, lock_confidence=0.05)
    assert key_trusted([0.0] * 5 + [20.0] * 5, lock_confidence=0.9)


def test_scene_expired_deadline_stops_before_decoding(monkeypatch):
    import time

    import pytest

    import ml.scene as scene_module

    def unexpected_read(*args):
        raise AssertionError("must not decode after the deadline")

    monkeypatch.setattr(scene_module, "read_frames", unexpected_read)
    with pytest.raises(TimeoutError, match="deadline"):
        scene_module.analyze_scene("unused.mov", None, deadline=time.monotonic() - 1)


def test_scene_stops_after_decode_overruns_deadline(monkeypatch):
    from types import SimpleNamespace

    import pytest

    import ml.scene as scene_module

    clock = [0.0]
    monkeypatch.setattr(scene_module.time, "monotonic", lambda: clock[0])

    def decode(*args):
        clock[0] = 11.0
        return []

    def unexpected_layout(*args, **kwargs):
        raise AssertionError("must not start layout after decoding exceeds the budget")

    monkeypatch.setattr(scene_module, "read_frames", decode)
    monkeypatch.setattr(scene_module, "locate_layout", unexpected_layout)
    perception = SimpleNamespace(info=SimpleNamespace(duration=1.0))
    with pytest.raises(TimeoutError, match="deadline"):
        scene_module.analyze_scene("unused.mov", perception, deadline=10.0)


def test_safe_region_normalizes_its_own_light_and_skips_lights_off():
    import numpy as np

    from ml.scene import _region_series

    rng = np.random.default_rng(7)
    reference = rng.integers(80, 180, (32, 32)).astype(np.uint8)
    shadow = (reference * 0.65).astype(np.uint8)
    dark = rng.integers(0, 10, (32, 32)).astype(np.uint8)  # structure lost in low light
    frames = [reference, shadow, dark]
    visible = [np.ones((32, 32), dtype=bool)] * 3
    raw, chosen = _region_series(frames, visible, [0.0, 1.0, 2.0], (0, 0, 32, 32),
                                gains=[1.0] * 3, skip=[False, False, True], structural=True)
    assert chosen == "start"
    assert raw == [0.0, 0.0, None]


def test_reference_change_cannot_calibrate_an_already_open_door():
    """Document the polarity limitation: static appearance alone gives no closed-state evidence."""
    import numpy as np

    from ml.scene import _region_series

    closed = np.full((32, 32), 150, dtype=np.uint8)
    opened = closed.copy()
    opened[:, ::4] = 40
    visible = [np.ones((32, 32), dtype=bool)] * 3
    raw, chosen = _region_series([opened, opened, closed], visible, [0.0, 1.0, 2.0], (0, 0, 32, 32),
                                gains=[1.0] * 3, skip=[False] * 3)
    assert chosen == "start"
    assert raw == [0.0, 0.0, 1.0]  # relative change, not proof that the last frame is open


def test_person_pixels_in_start_reference_do_not_open_an_unchanged_safe():
    import numpy as np

    from ml.scene import _finish, _region_series

    face = np.full((64, 64), 150, dtype=np.uint8)
    hidden_start = face.copy()
    hidden_start[::2, :12] = 20  # striped clothing in 18.75% of the start reference
    partially_visible = np.ones((64, 64), dtype=bool)
    partially_visible[:, :12] = False
    frames = [hidden_start] * 2 + [face] * 4
    visible = [partially_visible] * 2 + [np.ones_like(partially_visible)] * 4
    raw, chosen = _region_series(frames, visible, list(range(6)), (0, 0, 64, 64),
                                gains=[1.0] * 6, skip=[False] * 6, structural=True)
    assert chosen == "start"
    assert raw == [0.0] * 6
    assert _finish(raw, (0.07, 0.17)) == [0.0] * 6
