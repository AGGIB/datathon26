from ml.events import Event, extract_events, occupancy
from ml.track import Track, TrackPoint

BOX = (0.0, 0.0, 10.0, 20.0)


def track(tid, t0, t1, step=0.5, score=0.9, box=BOX):
    n = int(round((t1 - t0) / step)) + 1
    pts = tuple(TrackPoint(t=round(t0 + i * step, 3), box=box, score=score) for i in range(n))
    return Track(track_id=tid, points=pts)


def kinds(events):
    return [(e.kind, e.t) for e in events]


def test_occupancy_counts_tracks_alive_at_each_time():
    tracks = [track(1, 0.0, 4.0), track(2, 2.0, 3.0)]
    occ = occupancy(tracks, times=[0.0, 1.0, 2.0, 2.5, 3.5])
    assert occ == [1, 1, 2, 2, 1]


def test_person_entering_mid_video_is_an_enter_event():
    events = extract_events([track(1, 0.0, 10.0), track(2, 4.0, 10.0)], duration=10.0)
    assert ("enter", 4.0) in kinds(events)


def test_track_present_from_start_is_not_an_enter():
    events = extract_events([track(1, 0.0, 10.0)], duration=10.0)
    assert not [e for e in events if e.kind == "enter"]


def test_person_leaving_before_end_is_an_exit():
    events = extract_events([track(1, 0.0, 6.0)], duration=10.0)
    assert ("exit", 6.0) in kinds(events)


def test_room_empty_event_when_everyone_left():
    events = extract_events([track(1, 0.0, 6.0)], duration=10.0, min_empty_sec=1.0)
    empty = [e for e in events if e.kind == "room_empty"]
    assert len(empty) == 1 and empty[0].t == 6.0 and empty[0].t_end == 10.0


def test_short_dropout_is_not_room_empty():
    events = extract_events([track(1, 0.0, 6.0), track(2, 6.5, 10.0)], duration=10.0, min_empty_sec=1.0)
    assert not [e for e in events if e.kind == "room_empty"]


def test_crowd_event_when_three_people_present():
    tracks = [track(1, 0.0, 10.0), track(2, 0.0, 10.0), track(3, 3.0, 6.0)]
    events = extract_events(tracks, duration=10.0)
    crowd = [e for e in events if e.kind == "crowd"]
    assert crowd and crowd[0].t == 3.0 and crowd[0].count == 3


def test_short_or_weak_tracks_are_ignored_as_noise():
    tracks = [track(1, 0.0, 10.0), track(2, 4.0, 4.5), track(3, 5.0, 8.0, score=0.4)]
    events = extract_events(tracks, duration=10.0, min_track_sec=1.0, min_track_score=0.6)
    assert not [e for e in events if e.kind == "enter"]


def test_events_are_sorted_by_time():
    events = extract_events([track(1, 0.0, 6.0), track(2, 2.0, 9.0)], duration=10.0)
    assert [e.t for e in events] == sorted(e.t for e in events)
    assert all(isinstance(e, Event) for e in events)


def test_room_empty_at_start_when_first_person_walks_in_later():
    events = extract_events([track(1, 4.0, 10.0)], duration=10.0, min_empty_sec=1.0)
    empty = [e for e in events if e.kind in ("room_empty", "room_empty_at_start")]
    assert [(e.kind, e.t, e.t_end) for e in empty] == [("room_empty_at_start", 0.0, 4.0)]


def test_whole_video_is_empty_when_there_are_no_reliable_tracks():
    events = extract_events([track(1, 2.0, 2.5)], duration=10.0, min_empty_sec=1.0)
    empty = [(e.kind, e.t, e.t_end) for e in events if e.kind.startswith("room_empty")]
    assert empty == [("room_empty_at_start", 0.0, 10.0)]


def test_two_disjoint_crowds_give_two_crowd_events():
    tracks = [
        track(1, 0.0, 10.0),
        track(2, 0.0, 10.0),
        track(3, 0.0, 2.0),
        track(4, 5.0, 7.0),
    ]
    events = extract_events(tracks, duration=10.0, edge_sec=0.0)
    assert [e.t for e in events if e.kind == "crowd"] == [0.0, 5.0]


def test_crowd_event_reports_its_end():
    tracks = [track(1, 0.0, 10.0), track(2, 0.0, 10.0), track(3, 3.0, 6.0)]
    [crowd] = [e for e in extract_events(tracks, duration=10.0) if e.kind == "crowd"]
    assert crowd.t_end == 6.0


def test_crowd_that_breaks_up_and_reforms_between_boundaries_gives_two_events():
    elsewhere = (100.0, 0.0, 110.0, 20.0)
    tracks = [track(1, 0.0, 10.0), track(2, 0.0, 10.0), track(3, 3.0, 5.0), track(4, 6.0, 8.0, box=elsewhere)]
    crowds = [(e.t, e.t_end) for e in extract_events(tracks, duration=10.0) if e.kind == "crowd"]
    assert crowds == [(3.0, 5.0), (6.0, 8.0)]


def test_video_without_any_track_is_empty_for_its_whole_duration():
    events = extract_events([], duration=12.0)
    assert [(e.kind, e.t, e.t_end) for e in events] == [("room_empty_at_start", 0.0, 12.0)]


def test_short_dropout_of_the_only_person_is_neither_exit_nor_empty_room():
    events = extract_events([track(1, 0.0, 4.0), track(2, 5.25, 10.0)], duration=10.0)
    assert not [e for e in events if e.kind in ("exit", "enter", "room_empty")]


def test_empty_room_shorter_than_two_seconds_is_ignored():
    elsewhere = (100.0, 0.0, 110.0, 20.0)
    events = extract_events([track(1, 0.0, 4.0), track(2, 5.5, 10.0, box=elsewhere)], duration=10.0)
    assert not [e for e in events if e.kind == "room_empty"]
