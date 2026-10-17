import pytest

from ml.candidates import Candidate


def make(**kw):
    base = dict(type_id="safe_left_open", t_start=1.0, t_end=5.0, t_peak=2.0, confidence=0.8,
                observation="дверца сейфа остаётся открытой", evidence_times=(2.0,))
    return Candidate(**{**base, **kw})


def test_valid_candidate():
    assert make().t_peak == 2.0


@pytest.mark.parametrize("obs", ["", "   ", "...", " … ", "—", "12:30", "?!"])
def test_observation_must_contain_words(obs):
    with pytest.raises(ValueError):
        make(observation=obs)


def test_peak_must_be_first_evidence_time():
    with pytest.raises(ValueError):
        make(evidence_times=(1.0, 2.0))


@pytest.mark.parametrize("kw", [{"confidence": 1.5}, {"t_peak": 9.0}, {"evidence_times": ()}])
def test_invalid_values_rejected(kw):
    with pytest.raises(ValueError):
        make(**kw)
