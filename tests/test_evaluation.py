"""Phase 2 harness: no test-well data reaches training or calibration, and a known
event / known normal stretch is scored exactly as the metric definitions say."""
import numpy as np
import pandas as pd
import pytest

from ew3w.alerting import HysteresisConfig
from ew3w.evaluation import (
    Calibration,
    calibrate,
    calibration_normal,
    evaluate_wells,
    run_protocol,
    training_rows,
)
from ew3w.splits import PROTOCOLS, Fold

P = 10  # seconds per feature row
ALERT = HysteresisConfig(period_s=P)


def _rows(well, start, n, cls, event_class, inst, source="real", score=0.0, state=0.0):
    idx = pd.date_range(start, periods=n, freq=f"{P}s", name="timestamp").as_unit("ns")
    return pd.DataFrame(
        {"class": np.asarray(cls, dtype=np.float32) * np.ones(n, np.float32), "state": np.float32(state),
         "event_class": np.int8(event_class), "instance_id": inst, "segment": 0, "well": well,
         "source": source, "n_ok": 5.0, "x": np.asarray(score, dtype=np.float32) * np.ones(n, np.float32)},
        index=idx,
    )


def _dataset():
    """4 real wells (normal + one class-2 event each) + 2 simulated instances."""
    real = []
    for i, w in enumerate(["W1", "W2", "W3", "W4"]):
        rng = np.random.default_rng(i)
        real.append(_rows(w, "2020-01-01", 8640, 0, 0, f"0/{w}_n", score=rng.random(8640)))  # 1 day normal
        lab = np.r_[np.zeros(360), np.full(180, 102), np.full(180, 2)]
        sc = np.r_[rng.random(360), np.full(360, 50.0)]  # obvious event once it starts
        real.append(_rows(w, "2020-01-02", 720, lab, 2, f"2/{w}_e", score=sc))
    real = pd.concat(real)
    syn = pd.concat([
        _rows("SIM1", "2021-01-01", 2000, np.r_[np.zeros(1000), np.full(1000, 102)], 2, "2/SIM1", "simulated",
              score=np.r_[np.random.default_rng(9).random(1000), np.full(1000, 50.0)]),
        _rows("SIM2", "2021-02-01", 2000, 0, 2, "2/SIM2", "simulated", score=np.random.default_rng(8).random(2000)),
    ])
    for d in (real, syn):
        for c in ("instance_id", "well", "source"):
            d[c] = d[c].astype("category")
    return real, syn


FOLD = Fold(k=0, test_wells=("W1", "W2"), train_wells=("W3", "W4"))


class Spy:
    """Scorer that records every well it is shown during fit."""

    name, columns, needs_events = "spy", ["x"], True
    seen: set = set()

    def fit(self, train, normal):
        Spy.seen |= set(train["well"].astype(str)) | set(normal["well"].astype(str))
        return self

    def score(self, df):
        return df["x"].to_numpy(np.float64)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_no_test_well_reaches_training_or_calibration(protocol):
    real, syn = _dataset()
    for rows in (training_rows(protocol, real, syn, FOLD), calibration_normal(protocol, real, syn, FOLD)):
        assert not set(rows["well"].astype(str)) & set(FOLD.test_wells)
    Spy.seen = set()
    run_protocol(Spy, protocol, real, syn, [FOLD], ALERT, 1 / 7, log=lambda *_: None)
    assert not Spy.seen & set(FOLD.test_wells)


def test_protocol_contents():
    real, syn = _dataset()
    p1 = training_rows("P1_sim_only", real, syn, FOLD)
    assert set(p1["source"].astype(str)) == {"simulated"}
    p2 = training_rows("P2_sim_plus_site_normal", real, syn, FOLD)
    assert set(p2[p2["source"] == "real"]["event_class"]) == {0}  # no real events under P2
    assert set(calibration_normal("P1_sim_only", real, syn, FOLD)["source"].astype(str)) == {"simulated"}
    cal = calibration_normal("P3_real", real, syn, FOLD)
    assert set(cal["event_class"]) == {0} and (cal["class"] == 0).all()


def test_percentile_mapping():
    cal = Calibration(np.sort(np.arange(100.0)), 0.99, 0.9, 0, 1)
    p = cal.to_percentile(np.array([-1.0, 49.0, 99.0, 1e9, np.nan]))
    np.testing.assert_allclose(p[:4], [0.0, 0.495, 0.995, 1.0])  # mid-rank
    assert np.isnan(p[4])


def test_constant_scorer_never_alarms():
    """A scorer that outputs one constant carries no information: mid-rank puts it at 0.5."""
    cal = Calibration(np.full(1000, 0.3), 0.99, 0.9, 0, 1)
    np.testing.assert_allclose(cal.to_percentile(np.full(10, 0.3)), 0.5)


def test_calibration_meets_budget_and_is_lowest():
    real, _ = _dataset()
    normal = real[(real.event_class == 0)]
    cal = calibrate(normal["x"].to_numpy(np.float64), normal, ALERT, fa_budget_per_day=1.0)
    assert cal.train_fa_per_day <= 1.0 and cal.train_time_in_alarm <= 0.01 and cal.meets_budget
    assert cal.off_percentile < cal.on_percentile


def test_never_clearing_alarm_cannot_win_calibration():
    """Normal data with slow random-walk scores: a low threshold gives FEW but LONG alarms
    (few episodes, lots of time in alarm). The time-in-alarm budget must reject that."""
    rng = np.random.default_rng(3)
    n = 8640 * 4
    walk = np.cumsum(rng.normal(0, 1, n))
    normal = _rows("W9", "2020-01-01", n, 0, 0, "0/W9_n", score=walk)
    for c in ("instance_id", "well", "source"):
        normal[c] = normal[c].astype("category")
    cal = calibrate(normal["x"].to_numpy(np.float64), normal, ALERT, fa_budget_per_day=1.0)
    assert cal.train_time_in_alarm <= 0.01 or not cal.meets_budget
    assert cal.on_percentile >= 0.99


def test_budget_not_met_is_flagged(monkeypatch):
    """If even the strictest grid threshold breaks a budget, it is used AND flagged."""
    import ew3w.evaluation as ev

    monkeypatch.setattr(ev, "PERCENTILE_GRID", np.array([0.99, 0.995]))
    rng = np.random.default_rng(4)
    n = 8640 * 2
    normal = _rows("W9", "2020-01-01", n, 0, 0, "0/W9_n", score=np.cumsum(rng.normal(0, 1, n)))
    for c in ("instance_id", "well", "source"):
        normal[c] = normal[c].astype("category")
    cal = ev.calibrate(normal["x"].to_numpy(np.float64), normal, ALERT, fa_budget_per_day=1e-9, tia_budget=1e-9)
    assert not cal.meets_budget and cal.on_percentile == 0.995


def test_known_event_and_normal_are_scored_exactly():
    real, _ = _dataset()
    test = real[real["well"].isin(["W1"])]
    cal = Calibration(np.sort(np.random.default_rng(0).random(10_000)), 0.999999, 0.99999, 0, 1)
    out, fa = evaluate_wells(test, cal.to_percentile(test["x"].to_numpy(np.float64)), cal, ALERT)
    assert len(out) == 1 and len(fa) == 1
    o = out[0]
    # score jumps at onset (row 360); on_delay 60 s = 6 rows -> alert at row 365 = 50 s after onset
    assert o.detected and o.delay_s == 50 and o.lead_s == 180 * P - 50 and o.warned_before_established
    assert fa[0].episodes == 0 and fa[0].exposure_s == 8640 * P


class Memoriser:
    """Scores 0 on any well it was fit on, 1 elsewhere: the extreme overfitting case."""

    name, columns, needs_events, fitted = "memoriser", ["x"], True, True

    def fit(self, train, normal):
        self.seen = set(train["well"].astype(str)) | set(normal["well"].astype(str))
        return self

    def score(self, df):
        return np.where(df["well"].astype(str).isin(self.seen), 0.0, 1.0)


def test_fitted_scorers_are_calibrated_out_of_fold():
    """In-sample calibration would see only 0s (all wells memorised); out-of-fold
    calibration must see the 1s an unseen well produces."""
    real, syn = _dataset()
    r = run_protocol(Memoriser, "P3_real", real, syn, [FOLD], ALERT, 1 / 7, log=lambda *_: None)
    assert (r.calibrations[0].sorted_normal == 1.0).all()
