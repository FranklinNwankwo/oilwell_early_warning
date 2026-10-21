"""Phase 5 gate: the streaming service IS the evaluated system.

Features: streaming == batch (features.build_features) on a timeline with a data gap,
NaN, +inf, a historian-held sensor and a frozen sensor. Health codes, n_ok and NaN
patterns match exactly; values match to the last stored digit (pandas' rolling mean
adds/removes values online, so ~1 value in 200,000 lands on the other side of a
3-decimal rounding boundary; measured on real WELL-00011: 6 of 1.26 M values).
Calibration: TailCalibration gives the same alert decisions as the full Calibration.
Alerts: engine statuses == the evaluation path (HysteresisAlerter.run over batch rows).
"""
import numpy as np
import pandas as pd
import pytest

from ew3w.alerting import HysteresisAlerter, HysteresisConfig, Status
from ew3w.baselines import PhysicsRules
from ew3w.evaluation import Calibration, off_of
from ew3w.features import FeatureConfig, build_features
from ew3w.quality import Health
from ew3w.schema import SHARED_SENSORS
from ew3w.stream import (EarlyWarningEngine, FeatureStream, FrozenDetector, TailCalibration, shown_status,
                         ticks_frame)

CFG = FeatureConfig()
NAMES = CFG.feature_names()
EXACT_COLS = [c for c in NAMES if c.endswith(("__health", "__age")) or c == "n_ok"]


def _timeline(hours=9.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600)
    idx = pd.date_range("2020-01-01 00:00:03", periods=n, freq="s", name="timestamp").as_unit("ns")
    t = np.arange(n)
    df = pd.DataFrame({
        "P-PDG": 2.0e7 + rng.normal(0, 2e3, n),
        "P-TPT": 1.0e7 + rng.normal(0, 2e3, n) + np.where(t > 7 * 3600, -3e5, 0.0),
        # historian-held ~601 s; cools with the 7 h pressure drop (a flow drop: the flow_drop rule)
        "T-TPT": np.repeat(60 + rng.normal(0, 0.05, n // 601 + 1), 601)[:n] + np.where(t > 7 * 3600, -3.0, 0.0),
        "P-MON-CKP": 6.0e6 + rng.normal(0, 2e3, n),
        "T-JUS-CKP": 70 + rng.normal(0, 0.02, n),
    }, index=idx)
    df.iloc[2 * 3600 : 2 * 3600 + 900, 3] = np.nan  # P-MON-CKP missing 15 min
    df.iloc[5 * 3600 : 5 * 3600 + 120, 0] = np.inf  # P-PDG INVALID burst
    df.iloc[3 * 3600 : 4 * 3600 + 1800, 4] = df.iloc[3 * 3600, 4]  # T-JUS-CKP frozen 1.5 h
    df = df.astype(np.float32)
    df = df.drop(df.index[int(6.2 * 3600) : int(6.2 * 3600) + 47])  # a 47 s data gap -> new segment
    return df


@pytest.fixture(scope="module")
def both():
    tl = _timeline()
    batch = build_features(tl, CFG, detect_frozen=True)
    fs = FeatureStream(CFG, detect_frozen=True)
    t = pd.DatetimeIndex(tl.index).as_unit("ns").asi8
    X = tl[list(SHARED_SENSORS)].to_numpy()
    rows, ts = [], []
    for ti, x in zip(t, X):
        r = fs.push(ti, x)
        if r is not None:
            rows.append(r)
            ts.append(ti)
    return tl, batch, np.array(ts), np.vstack(rows)


def test_stream_emits_the_same_rows(both):
    _, batch, ts, S = both
    assert np.array_equal(ts, pd.DatetimeIndex(batch.index).as_unit("ns").asi8)
    assert S.shape == (len(batch), len(NAMES))


def test_stream_features_equal_batch(both):
    _, batch, _, S = both
    B = batch[NAMES].to_numpy(np.float32)
    assert np.array_equal(np.isnan(S), np.isnan(B)), "NaN pattern differs"
    fin = np.isfinite(B)
    diff = np.abs(S[fin].astype(np.float64) - B[fin])
    assert diff.max() <= 1e-3 + 1e-6, f"max diff {diff.max()}"  # at most one unit in the last stored digit
    assert (diff == 0).mean() > 0.9999
    for c in EXACT_COLS:
        j = NAMES.index(c)
        np.testing.assert_array_equal(S[:, j], B[:, j], err_msg=c)


def test_stream_sees_the_injected_faults(both):
    tl, batch, _, S = both
    h = pd.DataFrame(S[:, [NAMES.index(f"{s}__health") for s in SHARED_SENSORS]], index=batch.index,
                     columns=list(SHARED_SENSORS))
    assert (h["P-MON-CKP"] == Health.MISSING).any()
    assert (h["P-PDG"] == Health.INVALID).any()
    assert (h["T-JUS-CKP"] == Health.FROZEN).any()
    assert batch["segment"].nunique() == 2


def _cal(seed=0, n=200_000, on=0.999):
    raw = np.round(np.random.default_rng(seed).gamma(2.0, 1.0, n), 2)  # many ties, like rounded rule scores
    return Calibration(np.sort(raw), on, off_of(on), 0.1, 10.0)


def test_tail_calibration_is_exact_where_decisions_are_made():
    cal = _cal()
    tc = TailCalibration.from_calibration(cal)
    assert tc.tail.size < 0.2 * cal.sorted_normal.size
    q = np.r_[np.random.default_rng(1).gamma(2.0, 1.2, 50_000), cal.sorted_normal[::97], np.nan, np.inf]
    full, compact = cal.to_percentile(q), tc.to_percentile(q)
    hi = full >= tc.keep_from
    np.testing.assert_array_equal(compact[hi], full[hi])
    assert np.all(compact[np.isfinite(full) & ~hi] < tc.keep_from)
    assert np.array_equal(np.isnan(full), np.isnan(compact))
    a_cfg = HysteresisConfig(cal.on_percentile, cal.off_percentile, 60, 300, 10)
    s_full = HysteresisAlerter(a_cfg).run(full)
    s_comp = HysteresisAlerter(a_cfg).run(compact)
    np.testing.assert_array_equal(s_full, s_comp)


def test_tail_calibration_refuses_a_cut_above_the_off_threshold():
    with pytest.raises(ValueError):
        TailCalibration.from_calibration(_cal(), keep_from=0.995)


def _engine():
    rules = PhysicsRules()
    normal_scores = np.round(np.random.default_rng(2).gamma(2.0, 0.3, 100_000), 3)
    cal = Calibration(np.sort(normal_scores), 0.999, off_of(0.999), 0.1, 10.0)
    det = FrozenDetector("physics_rules", "Physics rules", rules, TailCalibration.from_calibration(cal))
    return det, cal


def test_engine_alerts_equal_the_evaluation_path():
    tl = _timeline()
    det, cal = _engine()
    eng = EarlyWarningEngine([det], CFG, detect_frozen=True)
    t = pd.DatetimeIndex(tl.index).as_unit("ns").asi8
    X = tl[list(SHARED_SENSORS)].to_numpy()
    ticks = [k for k in (eng.push(ti, x) for ti, x in zip(t, X)) if k is not None]
    got = ticks_frame(ticks, ["physics_rules"])
    batch = build_features(tl, CFG, detect_frozen=True)
    p = cal.to_percentile(det.scorer.score(batch))
    a_cfg = det.alert_config(HysteresisConfig(period_s=CFG.block_s))
    want = HysteresisAlerter(a_cfg).run(p, degraded=batch["n_ok"].to_numpy() < 2, segments=batch["segment"].to_numpy())
    np.testing.assert_array_equal(got["physics_rules__status"].to_numpy(), want)
    assert (want == Status.ALERT).any()  # the P-TPT drop at 7 h raises an alert
    assert got["segment"].nunique() == 2


def test_push_many_equals_push():
    tl = _timeline(hours=8.0)
    det, _ = _engine()
    t = pd.DatetimeIndex(tl.index).as_unit("ns").asi8
    X = tl[list(SHARED_SENSORS)].to_numpy()
    a = EarlyWarningEngine([det], CFG)
    one = [k for k in (a.push(ti, x) for ti, x in zip(t, X)) if k is not None]
    b = EarlyWarningEngine([det], CFG)
    many = b.push_many(t[:20_000], X[:20_000]) + b.push_many(t[20_000:], X[20_000:])
    pd.testing.assert_frame_equal(ticks_frame(one, ["physics_rules"]), ticks_frame(many, ["physics_rules"]))


def test_gap_resets_state_and_out_of_order_is_rejected():
    fs = FeatureStream(CFG, detect_frozen=True)
    t0 = pd.Timestamp("2020-01-01 00:00:00").value
    fs.push(t0, [1.0] * 5)
    seg = fs.segment
    fs.push(t0 + 5 * 10**9, [1.0] * 5)
    assert fs.segment == seg + 1
    with pytest.raises(ValueError):
        fs.push(t0, [1.0] * 5)


def test_display_rule_never_hides_an_alert():
    assert shown_status(Status.NORMAL, np.nan, 3)[0] == "DEGRADED"  # silent blindness becomes visible
    assert shown_status(Status.NORMAL, 0.2, 3)[0] == "NORMAL"
    assert shown_status(Status.DEGRADED, np.nan, 1)[0] == "DEGRADED"
    assert shown_status(Status.ALERT, np.nan, 3)[0] == "ALERT"  # an active alert is held, not relabelled
    assert shown_status(Status.CLEARING, 0.1, 0)[0] == "CLEARING"
