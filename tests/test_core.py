import numpy as np
import pandas as pd
import pytest

from ew3w.alerting import HysteresisAlerter, HysteresisConfig, Status, alert_active, alert_episodes
from ew3w.data import InstanceMeta, stitch_well
from ew3w.metrics import count_false_alarms, evaluate_event_instance, poisson_rate_ci
from ew3w.quality import FrozenRule, Health, frozen_mask, health_matrix
from ew3w.splits import assign_well_folds, make_folds, training_selection


# ---------------------------------------------------------------- quality
def test_frozen_is_causal_and_not_retroactive():
    x = np.r_[np.random.default_rng(0).normal(size=100), np.full(50, 3.0)]
    m = frozen_mask(x, FrozenRule(min_run_s=20))
    first = np.flatnonzero(m)[0]
    assert first == 100 + 19  # flagged at the 20th repeated sample, never before
    assert m[first:].all() and not m[:first].any()


def test_nan_breaks_frozen_run_and_is_not_frozen():
    x = np.r_[np.full(15, 1.0), np.nan, np.full(15, 1.0)]
    assert not frozen_mask(x, FrozenRule(min_run_s=20)).any()


def test_valve_states_never_frozen():
    df = pd.DataFrame({"ESTADO-M1": np.ones(2000), "P-TPT": np.ones(2000)})
    h = health_matrix(df, ["ESTADO-M1", "P-TPT"], FrozenRule(min_run_s=600), segment_col=None)
    assert (h["ESTADO-M1"] == Health.OK).all()
    assert (h["P-TPT"].iloc[700:] == Health.FROZEN).all()


def test_implausible_values_become_invalid(tmp_path):
    d = tmp_path / "1"
    d.mkdir()
    idx = pd.date_range("2020-01-01", periods=5, freq="s", name="timestamp")
    pd.DataFrame(
        {"P-PDG": [2e7, 1.18e42, 2e7, -1.4e8, np.nan], "T-TPT": [60, -999, 60, 3e4, 61], "class": 1.0, "state": 0.0}, index=idx
    ).to_parquet(d / "WELL-00006_20200101000000.parquet")
    from ew3w.data import load_instance

    df = load_instance(InstanceMeta(d / "WELL-00006_20200101000000.parquet", 1, "real", "WELL-00006"), ["P-PDG", "T-TPT"])
    h = health_matrix(df, ["P-PDG", "T-TPT"], segment_col=None)
    assert h["P-PDG"].tolist() == [Health.OK, Health.INVALID, Health.OK, Health.INVALID, Health.MISSING]
    assert h["T-TPT"].tolist() == [Health.OK, Health.INVALID, Health.OK, Health.INVALID, Health.OK]


# ---------------------------------------------------------------- alerting
def _cfg(**kw):
    base = dict(on_threshold=0.8, off_threshold=0.5, on_delay_s=5, off_delay_s=10)
    base.update(kw)
    return HysteresisConfig(**base)


def test_flicker_is_suppressed():
    s = np.tile([0.9, 0.1], 100)  # alternates every second
    st = HysteresisAlerter(_cfg()).run(s)
    assert not alert_active(st).any()


def test_raise_after_on_delay_and_hold_through_dips():
    s = np.r_[np.zeros(10), np.full(20, 0.9), np.full(5, 0.6), np.full(20, 0.9), np.zeros(30)]
    act = alert_active(HysteresisAlerter(_cfg()).run(s))
    assert np.flatnonzero(act)[0] == 10 + 4  # 5th consecutive high sample
    assert act[30:35].all()  # 0.6 is inside the dead-band: hold
    eps = alert_episodes(act)
    assert len(eps) == 1
    assert eps[0, 1] == 55 + 9  # 10th consecutive low sample (idx 64) is already NORMAL


def test_nan_does_not_clear_and_degraded_blocks_new_alerts():
    a = HysteresisAlerter(_cfg(on_delay_s=1, off_delay_s=2))
    assert a.step(0.9) == Status.ALERT
    assert a.step(np.nan) == Status.ALERT
    assert a.step(0.1) == Status.CLEARING
    assert a.step(0.1) == Status.NORMAL
    assert a.step(0.99, degraded=True) == Status.DEGRADED


def test_run_equals_step_and_resets_on_segment():
    rng = np.random.default_rng(1)
    s = rng.random(500)
    seg = np.repeat([0, 1], 250)
    batch = HysteresisAlerter(_cfg(on_delay_s=2, off_delay_s=3)).run(s, segments=seg)
    a = HysteresisAlerter(_cfg(on_delay_s=2, off_delay_s=3))
    stream = []
    for i, v in enumerate(s):
        if i == 250:
            a.reset()
        stream.append(a.step(v))
    assert np.array_equal(batch, np.array(stream, dtype=np.int8))


@pytest.mark.parametrize("seed", range(5))
def test_fast_run_matches_step_with_nans_degraded_and_segments(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    s = rng.random(n)
    s[rng.random(n) < 0.05] = np.nan
    deg = rng.random(n) < 0.03
    seg = np.cumsum(rng.random(n) < 0.002)
    cfg = _cfg(on_threshold=0.6, off_threshold=0.4, on_delay_s=3, off_delay_s=5)
    batch = HysteresisAlerter(cfg).run(s, degraded=deg, segments=seg)
    a = HysteresisAlerter(cfg)
    stream = []
    for i in range(n):
        if i and seg[i] != seg[i - 1]:
            a.reset()
        stream.append(a.step(s[i], bool(deg[i])))
    assert np.array_equal(batch, np.array(stream, dtype=np.int8))


def test_episode_split_at_segment_gap():
    act = np.ones(10, bool)
    seg = np.r_[np.zeros(5), np.ones(5)]
    assert alert_episodes(act, seg).tolist() == [[0, 5], [5, 10]]


# ---------------------------------------------------------------- metrics
def _event(n_norm=100, n_tr=50, n_ss=50, cls=2):
    lab = np.r_[np.zeros(n_norm), np.full(n_tr, 100 + cls), np.full(n_ss, cls)].astype(float)
    return np.arange(lab.size, dtype=float), lab


def test_lead_and_delay():
    t, lab = _event()
    act = np.zeros(t.size, bool)
    act[120:] = True
    o = evaluate_event_instance(t, lab, act, 2, grace_s=10)
    assert o.detected and o.delay_s == 20 and o.lead_s == 30 and o.warned_before_established


def test_early_alert_credit_capped_at_grace():
    t, lab = _event()
    act = np.ones(t.size, bool)  # alarming all the time
    o = evaluate_event_instance(t, lab, act, 2, grace_s=10)
    assert o.delay_s == -10 and o.lead_s == 60
    assert o.pre_onset_alerts == 1


def test_censored_event():
    t, lab = _event(n_ss=0)
    act = np.zeros(t.size, bool)
    act[110:] = True
    o = evaluate_event_instance(t, lab, act, 2, grace_s=0)
    assert o.detected and o.censored and np.isnan(o.lead_s) and not o.warned_before_established


def test_no_transient_class_has_no_lead():
    lab = np.full(100, 4.0)
    act = np.zeros(100, bool)
    act[30:] = True
    o = evaluate_event_instance(np.arange(100.0), lab, act, 4, grace_s=0)
    assert not o.onset_observable and o.delay_s == 30 and np.isnan(o.lead_s)


def test_missed_event():
    t, lab = _event()
    o = evaluate_event_instance(t, lab, np.zeros(t.size, bool), 2)
    assert not o.detected and np.isnan(o.delay_s)


def test_false_alarms_only_count_episodes_starting_on_normal():
    lab = np.r_[np.full(10, np.nan), np.zeros(90)]
    state = np.r_[np.zeros(60), np.ones(40)]
    act = np.zeros(100, bool)
    act[5:15] = True  # starts on unlabeled: ignored
    act[30:35] = True  # Open
    act[70:72] = True  # Shut-In
    c = count_false_alarms(lab, state, act)
    assert (c.episodes_open, c.episodes_not_open) == (1, 1)
    assert c.exposure_open_s == 50 and c.exposure_not_open_s == 40


def test_poisson_ci_zero_events():
    r, lo, hi = poisson_rate_ci(0, 116.0)
    assert r == 0 and lo == 0 and hi == pytest.approx(3.689 / 116.0, rel=1e-3)


# ---------------------------------------------------------------- data + splits
def _write(path, start, n, cls_vals):
    idx = pd.date_range(start, periods=n, freq="s", name="timestamp")
    df = pd.DataFrame({"P-TPT": np.arange(n, dtype=float), "class": cls_vals, "state": 0.0}, index=idx)
    df.to_parquet(path)


def test_stitch_dedups_overlap_keeping_labeled(tmp_path):
    d = tmp_path / "0"
    d.mkdir()
    # instance A: 1h unlabeled + 2h normal; instance B starts 1h before A ends
    _write(d / "WELL-00001_20200101000000.parquet", "2020-01-01 00:00:00", 3 * 3600, np.r_[np.full(3600, np.nan), np.zeros(7200)])
    _write(d / "WELL-00001_20200101020000.parquet", "2020-01-01 02:00:00", 3 * 3600, np.r_[np.full(3600, np.nan), np.zeros(7200)])
    metas = [InstanceMeta(p, 0, "real", "WELL-00001") for p in sorted(d.glob("*.parquet"))]
    tl = stitch_well(metas, sensors=["P-TPT"])
    assert tl.index.is_unique and len(tl) == 5 * 3600
    assert int((tl["class"] == 0).sum()) == 4 * 3600  # no double counting
    assert tl["segment"].nunique() == 1


def test_folds_are_well_disjoint_and_p2_has_no_real_events():
    cat = pd.DataFrame(
        {
            "well": [f"W{i}" for i in range(10)] * 3,
            "event_class": [0] * 10 + [2] * 10 + [7] * 10,
            "normal_days": [5.0] * 10 + [0.0] * 20,
            "source": "real",
        }
    )
    folds = make_folds(assign_well_folds(cat, n_folds=5))
    for f in folds:
        assert not set(f.test_wells) & set(f.train_wells)
    synth = pd.DataFrame({"well": [None], "event_class": [2], "normal_days": [0.0], "source": ["simulated"]})
    full = pd.concat([cat, synth], ignore_index=True)
    tr = training_selection("P2_sim_plus_site_normal", full, folds[0])
    assert set(tr[tr.source == "real"].event_class) == {0}
    assert not tr.well.isin(folds[0].test_wells).any()


def test_dataset_path_accepts_repo_root_or_dataset_folder(tmp_path):
    from ew3w.data import resolve_dataset_dir

    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "dataset.ini").write_text("[VERSION]\nDATASET = 2.0.0\n")
    assert resolve_dataset_dir(tmp_path) == tmp_path / "dataset"
    assert resolve_dataset_dir(tmp_path / "dataset") == tmp_path / "dataset"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_dataset_dir(tmp_path / "nope")


# ---------------------------------------------------------------- regression tests from the code review
def test_default_frozen_rule_ignores_ten_minute_historian_holds():
    # Healthy low-rate sensor: value held ~601 s, then steps (seen thousands of times in real P-TPT/T-TPT)
    x = np.repeat(np.arange(20, dtype=float), 601)
    assert not frozen_mask(x).any()
    stuck = np.r_[x, np.full(4000, 19.0)]  # genuinely stuck for > 1 h
    assert frozen_mask(stuck).any()


def test_detect_frozen_off_for_synthetic_data():
    df = pd.DataFrame({"P-TPT": np.ones(5000)})
    h = health_matrix(df, ["P-TPT"], segment_col=None, detect_frozen=False)
    assert (h["P-TPT"] == Health.OK).all()


def test_unbounded_scores_and_threshold_validation():
    HysteresisConfig(on_threshold=3.0, off_threshold=2.0)  # z-score style thresholds are allowed
    HysteresisConfig(on_threshold=-0.1, off_threshold=-0.3)  # isolation-forest style
    with pytest.raises(ValueError):
        HysteresisConfig(on_threshold=0.5, off_threshold=0.5)
    with pytest.raises(ValueError):
        HysteresisConfig(on_threshold=float("nan"), off_threshold=0.1)


def test_run_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        HysteresisAlerter(_cfg()).run(np.zeros(10), degraded=np.zeros(9, bool))


def test_no_grace_credit_without_observable_onset():
    lab = np.r_[np.full(50, np.nan), np.full(100, 4.0)]
    act = np.zeros(150, bool)
    act[10:20] = True  # alert in the unlabeled lead-in (belongs to another instance)
    o = evaluate_event_instance(np.arange(150.0), lab, act, 4, grace_s=600)
    assert not o.detected


def test_multiple_event_runs_rejected():
    lab = np.r_[np.full(10, 102.0), np.zeros(10), np.full(10, 102.0)]
    with pytest.raises(ValueError):
        evaluate_event_instance(np.arange(30.0), lab, np.zeros(30, bool), 2)
