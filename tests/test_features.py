"""Phase 1 gate: features are causal, streaming == batch, and robust to bad sensors."""
import numpy as np
import pandas as pd
import pytest

from ew3w.alerting import HysteresisConfig
from ew3w.features import SIGNALS, FeatureConfig, build_features
from ew3w.metrics import count_false_alarms
from ew3w.schema import SHARED_SENSORS

CFG = FeatureConfig()


def _timeline(hours=9.0, seed=0, start="2020-01-01 00:00:03"):
    """Synthetic 1 Hz well: noisy pressures, a 601 s historian-held temperature, a
    pressure step at 7 h, a NaN gap, an INVALID (+inf) burst, and labels/state."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600)
    idx = pd.date_range(start, periods=n, freq="s", name="timestamp")
    t = np.arange(n)
    p_tpt = 1.0e7 + rng.normal(0, 2e3, n) + np.where(t > 7 * 3600, -3e5, 0.0)
    p_pdg = 2.0e7 + rng.normal(0, 2e3, n)
    p_ckp = 6.0e6 + rng.normal(0, 2e3, n)
    t_tpt = np.repeat(60 + rng.normal(0, 0.05, n // 601 + 1), 601)[:n]  # held ~601 s, like real T-TPT
    t_jus = 70 + rng.normal(0, 0.02, n)
    p_ckp[2 * 3600 : 2 * 3600 + 900] = np.nan  # missing for 15 min
    p_pdg[5 * 3600 : 5 * 3600 + 120] = np.inf  # INVALID fill values
    df = pd.DataFrame(
        {"P-PDG": p_pdg, "P-TPT": p_tpt, "T-TPT": t_tpt, "P-MON-CKP": p_ckp, "T-JUS-CKP": t_jus}, index=idx
    ).astype(np.float32)
    df["class"] = np.where(t > 7 * 3600, 102.0, 0.0).astype(np.float32)
    df["state"] = np.float32(0)
    df["event_class"] = np.int8(2)
    df["instance_id"] = "2/WELL-TEST"
    return df


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-9, atol=1e-6):
    cols = CFG.feature_names()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a[cols].to_numpy(np.float64), b[cols].to_numpy(np.float64), rtol=rtol, atol=atol, equal_nan=True)


@pytest.fixture(scope="module")
def full():
    tl = _timeline()
    return tl, build_features(tl, CFG, detect_frozen=True)


def test_rows_are_epoch_aligned_complete_blocks(full):
    tl, f = full
    secs = f.index.as_unit("s").asi8
    assert np.all(secs % CFG.block_s == CFG.block_s - 1)  # timestamped at each block's last second
    assert f.index[0] >= tl.index[0] + pd.Timedelta(seconds=CFG.block_s - 1)
    assert list(f.columns[: len(CFG.feature_names())]) == CFG.feature_names()


@pytest.mark.parametrize("cut_h", [0.5, 3.0, 6.9, 7.2, 8.5])
def test_prefix_invariance_no_future_leakage(full, cut_h):
    """Features up to time t must not change when data after t is removed."""
    tl, f = full
    cut = int(cut_h * 3600)
    pre = build_features(tl.iloc[:cut], CFG, detect_frozen=True)
    _assert_same(pre, f.loc[pre.index])
    assert len(pre) == (f.index <= tl.index[cut - 1]).sum()


def test_future_perturbation_does_not_change_past(full):
    tl, f = full
    cut = int(6.5 * 3600)
    tl2 = tl.copy()
    tl2.iloc[cut:, :5] = tl2.iloc[cut:, :5] * 1.5  # wreck the future
    f2 = build_features(tl2, CFG, detect_frozen=True)
    past = f.index <= tl.index[cut - 1]
    _assert_same(f2[past], f[past])
    assert not np.allclose(f2[~past]["P-TPT__lvl_300"], f[~past]["P-TPT__lvl_300"], equal_nan=True)


def test_trailing_buffer_reproduces_full_history(full):
    """A live service keeping only cfg.history_s seconds produces identical features."""
    tl, f = full
    buf = tl.iloc[-CFG.history_s :]
    fb = build_features(buf, CFG, detect_frozen=True)
    # rolling sums restart at the buffer start: float error ~1e-12 can flip the 3rd-decimal
    # rounding, so allow one rounding step (0.001 robust-scale units)
    _assert_same(fb.iloc[[-1]], f.iloc[[-1]], rtol=0, atol=1.01 * 10.0 ** -CFG.decimals)


def test_history_s_is_tight_enough_to_matter(full):
    """Sanity: with too little history the last row DOES differ, so the test above is meaningful."""
    tl, f = full
    fb = build_features(tl.iloc[-3600:], CFG, detect_frozen=True)
    a = fb.iloc[-1][CFG.feature_names()].to_numpy(np.float64)
    b = f.iloc[-1][CFG.feature_names()].to_numpy(np.float64)
    assert not np.allclose(a, b, equal_nan=True)


def test_state_resets_at_gaps():
    tl = _timeline(hours=9.0)
    gap_at = 4 * 3600
    tl2 = pd.concat([tl.iloc[:gap_at], tl.iloc[gap_at + 600 :]])  # 10-min data gap
    f = build_features(tl2, CFG, detect_frozen=True)
    after = build_features(tl.iloc[gap_at + 600 :], CFG, detect_frozen=True)
    _assert_same(f.loc[after.index], after)


def test_bad_values_never_leak_into_features(full):
    _, f = full
    x = f[CFG.feature_names()].to_numpy(np.float64)
    assert not np.isinf(x).any()
    burst = (f.index >= "2020-01-01 05:00:10") & (f.index < "2020-01-01 05:02:00")
    assert (f.loc[burst, "P-PDG__health"] == 3).all()  # INVALID
    # 60 s window needs >= 50% usable blocks: once 30 s of the burst have passed it is NaN
    late = (f.index >= "2020-01-01 05:00:39") & (f.index < "2020-01-01 05:02:00")
    assert f.loc[late, "dP_PDG_TPT__lvl_60"].isna().all()  # pair needs both sensors OK
    assert (f.loc[burst, "n_ok"] == 4).all()


def test_held_signal_age_feature(full):
    _, f = full
    age = f["T-TPT__age"].dropna()
    assert age.max() >= 590 and age.max() <= 600  # ~601 s historian holds are visible, not frozen
    assert (f["T-TPT__health"] == 0).all()


def test_scale_floor_makes_noise_free_signals_comparable():
    """Simulated steady states are noise-free. A 1 bar step must read ~2 floor units
    (1e5 Pa / 5e4 Pa floor), not explode because the raw scale is ~0."""
    n = 9 * 3600
    idx = pd.date_range("2020-01-01", periods=n, freq="s", name="timestamp")
    p = np.where(np.arange(n) > 8 * 3600, 1.0e7 - 1.0e5, 1.0e7)
    df = pd.DataFrame({s: np.float32(50.0) for s in SHARED_SENSORS}, index=idx)
    df["P-TPT"] = p.astype(np.float32)
    f = build_features(df, CFG, detect_frozen=False)
    lvl = f["P-TPT__lvl_300"].iloc[-1]
    assert lvl == pytest.approx(-2.0, abs=2e-3)
    assert np.isfinite(f[[c for c in CFG.feature_names() if "__lvl_" in c]].iloc[-1].dropna()).all()


def test_signals_cover_pairs_and_names_unique():
    names = CFG.feature_names()
    assert len(names) == len(set(names))
    assert len(SIGNALS) == 7 and len(names) == 7 * 3 * 3 + 5 * 3 + 1


def test_config_key_changes_with_any_parameter():
    assert FeatureConfig().key == FeatureConfig().key
    assert FeatureConfig(pressure_floor=1e5).key != FeatureConfig().key
    with pytest.raises(ValueError):
        FeatureConfig(windows_s=(65,))


def test_alerter_and_exposure_respect_period():
    c = HysteresisConfig(on_delay_s=60, off_delay_s=300, period_s=10)
    assert (c.on_steps, c.off_steps) == (6, 30)
    lab = np.zeros(8640)  # one day of 10 s rows
    fa = count_false_alarms(lab, np.zeros(8640), np.zeros(8640, bool), period_s=10)
    assert fa.exposure_s == 86_400
