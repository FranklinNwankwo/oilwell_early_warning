"""Phase 4: injected failures are exactly what the production pipeline would produce."""
import numpy as np
import pandas as pd
import pytest

from ew3w.degradation import affected_columns, drop_sensors, freeze_raw, missing_conditions
from ew3w.features import FeatureConfig, build_features
from ew3w.quality import Health
from ew3w.schema import SHARED_SENSORS

CFG = FeatureConfig()


def _timeline(hours=9.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600)
    idx = pd.date_range("2020-01-01 00:00:03", periods=n, freq="s", name="timestamp").as_unit("ns")
    t = np.arange(n)
    df = pd.DataFrame({
        "P-PDG": 2.0e7 + rng.normal(0, 2e3, n),
        "P-TPT": 1.0e7 + rng.normal(0, 2e3, n) + np.where(t > 7 * 3600, -3e5, 0.0),
        "T-TPT": 60 + rng.normal(0, 0.05, n),
        "P-MON-CKP": 6.0e6 + rng.normal(0, 2e3, n),
        "T-JUS-CKP": 70 + rng.normal(0, 0.02, n),
    }, index=idx).astype(np.float32)
    df["class"] = np.where(t > 7 * 3600, 102.0, np.where(t < 3600, np.nan, 0.0)).astype(np.float32)
    df["state"] = np.float32(0)
    df["event_class"] = np.int8(2)
    df["instance_id"] = "2/W"
    return df


@pytest.mark.parametrize("sensors", [("P-PDG",), ("P-TPT", "T-JUS-CKP"), ("P-PDG", "P-TPT", "T-TPT", "P-MON-CKP")])
def test_drop_equals_rebuild_from_raw(sensors):
    """Feature-level 'missing from the start' must equal rebuilding with the raw sensor gone."""
    tl = _timeline()
    clean = build_features(tl, CFG, detect_frozen=True)
    raw = tl.copy()
    raw[list(sensors)] = np.nan
    rebuilt = build_features(raw, CFG, detect_frozen=True)
    dropped = drop_sensors(clean, sensors)
    cols = CFG.feature_names()
    np.testing.assert_array_equal(dropped[cols].to_numpy(np.float64), rebuilt[cols].to_numpy(np.float64))


def test_drop_marks_health_and_n_ok():
    clean = build_features(_timeline(), CFG, detect_frozen=True)
    d = drop_sensors(clean, ("P-TPT",))
    assert (d["P-TPT__health"] == Health.MISSING).all()
    assert (d["n_ok"] == clean["n_ok"] - (clean["P-TPT__health"] == Health.OK)).all()
    assert d[affected_columns("P-TPT")].isna().all().all()
    assert not d[[c for c in affected_columns("T-TPT")]].isna().all().any()  # untouched sensor keeps its features


def test_missing_conditions_cover_all_subsets():
    c = missing_conditions()
    assert len(c) == 30 and len(set(c)) == 30
    assert {len(x) for x in c} == {1, 2, 3, 4}


def test_freeze_raw_holds_value_from_before_onset_to_end():
    tl = _timeline()
    fz = freeze_raw(tl, "P-TPT", seed=3)
    x = fz["P-TPT"].to_numpy()
    onset = int(np.flatnonzero(tl["class"].to_numpy() == 102)[0])
    i_f = int(np.flatnonzero(x != tl["P-TPT"].to_numpy())[0]) - 1  # last sample before values start to differ
    assert onset - 3600 <= i_f <= onset
    assert np.all(x[i_f:] == x[i_f])
    np.testing.assert_array_equal(x[:i_f], tl["P-TPT"].to_numpy()[:i_f])
    others = [s for s in SHARED_SENSORS if s != "P-TPT"]
    assert fz[others].equals(tl[others])


def test_frozen_sensor_is_flagged_after_one_hour_and_masked():
    tl = _timeline(hours=12)
    tl["class"] = np.where(np.arange(len(tl)) < 3600, np.nan, 0.0).astype(np.float32)
    tl["event_class"] = np.int8(0)
    fz = freeze_raw(tl, "P-PDG", seed=1)
    f = build_features(fz, CFG, detect_frozen=True)
    i_f = int(np.flatnonzero(fz["P-PDG"].to_numpy() != tl["P-PDG"].to_numpy())[0]) - 1
    t_f = fz.index[i_f]
    # flagged at 3,600 s; the 5-min window turns NaN once over half of it is flagged
    flagged = f.index > t_f + pd.Timedelta(seconds=3600 + 20)
    late = f.index > t_f + pd.Timedelta(seconds=3600 + 300)
    early = (f.index > t_f + pd.Timedelta(seconds=120)) & (f.index < t_f + pd.Timedelta(seconds=3000))
    assert (f.loc[flagged, "P-PDG__health"] == Health.FROZEN).all()
    assert f.loc[late, "P-PDG__lvl_300"].isna().all()
    assert (f.loc[early, "P-PDG__std_60"] == 0).all()  # before the flag: the model sees a dead-flat signal
