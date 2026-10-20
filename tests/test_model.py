"""Phase 3: missingness-matched dropout removes the "which sensors are present" shortcut."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ew3w.baselines import VALUE_COLS
from ew3w.model import HEALTH_CODE_COLS, GBDTEarlyWarning, GBDTNoDropout, apply_mask, availability, cap_rows
from ew3w.schema import SHARED_SENSORS


def _frame(n, source, cls, inst_prefix, p_pdg_ok, rng):
    """Value features are pure noise for BOTH classes; only availability can differ."""
    X = rng.normal(0, 1, (n, len(VALUE_COLS))).astype(np.float32)
    health = np.zeros((n, 5), np.float32)
    health[:, SHARED_SENSORS.index("P-PDG")] = np.where(rng.random(n) < p_pdg_ok, 0, 1)
    df = pd.DataFrame(X, columns=VALUE_COLS)
    for j, c in enumerate(HEALTH_CODE_COLS):
        df[c] = health[:, j]
    av = availability(df)
    df[VALUE_COLS] = apply_mask(df[VALUE_COLS].to_numpy(np.float32, copy=True), av)
    df["class"] = np.float32(cls)
    df["instance_id"] = [f"{inst_prefix}{i // 200}" for i in range(n)]
    df["source"] = source
    df["well"] = df["instance_id"]
    df["n_ok"] = (health == 0).sum(axis=1).astype(np.float32)
    df.index = pd.date_range("2020-01-01", periods=n, freq="10s")
    return df


def _train_and_probe(model_cls):
    rng = np.random.default_rng(0)
    sim_events = _frame(6000, "simulated", 102, "S", p_pdg_ok=1.0, rng=rng)  # sim: always healthy
    real_normal = _frame(6000, "real", 0, "R", p_pdg_ok=0.2, rng=rng)  # real: P-PDG mostly dead
    train = pd.concat([sim_events, real_normal])
    m = model_cls(per_group=10_000, max_iter=60).fit(train, real_normal)
    probe = _frame(4000, "real", 0, "P", p_pdg_ok=0.5, rng=rng)  # unseen real rows, labels irrelevant
    s = m.score(probe)
    has_pdg = availability(probe)[:, SHARED_SENSORS.index("P-PDG")]
    return roc_auc_score(has_pdg, s)  # how much does the score "know" about P-PDG presence?


def test_without_dropout_model_learns_the_missingness_shortcut():
    assert _train_and_probe(GBDTNoDropout) > 0.9


def test_matched_dropout_removes_the_shortcut():
    assert abs(_train_and_probe(GBDTEarlyWarning) - 0.5) < 0.1


def test_apply_mask_blanks_sensor_and_its_pairs():
    X = np.ones((2, len(VALUE_COLS)), np.float32)
    av = np.ones((2, 5), bool)
    av[0, SHARED_SENSORS.index("P-TPT")] = False
    apply_mask(X, av)
    nan_cols = {VALUE_COLS[j].split("__")[0] for j in np.flatnonzero(np.isnan(X[0]))}
    assert nan_cols == {"P-TPT", "dP_PDG_TPT", "dP_TPT_CKP"}
    assert not np.isnan(X[1]).any()


def test_cap_rows_limits_each_instance_phase():
    idx = pd.date_range("2020-01-01", periods=1000, freq="10s")
    df = pd.DataFrame({"class": np.r_[np.zeros(500), np.full(300, 102.0), np.full(200, 2.0)],
                       "instance_id": "A"}, index=idx)
    out = cap_rows(df, 100, seed=0)
    assert len(out) == 300 and out.index.is_monotonic_increasing
    assert cap_rows(df, 100, seed=0).index.equals(out.index)  # deterministic


def test_single_class_training_is_refused():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        GBDTEarlyWarning(max_iter=5).fit(_frame(500, "real", 0, "R", 0.5, rng), _frame(10, "real", 0, "R", 0.5, rng))


def test_paired_difference_counts_and_sign():
    from ew3w.metrics import paired_difference

    base = dict(onset_observable=True, censored=False)
    a = pd.DataFrame([dict(instance_id=f"i{k}", well=f"W{k % 3}", warned_before_established=k < 8, **base) for k in range(10)])
    b = pd.DataFrame([dict(instance_id=f"i{k}", well=f"W{k % 3}", warned_before_established=k < 5, **base) for k in range(10)])
    r = paired_difference(a, b, n_boot=500)
    assert r["n"] == 10 and r["a_only"] == 3 and r["b_only"] == 0
    assert r["diff"] == pytest.approx(0.3) and r["lo"] <= 0.3 <= r["hi"]
