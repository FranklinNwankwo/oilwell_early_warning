"""Integration tests against the real 3W 2.0.0 dataset. Skipped unless EW3W_DATASET
points at your 3W clone (repo root or its 'dataset' folder), e.g. (PowerShell):
    $env:EW3W_DATASET = "C:/Users/<you>/Desktop/3W"
"""
import os

import numpy as np
import pytest

from ew3w.alerting import HysteresisAlerter, HysteresisConfig, alert_active
from ew3w.data import discover, load_instance, stitch_well
from ew3w.metrics import count_false_alarms, evaluate_event_instance
from ew3w.quality import Health, health_matrix

DATASET = os.environ.get("EW3W_DATASET")
pytestmark = pytest.mark.skipif(not DATASET, reason="set EW3W_DATASET to run real-data tests")


@pytest.fixture(scope="module")
def metas():
    try:
        return discover(DATASET)
    except FileNotFoundError as e:
        # The variable was set on purpose, so a bad path is a failure, not a skip.
        pytest.fail(f"EW3W_DATASET is set but unusable. {e}", pytrace=False)


def test_counts(metas):
    src = [m.source for m in metas]
    assert (len(metas), src.count("real"), src.count("simulated"), src.count("drawn")) == (2228, 1119, 1089, 20)
    assert len({m.well for m in metas if m.well}) == 40


def test_every_scorable_real_event_instance_evaluates(metas):
    scored = skipped = 0
    for m in metas:
        if m.source != "real" or m.event_class == 0:
            continue
        df = load_instance(m, sensors=[])
        lab = df["class"].to_numpy(dtype=float)
        if not np.isin(lab, [m.event_class, 100 + m.event_class]).any():
            skipped += 1
            continue
        t = (df.index - df.index[0]).total_seconds().to_numpy()
        o = evaluate_event_instance(t, lab, np.zeros(len(df), bool), m.event_class, instance_id=m.instance_id)
        assert not o.detected
        scored += 1
    assert skipped == 43 and scored == 1119 - 594 - 43


def test_normal_exposure_and_no_alarms_with_zero_score(metas):
    by_well = {}
    for m in metas:
        if m.source == "real" and m.event_class == 0:
            by_well.setdefault(m.well, []).append(m)
    days = 0.0
    for w, ms in by_well.items():
        tl = stitch_well(ms, sensors=[])
        assert tl.index.is_unique
        st = HysteresisAlerter(HysteresisConfig()).run(np.zeros(len(tl)), segments=tl["segment"].to_numpy())
        c = count_false_alarms(tl["class"].to_numpy(), tl["state"].to_numpy(), alert_active(st), well=w)
        assert c.episodes == 0
        days += c.exposure_s / 86_400
    assert round(days, 1) == 116.0


def test_fill_values_flagged_invalid(metas):
    m = next(m for m in metas if m.instance_id == "7/WELL-00006_20180620152903")
    df = load_instance(m, ["P-PDG"])
    h = health_matrix(df, ["P-PDG"], segment_col=None)
    assert (h["P-PDG"] == Health.INVALID).mean() > 0.99
