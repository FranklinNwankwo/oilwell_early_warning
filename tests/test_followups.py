"""Phase 6: the warm-up variant and the operator-visible false-alarm count."""
import numpy as np
import pandas as pd

from ew3w.baselines import PhysicsRules, PhysicsRulesWarmup
from ew3w.metrics import count_false_alarms, visible_false_alarm_episodes


def _rows(osc_level_known: bool) -> pd.DataFrame:
    cols = set(PhysicsRules.columns) | set(PhysicsRulesWarmup.columns)
    d = {c: [np.nan] for c in cols}
    d["P-MON-CKP__std_1800"] = [3.2]  # noisy choke pressure divided by the floor scale
    d["P-MON-CKP__lvl_1800"] = [0.1 if osc_level_known else np.nan]
    return pd.DataFrame(d)


def test_warmup_variant_needs_a_baseline_for_the_oscillation_rule():
    assert PhysicsRules().score(_rows(False))[0] == 3.2  # evaluated rules: scores on the floor scale
    assert np.isnan(PhysicsRulesWarmup().score(_rows(False))[0])  # variant: no evidence yet
    assert PhysicsRulesWarmup().score(_rows(True))[0] == 3.2  # once the baseline exists, identical


def test_warmup_variant_changes_nothing_else():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(0, 2, 500) for c in PhysicsRulesWarmup.columns})
    np.testing.assert_array_equal(PhysicsRules().score(df), PhysicsRulesWarmup().score(df))


def test_visible_count_includes_alarms_raised_in_the_unlabeled_hour():
    labels = np.r_[np.full(6, np.nan), np.zeros(14)]
    active = np.zeros(20, bool)
    active[3:9] = True  # raised in the unlabeled lead-in, still on when normal labels begin
    active[12:15] = True  # raised on normal data
    official = count_false_alarms(labels, np.zeros(20), active)
    visible = visible_false_alarm_episodes(labels, active)
    assert official.episodes == 1
    assert len(visible) == 2
    assert [v["starts_on_normal"] for v in visible] == [False, True]
    assert visible[0]["duration_s"] == 6


def test_alarm_entirely_inside_the_unlabeled_hour_is_not_counted():
    labels = np.r_[np.full(6, np.nan), np.zeros(4)]
    active = np.zeros(10, bool)
    active[1:4] = True
    assert visible_false_alarm_episodes(labels, active) == []
