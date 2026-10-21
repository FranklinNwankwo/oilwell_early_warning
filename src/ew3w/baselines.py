"""Phase 2 baselines. A model only counts as a result if it beats all four at the same
false-alarm budget (evaluation.calibrate puts every scorer on one percentile scale).

  health_only   Gradient boosting on sensor-HEALTH features only (which sensors are
                present / frozen / invalid, no values). The shortcut to beat: the audit
                found health alone separates event from normal instances at ROC-AUC 0.740
                even grouped by well.
  physics_rules Hand-written rules from what each event does to the shared sensors.
                No training; what a control-room engineer could deploy tomorrow.
  zscore        Max |5-min level vs lagged 6 h baseline| over the 5 sensors. No training,
                needs no events at all.
  iforest       Isolation forest trained on NORMAL rows only (value features).

All four read only cached features (features.py), so they inherit causality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from .evaluation import event_target
from .features import FeatureConfig
from .schema import SHARED_SENSORS

_CFG = FeatureConfig()
HEALTH_COLS = [f"{s}__{k}" for s in SHARED_SENSORS for k in ("ok_300", "health")] + ["n_ok"]
VALUE_COLS = [c for c in _CFG.feature_names() if any(f"__{k}_" in c for k in ("lvl", "dlt", "std"))]


def _subsample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed) if len(df) > n else df


class HealthOnly:
    name = "health_only"
    needs_events = True
    fitted = True
    columns = HEALTH_COLS

    def __init__(self, max_rows: int = 1_000_000, seed: int = 0):
        self.max_rows, self.seed = max_rows, seed
        self.model: HistGradientBoostingClassifier | None = None
        self.constant: float | None = None

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "HealthOnly":
        d = _subsample(train, self.max_rows, self.seed)
        y = event_target(d)
        keep = np.isfinite(y)
        X, y = d.loc[keep, self.columns].to_numpy(np.float32), y[keep].astype(int)
        if np.unique(y).size < 2:
            self.constant = float(y.mean())
            return self
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                                    random_state=self.seed).fit(X, y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), self.constant if self.constant is not None else 0.0)
        return self.model.predict_proba(df[self.columns].to_numpy(np.float32))[:, 1]


def _col(df: pd.DataFrame, c: str) -> np.ndarray:
    return df[c].to_numpy(np.float64)


class PhysicsRules:
    """Each rule is a fuzzy AND (np.minimum, NaN if any input is unavailable) of signed
    robust-scale deviations; the score is the strongest rule that can be evaluated.

    rule                 sensors (sign)                         events it targets
    flow_drop            P-TPT (-), T-TPT (-)                    2 DHSV closure, 5 productivity loss
    column_block         P-PDG (+), P-TPT (-)                    2 DHSV closure, 1 BSW increase (heavier column)
    choke_restriction    P-MON-CKP (+), T-JUS-CKP (-), 5 min     6 quick restriction in PCK
    choke_scaling        P-MON-CKP (+), T-JUS-CKP (-), 30 min    7 scaling in PCK (slow)
    line_block           P-TPT - P-MON-CKP (+)                   8 hydrate in production line
    oscillation          std of P-TPT or P-MON-CKP, 30 min       3 severe slugging, 4 flow instability
    Class 9 (hydrate in service line) acts on gas-lift sensors that are not in the shared
    set, so no rule targets it.
    """

    name = "physics_rules"
    needs_events = False
    fitted = False
    columns = [
        "P-TPT__lvl_300", "T-TPT__lvl_300", "P-PDG__lvl_300", "P-MON-CKP__lvl_300", "T-JUS-CKP__lvl_300",
        "P-MON-CKP__lvl_1800", "T-JUS-CKP__lvl_1800", "dP_TPT_CKP__lvl_300", "P-TPT__std_1800", "P-MON-CKP__std_1800",
    ]

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "PhysicsRules":
        return self

    def rule_scores(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        g = lambda c: _col(df, c)  # noqa: E731
        return {
            "flow_drop": np.minimum(-g("P-TPT__lvl_300"), -g("T-TPT__lvl_300")),
            "column_block": np.minimum(g("P-PDG__lvl_300"), -g("P-TPT__lvl_300")),
            "choke_restriction": np.minimum(g("P-MON-CKP__lvl_300"), -g("T-JUS-CKP__lvl_300")),
            "choke_scaling": np.minimum(g("P-MON-CKP__lvl_1800"), -g("T-JUS-CKP__lvl_1800")),
            "line_block": g("dP_TPT_CKP__lvl_300"),
            "oscillation": np.fmax(g("P-TPT__std_1800"), g("P-MON-CKP__std_1800")),
        }

    def score(self, df: pd.DataFrame) -> np.ndarray:
        r = np.vstack(list(self.rule_scores(df).values()))
        with np.errstate(invalid="ignore"):
            return np.fmax.reduce(r, axis=0)  # NaN only if no rule is evaluable


class ZScore:
    name = "zscore"
    needs_events = False
    fitted = False
    columns = [f"{s}__lvl_300" for s in SHARED_SENSORS]

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "ZScore":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = np.abs(df[self.columns].to_numpy(np.float64))
        with np.errstate(invalid="ignore"):
            return np.fmax.reduce(x, axis=1)


class IForest:
    name = "iforest"
    needs_events = False
    fitted = True
    columns = VALUE_COLS

    def __init__(self, max_rows: int = 200_000, seed: int = 0):
        self.max_rows, self.seed = max_rows, seed
        self.model: IsolationForest | None = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        # NaN -> 0 = "at baseline": an unavailable signal contributes no anomaly evidence.
        return np.nan_to_num(df[self.columns].to_numpy(np.float32), nan=0.0)

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "IForest":
        d = _subsample(normal, self.max_rows, self.seed)
        self.model = IsolationForest(n_estimators=200, max_samples=512, random_state=self.seed, n_jobs=-1).fit(self._X(d))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        s = -self.model.score_samples(self._X(df))
        s[df["n_ok"].to_numpy() == 0] = np.nan  # nothing observable: no evidence, not "normal"
        return s


SCORERS = {"health_only": HealthOnly, "physics_rules": PhysicsRules, "zscore": ZScore, "iforest": IForest}


class PhysicsRulesWarmup(PhysicsRules):
    """Phase 6 variant (the evaluated PhysicsRules is unchanged). The oscillation rule
    divides a 30-min standard deviation by the signal's robust scale. Until a baseline
    exists (the first ~hour after every data gap) that scale falls back to its floor
    (0.5 bar), so ordinary choke-pressure noise scores high: the Phase 5 demo's false
    alarm. Here the oscillation rule only scores once the same signal's 30-min level,
    which needs that baseline, is available. Every other rule already needs it."""

    name = "physics_rules_warmup"
    columns = PhysicsRules.columns + ["P-TPT__lvl_1800", "P-MON-CKP__lvl_1800"]

    def rule_scores(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        r = super().rule_scores(df)
        g = lambda c: _col(df, c)  # noqa: E731
        osc_t = np.where(np.isfinite(g("P-TPT__lvl_1800")), g("P-TPT__std_1800"), np.nan)
        osc_c = np.where(np.isfinite(g("P-MON-CKP__lvl_1800")), g("P-MON-CKP__std_1800"), np.nan)
        r["oscillation"] = np.fmax(osc_t, osc_c)
        return r


SCORERS["physics_rules_warmup"] = PhysicsRulesWarmup
