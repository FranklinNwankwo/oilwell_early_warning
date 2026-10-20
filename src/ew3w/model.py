"""Phase 3 main model: gradient-boosted trees on value features, with sensor dropout
that removes missingness as a label shortcut.

The shortcut, concretely. Under P2 the training set is simulated EVENTS plus real
NORMAL rows. Simulated sensors are always healthy; real ones often are not (real
normal P-PDG is usable 18% of the time). A tree model given NaN-able features learns
"P-PDG present => event", which is a statement about data sources, not about wells.
Phase 2 showed the same trap in miniature: the health-only baseline.

The fix is MISSINGNESS-MATCHED DROPOUT. Every synthetic training row receives the
sensor-availability pattern of a randomly drawn real row from the training wells
(the 5-sensor OK/not-OK vector), and its features for unavailable sensors, and for
pair signals involving them, are set to NaN. After that, availability carries no
information about the label, because both classes draw it from the same distribution.
Under P1 (no real data may be used at all), each sensor is dropped independently
with a fixed prior probability instead.

Inputs are the 63 value features only (level / change / variability, 7 signals x 3
windows). Health codes, usable fractions, staleness and n_ok are NOT inputs: they
describe data plumbing, and they are exactly what the shortcut runs on.

Row sampling: training rows are capped per (instance, label) so that one long event
(class 7 transients last days; simulated class 5 has 439 instances) cannot dominate,
and so that each fit takes ~1 minute on a laptop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .baselines import VALUE_COLS
from .evaluation import event_target
from .features import PAIR_SIGNALS
from .schema import SHARED_SENSORS

HEALTH_CODE_COLS = [f"{s}__health" for s in SHARED_SENSORS]


def _signal_columns(sensor: str) -> list[str]:
    """Value columns that become NaN when `sensor` is unavailable (its own + its pairs)."""
    sigs = [sensor] + [name for name, a, b in PAIR_SIGNALS if sensor in (a, b)]
    return [c for c in VALUE_COLS if c.split("__")[0] in sigs]


SENSOR_COLUMN_INDEX = {s: [VALUE_COLS.index(c) for c in _signal_columns(s)] for s in SHARED_SENSORS}


def availability(df: pd.DataFrame) -> np.ndarray:
    """(n, 5) bool: sensor usable (health code OK) at each row."""
    return df[HEALTH_CODE_COLS].to_numpy() == 0


def apply_mask(X: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Set value columns of unavailable sensors (and their pair signals) to NaN, in place."""
    for j, s in enumerate(SHARED_SENSORS):
        rows = ~avail[:, j]
        if rows.any():
            idx = SENSOR_COLUMN_INDEX[s]
            X[np.ix_(rows, idx)] = np.nan
    return X


def cap_rows(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """At most `per_group` rows per (instance, label phase): every instance counts, no
    instance dominates. Deterministic."""
    y = event_target(df)
    phase = np.where(np.isnan(y), -1, np.where(df["class"].to_numpy() > 100, 2, y)).astype(np.int8)
    key = df["instance_id"].astype(str).to_numpy()
    order = np.random.default_rng(seed).permutation(len(df))
    g = pd.DataFrame({"k": key[order], "p": phase[order], "i": order})
    keep = g.groupby(["k", "p"], sort=False).head(per_group)["i"].to_numpy()
    return df.iloc[np.sort(keep)]


class GBDTEarlyWarning:
    name = "gbdt"
    columns = VALUE_COLS + HEALTH_CODE_COLS  # health codes are loaded only to build masks
    needs_events = True
    fitted = True

    def __init__(self, dropout: str | None = "matched", per_group: int = 200, p_drop_prior: float = 0.4,
                 max_iter: int = 300, seed: int = 0):
        if dropout not in ("matched", None):
            raise ValueError("dropout must be 'matched' or None")
        self.dropout, self.per_group, self.p_drop_prior = dropout, per_group, p_drop_prior
        self.max_iter, self.seed = max_iter, seed
        self.model: HistGradientBoostingClassifier | None = None
        self.mask_source = "none"

    def prepare_training(self, train: pd.DataFrame) -> pd.DataFrame:
        """Called once per fold by the harness: labeled rows, capped per (instance, phase).
        Idempotent, so fit() can apply it again safely."""
        return cap_rows(train[train["class"].notna()], self.per_group, self.seed)

    def _training_matrix(self, train: pd.DataFrame, normal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # `train` already contains the calibration-normal rows under every protocol
        # (evaluation.training_rows is a superset of evaluation.calibration_normal).
        rows = self.prepare_training(train)
        y = event_target(rows).astype(int)
        X = rows[VALUE_COLS].to_numpy(np.float32, copy=True)  # pandas copy-on-write may hand back a read-only view
        if self.dropout == "matched":
            rng = np.random.default_rng(self.seed + 1)
            synth = (rows["source"].astype(str) != "real").to_numpy()
            real_avail = availability(rows[~synth])
            if synth.any():
                if len(real_avail):
                    # transplant real availability patterns onto synthetic rows
                    pick = rng.integers(0, len(real_avail), synth.sum())
                    av = real_avail[pick]
                    self.mask_source = "real-normal patterns"
                else:
                    # P1: no real data allowed; independent prior dropout, keep >= 1 sensor
                    av = rng.random((synth.sum(), len(SHARED_SENSORS))) >= self.p_drop_prior
                    none = ~av.any(axis=1)
                    av[none, rng.integers(0, len(SHARED_SENSORS), none.sum())] = True
                    self.mask_source = f"prior p={self.p_drop_prior}"
                X[synth] = apply_mask(X[synth], av)
        return X, y

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "GBDTEarlyWarning":
        X, y = self._training_matrix(train, normal)
        if np.unique(y).size < 2:
            raise ValueError("training rows contain a single class")
        self.model = self._new_model().fit(X, y)
        return self

    def _new_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=self.seed,
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(df[VALUE_COLS].to_numpy(np.float32))[:, 1]
        p[df["n_ok"].to_numpy() == 0] = np.nan  # nothing observable: no evidence
        return p


class GBDTNoDropout(GBDTEarlyWarning):
    """Ablation: identical model without missingness-matched dropout."""

    name = "gbdt_no_dropout"

    def __init__(self, **kw):
        super().__init__(dropout=None, **kw)


class GBDTSimTrained(GBDTEarlyWarning):
    """Same model, trained on SIMULATED/DRAWN rows only; the site's normal history is used
    only to set the threshold (calibration), never as training data.

    Why this variant exists: under P2 the default model trains on simulated events plus
    REAL normal rows, so "looks real" and "is normal" are confounded and the model can
    learn the data source instead of the event. Physics rules, which train on nothing,
    use site normal data purely for calibration, and they are the strongest P2 baseline.
    This variant applies the same discipline to the learned model.

    fitted = False here means "never fitted on calibration rows", so the harness
    calibrates directly on its scores: those real normal rows are out-of-sample for it.
    """

    name = "gbdt_sim_trained"
    fitted = False

    def fit(self, train: pd.DataFrame, normal: pd.DataFrame) -> "GBDTSimTrained":
        synth = train[train["source"].astype(str) != "real"]
        if synth.empty:
            raise ValueError("gbdt_sim_trained needs synthetic training rows (P1, P2 or P4)")
        patterns = train[train["source"].astype(str) == "real"]
        if patterns.empty:  # P1: no real data at all -> prior dropout, as in the parent class
            return super().fit(synth, normal)
        return self._fit_with_patterns(synth, patterns)

    def _fit_with_patterns(self, synth: pd.DataFrame, patterns: pd.DataFrame) -> "GBDTSimTrained":
        rows = self.prepare_training(synth)
        y = event_target(rows).astype(int)
        X = rows[VALUE_COLS].to_numpy(np.float32, copy=True)
        rng = np.random.default_rng(self.seed + 1)
        av = availability(patterns)[rng.integers(0, len(patterns), len(rows))]
        X = apply_mask(X, av)
        self.mask_source = "site-normal patterns (real data used for masks and calibration only)"
        if np.unique(y).size < 2:
            raise ValueError("training rows contain a single class")
        self.model = self._new_model().fit(X, y)
        return self
