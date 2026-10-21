"""Instance discovery, loading, and stitching of real instances into well timelines.

Audit facts this module is built around (reports/data_audit.md):
  * 1,119 real, 1,089 simulated and 20 hand-drawn instances, all at 1 Hz.
  * Every real instance starts with 3,600 unlabeled samples (class = NaN). When two
    instances of the same well are consecutive, that unlabeled hour is a duplicate of
    the previous instance's last hour (910 of 1,119 real instances overlap their
    predecessor, always only inside the unlabeled hour). Counting files naively
    double-counts exposure: 140.7 normal well-days naive vs 116.0 de-duplicated.
  * 7 overlaps cross class folders (e.g. a normal instance follows a class-4 one), so
    a random split by *instance* leaks the same seconds into train and test. Split by well.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .schema import (
    ALL_SENSORS,
    DRAWN_PRESSURE_TO_PA,
    PRESSURE_SENSORS,
    SAMPLE_PERIOD_S,
    plausible_range,
    source_from_stem,
)


@dataclass(frozen=True)
class InstanceMeta:
    path: Path
    event_class: int  # folder class: 0 (normal) or 1..9
    source: str  # real | simulated | drawn
    well: str | None  # e.g. "WELL-00001"; None for simulated/drawn

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def instance_id(self) -> str:
        return f"{self.event_class}/{self.stem}"


def resolve_dataset_dir(path: str | Path) -> Path:
    """Accept either the 3W repo root or its 'dataset' subfolder; return the subfolder."""
    p = Path(path).expanduser()
    for cand in (p, p / "dataset"):
        if (cand / "dataset.ini").is_file():
            return cand
    if not p.exists():
        why = "that folder does not exist"
    else:
        why = "it contains no dataset.ini (neither it nor its 'dataset' subfolder is the 3W dataset)"
    raise FileNotFoundError(
        f"Cannot use '{p}' as the 3W dataset: {why}. Point it at your 3W clone, "
        f"e.g. C:/Users/<you>/Desktop/3W or C:/Users/<you>/Desktop/3W/dataset "
        f"(the folder that holds dataset.ini and the 0..9 subfolders)."
    )


def discover(dataset_dir: str | Path) -> list[InstanceMeta]:
    """List every instance under <3W repo>/dataset/{0..9}/*.parquet.

    `dataset_dir` may be the 3W repo root or its 'dataset' subfolder.
    """
    root = resolve_dataset_dir(dataset_dir)
    metas: list[InstanceMeta] = []
    for cls in range(10):
        for p in sorted((root / str(cls)).glob("*.parquet")):
            src = source_from_stem(p.stem)
            well = p.stem.split("_")[0] if src == "real" else None
            metas.append(InstanceMeta(p, cls, src, well))
    if not metas:
        raise FileNotFoundError(f"No parquet instances found under {root}")
    return metas


def catalog(metas: Sequence[InstanceMeta]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": [m.instance_id for m in metas],
            "event_class": [m.event_class for m in metas],
            "source": [m.source for m in metas],
            "well": [m.well for m in metas],
            "path": [str(m.path) for m in metas],
        }
    )


def load_instance(meta: InstanceMeta, sensors: Sequence[str] | None = None) -> pd.DataFrame:
    """Load one instance as float32 sensors + 'class' (float, NaN=unlabeled) + 'state'.

    The index is a tz-naive DatetimeIndex named 'timestamp'. Sensors that the source
    never recorded stay as all-NaN columns, so every frame has the same schema.
    Units are harmonised here: hand-drawn pressures are converted from bar to Pa.
    """
    sensors = list(ALL_SENSORS if sensors is None else sensors)
    df = pd.read_parquet(meta.path, columns=[*sensors, "class", "state"])
    # Normalise to ns so every frame in the project has the same timestamp resolution.
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="timestamp").as_unit("ns")
    out = pd.DataFrame(index=df.index)
    for s in sensors:
        x = df[s].to_numpy(dtype=np.float64)
        if meta.source == "drawn" and s in PRESSURE_SENSORS:
            x = x * DRAWN_PRESSURE_TO_PA  # hand-drawn pressures are in bar (see schema.py)
        lo, hi = plausible_range(s)
        # Implausible values become +inf: survives the float32 cast, and quality.py
        # reports it as INVALID (distinct from MISSING = NaN).
        x = np.where(np.isfinite(x) & ((x < lo) | (x > hi)), np.inf, x)
        out[s] = x.astype(np.float32)
    out["class"] = df["class"].astype("float32")
    out["state"] = df["state"].astype("float32")
    out["event_class"] = np.int8(meta.event_class)
    out["instance_id"] = meta.instance_id
    if not out.index.is_monotonic_increasing:
        raise ValueError(f"{meta.instance_id}: timestamps not monotonic")
    return out


def iter_instances(
    metas: Iterable[InstanceMeta], sensors: Sequence[str] | None = None
) -> Iterator[tuple[InstanceMeta, pd.DataFrame]]:
    for m in metas:
        yield m, load_instance(m, sensors)


def stitch_well(metas: Sequence[InstanceMeta], sensors: Sequence[str] | None = None) -> pd.DataFrame:
    """Merge all real instances of ONE well into a single de-duplicated timeline.

    Where instances overlap, the earlier instance's (labeled) sample wins over the
    later instance's unlabeled lead-in. A 'segment' column increments at every gap
    > 1 sample, so stateful consumers (features, alerting) must reset per segment.
    """
    wells = {m.well for m in metas}
    if len(wells) != 1 or None in wells:
        raise ValueError(f"stitch_well expects real instances of exactly one well, got {wells}")
    frames = [load_instance(m, sensors) for m in sorted(metas, key=lambda m: m.stem)]
    df = pd.concat(frames).sort_index(kind="stable")
    # Among duplicate timestamps keep the labeled one; ties -> earliest instance.
    df["_unlab"] = df["class"].isna()
    df = df.sort_values(["_unlab"], kind="stable").sort_index(kind="stable")
    df = df[~df.index.duplicated(keep="first")].drop(columns="_unlab")
    gaps = df.index.to_series().diff().dt.total_seconds().fillna(SAMPLE_PERIOD_S)
    df["segment"] = (gaps > SAMPLE_PERIOD_S).cumsum().astype(np.int32)
    df["well"] = next(iter(wells))
    return df


def labeled_exposure_days(timeline: pd.DataFrame, label: int = 0) -> float:
    """Seconds carrying `label` in a de-duplicated timeline, in days (1 Hz data)."""
    return float((timeline["class"] == label).sum()) * SAMPLE_PERIOD_S / 86_400.0


def seconds_since_start(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - index[0]).total_seconds()).to_numpy(dtype=np.float64)
