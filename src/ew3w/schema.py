"""Single source of truth for 3W 2.0.0 constants.

Every number here was verified against dataset/dataset.ini and a full profile of
all 2,228 instances (see reports/data_audit.md). Do not hard-code these elsewhere.
"""
from __future__ import annotations

SAMPLE_PERIOD_S = 1  # every instance (real, simulated, hand-drawn) is sampled at 1 Hz

ALL_SENSORS: tuple[str, ...] = (
    "ABER-CKGL", "ABER-CKP", "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO",
    "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO",
    "P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP", "P-MON-CKGL", "P-MON-CKP",
    "P-MON-SDV-P", "P-PDG", "PT-P", "P-TPT", "QBS", "QGL", "T-JUS-CKP", "T-MON-CKP",
    "T-PDG", "T-TPT",
)

# The only continuous sensors present in (almost) every simulated and hand-drawn
# instance. Exceptions, from the audit: simulated class 6 has no T-TPT; real class 8
# never has T-JUS-CKP; real class 2 has T-JUS-CKP in only 5% of instances.
SHARED_SENSORS: tuple[str, ...] = ("P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP")

PRESSURE_SENSORS = frozenset(s for s in ALL_SENSORS if s.startswith("P"))
FLOW_SENSORS = frozenset({"QBS", "QGL"})
OPENING_SENSORS = frozenset({"ABER-CKGL", "ABER-CKP"})
TEMPERATURE_SENSORS = frozenset(s for s in ALL_SENSORS if s.startswith("T-"))
VALVE_STATE_SENSORS = frozenset(s for s in ALL_SENSORS if s.startswith("ESTADO-"))


def plausible_range(sensor: str) -> tuple[float, float]:
    """Generous physical bounds. Anything outside is a historian fill value, not a
    measurement: the audit found P-PDG = 1.18e42 Pa and T-PDG = 1.7e38 degC in
    WELL-00006, and T-PDG up to 1.4e20 in WELL-00019/WELL-00028 (8 instance-sensors)."""
    if sensor in VALVE_STATE_SENSORS:
        return (0.0, 1.0)
    if sensor in PRESSURE_SENSORS:
        return (-1.0e6, 2.0e8)  # Pa; ~2,000 bar ceiling, slight negative allowed for gauge offset
    if sensor in TEMPERATURE_SENSORS:
        return (-50.0, 400.0)  # degC
    if sensor in FLOW_SENSORS:
        return (-1.0, 1.0e3)  # m3/s
    if sensor in OPENING_SENSORS:
        return (-5.0, 105.0)  # %
    raise KeyError(sensor)


NORMAL = 0
TRANSIENT_OFFSET = 100
EVENT_NAMES = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}
EVENT_CLASSES = tuple(range(1, 10))
# Classes 3 and 4 have no transient label: the instance is "established" from its
# first labeled sample, so onset is unobservable and lead time is undefined.
NO_TRANSIENT_CLASSES = frozenset({3, 4})

# Well operational status ("state" column), 3W 2.0.0 paper, Table 5.
STATE_NAMES = {
    0: "Open", 1: "Shut-In", 2: "Flushing Diesel", 3: "Flushing Gas", 4: "Bullheading",
    5: "Closed With Diesel", 6: "Closed With Gas", 7: "Restart", 8: "Depressurization",
}
STATE_OPEN = 0

SOURCES = ("real", "simulated", "drawn")


def source_from_stem(stem: str) -> str:
    if stem.startswith("WELL-"):
        return "real"
    if stem.startswith("SIMULATED_"):
        return "simulated"
    if stem.startswith("DRAWN_"):
        return "drawn"
    raise ValueError(f"Unrecognised 3W instance name: {stem!r}")
