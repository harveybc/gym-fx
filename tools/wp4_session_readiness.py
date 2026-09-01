"""WP4 historical session-readiness package (order
agent-multi@0ca5f7af §5). CPU-only, NO model construction, NO
training, NO venue. Its purpose is to quantify EXACTLY what session
evidence exists for the weekly-flat WP4 protocol and what remains
missing — never to authorize an economic grid.

Three PROVENANCE CLASSES are kept strictly separate and one is
never promoted by joining it to another:

- BROKER_SESSION_ENVELOPE: signed venue session evidence from the
  MT5 collector (authoritative). None exists until the collector is
  activated.
- OPERATOR_EXCEPTION_CALENDAR: a reviewed operator calendar for
  exceptional closures (authoritative for the exceptions it names).
- OBSERVED_GAP: a missing interval in historical bars. Calibration
  data only — every value derived from absent bars is stamped
  GAP_OBSERVED_NOT_SESSION_AUTHORITY and can never stand in for a
  broker session envelope (work plan 42 section 3).

Only BROKER_SESSION_ENVELOPE (and, for the exceptions it names,
OPERATOR_EXCEPTION_CALENDAR) is session authority. The data-readiness
verdict is AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION only when
direct session authority actually supports the required paired
weekly units.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

WP4_MIN_PAIRED_WEEKS = 30          # the protocol minimum; not changed here
WEEKEND_MIN_HOURS = 40.0           # a broker-style weekly closure
GAP_STAMP = "GAP_OBSERVED_NOT_SESSION_AUTHORITY"

PROVENANCE_BROKER = "BROKER_SESSION_ENVELOPE"
PROVENANCE_OPERATOR = "OPERATOR_EXCEPTION_CALENDAR"
PROVENANCE_OBSERVED = "OBSERVED_GAP"
_AUTHORITATIVE = (PROVENANCE_BROKER, PROVENANCE_OPERATOR)

READINESS_STATES = (
    "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING",
    "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY",
    "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION",
)


class ProvenanceError(ValueError):
    """A provenance class was promoted by joining it to another."""


class JoinContractError(ValueError):
    """The session/bar join refuses: overlap, missing timezone,
    contradictory intervals or look-ahead."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ #
# P3.1/P3.2/P3.3/P3.4: observed-gap inventory                        #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ObservedClosureUnit:
    """One observed gap in the historical bars. NON-AUTHORITATIVE:
    stamped GAP_OBSERVED_NOT_SESSION_AUTHORITY. It is calibration
    data about collection, never a session boundary."""

    last_pre_close_at: str
    first_reopen_at: str
    gap_hours: float
    first_open_gap_return: float
    pre_close_spread: Optional[float]
    reopen_realized_vol: Optional[float]
    quote_continuity_ok: bool
    kind: str                      # weekend | holiday_or_shortened
    crosses_dst: bool
    provenance: str = PROVENANCE_OBSERVED
    stamp: str = GAP_STAMP

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "last_pre_close_at", "first_reopen_at", "gap_hours",
            "first_open_gap_return", "pre_close_spread",
            "reopen_realized_vol", "quote_continuity_ok", "kind",
            "crosses_dst", "provenance", "stamp")}


def _dst_offset_hours(stamp: pd.Timestamp, tz: str) -> float:
    if tz is None:
        return 0.0
    local = stamp.tz_convert(tz) if stamp.tzinfo else \
        stamp.tz_localize("UTC").tz_convert(tz)
    return local.utcoffset().total_seconds() / 3600.0


def inventory_weekly_closures(frame: pd.DataFrame, *,
                              bar_hours: float,
                              datetime_col: str = "DATE_TIME",
                              close_col: str = "CLOSE",
                              spread_col: Optional[str] = None,
                              vol_col: Optional[str] = None,
                              calendar_tz: Optional[str] = None
                              ) -> dict:
    """Inventory every OBSERVED closure-shaped gap and derive the
    per-unit metrics. Authoritative units are NOT produced here —
    an observed gap is never session authority. Timestamps that are
    tz-naive are read as UTC.

    Returns a dict with the observed units, counts by kind, and the
    DST-transition units flagged explicitly (P3.3)."""
    if datetime_col not in frame.columns:
        raise JoinContractError(
            f"no datetime column {datetime_col!r}")
    ts = pd.to_datetime(frame[datetime_col])
    if getattr(ts.dtype, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    order = ts.argsort()
    ts = ts.iloc[order].reset_index(drop=True)
    closes = pd.to_numeric(
        frame[close_col].iloc[order].reset_index(drop=True),
        errors="coerce")
    spread = (pd.to_numeric(frame[spread_col].iloc[order]
                            .reset_index(drop=True), errors="coerce")
              if spread_col and spread_col in frame.columns
              else None)
    vol = (pd.to_numeric(frame[vol_col].iloc[order]
                         .reset_index(drop=True), errors="coerce")
           if vol_col and vol_col in frame.columns else None)
    bar = pd.Timedelta(hours=bar_hours)
    units = []
    for i in range(1, len(ts)):
        delta = ts.iloc[i] - ts.iloc[i - 1]
        if delta <= bar:
            continue
        gap_hours = delta.total_seconds() / 3600.0
        pre_c = closes.iloc[i - 1]
        open_c = closes.iloc[i]
        gap_ret = (float(open_c / pre_c - 1.0)
                   if pd.notna(pre_c) and pd.notna(open_c)
                   and pre_c else float("nan"))
        realized = None
        if vol is not None and pd.notna(vol.iloc[i]):
            realized = float(vol.iloc[i])
        pre_spread = None
        if spread is not None and pd.notna(spread.iloc[i - 1]):
            pre_spread = float(spread.iloc[i - 1])
        crosses_dst = False
        if calendar_tz is not None:
            crosses_dst = (_dst_offset_hours(ts.iloc[i - 1],
                                             calendar_tz)
                           != _dst_offset_hours(ts.iloc[i],
                                                calendar_tz))
        kind = ("weekend" if gap_hours >= WEEKEND_MIN_HOURS
                else "holiday_or_shortened")
        units.append(ObservedClosureUnit(
            last_pre_close_at=str(ts.iloc[i - 1]),
            first_reopen_at=str(ts.iloc[i]),
            gap_hours=round(gap_hours, 4),
            first_open_gap_return=(round(gap_ret, 8)
                                   if gap_ret == gap_ret
                                   else None),
            pre_close_spread=pre_spread,
            reopen_realized_vol=realized,
            quote_continuity_ok=bool(pd.notna(open_c)),
            kind=kind, crosses_dst=crosses_dst))
    weekend = [u for u in units if u.kind == "weekend"]
    holiday = [u for u in units if u.kind != "weekend"]
    return {
        "bars": int(len(ts)),
        "span": [str(ts.iloc[0]), str(ts.iloc[-1])] if len(ts)
                else [],
        "observed_units": [u.as_dict() for u in units],
        "weekend_units": len(weekend),
        "holiday_or_shortened_units": len(holiday),
        "dst_crossing_units": sum(1 for u in units if u.crosses_dst),
        "authority_note": "every unit here is OBSERVED_GAP, stamped "
                          f"{GAP_STAMP}; none is session authority",
    }


# ------------------------------------------------------------------ #
# P3.5: provenance classes never promoted by joining                 #
# ------------------------------------------------------------------ #

def refuse_provenance_promotion(unit_provenance: str,
                                used_as: str) -> None:
    """A non-authoritative observed gap can never be USED AS session
    authority; the operator calendar authorizes only its own named
    exceptions. Anything else refuses."""
    if used_as == "session_authority" and \
            unit_provenance not in _AUTHORITATIVE:
        raise ProvenanceError(
            f"provenance {unit_provenance!r} may not be used as "
            "session authority — an observed gap is calibration "
            "data, never a broker session envelope")


# ------------------------------------------------------------------ #
# P3.6: the executable join contract                                 #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class SessionEnvelopeInterval:
    close_at: pd.Timestamp
    reopen_at: pd.Timestamp
    provenance: str

    def __post_init__(self):
        for name in ("close_at", "reopen_at"):
            stamp = getattr(self, name)
            if not isinstance(stamp, pd.Timestamp):
                raise JoinContractError(
                    f"{name} must be a pandas Timestamp")
            if stamp.tzinfo is None:
                raise JoinContractError(
                    f"{name} is timezone-naive — a session interval "
                    "without a timezone is refused")
        if not self.close_at < self.reopen_at:
            raise JoinContractError(
                "contradictory interval: close_at must precede "
                "reopen_at")
        if self.provenance not in _AUTHORITATIVE:
            raise JoinContractError(
                f"a join interval must be authoritative provenance, "
                f"got {self.provenance!r}")


def build_join_contract(intervals: Sequence[SessionEnvelopeInterval]
                        ) -> tuple:
    """Order authoritative session intervals and refuse overlap and
    contradiction. Returns the canonical tuple used to join future
    collector envelopes against historical bars."""
    canonical = sorted(intervals, key=lambda iv: iv.close_at)
    for a, b in zip(canonical, canonical[1:]):
        if b.close_at < a.reopen_at:
            raise JoinContractError(
                f"overlapping session intervals: {a.reopen_at} vs "
                f"{b.close_at}")
    return tuple(canonical)


def join_bar_to_session(bar_at: pd.Timestamp,
                        contract: tuple, *,
                        evidence_known_up_to: pd.Timestamp) -> dict:
    """Classify one historical bar against the authoritative
    contract. LOOK-AHEAD refuses: a bar may only be judged by
    session evidence whose close_at is already known at the bar's
    own time (evidence_known_up_to)."""
    if bar_at.tzinfo is None or evidence_known_up_to.tzinfo is None:
        raise JoinContractError(
            "bar and evidence horizon must be timezone-aware")
    for interval in contract:
        if interval.close_at > evidence_known_up_to:
            raise JoinContractError(
                f"look-ahead refused: session interval closing at "
                f"{interval.close_at} is not yet known at "
                f"{evidence_known_up_to}")
        if interval.close_at <= bar_at < interval.reopen_at:
            return {"in_closure": True,
                    "closure_close_at": str(interval.close_at),
                    "closure_reopen_at": str(interval.reopen_at),
                    "provenance": interval.provenance}
    return {"in_closure": False}


# ------------------------------------------------------------------ #
# P3.7: paired weekly unit count vs the protocol minimum             #
# ------------------------------------------------------------------ #

def count_paired_weekly_units(*, authoritative_units: int) -> dict:
    """Only AUTHORITATIVE units count toward the WP4 paired-week
    protocol. Fewer than the minimum is a typed INCONCLUSIVE with
    the EXACT deficit — the minimum is not changed here."""
    have = int(authoritative_units)
    deficit = max(0, WP4_MIN_PAIRED_WEEKS - have)
    return {
        "minimum_required": WP4_MIN_PAIRED_WEEKS,
        "authoritative_paired_weeks_available": have,
        "exact_deficit": deficit,
        "status": ("SUFFICIENT" if deficit == 0 else "INCONCLUSIVE"),
        "note": "observed gaps do NOT count; only broker session "
                "envelopes (and operator-calendar exceptions) are "
                "authoritative paired weekly units",
    }


# ------------------------------------------------------------------ #
# P3.9: the data-readiness verdict                                   #
# ------------------------------------------------------------------ #

def data_readiness_verdict(*, collector_active: bool,
                           authoritative_units: int,
                           observed_units: int) -> dict:
    """Only the three allowed states. AUTHORITATIVE... requires that
    direct session authority actually supports the required units."""
    paired = count_paired_weekly_units(
        authoritative_units=authoritative_units)
    if collector_active and paired["status"] == "SUFFICIENT":
        state = "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
    elif collector_active:
        state = "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING"
    else:
        state = "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
    assert state in READINESS_STATES
    return {
        "state": state,
        "collector_active": bool(collector_active),
        "authoritative_paired_weeks": authoritative_units,
        "observed_nonauthoritative_gaps": observed_units,
        "paired_week_accounting": paired,
        "economic_grid_authorized": False,
        "note": "no W1/W2 economic grid may start from this "
                "package; it quantifies missing evidence only",
    }


def build_readiness_package(frame: pd.DataFrame, *, bar_hours: float,
                            collector_active: bool = False,
                            **inventory_kwargs) -> dict:
    """The full P3 package over one historical dataset. No dataset
    can be authoritative here — the collector is the only source of
    BROKER_SESSION_ENVELOPE units, so authoritative_units is 0 until
    it is active and its envelopes are joined."""
    inv = inventory_weekly_closures(frame, bar_hours=bar_hours,
                                    **inventory_kwargs)
    verdict = data_readiness_verdict(
        collector_active=collector_active,
        authoritative_units=0,
        observed_units=len(inv["observed_units"]))
    package = {
        "schema": "gymfx.wp4.session_readiness.v1",
        "inventory": {k: v for k, v in inv.items()
                      if k != "observed_units"},
        "observed_unit_count": len(inv["observed_units"]),
        "verdict": verdict,
        "provenance_classes": [PROVENANCE_BROKER,
                               PROVENANCE_OPERATOR,
                               PROVENANCE_OBSERVED],
    }
    package["digest"] = _sha(package)
    return package
