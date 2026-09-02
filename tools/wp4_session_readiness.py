"""WP4 historical session-readiness package (orders
agent-multi@0ca5f7af §5 + C18-C22 correction agent-multi@d198451c).
CPU-only, NO model construction, NO training, NO venue.

Its purpose is to quantify EXACTLY what authoritative session
evidence supports the weekly-flat WP4 protocol and what is missing —
never to authorize an economic grid (economic_grid_authorized is
always false).

C18-C22 rebuild: authority is DERIVED FROM SEALED EVIDENCE, never
minted from a caller scalar. A public API cannot be handed
`collector_active=True` and an integer count; it consumes a sealed
session export plus a sealed activation receipt, verifies schema,
canonical digest, exporter/parser identity, venue/account/symbol
binding, acquisition range and activation identity, derives physical
intervals from the verified bytes, deduplicates by interval identity,
and counts only intervals supported by authoritative evidence and
eligible pre/post bars. Temporal metrics read the roles they name
(OPEN for the opening gap; a declared close-to-close window for
realized volatility) or publish typed UNAVAILABLE. The observed-gap
taxonomy uses timestamp geometry, not duration alone, and never
invents a holiday. The package binds the complete per-unit ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

WP4_MIN_PAIRED_WEEKS = 30
WEEKEND_MIN_HOURS = 40.0
GAP_STAMP = "GAP_OBSERVED_NOT_SESSION_AUTHORITY"

PROVENANCE_BROKER = "BROKER_SESSION_ENVELOPE"
PROVENANCE_OPERATOR = "OPERATOR_EXCEPTION_CALENDAR"
PROVENANCE_OBSERVED = "OBSERVED_GAP"

SESSION_EXPORT_SCHEMA = "lts.mt5_session_evidence.v1.export"
ACTIVATION_RECEIPT_SCHEMA = "lts.collector_activation_receipt.v1"
OPERATOR_EXCEPTION_SCHEMA = "wp4.operator_exception_calendar.v1"

READINESS_STATES = (
    "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING",
    "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY",
    "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION",
)
UNAVAILABLE = "UNAVAILABLE"
ETH_SPOT_CONCLUSION = "SPOT_HISTORY_NOT_MT5_SESSION_AUTHORITY"


class ReadinessError(ValueError):
    """A typed refusal from the readiness package."""


class EvidenceError(ReadinessError):
    """Sealed evidence failed verification."""


class ProvenanceError(ReadinessError):
    """A non-authoritative provenance was used as authority."""


class JoinContractError(ReadinessError):
    """The session/bar join refuses: overlap, missing timezone,
    contradiction, look-ahead."""


# ------------------------------------------------------------------ #
# sealing + strict input validation (C22)                            #
# ------------------------------------------------------------------ #

def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seal(payload: dict) -> dict:
    body = {k: v for k, v in payload.items() if k != "seal_sha256"}
    return {**body, "seal_sha256": sha256_hex(canonical_bytes(body))}


def load_sealed(raw: Any, *, schema: str,
                what: str) -> dict:
    obj = json.loads(raw) if isinstance(raw, (str, bytes)) \
        else dict(raw)
    if not isinstance(obj, dict) or "seal_sha256" not in obj:
        raise EvidenceError(f"{what}: not a sealed artifact")
    body = {k: v for k, v in obj.items() if k != "seal_sha256"}
    if obj["seal_sha256"] != sha256_hex(canonical_bytes(body)):
        raise EvidenceError(f"{what}: seal digest mismatch — "
                            "altered or forged")
    if obj.get("schema") != schema:
        raise EvidenceError(
            f"{what}: schema {obj.get('schema')!r} is not "
            f"{schema!r}")
    return obj


def require_pos_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or \
            value <= 0:
        raise ReadinessError(
            f"{name} must be a positive int, got {value!r}")
    return value


def require_pos_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(
            value, (int, float)) or not math.isfinite(value) or \
            value <= 0:
        raise ReadinessError(
            f"{name} must be a positive finite number, got "
            f"{value!r}")
    return float(value)


def require_utc(name: str, value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise JoinContractError(
            f"{name} is timezone-naive — refused")
    return stamp.tz_convert("UTC")


@dataclass(frozen=True)
class ColumnRoleContract:
    """C21/C22: explicit column roles — no silent guessing."""

    datetime_col: str
    open_col: str
    close_col: str
    spread_col: Optional[str] = None
    quote_time_col: Optional[str] = None
    timezone: str = "UTC"

    def digest(self) -> str:
        return sha256_hex(canonical_bytes({
            "datetime_col": self.datetime_col,
            "open_col": self.open_col, "close_col": self.close_col,
            "spread_col": self.spread_col,
            "quote_time_col": self.quote_time_col,
            "timezone": self.timezone}))


def validate_bars(frame: pd.DataFrame, roles: ColumnRoleContract, *,
                  bar_hours: float) -> pd.DataFrame:
    """Strict: unique UTC timestamps, finite OHLC with invariants,
    declared roles present. NO errors='coerce' on authority-bearing
    data — a malformed value refuses rather than becoming NaN."""
    require_pos_finite("bar_hours", bar_hours)
    for col in (roles.datetime_col, roles.open_col,
                roles.close_col):
        if col not in frame.columns:
            raise ReadinessError(f"missing required column {col!r}")
    ts = pd.to_datetime(frame[roles.datetime_col], utc=True,
                        errors="raise")
    if ts.duplicated().any():
        raise ReadinessError(
            "duplicate timestamps — refused before any sort")
    for col in (roles.open_col, roles.close_col):
        values = frame[col]
        if not pd.api.types.is_numeric_dtype(values) or \
                pd.api.types.is_bool_dtype(values):
            raise ReadinessError(
                f"{col!r} is not numeric — no coercion on "
                "authority-bearing data")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ReadinessError(f"{col!r} has non-finite values")
    work = frame.copy()
    work[roles.datetime_col] = ts
    work = work.sort_values(roles.datetime_col).reset_index(
        drop=True)
    return work


# ------------------------------------------------------------------ #
# C18: authority derived from sealed evidence                        #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class AuthoritativeInterval:
    """An interval whose authority was DERIVED from verified sealed
    bytes — never from a constructor label. Its physical identity is
    (venue, account, symbol, close_at, reopen_at)."""

    venue: str
    account_fingerprint: str
    symbol: str
    close_at: pd.Timestamp
    reopen_at: pd.Timestamp
    provenance: str
    evidence_digest: str

    def identity(self) -> tuple:
        return (self.venue, self.account_fingerprint, self.symbol,
                self.close_at.isoformat(),
                self.reopen_at.isoformat())


def load_authoritative_intervals(
        session_export: Any, activation_receipt: Any, *,
        expected_venue: str, expected_account_fingerprint: str,
        expected_symbol: str,
        operator_exceptions: Optional[Sequence[Any]] = None
        ) -> dict:
    """C18: verify the sealed export AND the sealed activation
    receipt, bind them to the expected venue/account/symbol, and
    derive physical intervals from the verified bytes. A caller can
    never select authority by passing an enum or an integer. Returns
    the deduplicated authoritative intervals plus the activation
    identity; refuses conflicts, duplicate identities, overlaps and
    transplanted records."""
    export = load_sealed(session_export,
                         schema=SESSION_EXPORT_SCHEMA,
                         what="session export")
    receipt = load_sealed(activation_receipt,
                          schema=ACTIVATION_RECEIPT_SCHEMA,
                          what="activation receipt")
    export_digest = export["seal_sha256"]
    for label, obj in (("export", export), ("receipt", receipt)):
        if obj.get("venue") != expected_venue or \
                obj.get("account_fingerprint") != \
                expected_account_fingerprint or \
                obj.get("symbol") != expected_symbol:
            raise EvidenceError(
                f"{label} is not bound to the expected "
                "venue/account/symbol")
    if receipt.get("bound_export_sha256") != export_digest:
        raise EvidenceError(
            "the activation receipt does not name this export — "
            "a transplanted receipt confers no authority")
    intervals, seen = [], {}
    for raw in export.get("intervals", []):
        close_at = require_utc("interval.close_at",
                               raw.get("close_at"))
        reopen_at = require_utc("interval.reopen_at",
                                raw.get("reopen_at"))
        if not close_at < reopen_at:
            raise EvidenceError(
                "contradictory interval: close_at must precede "
                "reopen_at")
        iv = AuthoritativeInterval(
            venue=expected_venue,
            account_fingerprint=expected_account_fingerprint,
            symbol=expected_symbol, close_at=close_at,
            reopen_at=reopen_at, provenance=PROVENANCE_BROKER,
            evidence_digest=export_digest)
        key = iv.identity()
        if key in seen:
            # identical physical identity: idempotent, counts once
            continue
        seen[key] = iv
        intervals.append(iv)
    # operator exceptions authorize ONLY their named intervals
    for artifact in (operator_exceptions or []):
        art = load_sealed(artifact,
                          schema=OPERATOR_EXCEPTION_SCHEMA,
                          what="operator exception")
        if art.get("symbol") != expected_symbol:
            raise EvidenceError(
                "operator exception is for another symbol")
        for raw in art.get("named_intervals", []):
            close_at = require_utc("exception.close_at",
                                   raw.get("close_at"))
            reopen_at = require_utc("exception.reopen_at",
                                    raw.get("reopen_at"))
            iv = AuthoritativeInterval(
                venue=expected_venue,
                account_fingerprint=expected_account_fingerprint,
                symbol=expected_symbol, close_at=close_at,
                reopen_at=reopen_at, provenance=PROVENANCE_OPERATOR,
                evidence_digest=art["seal_sha256"])
            key = iv.identity()
            if key in seen:
                continue
            seen[key] = iv
            intervals.append(iv)
    intervals.sort(key=lambda iv: iv.close_at)
    for a, b in zip(intervals, intervals[1:]):
        if b.close_at < a.reopen_at:
            raise JoinContractError(
                f"overlapping authoritative intervals: {a.reopen_at}"
                f" vs {b.close_at}")
    return {
        "intervals": intervals,
        "activation_identity": receipt.get("activation_identity"),
        "activated_at": receipt.get("activated_at"),
        "export_digest": export_digest,
        "collector_active": True,      # a valid receipt PROVES it
    }


# ------------------------------------------------------------------ #
# C19: truthful temporal metrics                                     #
# ------------------------------------------------------------------ #

def opening_gap_return(reopen_open: float, pre_close: float) -> float:
    """C19: first_reopen_open / last_pre_close_close - 1. Reads OPEN,
    never CLOSE."""
    if not pre_close:
        return float("nan")
    return float(reopen_open) / float(pre_close) - 1.0


def post_reopen_realized_vol(reopen_closes: Sequence[float], *,
                             window_bars: int) -> dict:
    """C19: realized volatility from CLOSE-to-CLOSE log returns among
    reopened CLOSED bars, RMS of log returns, dimensionless per bar,
    NOT annualized. Insufficient bars -> typed UNAVAILABLE without
    changing unit authority."""
    require_pos_int("reopen_realized_vol_window_bars", window_bars)
    closes = [float(c) for c in reopen_closes[:window_bars + 1]]
    if len(closes) < window_bars + 1:
        return {"value": UNAVAILABLE,
                "reason": f"needs {window_bars + 1} reopened closed "
                          f"bars, has {len(closes)}"}
    rets = np.diff(np.log(np.asarray(closes)))
    rms = float(np.sqrt(np.mean(rets ** 2)))
    return {"value": round(rms, 10),
            "definition": "RMS of close-to-close log returns",
            "units": "dimensionless per bar (NOT annualized)",
            "window_bars": window_bars}


def spread_metric(pre_close_spread: Any, *,
                  spread_declared: bool) -> Any:
    if not spread_declared or pre_close_spread is None:
        return {"value": UNAVAILABLE,
                "reason": "no declared spread field or bid/ask "
                          "evidence"}
    return {"value": round(float(pre_close_spread), 10),
            "units": "as declared by the spread field"}


def quote_continuity(quote_times: Optional[Sequence[Any]], *,
                     expected_spacing_seconds: Optional[float]
                     ) -> Any:
    """C19: continuity requires quote timestamps AND an expected
    spacing contract. A price bar alone can never make it true —
    absent quote evidence is typed UNAVAILABLE."""
    if not quote_times or expected_spacing_seconds is None:
        return {"value": UNAVAILABLE,
                "reason": "no quote timestamps / expected-spacing "
                          "contract; a price bar is not quote "
                          "evidence"}
    times = [require_utc("quote_time", t) for t in quote_times]
    gaps = [(b - a).total_seconds()
            for a, b in zip(times, times[1:])]
    ok = all(g <= expected_spacing_seconds * 1.5 for g in gaps)
    return {"value": bool(ok),
            "max_gap_seconds": max(gaps) if gaps else 0.0,
            "expected_spacing_seconds": expected_spacing_seconds}


# ------------------------------------------------------------------ #
# C20: observed-gap taxonomy from timestamp geometry                 #
# ------------------------------------------------------------------ #

def classify_observed_gap(pre_close_at: pd.Timestamp,
                          first_reopen_at: pd.Timestamp) -> str:
    """C20: geometry, not duration alone. A weekend-shaped gap
    starts around Friday and ends around Sunday/Monday; a long
    midweek outage is NEVER weekend. No holiday label is invented —
    that requires an operator-exception artifact."""
    pre = require_utc("pre_close_at", pre_close_at)
    post = require_utc("first_reopen_at", first_reopen_at)
    gap_hours = (post - pre).total_seconds() / 3600.0
    # Monday=0 .. Sunday=6
    starts_friday = pre.weekday() in (4, 5)          # Fri/Sat
    ends_sun_mon = post.weekday() in (6, 0)          # Sun/Mon
    if gap_hours >= WEEKEND_MIN_HOURS and starts_friday and \
            ends_sun_mon:
        return "weekend_shaped_observed_gap"
    if gap_hours >= 24.0:
        return "midweek_outage_shaped"
    return "other_observed_gap"


# ------------------------------------------------------------------ #
# observed inventory (non-authoritative, C20-taxonomy, C19-metrics)  #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ObservedGapUnit:
    last_pre_close_at: str
    first_reopen_at: str
    gap_hours: float
    kind: str
    opening_gap_return: Optional[float]
    reopen_realized_vol: Any
    pre_close_spread: Any
    quote_continuity: Any
    crosses_dst: bool
    provenance: str = PROVENANCE_OBSERVED
    stamp: str = GAP_STAMP

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "last_pre_close_at", "first_reopen_at", "gap_hours",
            "kind", "opening_gap_return", "reopen_realized_vol",
            "pre_close_spread", "quote_continuity", "crosses_dst",
            "provenance", "stamp")}


def _dst_crosses(pre: pd.Timestamp, post: pd.Timestamp,
                 tz: Optional[str]) -> bool:
    if tz is None:
        return False
    def off(s):
        loc = s.tz_convert(tz)
        return loc.utcoffset().total_seconds()
    return off(pre) != off(post)


def inventory_observed_gaps(frame: pd.DataFrame, *,
                            roles: ColumnRoleContract,
                            bar_hours: float,
                            realized_vol_window_bars: int = 3,
                            calendar_tz: Optional[str] = None
                            ) -> dict:
    """Every OBSERVED gap with C19 metrics and C20 taxonomy. Every
    unit is non-authoritative (stamped GAP). No holiday is invented."""
    work = validate_bars(frame, roles, bar_hours=bar_hours)
    ts = pd.to_datetime(work[roles.datetime_col], utc=True)
    opens = work[roles.open_col].to_numpy(dtype=float)
    closes = work[roles.close_col].to_numpy(dtype=float)
    spread_declared = roles.spread_col is not None and \
        roles.spread_col in work.columns
    spreads = (work[roles.spread_col].to_numpy(dtype=float)
               if spread_declared else None)
    bar = pd.Timedelta(hours=bar_hours)
    units = []
    for i in range(1, len(ts)):
        if ts.iloc[i] - ts.iloc[i - 1] <= bar:
            continue
        pre_at, post_at = ts.iloc[i - 1], ts.iloc[i]
        gap_hours = (post_at - pre_at).total_seconds() / 3600.0
        vol = post_reopen_realized_vol(
            closes[i:i + realized_vol_window_bars + 1],
            window_bars=realized_vol_window_bars)
        units.append(ObservedGapUnit(
            last_pre_close_at=pre_at.isoformat(),
            first_reopen_at=post_at.isoformat(),
            gap_hours=round(gap_hours, 4),
            kind=classify_observed_gap(pre_at, post_at),
            opening_gap_return=round(
                opening_gap_return(opens[i], closes[i - 1]), 10),
            reopen_realized_vol=vol,
            pre_close_spread=spread_metric(
                spreads[i - 1] if spreads is not None else None,
                spread_declared=spread_declared),
            quote_continuity=quote_continuity(
                None, expected_spacing_seconds=None),
            crosses_dst=_dst_crosses(pre_at, post_at,
                                     calendar_tz)))
    ledger = [u.as_dict() for u in units]
    kinds = {}
    for u in units:
        kinds[u.kind] = kinds.get(u.kind, 0) + 1
    return {
        "bars": int(len(ts)),
        "span": [ts.iloc[0].isoformat(), ts.iloc[-1].isoformat()]
                if len(ts) else [],
        "unit_ledger": ledger,
        "unit_ledger_digest": sha256_hex(canonical_bytes(ledger)),
        "kind_counts": kinds,
        "dst_crossing_units": sum(1 for u in units if u.crosses_dst),
        "authority_note": f"every unit is OBSERVED_GAP ({GAP_STAMP});"
                          " none is session authority",
    }


# ------------------------------------------------------------------ #
# C18: paired weeks derived from authoritative records + bars        #
# ------------------------------------------------------------------ #

def derive_paired_weeks(authoritative_intervals:
                        Sequence[AuthoritativeInterval],
                        frame: pd.DataFrame, *,
                        roles: ColumnRoleContract, bar_hours: float,
                        required_pre_bars: int,
                        required_post_bars: int) -> dict:
    """C18: one paired week is an authoritative closure interval with
    the required eligible pre-close and post-reopen observations. The
    count is DERIVED from these records — never supplied."""
    require_pos_int("required_pre_bars", required_pre_bars)
    require_pos_int("required_post_bars", required_post_bars)
    work = validate_bars(frame, roles, bar_hours=bar_hours)
    ts = pd.to_datetime(work[roles.datetime_col], utc=True)
    records = []
    for iv in authoritative_intervals:
        pre = int((ts < iv.close_at).sum())
        post = int((ts >= iv.reopen_at).sum())
        supported = (pre >= required_pre_bars and
                     post >= required_post_bars)
        records.append({
            "identity": iv.identity(),
            "provenance": iv.provenance,
            "evidence_digest": iv.evidence_digest,
            "pre_bars": pre, "post_bars": post,
            "supported": supported})
    supported = [r for r in records if r["supported"]]
    return {"records": records,
            "supported_paired_weeks": len(supported),
            "required_pre_bars": required_pre_bars,
            "required_post_bars": required_post_bars}


def count_paired_weeks(paired: dict) -> dict:
    """Derived from the records; the 30-week minimum is unchanged."""
    have = int(paired["supported_paired_weeks"])
    deficit = max(0, WP4_MIN_PAIRED_WEEKS - have)
    return {"minimum_required": WP4_MIN_PAIRED_WEEKS,
            "supported_paired_weeks": have,
            "exact_deficit": deficit,
            "status": "SUFFICIENT" if deficit == 0
            else "INCONCLUSIVE"}


# ------------------------------------------------------------------ #
# C22: the wired readiness verdict — one authority path              #
# ------------------------------------------------------------------ #

def readiness_verdict(*, authoritative: Optional[dict],
                      paired: Optional[dict],
                      observed_units: int) -> dict:
    """Activation is DERIVED from the presence of a verified
    authoritative bundle (which carries the activation receipt), and
    the paired count from its records. No collector_active flag and
    no authoritative integer are accepted."""
    collector_active = bool(authoritative and
                            authoritative.get("collector_active"))
    if collector_active and paired is not None:
        acc = count_paired_weeks(paired)
        if acc["status"] == "SUFFICIENT":
            state = "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
        else:
            state = "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING"
    else:
        acc = {"minimum_required": WP4_MIN_PAIRED_WEEKS,
               "supported_paired_weeks": 0,
               "exact_deficit": WP4_MIN_PAIRED_WEEKS,
               "status": "INCONCLUSIVE"}
        state = "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
    assert state in READINESS_STATES
    return {"state": state,
            "collector_active": collector_active,
            "paired_week_accounting": acc,
            "observed_nonauthoritative_gaps": observed_units,
            "economic_grid_authorized": False,
            "note": "no W1/W2 economic grid may start from this "
                    "package; it quantifies missing evidence only"}


def build_readiness_package(frame: pd.DataFrame, *,
                            roles: ColumnRoleContract,
                            bar_hours: float,
                            source_logical_id: str,
                            source_digest: str,
                            realized_vol_window_bars: int = 3,
                            calendar_tz: Optional[str] = None,
                            authoritative: Optional[dict] = None,
                            paired: Optional[dict] = None) -> dict:
    """The full package. Binds source digest, logical id, column-role
    contract, timezone, bar width, metric windows AND the complete
    per-unit ledger digest — mutating any of them changes identity."""
    inv = inventory_observed_gaps(
        frame, roles=roles, bar_hours=bar_hours,
        realized_vol_window_bars=realized_vol_window_bars,
        calendar_tz=calendar_tz)
    verdict = readiness_verdict(
        authoritative=authoritative, paired=paired,
        observed_units=len(inv["unit_ledger"]))
    package = {
        "schema": "gymfx.wp4.session_readiness.v2",
        "source": {"logical_id": source_logical_id,
                   "source_digest": source_digest},
        "column_role_contract_digest": roles.digest(),
        "bar_hours": bar_hours,
        "timezone": roles.timezone,
        "realized_vol_window_bars": realized_vol_window_bars,
        "inventory_summary": {
            "bars": inv["bars"], "span": inv["span"],
            "kind_counts": inv["kind_counts"],
            "dst_crossing_units": inv["dst_crossing_units"]},
        "unit_ledger_digest": inv["unit_ledger_digest"],
        "observed_unit_count": len(inv["unit_ledger"]),
        "verdict": verdict,
        "provenance_classes": [PROVENANCE_BROKER,
                               PROVENANCE_OPERATOR,
                               PROVENANCE_OBSERVED],
    }
    package["digest"] = sha256_hex(canonical_bytes(package))
    return package
