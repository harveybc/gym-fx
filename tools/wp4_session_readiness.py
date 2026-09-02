"""WP4 historical session-readiness package (orders
agent-multi@0ca5f7af §5, C18-C22 @d198451c, C23-C27 @758d6799).
CPU-only, NO model construction, NO training, NO venue.

Quantifies EXACTLY what authoritative MT5 session evidence supports
the weekly-flat WP4 protocol and what is missing. It never authorizes
an economic grid (economic_grid_authorized is always false).

C23-C27: AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION can be born
ONLY from evidence with an EXTERNAL ROOT OF TRUST the payload cannot
invent — a detached Ed25519 signature over the export and the receipt
under a public key FIXED BY THE REVIEWED ORDER (private key never in
a public repo). A self-consistent synthetic bundle refuses because it
cannot produce a valid signature under the fixed key. There is ONE
derivation path: the package consumes evidence BYTES plus the trust
contract and derives internally activation -> intervals -> LOCAL
causal paired windows -> count -> verdict; no caller may hand it a
precomputed authority dict or count. Paired weeks require the exact
adjacent pre-close and post-reopen bars on the declared grid with no
bar inside the closure; a remote bar never satisfies a local window.
The final digest binds the trust contract, activation proof, export,
authoritative intervals, the pairing ledger (bar timestamps and
digests per side), the separate observed-gap ledger, the verified
source digest, the column-role contract, timezone, bar width and
metric windows.
"""
from __future__ import annotations

import binascii
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey)

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

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?"
    r"(Z|[+-]\d{2}:\d{2})$")


class ReadinessError(ValueError):
    """A typed refusal from the readiness package."""


class EvidenceError(ReadinessError):
    """Sealed/signed evidence failed verification."""


class TrustError(EvidenceError):
    """The external root of trust rejected the evidence."""


class JoinContractError(ReadinessError):
    """The session/bar pairing refuses: overlap, missing timezone,
    contradiction, look-ahead, a bar inside a closure."""


# ------------------------------------------------------------------ #
# strict primitives (C27)                                            #
# ------------------------------------------------------------------ #

def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str, allow_nan=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _no_dup_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ReadinessError(
                f"duplicate JSON key {key!r} — refused")
        seen[key] = value
    return seen


def _refuse_non_finite(value):
    raise ReadinessError(f"non-finite JSON constant {value!r}")


def strict_json_loads(raw: Any) -> dict:
    """C27: reject duplicate keys and non-finite constants."""
    if isinstance(raw, (dict, list)):
        # already parsed by a trusted caller; re-serialize to enforce
        raw = json.dumps(raw, allow_nan=False)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw, object_pairs_hook=_no_dup_keys,
                      parse_constant=_refuse_non_finite)


def require_canonical_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ReadinessError(
            f"{name} is not a canonical lowercase 64-hex digest")
    return value


def require_rfc3339_utc(name: str, value: Any) -> datetime:
    if not isinstance(value, str) or not _RFC3339.match(value):
        raise ReadinessError(f"{name} is not RFC3339: {value!r}")
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ReadinessError(f"{name} has no timezone")
    return stamp.tz_convert("UTC").to_pydatetime()


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
        raise JoinContractError(f"{name} is timezone-naive")
    return stamp.tz_convert("UTC")


def require_logical_id(name: str, value: Any) -> str:
    """C27: a logical identity, never a filesystem path/host."""
    if not isinstance(value, str) or not value:
        raise ReadinessError(f"{name} must be a non-empty string")
    if value.startswith("/") or value.startswith("~") or \
            ".." in value or "\\" in value or "://" in value or \
            value.startswith("\\\\"):
        raise ReadinessError(
            f"{name} looks like an absolute path/traversal/host, "
            f"not a logical id: {value!r}")
    return value


# ------------------------------------------------------------------ #
# C23: external root of trust — detached Ed25519 signatures          #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class TrustContract:
    """FIXED BY THE REVIEWED ORDER, external to every bundle. The
    public key, the identities and the bindings here cannot be
    written by the evidence producer, so a synthetic self-consistent
    bundle cannot satisfy them."""

    public_key_hex: str
    venue: str
    account_fingerprint: str
    symbol: str
    exporter_identity: str
    parser_identity: str
    code_identity: str
    max_activation_age_days: float = 3650.0

    def public_key(self) -> Ed25519PublicKey:
        try:
            return Ed25519PublicKey.from_public_bytes(
                binascii.unhexlify(self.public_key_hex))
        except Exception as exc:
            raise TrustError(
                f"trust contract public key is malformed: "
                f"{exc}") from exc

    def digest(self) -> str:
        return sha256_hex(canonical_bytes({
            "public_key_hex": self.public_key_hex,
            "venue": self.venue,
            "account_fingerprint": self.account_fingerprint,
            "symbol": self.symbol,
            "exporter_identity": self.exporter_identity,
            "parser_identity": self.parser_identity,
            "code_identity": self.code_identity,
            "max_activation_age_days": self.max_activation_age_days}))


def verify_signed_artifact(raw: Any, trust: TrustContract, *,
                           schema: str, required_fields: Sequence[str],
                           what: str) -> dict:
    """C23/C27: strict-parse, then verify the DETACHED signature over
    the canonical body under the trust contract's fixed public key.
    A field written by the producer (an 'exporter_identity' label)
    is trusted only after it is checked against the trust contract —
    never taken on its own word."""
    obj = strict_json_loads(raw)
    if "signature" not in obj:
        raise TrustError(f"{what}: no detached signature")
    signature = obj["signature"]
    if not isinstance(signature, str):
        raise TrustError(f"{what}: signature is not hex")
    body = {k: v for k, v in obj.items() if k != "signature"}
    expected = set(required_fields) | {"schema"}
    if set(body) != expected:
        raise ReadinessError(
            f"{what}: fields {sorted(body)} do not match the "
            f"required {sorted(expected)}")
    if body.get("schema") != schema:
        raise ReadinessError(
            f"{what}: schema {body.get('schema')!r} is not "
            f"{schema!r}")
    try:
        trust.public_key().verify(
            binascii.unhexlify(signature), canonical_bytes(body))
    except (InvalidSignature, binascii.Error, ValueError) as exc:
        raise TrustError(
            f"{what}: detached signature does not verify under the "
            "order-fixed public key — a self-consistent bundle "
            "cannot mint authority") from exc
    return body


def _bind_identity(body: dict, trust: TrustContract, what: str,
                   *, check_exporter_parser: bool) -> None:
    if body.get("venue") != trust.venue or \
            body.get("account_fingerprint") != \
            trust.account_fingerprint or \
            body.get("symbol") != trust.symbol:
        raise TrustError(
            f"{what}: not bound to the trust venue/account/symbol")
    if check_exporter_parser:
        if body.get("exporter_identity") != trust.exporter_identity:
            raise TrustError(f"{what}: exporter identity mismatch")
        if body.get("parser_identity") != trust.parser_identity:
            raise TrustError(f"{what}: parser identity mismatch")
        if body.get("code_identity") != trust.code_identity:
            raise TrustError(f"{what}: code identity mismatch")


# ------------------------------------------------------------------ #
# C24: the single derivation path                                    #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class AuthoritativeInterval:
    venue: str
    account_fingerprint: str
    symbol: str
    close_at: pd.Timestamp
    reopen_at: pd.Timestamp
    provenance: str
    evidence_digest: str

    def identity(self) -> tuple:
        return (self.venue, self.account_fingerprint, self.symbol,
                self.close_at.isoformat(), self.reopen_at.isoformat())


_EXPORT_FIELDS = ("venue", "account_fingerprint", "symbol",
                  "exporter_identity", "parser_identity",
                  "code_identity", "acquisition_range", "intervals")
_RECEIPT_FIELDS = ("venue", "account_fingerprint", "symbol",
                   "exporter_identity", "parser_identity",
                   "code_identity", "activation_identity",
                   "activated_at", "bound_export_sha256")
_EXCEPTION_FIELDS = ("venue", "account_fingerprint", "symbol",
                     "exporter_identity", "parser_identity",
                     "code_identity", "named_intervals")


def _load_authoritative_bundle(export_raw: Any, receipt_raw: Any,
                               trust: TrustContract, *,
                               operator_exceptions:
                               Optional[Sequence[Any]],
                               now: datetime) -> dict:
    export = verify_signed_artifact(
        export_raw, trust, schema=SESSION_EXPORT_SCHEMA,
        required_fields=_EXPORT_FIELDS, what="session export")
    receipt = verify_signed_artifact(
        receipt_raw, trust, schema=ACTIVATION_RECEIPT_SCHEMA,
        required_fields=_RECEIPT_FIELDS, what="activation receipt")
    _bind_identity(export, trust, "export",
                   check_exporter_parser=True)
    _bind_identity(receipt, trust, "receipt",
                   check_exporter_parser=True)
    export_digest = sha256_hex(canonical_bytes(export))
    if require_canonical_digest("bound_export_sha256",
                                receipt.get("bound_export_sha256")) \
            != export_digest:
        raise TrustError(
            "the receipt does not name this export's canonical "
            "digest — a transplanted receipt confers no authority")
    activated = require_rfc3339_utc("activated_at",
                                    receipt.get("activated_at"))
    if activated > now:
        raise TrustError("activation is in the future")
    age_days = (now - activated).total_seconds() / 86400.0
    if age_days > trust.max_activation_age_days:
        raise TrustError(
            f"activation is {age_days:.1f} days old, older than the "
            f"trust window {trust.max_activation_age_days}")
    if not isinstance(receipt.get("activation_identity"), str) or \
            not receipt["activation_identity"]:
        raise TrustError("activation_identity is empty")
    acq = export.get("acquisition_range")
    if not isinstance(acq, list) or len(acq) != 2:
        raise ReadinessError("acquisition_range must be [start, end]")
    acq_start = require_rfc3339_utc("acquisition_range[0]", acq[0])
    acq_end = require_rfc3339_utc("acquisition_range[1]", acq[1])
    if not acq_start < acq_end:
        raise ReadinessError("acquisition_range is not ordered")

    intervals, seen = [], {}
    for raw in export.get("intervals", []):
        if not isinstance(raw, dict) or \
                set(raw) != {"close_at", "reopen_at"}:
            raise ReadinessError(
                "interval must be exactly {close_at, reopen_at}")
        close_at = require_utc("interval.close_at", raw["close_at"])
        reopen_at = require_utc("interval.reopen_at",
                                raw["reopen_at"])
        if not close_at < reopen_at:
            raise EvidenceError(
                "contradictory interval: close_at must precede "
                "reopen_at")
        if close_at < pd.Timestamp(acq_start) or \
                reopen_at > pd.Timestamp(acq_end):
            raise EvidenceError(
                "interval lies outside the acquisition range")
        iv = AuthoritativeInterval(
            trust.venue, trust.account_fingerprint, trust.symbol,
            close_at, reopen_at, PROVENANCE_BROKER, export_digest)
        if iv.identity() in seen:
            continue
        seen[iv.identity()] = iv
        intervals.append(iv)

    for artifact in (operator_exceptions or []):
        art = verify_signed_artifact(
            artifact, trust, schema=OPERATOR_EXCEPTION_SCHEMA,
            required_fields=_EXCEPTION_FIELDS,
            what="operator exception")
        _bind_identity(art, trust, "operator exception",
                       check_exporter_parser=True)
        art_digest = sha256_hex(canonical_bytes(art))
        for raw in art.get("named_intervals", []):
            close_at = require_utc("exception.close_at",
                                   raw["close_at"])
            reopen_at = require_utc("exception.reopen_at",
                                    raw["reopen_at"])
            iv = AuthoritativeInterval(
                trust.venue, trust.account_fingerprint, trust.symbol,
                close_at, reopen_at, PROVENANCE_OPERATOR, art_digest)
            if iv.identity() in seen:
                continue
            seen[iv.identity()] = iv
            intervals.append(iv)

    intervals.sort(key=lambda iv: iv.close_at)
    for a, b in zip(intervals, intervals[1:]):
        if b.close_at < a.reopen_at:
            raise JoinContractError(
                f"overlapping authoritative intervals: {a.reopen_at}"
                f" vs {b.close_at}")
    return {
        "intervals": intervals,
        "activation_identity": receipt["activation_identity"],
        "activated_at": receipt["activated_at"],
        "export_digest": export_digest,
        "acquisition_range": [acq[0], acq[1]],
        "trust_digest": trust.digest(),
    }


# ------------------------------------------------------------------ #
# C25: local causal pairing                                          #
# ------------------------------------------------------------------ #

def _row_digest(stamp: pd.Timestamp, open_v: float,
                close_v: float) -> str:
    return sha256_hex(canonical_bytes(
        [stamp.isoformat(), float(open_v), float(close_v)]))


def _derive_local_pairing(intervals: Sequence[AuthoritativeInterval],
                          frame: pd.DataFrame, *,
                          roles: "ColumnRoleContract",
                          bar_hours: float, required_pre_bars: int,
                          required_post_bars: int,
                          acquisition_range: Sequence[str]) -> dict:
    require_pos_int("required_pre_bars", required_pre_bars)
    require_pos_int("required_post_bars", required_post_bars)
    work = validate_bars(frame, roles, bar_hours=bar_hours)
    ts = pd.to_datetime(work[roles.datetime_col], utc=True)
    opens = work[roles.open_col].to_numpy(dtype=float)
    closes = work[roles.close_col].to_numpy(dtype=float)
    by_stamp = {ts.iloc[i]: i for i in range(len(ts))}
    bar = pd.Timedelta(hours=bar_hours)
    acq_start = require_utc("acq[0]", acquisition_range[0])
    acq_end = require_utc("acq[1]", acquisition_range[1])
    records = []
    for iv in intervals:
        # C25: a bar physically INSIDE the authoritative closure is a
        # contradiction — refuse, never a silent supported=False
        inside = ts[(ts > iv.close_at) & (ts < iv.reopen_at)]
        if len(inside):
            raise JoinContractError(
                f"a bar exists inside the authoritative closure "
                f"[{iv.close_at}, {iv.reopen_at}]: {inside.iloc[0]}")
        if iv.close_at < acq_start or iv.reopen_at > acq_end:
            raise JoinContractError(
                "paired interval outside the acquisition range")

        def adjacent(anchor, count, step):
            rows = []
            cur = anchor
            for _ in range(count):
                if cur not in by_stamp:
                    return None
                j = by_stamp[cur]
                rows.append({"at": cur.isoformat(),
                             "row_digest": _row_digest(
                                 cur, opens[j], closes[j])})
                cur = cur + step
            return rows
        # pre-close: the bar at close_at - bar, then backwards
        pre = adjacent(iv.close_at - bar, required_pre_bars, -bar)
        # post-reopen: the bar at reopen_at, then forwards
        post = adjacent(iv.reopen_at, required_post_bars, bar)
        supported = pre is not None and post is not None
        records.append({
            "identity": list(iv.identity()),
            "provenance": iv.provenance,
            "evidence_digest": iv.evidence_digest,
            "pre_close_bars": pre, "post_reopen_bars": post,
            "supported": supported})
    supported = [r for r in records if r["supported"]]
    return {"records": records,
            "supported_paired_weeks": len(supported),
            "required_pre_bars": required_pre_bars,
            "required_post_bars": required_post_bars,
            "pairing_ledger_digest": sha256_hex(
                canonical_bytes(records))}


def _count_paired_weeks(pairing: dict) -> dict:
    """Recomputed from the verified records at the point of use."""
    have = sum(1 for r in pairing["records"] if r["supported"])
    deficit = max(0, WP4_MIN_PAIRED_WEEKS - have)
    return {"minimum_required": WP4_MIN_PAIRED_WEEKS,
            "supported_paired_weeks": have,
            "exact_deficit": deficit,
            "status": "SUFFICIENT" if deficit == 0
            else "INCONCLUSIVE"}


# ------------------------------------------------------------------ #
# C19/C20: observed inventory + truthful metrics (unchanged core)    #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ColumnRoleContract:
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
    require_pos_finite("bar_hours", bar_hours)
    for col in (roles.datetime_col, roles.open_col, roles.close_col):
        if col not in frame.columns:
            raise ReadinessError(f"missing required column {col!r}")
    ts = pd.to_datetime(frame[roles.datetime_col], utc=True,
                        errors="raise")
    if ts.duplicated().any():
        raise ReadinessError("duplicate timestamps — refused")
    for col in (roles.open_col, roles.close_col):
        values = frame[col]
        if not pd.api.types.is_numeric_dtype(values) or \
                pd.api.types.is_bool_dtype(values):
            raise ReadinessError(
                f"{col!r} is not numeric — no coercion on "
                "authority-bearing data")
        arr = values.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ReadinessError(f"{col!r} has non-finite values")
        if (arr <= 0).any():
            raise ReadinessError(f"{col!r} has non-positive prices")
    work = frame.copy()
    work[roles.datetime_col] = ts
    return work.sort_values(roles.datetime_col).reset_index(
        drop=True)


def opening_gap_return(reopen_open: float, pre_close: float) -> float:
    if not pre_close:
        return float("nan")
    return float(reopen_open) / float(pre_close) - 1.0


def post_reopen_realized_vol(reopen_closes: Sequence[float], *,
                             window_bars: int) -> dict:
    require_pos_int("reopen_realized_vol_window_bars", window_bars)
    closes = [float(c) for c in reopen_closes[:window_bars + 1]]
    if len(closes) < window_bars + 1:
        return {"value": UNAVAILABLE,
                "reason": f"needs {window_bars + 1} reopened closed "
                          f"bars, has {len(closes)}"}
    rets = np.diff(np.log(np.asarray(closes)))
    return {"value": round(float(np.sqrt(np.mean(rets ** 2))), 10),
            "definition": "RMS of close-to-close log returns",
            "units": "dimensionless per bar (NOT annualized)",
            "window_bars": window_bars}


def quote_continuity(quote_times: Optional[Sequence[Any]], *,
                     expected_spacing_seconds: Optional[float]
                     ) -> Any:
    """C27: continuity requires quote timestamps that are ORDERED and
    SUFFICIENT under a bound spacing contract. Fewer than two quotes,
    an out-of-order series or an absent contract are typed
    UNAVAILABLE — a price bar can never make it true."""
    if not quote_times or expected_spacing_seconds is None or \
            len(quote_times) < 2:
        return {"value": UNAVAILABLE,
                "reason": "no ordered/sufficient quote timestamps "
                          "under a bound spacing contract"}
    times = [require_utc("quote_time", t) for t in quote_times]
    if any(b <= a for a, b in zip(times, times[1:])):
        return {"value": UNAVAILABLE,
                "reason": "quote timestamps are not strictly "
                          "ordered"}
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    return {"value": all(g <= expected_spacing_seconds * 1.5
                         for g in gaps),
            "max_gap_seconds": max(gaps),
            "expected_spacing_seconds": expected_spacing_seconds}


def classify_observed_gap(pre_close_at: pd.Timestamp,
                          first_reopen_at: pd.Timestamp) -> str:
    pre = require_utc("pre_close_at", pre_close_at)
    post = require_utc("first_reopen_at", first_reopen_at)
    gap_hours = (post - pre).total_seconds() / 3600.0
    if gap_hours >= WEEKEND_MIN_HOURS and pre.weekday() in (4, 5) \
            and post.weekday() in (6, 0):
        return "weekend_shaped_observed_gap"
    if gap_hours >= 24.0:
        return "midweek_outage_shaped"
    return "other_observed_gap"


def _dst_crosses(pre: pd.Timestamp, post: pd.Timestamp,
                 tz: Optional[str]) -> bool:
    if tz is None:
        return False
    return pre.tz_convert(tz).utcoffset() != \
        post.tz_convert(tz).utcoffset()


def inventory_observed_gaps(frame: pd.DataFrame, *,
                            roles: ColumnRoleContract,
                            bar_hours: float,
                            realized_vol_window_bars: int = 3,
                            calendar_tz: Optional[str] = None
                            ) -> dict:
    work = validate_bars(frame, roles, bar_hours=bar_hours)
    ts = pd.to_datetime(work[roles.datetime_col], utc=True)
    opens = work[roles.open_col].to_numpy(dtype=float)
    closes = work[roles.close_col].to_numpy(dtype=float)
    spread_declared = roles.spread_col is not None and \
        roles.spread_col in work.columns
    spreads = (work[roles.spread_col].to_numpy(dtype=float)
               if spread_declared else None)
    bar = pd.Timedelta(hours=bar_hours)
    ledger = []
    for i in range(1, len(ts)):
        if ts.iloc[i] - ts.iloc[i - 1] <= bar:
            continue
        pre_at, post_at = ts.iloc[i - 1], ts.iloc[i]
        ledger.append({
            "last_pre_close_at": pre_at.isoformat(),
            "first_reopen_at": post_at.isoformat(),
            "gap_hours": round((post_at - pre_at).total_seconds()
                               / 3600.0, 4),
            "kind": classify_observed_gap(pre_at, post_at),
            "opening_gap_return": round(
                opening_gap_return(opens[i], closes[i - 1]), 10),
            "reopen_realized_vol": post_reopen_realized_vol(
                closes[i:i + realized_vol_window_bars + 1],
                window_bars=realized_vol_window_bars),
            "pre_close_spread": ({"value": round(float(
                spreads[i - 1]), 10)} if spread_declared else
                {"value": UNAVAILABLE}),
            "quote_continuity": quote_continuity(
                None, expected_spacing_seconds=None),
            "crosses_dst": _dst_crosses(pre_at, post_at,
                                        calendar_tz),
            "provenance": PROVENANCE_OBSERVED, "stamp": GAP_STAMP})
    kinds = {}
    for u in ledger:
        kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    return {
        "bars": int(len(ts)),
        "span": [ts.iloc[0].isoformat(), ts.iloc[-1].isoformat()]
                if len(ts) else [],
        "observed_gap_ledger": ledger,
        "observed_gap_ledger_digest": sha256_hex(
            canonical_bytes(ledger)),
        "kind_counts": kinds,
        "dst_crossing_units": sum(1 for u in ledger
                                  if u["crosses_dst"]),
    }


# ------------------------------------------------------------------ #
# C24/C26: the one wired package                                     #
# ------------------------------------------------------------------ #

def build_readiness_package(*, source_bytes: bytes,
                            source_logical_id: str,
                            roles: ColumnRoleContract,
                            bar_hours: float,
                            frame: pd.DataFrame,
                            realized_vol_window_bars: int = 3,
                            calendar_tz: Optional[str] = None,
                            session_export: Optional[Any] = None,
                            activation_receipt: Optional[Any] = None,
                            trust: Optional[TrustContract] = None,
                            required_pre_bars: int = 4,
                            required_post_bars: int = 4,
                            operator_exceptions:
                            Optional[Sequence[Any]] = None,
                            now: Optional[datetime] = None) -> dict:
    """The ONE derivation path. Authority — if any — is derived here
    from the signed bytes and the trust contract; the caller cannot
    hand in an authority dict or a count. C26: the digest binds the
    trust contract, activation proof, export, authoritative
    intervals, the pairing ledger, the observed ledger, the VERIFIED
    source digest and every contract field."""
    require_logical_id("source_logical_id", source_logical_id)
    if not isinstance(source_bytes, (bytes, bytearray)):
        raise ReadinessError(
            "source_digest must be verified from source BYTES, not "
            "supplied as prose")
    source_digest = sha256_hex(bytes(source_bytes))
    now = now or datetime.now(timezone.utc)

    inv = inventory_observed_gaps(
        frame, roles=roles, bar_hours=bar_hours,
        realized_vol_window_bars=realized_vol_window_bars,
        calendar_tz=calendar_tz)

    authoritative_block = None
    verdict_state = "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
    accounting = {"minimum_required": WP4_MIN_PAIRED_WEEKS,
                  "supported_paired_weeks": 0,
                  "exact_deficit": WP4_MIN_PAIRED_WEEKS,
                  "status": "INCONCLUSIVE"}
    if session_export is not None and activation_receipt is not None \
            and trust is not None:
        bundle = _load_authoritative_bundle(
            session_export, activation_receipt, trust,
            operator_exceptions=operator_exceptions, now=now)
        pairing = _derive_local_pairing(
            bundle["intervals"], frame, roles=roles,
            bar_hours=bar_hours, required_pre_bars=required_pre_bars,
            required_post_bars=required_post_bars,
            acquisition_range=bundle["acquisition_range"])
        accounting = _count_paired_weeks(pairing)
        verdict_state = (
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
            if accounting["status"] == "SUFFICIENT"
            else "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING")
        authoritative_block = {
            "trust_digest": bundle["trust_digest"],
            "export_digest": bundle["export_digest"],
            "activation_identity": bundle["activation_identity"],
            "activated_at": bundle["activated_at"],
            "acquisition_range": bundle["acquisition_range"],
            "authoritative_intervals": [
                list(iv.identity()) for iv in bundle["intervals"]],
            "authoritative_pairing_digest":
                pairing["pairing_ledger_digest"],
            "pairing_records": pairing["records"],
            "required_pre_bars": required_pre_bars,
            "required_post_bars": required_post_bars}
    assert verdict_state in READINESS_STATES

    package = {
        "schema": "gymfx.wp4.session_readiness.v3",
        "source": {"logical_id": source_logical_id,
                   "source_digest": source_digest},
        "column_role_contract_digest": roles.digest(),
        "bar_hours": bar_hours,
        "timezone": roles.timezone,
        "realized_vol_window_bars": realized_vol_window_bars,
        "inventory_summary": {"bars": inv["bars"], "span": inv["span"],
                              "kind_counts": inv["kind_counts"],
                              "dst_crossing_units":
                                  inv["dst_crossing_units"]},
        "observed_gap_ledger_digest":
            inv["observed_gap_ledger_digest"],
        "observed_gap_count": len(inv["observed_gap_ledger"]),
        "authoritative": authoritative_block,
        "verdict": {"state": verdict_state,
                    "collector_active": authoritative_block
                    is not None,
                    "paired_week_accounting": accounting,
                    "observed_nonauthoritative_gaps":
                        len(inv["observed_gap_ledger"]),
                    "economic_grid_authorized": False},
        "provenance_classes": [PROVENANCE_BROKER, PROVENANCE_OPERATOR,
                               PROVENANCE_OBSERVED],
        "eth_conclusion_when_spot": ETH_SPOT_CONCLUSION,
    }
    package["digest"] = sha256_hex(canonical_bytes(package))
    return package
