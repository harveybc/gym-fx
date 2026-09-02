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
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey)

# C28: the trust manifest is FIXED BY THE EXECUTING PATH, not the
# caller. The production path loads this committed manifest and
# verifies it against the digest pinned in code below. The manifest
# ships in status NOT_PROVISIONED_NON_AUTHORIZING: no operational key
# exists, so NO bundle can produce collector_active=True until a
# separate operator key ceremony fixes a new manifest and pin.
PINNED_TRUST_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples" / "wp4_trust" / "session_trust_manifest.json")
PINNED_TRUST_MANIFEST_DIGEST = (
    "55b8cfc4301080e3d0e7758ef3174abfcb3e1fe45b8583d076e2ad675be511b8")
TRUST_MANIFEST_SCHEMA = "wp4.session_trust_manifest.v1"
STATUS_NOT_PROVISIONED = "NOT_PROVISIONED_NON_AUTHORIZING"
STATUS_PROVISIONED = "PROVISIONED_AUTHORIZING"

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
# C28: external root of trust FIXED BY THE EXECUTING PATH             #
# ------------------------------------------------------------------ #

def require_real_positive(name: str, value: Any) -> float:
    """C32: a real positive finite number that is NOT a bool."""
    if isinstance(value, bool) or not isinstance(
            value, (int, float)) or not math.isfinite(value) or \
            value <= 0:
        raise ReadinessError(
            f"{name} must be a real positive finite number "
            f"(not a bool), got {value!r}")
    return float(value)


@dataclass(frozen=True)
class ResolvedTrust:
    """The trust resolved from a loaded, pin-verified manifest. It
    carries authority ONLY when the manifest is provisioned with a
    real key. NOT_PROVISIONED yields authorizing=False, so no bundle
    can activate the collector."""

    authorizing: bool
    manifest_digest: str
    status: str
    public_key_hex: Optional[str]
    venue: Optional[str]
    account_fingerprint: Optional[str]
    symbol: Optional[str]
    exporter_code_digest: Optional[str]
    parser_code_digest: Optional[str]
    code_identity_digest: Optional[str]
    max_activation_age_days: float
    approving_order_reference: str
    approving_order_digest: str

    def public_key(self) -> Ed25519PublicKey:
        if not self.public_key_hex:
            raise TrustError("no public key: trust is not "
                             "provisioned")
        try:
            return Ed25519PublicKey.from_public_bytes(
                binascii.unhexlify(self.public_key_hex))
        except Exception as exc:
            raise TrustError(
                f"manifest public key malformed: {exc}") from exc


_MANIFEST_FIELDS = ("schema", "status", "public_key_hex", "venue",
                    "account_fingerprint", "symbol",
                    "exporter_code_digest", "parser_code_digest",
                    "code_identity_digest", "max_activation_age_days",
                    "approving_order_reference",
                    "approving_order_digest")


def _resolve_manifest(obj: dict, *, expected_digest: str) -> \
        ResolvedTrust:
    body = {k: v for k, v in obj.items() if k != "manifest_digest"}
    if set(body) != set(_MANIFEST_FIELDS):
        raise TrustError(
            f"trust manifest fields {sorted(body)} do not match the "
            f"required {sorted(_MANIFEST_FIELDS)}")
    if body["schema"] != TRUST_MANIFEST_SCHEMA:
        raise TrustError("trust manifest schema mismatch")
    self_digest = sha256_hex(canonical_bytes(body))
    if obj.get("manifest_digest") != self_digest:
        raise TrustError("trust manifest self-digest mismatch")
    if self_digest != expected_digest:
        raise TrustError(
            f"trust manifest digest {self_digest[:16]}… is not the "
            f"pinned {expected_digest[:16]}… — an unfixed manifest "
            "confers no authority")
    require_real_positive("max_activation_age_days",
                          body["max_activation_age_days"])
    if not isinstance(body["approving_order_reference"], str) or \
            not body["approving_order_reference"]:
        raise TrustError("approving_order_reference is empty")
    require_canonical_digest("approving_order_digest",
                             body["approving_order_digest"])
    status = body["status"]
    if status not in (STATUS_NOT_PROVISIONED, STATUS_PROVISIONED):
        raise TrustError(f"unknown manifest status {status!r}")
    authorizing = status == STATUS_PROVISIONED
    if authorizing:
        require_canonical_digest("public_key... via identity checks",
                                 body["exporter_code_digest"])
        require_canonical_digest("parser_code_digest",
                                 body["parser_code_digest"])
        require_canonical_digest("code_identity_digest",
                                 body["code_identity_digest"])
        for name in ("public_key_hex", "venue",
                     "account_fingerprint", "symbol"):
            if not body[name]:
                raise TrustError(
                    f"provisioned manifest missing {name}")
    return ResolvedTrust(
        authorizing=authorizing, manifest_digest=self_digest,
        status=status, public_key_hex=body["public_key_hex"],
        venue=body["venue"],
        account_fingerprint=body["account_fingerprint"],
        symbol=body["symbol"],
        exporter_code_digest=body["exporter_code_digest"],
        parser_code_digest=body["parser_code_digest"],
        code_identity_digest=body["code_identity_digest"],
        max_activation_age_days=float(
            body["max_activation_age_days"]),
        approving_order_reference=body["approving_order_reference"],
        approving_order_digest=body["approving_order_digest"])


def load_pinned_production_trust() -> ResolvedTrust:
    """The PRODUCTION path: load the committed manifest fixed by the
    executing path and verify it against the code-pinned digest. It
    ships NOT_PROVISIONED, so it never authorizes until a separate
    key ceremony fixes a new manifest and pin."""
    raw = PINNED_TRUST_MANIFEST_PATH.read_bytes()
    return _resolve_manifest(
        strict_json_loads(raw),
        expected_digest=PINNED_TRUST_MANIFEST_DIGEST)


def load_trust_manifest_TEST_ONLY(path: Any, *,
                                  expected_digest: str) -> \
        ResolvedTrust:
    """TEST-ONLY trust injection. The production path never calls
    this — it is the isolated-fixture door the acceptance battery
    uses to exercise a PROVISIONED manifest. Its name carries the
    warning so no production code reaches for it."""
    raw = Path(path).read_bytes()
    return _resolve_manifest(strict_json_loads(raw),
                             expected_digest=expected_digest)


def verify_signed_artifact(raw: Any, trust: ResolvedTrust, *,
                           schema: str, required_fields: Sequence[str],
                           what: str) -> dict:
    """C23/C28/C27: strict-parse, then verify the DETACHED signature
    over the canonical body under the PINNED manifest's public key.
    A field written by the producer is trusted only after it is
    checked against the resolved trust — never taken on its word."""
    if not trust.authorizing:
        raise TrustError(
            f"{what}: the trust manifest is {trust.status} — no key "
            "is provisioned and no bundle can be authoritative")
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
            "pinned manifest public key — a self-consistent bundle "
            "cannot mint authority") from exc
    return body


def _bind_identity(body: dict, trust: ResolvedTrust, what: str,
                   *, check_exporter_parser: bool) -> None:
    if body.get("venue") != trust.venue or \
            body.get("account_fingerprint") != \
            trust.account_fingerprint or \
            body.get("symbol") != trust.symbol:
        raise TrustError(
            f"{what}: not bound to the trust venue/account/symbol")
    if check_exporter_parser:
        # C28/C32: the identities the manifest fixes are canonical
        # code digests, matched exactly against the producer's fields
        if body.get("exporter_identity") != \
                trust.exporter_code_digest:
            raise TrustError(f"{what}: exporter code digest mismatch")
        if body.get("parser_identity") != trust.parser_code_digest:
            raise TrustError(f"{what}: parser code digest mismatch")
        if body.get("code_identity") != trust.code_identity_digest:
            raise TrustError(f"{what}: code identity digest mismatch")


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
                  "code_identity", "acquisition_range",
                  "exported_at", "observed_through", "intervals")
_RECEIPT_FIELDS = ("venue", "account_fingerprint", "symbol",
                   "exporter_identity", "parser_identity",
                   "code_identity", "activation_identity",
                   "activated_at", "bound_export_sha256")
_EXCEPTION_FIELDS = ("venue", "account_fingerprint", "symbol",
                     "exporter_identity", "parser_identity",
                     "code_identity", "named_intervals")


def _load_authoritative_bundle(export_raw: Any, receipt_raw: Any,
                               trust: ResolvedTrust, *,
                               operator_exceptions:
                               Optional[Sequence[Any]],
                               evaluation_as_of: datetime) -> dict:
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
    # C30: the as-of temporal contract. evaluation_as_of comes from
    # the reviewed invocation, not the bundle.
    activated = require_rfc3339_utc("activated_at",
                                    receipt["activated_at"])
    exported_at = require_rfc3339_utc("exported_at",
                                      export["exported_at"])
    observed_through = require_rfc3339_utc("observed_through",
                                           export["observed_through"])
    as_of = pd.Timestamp(evaluation_as_of).tz_convert("UTC") \
        if pd.Timestamp(evaluation_as_of).tzinfo \
        else pd.Timestamp(evaluation_as_of, tz="UTC")
    chain = [("activated_at", pd.Timestamp(activated)),
             ("observed_through", pd.Timestamp(observed_through)),
             ("exported_at", pd.Timestamp(exported_at)),
             ("evaluation_as_of", as_of)]
    for (na, a), (nb, b) in zip(chain, chain[1:]):
        if not a <= b:
            raise TrustError(
                f"as-of violation: {na} {a} must be <= {nb} {b} — "
                "future evidence refuses, it is not merely "
                "unsupported")
    if not isinstance(receipt["activation_identity"], str) or \
            not receipt["activation_identity"]:
        raise TrustError("activation_identity is empty")
    age_days = (as_of - pd.Timestamp(activated)).total_seconds() \
        / 86400.0
    if age_days > trust.max_activation_age_days:
        raise TrustError(
            f"activation is {age_days:.1f} days old, older than the "
            f"trust window {trust.max_activation_age_days}")
    acq = export["acquisition_range"]
    if not isinstance(acq, list) or len(acq) != 2:
        raise ReadinessError("acquisition_range must be [start, end]")
    acq_start = pd.Timestamp(require_rfc3339_utc(
        "acquisition_range[0]", acq[0]))
    acq_end = pd.Timestamp(require_rfc3339_utc(
        "acquisition_range[1]", acq[1]))
    if not acq_start <= acq_end:
        raise ReadinessError("acquisition_range is not ordered")
    if not acq_end <= pd.Timestamp(observed_through):
        raise TrustError(
            "acquisition_range extends past observed_through")

    def _intervals_from(raw_list, provenance, evidence_digest,
                        what):
        for raw in raw_list:
            if not isinstance(raw, dict) or \
                    set(raw) != {"close_at", "reopen_at"}:
                raise ReadinessError(
                    f"{what} must be exactly {{close_at, reopen_at}}")
            close_at = require_utc(f"{what}.close_at", raw["close_at"])
            reopen_at = require_utc(f"{what}.reopen_at",
                                    raw["reopen_at"])
            if not close_at < reopen_at:
                raise EvidenceError(
                    f"contradictory {what}: close_at must precede "
                    "reopen_at")
            if close_at < acq_start or reopen_at > acq_end:
                raise EvidenceError(
                    f"{what} lies outside the acquisition range")
            # C30: no future authoritative interval
            if reopen_at > pd.Timestamp(observed_through):
                raise TrustError(
                    f"{what} reopens at {reopen_at} after "
                    f"observed_through {observed_through} — future "
                    "evidence refuses")
            yield AuthoritativeInterval(
                trust.venue, trust.account_fingerprint, trust.symbol,
                close_at, reopen_at, provenance, evidence_digest)

    intervals, seen = [], {}
    for iv in _intervals_from(export["intervals"], PROVENANCE_BROKER,
                              export_digest, "interval"):
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
        for iv in _intervals_from(art["named_intervals"],
                                  PROVENANCE_OPERATOR, art_digest,
                                  "exception"):
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
        "exported_at": export["exported_at"],
        "observed_through": export["observed_through"],
        "evaluation_as_of": as_of.isoformat(),
        "export_digest": export_digest,
        "acquisition_range": [acq[0], acq[1]],
        "trust_digest": trust.manifest_digest,
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
                          acquisition_range: Sequence[str],
                          evaluation_as_of: pd.Timestamp) -> dict:
    require_pos_int("required_pre_bars", required_pre_bars)
    require_pos_int("required_post_bars", required_post_bars)
    work = validate_bars(frame, roles, bar_hours=bar_hours)
    ts = pd.to_datetime(work[roles.datetime_col], utc=True)
    # C30: no bar used may lie past the as-of horizon
    if len(ts) and ts.max() > evaluation_as_of:
        raise TrustError(
            f"a bar at {ts.max()} lies past evaluation_as_of "
            f"{evaluation_as_of} — future evidence refuses")
    opens = work[roles.open_col].to_numpy(dtype=float)
    closes = work[roles.close_col].to_numpy(dtype=float)
    by_stamp = {ts.iloc[i]: i for i in range(len(ts))}
    bar = pd.Timedelta(hours=bar_hours)
    acq_start = require_utc("acq[0]", acquisition_range[0])
    acq_end = require_utc("acq[1]", acquisition_range[1])
    records = []
    for iv in intervals:
        # C31: the closure is [close_at, reopen_at). A bar with
        # close_at <= ts < reopen_at is INSIDE the closure and is a
        # contradiction — refuse, never a silent supported=False.
        inside = ts[(ts >= iv.close_at) & (ts < iv.reopen_at)]
        if len(inside):
            raise JoinContractError(
                f"a bar exists inside the authoritative closure "
                f"[{iv.close_at}, {iv.reopen_at}): {inside.iloc[0]}")
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
    """C27/C32: continuity requires quote timestamps that are ORDERED
    and SUFFICIENT under a REAL positive spacing (a bool/string/zero/
    NaN is not a spacing). Fewer than two quotes, an out-of-order
    series or an absent/invalid contract are typed UNAVAILABLE — a
    price bar can never make it true."""
    if expected_spacing_seconds is not None:
        try:
            require_real_positive("expected_spacing_seconds",
                                  expected_spacing_seconds)
        except ReadinessError:
            return {"value": UNAVAILABLE,
                    "reason": "expected_spacing_seconds is not a "
                              "real positive number"}
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

@dataclass(frozen=True)
class VerifiedSource:
    """C29: ONE data population. Bytes, their digest and the parsed
    frame are inseparable — the public constructor parses the frame
    FROM the bytes, so a caller can never hand a frame that is not
    the hashed source. Two sources with distinct rows yield distinct
    digests even without observed gaps."""

    source_bytes: bytes
    source_digest: str
    source_logical_id: str
    roles: "ColumnRoleContract"
    frame: pd.DataFrame

    @staticmethod
    def from_csv_bytes(raw: bytes, *, roles: "ColumnRoleContract",
                       source_logical_id: str,
                       usecols: Optional[Sequence[str]] = None
                       ) -> "VerifiedSource":
        require_logical_id("source_logical_id", source_logical_id)
        if not isinstance(raw, (bytes, bytearray)):
            raise ReadinessError(
                "source must be provided as BYTES and parsed here, "
                "never as a detached frame")
        digest = sha256_hex(bytes(raw))
        cols = list(usecols) if usecols else [
            roles.datetime_col, roles.open_col, roles.close_col]
        frame = pd.read_csv(io.BytesIO(bytes(raw)), usecols=cols)
        return VerifiedSource(bytes(raw), digest, source_logical_id,
                              roles, frame)


def _build_package(source: VerifiedSource, trust: ResolvedTrust, *,
                   bar_hours: float, evaluation_as_of: datetime,
                   realized_vol_window_bars: int,
                   calendar_tz: Optional[str],
                   session_export: Optional[Any],
                   activation_receipt: Optional[Any],
                   required_pre_bars: int, required_post_bars: int,
                   operator_exceptions: Optional[Sequence[Any]]
                   ) -> dict:
    roles = source.roles
    frame = source.frame
    as_of = (pd.Timestamp(evaluation_as_of).tz_convert("UTC")
             if pd.Timestamp(evaluation_as_of).tzinfo
             else pd.Timestamp(evaluation_as_of, tz="UTC"))
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
    if session_export is not None and \
            activation_receipt is not None and trust.authorizing:
        bundle = _load_authoritative_bundle(
            session_export, activation_receipt, trust,
            operator_exceptions=operator_exceptions,
            evaluation_as_of=as_of)
        pairing = _derive_local_pairing(
            bundle["intervals"], frame, roles=roles,
            bar_hours=bar_hours, required_pre_bars=required_pre_bars,
            required_post_bars=required_post_bars,
            acquisition_range=bundle["acquisition_range"],
            evaluation_as_of=as_of)
        accounting = _count_paired_weeks(pairing)
        verdict_state = (
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
            if accounting["status"] == "SUFFICIENT"
            else "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING")
        authoritative_block = {
            "trust_manifest_digest": bundle["trust_digest"],
            "export_digest": bundle["export_digest"],
            "activation_identity": bundle["activation_identity"],
            "activated_at": bundle["activated_at"],
            "exported_at": bundle["exported_at"],
            "observed_through": bundle["observed_through"],
            "evaluation_as_of": bundle["evaluation_as_of"],
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
        "schema": "gymfx.wp4.session_readiness.v4",
        "source": {"logical_id": source.source_logical_id,
                   "source_digest": source.source_digest},
        "column_role_contract_digest": roles.digest(),
        "bar_hours": bar_hours,
        "timezone": roles.timezone,
        "evaluation_as_of": as_of.isoformat(),
        "trust_manifest_digest": trust.manifest_digest,
        "trust_status": trust.status,
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


def build_readiness_package(source: VerifiedSource, *,
                            bar_hours: float,
                            evaluation_as_of: datetime,
                            realized_vol_window_bars: int = 3,
                            calendar_tz: Optional[str] = None,
                            session_export: Optional[Any] = None,
                            activation_receipt: Optional[Any] = None,
                            required_pre_bars: int = 4,
                            required_post_bars: int = 4,
                            operator_exceptions:
                            Optional[Sequence[Any]] = None) -> dict:
    """PRODUCTION path. It loads the PINNED trust manifest itself and
    NEVER accepts a caller-supplied trust — the manifest ships
    NOT_PROVISIONED, so no bundle can activate the collector until a
    separate key ceremony. evaluation_as_of comes from the reviewed
    invocation and is bound into the package."""
    trust = load_pinned_production_trust()
    return _build_package(
        source, trust, bar_hours=bar_hours,
        evaluation_as_of=evaluation_as_of,
        realized_vol_window_bars=realized_vol_window_bars,
        calendar_tz=calendar_tz, session_export=session_export,
        activation_receipt=activation_receipt,
        required_pre_bars=required_pre_bars,
        required_post_bars=required_post_bars,
        operator_exceptions=operator_exceptions)


def build_readiness_package_with_trust_TEST_ONLY(
        source: VerifiedSource, trust: ResolvedTrust, *,
        bar_hours: float, evaluation_as_of: datetime,
        realized_vol_window_bars: int = 3,
        calendar_tz: Optional[str] = None,
        session_export: Optional[Any] = None,
        activation_receipt: Optional[Any] = None,
        required_pre_bars: int = 4, required_post_bars: int = 4,
        operator_exceptions: Optional[Sequence[Any]] = None) -> dict:
    """TEST-ONLY door for an isolated PROVISIONED fixture manifest.
    The production path never calls this; its name carries the
    warning so no production code injects trust."""
    return _build_package(
        source, trust, bar_hours=bar_hours,
        evaluation_as_of=evaluation_as_of,
        realized_vol_window_bars=realized_vol_window_bars,
        calendar_tz=calendar_tz, session_export=session_export,
        activation_receipt=activation_receipt,
        required_pre_bars=required_pre_bars,
        required_post_bars=required_post_bars,
        operator_exceptions=operator_exceptions)
