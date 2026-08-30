"""Parser-derived direct evidence under a POLICY-OWNED contract
(order B1/B2).

Two rules make fabrication structurally impossible:

**B1 — one representation of each fact.** Interpreted values are NEVER
constructor arguments. An allowlisted parser, selected by venue and
evidence type, derives SL/TP acceptance and position/order counts
*exclusively* from the canonical payload, and the digest covers those
exact canonical bytes. There is no way to state ``payload.positions=7``
and ``positions_total=0`` in one valid object, because only the
payload exists.

**B2 — policy owns freshness and source authority.** Evidence reports
when it was observed; it can never extend its own lifetime. The
maximum age and the allowlisted source live in a validated
``EvidencePolicy`` bound to venue/account/symbol and passed by the
custody caller. Parser and schema version are bound into the record.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from types import MappingProxyType
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from app.session_exposure import (SessionEvidenceError, require_count,
                                  require_identity, require_real,
                                  require_utc)


class EvidenceError(SessionEvidenceError):
    """The direct-evidence envelope is unusable — typed refusal."""


def decode_authority_bytes(raw: bytes) -> dict:
    """P1: authority begins at the ORIGINAL BYTES.

    Duplicate keys, non-finite JSON numbers (NaN/Infinity), invalid
    encoding and non-object roots all REFUSE. A pre-parsed mapping is
    lossy — duplicates were already collapsed — so it is never
    accepted on an authority path."""
    if not isinstance(raw, (bytes, bytearray)):
        raise EvidenceError(
            "authority requires original JSON bytes, not a "
            f"pre-parsed {type(raw).__name__} — duplicates would "
            "already be lost")
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"invalid encoding: {exc}") from exc

    def _no_duplicates(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                raise EvidenceError(
                    f"duplicate authority key {key!r} in the raw "
                    "payload — refused")
            seen[key] = value
        return seen

    def _no_constants(token):
        raise EvidenceError(
            f"non-finite JSON constant {token!r} refused")

    try:
        decoded = json.loads(text, object_pairs_hook=_no_duplicates,
                             parse_constant=_no_constants)
    except EvidenceError:
        raise
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON bytes: {exc}") from exc
    if not isinstance(decoded, dict):
        raise EvidenceError(
            "authority payload must be a JSON object")
    return decoded


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload, Mapping):
        raise EvidenceError("raw payload must be a mapping")
    return json.dumps(dict(payload), sort_keys=True,
                      separators=(",", ":"),
                      default=str).encode()


def payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise EvidenceError(
            f"{name}: a strict bool is required in the payload, got "
            f"{type(value).__name__} {value!r}")
    return value


# ---------------------------------------------------------------- #
# B1: allowlisted parsers — the ONLY path from payload to facts     #
# ---------------------------------------------------------------- #

def _parse_mt5_protection_v1(payload: Mapping[str, Any]) -> dict:
    """mt5_bridge_report protection schema v1. Authority-bearing
    fields: exactly {sl_accepted, tp_accepted, ticket}. Unknown,
    duplicate, missing or extra authority fields REFUSE."""
    required = {"sl_accepted", "tp_accepted", "ticket"}
    keys = set(payload)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise EvidenceError(
            f"protection payload missing authority fields {missing}")
    if extra:
        raise EvidenceError(
            f"protection payload carries extra authority-bearing "
            f"fields {extra} — refused")
    return {
        "stop_loss_accepted": _strict_bool("sl_accepted",
                                           payload["sl_accepted"]),
        "take_profit_accepted": _strict_bool(
            "tp_accepted", payload["tp_accepted"]),
        "ticket": require_identity("ticket",
                                   str(payload["ticket"])),
    }


def _parse_mt5_reconciliation_v1(payload: Mapping[str, Any]) -> dict:
    """mt5_bridge_report reconciliation schema v1. Authority-bearing
    fields: exactly {positions_total, orders_total}."""
    required = {"positions_total", "orders_total"}
    keys = set(payload)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise EvidenceError(
            f"reconciliation payload missing authority fields "
            f"{missing}")
    if extra:
        raise EvidenceError(
            f"reconciliation payload carries extra authority-bearing "
            f"fields {extra} — refused")
    return {
        "positions_total": require_count(
            "positions_total", payload["positions_total"], minimum=0),
        "orders_total": require_count(
            "orders_total", payload["orders_total"], minimum=0),
    }


# P2: SEALED schema definitions. Authority-bearing key sets are part
# of the identity, not a comment.
SCHEMAS = MappingProxyType({
    ("mt5_demo", "native_protection", "v1"): frozenset(
        {"sl_accepted", "tp_accepted", "ticket"}),
    ("mt5_demo", "reconciliation", "v1"): frozenset(
        {"positions_total", "orders_total"}),
})

# IMMUTABLE registry: a monkeypatch of the mapping is impossible, and
# a replaced function is caught because identity is recomputed from
# the EXECUTING parser's SOURCE at every use.
PARSERS: Mapping[tuple, Callable[[Mapping[str, Any]], dict]] = \
    MappingProxyType({
        ("mt5_demo", "native_protection", "v1"):
            _parse_mt5_protection_v1,
        ("mt5_demo", "reconciliation", "v1"):
            _parse_mt5_reconciliation_v1,
    })


def parser_identity(key: tuple, parser: Callable) -> str:
    """P2: identity covers the EXECUTABLE PARSER SOURCE plus the
    sealed schema and key — recomputed AT USE, never cached from a
    function name. Same name with different code refuses."""
    try:
        source = inspect.getsource(parser).encode()
    except (OSError, TypeError) as exc:
        raise EvidenceError(
            "parser source is unavailable — identity cannot be "
            "established") from exc
    schema = SCHEMAS.get(key)
    if schema is None:
        raise EvidenceError(f"no sealed schema for {key}")
    material = b"|".join([
        "|".join(key).encode(),
        ",".join(sorted(schema)).encode(),
        hashlib.sha256(source).hexdigest().encode()])
    return hashlib.sha256(material).hexdigest()[:32]


# P2: SEALED expected identities, COMMITTED as constants. Recomputing
# identity at use catches evidence derived under a different parser,
# but a substituted parser is self-consistent for NEW evidence — so
# the recomputed identity is checked against these sealed values. Any
# drift (substitution OR a legitimate parser edit) refuses until the
# constant is deliberately updated in review, exactly like the
# executable allowlist of the dispatch driver.
SEALED_PARSER_IDENTITIES = MappingProxyType({
    ("mt5_demo", "native_protection", "v1"):
        "1731381345c0f53f8bbb219c91fa9a19",
    ("mt5_demo", "reconciliation", "v1"):
        "8c2be53face3664e546647cf23115850",
})


def resolve_parser(key: tuple):
    """Sealed resolver: returns (parser, identity) or refuses. The
    executing parser's recomputed identity MUST equal the sealed
    committed constant."""
    parser = PARSERS.get(key)
    if parser is None:
        raise EvidenceError(
            f"unknown source schema {key} — no allowlisted parser")
    identity = parser_identity(key, parser)
    sealed = SEALED_PARSER_IDENTITIES.get(key)
    if sealed is None:
        raise EvidenceError(
            f"no sealed parser identity committed for {key}")
    if identity != sealed:
        raise EvidenceError(
            f"executing parser for {key} has identity "
            f"{identity[:12]}… but the SEALED committed identity is "
            f"{sealed[:12]}… — parser substitution or unreviewed "
            "code change refused")
    return parser, identity


# ---------------------------------------------------------------- #
# B2: policy-owned freshness and source authority                   #
# ---------------------------------------------------------------- #

@dataclass(frozen=True)
class EvidencePolicy:
    """Validated, POLICY-side contract. Evidence never sets these."""

    venue: str
    account_fingerprint: str
    symbol: str
    allowed_sources: tuple
    max_age_seconds: float
    schema_version: str

    def __post_init__(self):
        for name in ("venue", "account_fingerprint", "symbol",
                     "schema_version"):
            require_identity(name, getattr(self, name))
        require_real("max_age_seconds", self.max_age_seconds,
                     positive=True)
        if not isinstance(self.allowed_sources, tuple) or \
                not self.allowed_sources:
            raise EvidenceError(
                "allowed_sources must be a non-empty tuple")
        for source in self.allowed_sources:
            require_identity("allowed_source", source)

    @property
    def policy_digest(self) -> str:
        """P3: digest of the COMPLETE validated policy — venue,
        account, symbol, allowlist, maximum age, schema and the
        parser identities it authorizes. Persisted at claim and
        re-verified at finish."""
        parsers = {}
        for key in sorted(SCHEMAS):
            if key[0] == self.venue and key[2] == self.schema_version:
                _fn, identity = resolve_parser(key)
                parsers["|".join(key)] = identity
        return hashlib.sha256(json.dumps({
            "venue": self.venue,
            "account_fingerprint": self.account_fingerprint,
            "symbol": self.symbol,
            "allowed_sources": sorted(self.allowed_sources),
            "max_age_seconds": self.max_age_seconds,
            "schema_version": self.schema_version,
            "parser_identities": parsers,
        }, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def build(*, venue: str, account_fingerprint: str, symbol: str,
              allowed_sources, max_age_seconds: float,
              schema_version: str) -> "EvidencePolicy":
        return EvidencePolicy(
            venue=venue, account_fingerprint=account_fingerprint,
            symbol=symbol,
            allowed_sources=tuple(allowed_sources),
            max_age_seconds=max_age_seconds,
            schema_version=schema_version)


@dataclass(frozen=True)
class DirectEvidence:
    """Immutable envelope. Facts are DERIVED, never supplied."""

    venue: str
    account_fingerprint: str
    symbol: str
    position_identity: str
    evidence_type: str
    schema_version: str
    observed_at: datetime
    source: str
    evidence_id: str
    raw_payload_bytes: bytes
    raw_sha256: str
    canonical_payload: bytes
    payload_sha256: str
    parser_digest: str
    _facts: tuple           # derived (key, value) pairs — read-only

    def __post_init__(self):
        for name in ("venue", "account_fingerprint", "symbol",
                     "position_identity", "evidence_type",
                     "schema_version", "source", "evidence_id",
                     "payload_sha256", "parser_digest"):
            require_identity(name, getattr(self, name))
        require_utc("observed_at", self.observed_at)
        for name, blob in (("raw_payload_bytes",
                            self.raw_payload_bytes),
                           ("canonical_payload",
                            self.canonical_payload)):
            if not isinstance(blob, bytes):
                raise EvidenceError(f"{name} must be bytes")
        if hashlib.sha256(self.raw_payload_bytes).hexdigest() != \
                self.raw_sha256:
            raise EvidenceError(
                "raw_sha256 does not cover the ORIGINAL bytes")
        if hashlib.sha256(self.canonical_payload).hexdigest() != \
                self.payload_sha256:
            raise EvidenceError(
                "payload_sha256 does not cover the canonical bytes")
        key = (self.venue, self.evidence_type, self.schema_version)
        # P2: identity is RECOMPUTED from the EXECUTING parser source
        # at every use — a substituted implementation cannot reuse a
        # cached digest
        parser, identity = resolve_parser(key)
        if identity != self.parser_digest:
            raise EvidenceError(
                "parser identity mismatch — the executing parser is "
                "not the one this evidence was derived under")
        # P1/B1: facts are RE-DERIVED from the ORIGINAL BYTES (with
        # duplicate/NaN detection), never from a lossy mapping
        decoded = decode_authority_bytes(self.raw_payload_bytes)
        if canonical_bytes(decoded) != self.canonical_payload:
            raise EvidenceError(
                "canonical payload does not derive from the original "
                "bytes")
        derived = parser(decoded)
        if tuple(sorted(derived.items())) != self._facts:
            raise EvidenceError(
                "derived facts do not match the canonical payload — "
                "there is exactly ONE representation of each fact")

    @property
    def facts(self) -> dict:
        return dict(self._facts)

    @staticmethod
    def parse(*, venue: str, account_fingerprint: str, symbol: str,
              position_identity: str, evidence_type: str,
              schema_version: str, observed_at: Any, source: str,
              evidence_id: str,
              raw_bytes: bytes) -> "DirectEvidence":
        """The ONLY construction path: consume ORIGINAL BYTES, decode
        with duplicate/NaN detection, derive the facts, bind the
        executing parser identity."""
        key = (venue, evidence_type, schema_version)
        parser, identity = resolve_parser(key)
        decoded = decode_authority_bytes(raw_bytes)
        canonical = canonical_bytes(decoded)
        facts = parser(decoded)
        return DirectEvidence(
            venue=venue, account_fingerprint=account_fingerprint,
            symbol=symbol, position_identity=position_identity,
            evidence_type=evidence_type,
            schema_version=schema_version,
            observed_at=require_utc("observed_at", observed_at),
            source=source, evidence_id=evidence_id,
            raw_payload_bytes=bytes(raw_bytes),
            raw_sha256=hashlib.sha256(bytes(raw_bytes)).hexdigest(),
            canonical_payload=canonical,
            payload_sha256=hashlib.sha256(canonical).hexdigest(),
            parser_digest=identity,
            _facts=tuple(sorted(facts.items())))

    # ---- policy-enforced verification (B2) ----------------------
    def verify(self, policy: EvidencePolicy, *, now: Any,
               position_identity: str) -> float:
        if not isinstance(policy, EvidencePolicy):
            raise EvidenceError(
                "a validated EvidencePolicy is required")
        for label, expected, actual in (
                ("venue", policy.venue, self.venue),
                ("account_fingerprint", policy.account_fingerprint,
                 self.account_fingerprint),
                ("symbol", policy.symbol, self.symbol),
                ("schema_version", policy.schema_version,
                 self.schema_version),
                ("position_identity", position_identity,
                 self.position_identity)):
            if expected != actual:
                raise EvidenceError(
                    f"evidence {label} mismatch: policy/claim "
                    f"expects {expected!r}, evidence carries "
                    f"{actual!r}")
        if self.source not in policy.allowed_sources:
            raise EvidenceError(
                f"source {self.source!r} is not in the policy "
                f"allowlist {list(policy.allowed_sources)}")
        moment = require_utc("now", now)
        age = (moment - self.observed_at).total_seconds()
        if age < 0:
            raise EvidenceError(
                "evidence observed in the future — refused")
        # B2: the POLICY owns the lifetime; evidence cannot extend it
        if age > policy.max_age_seconds:
            raise EvidenceError(
                f"stale evidence: age {age:.1f}s exceeds the "
                f"POLICY maximum {policy.max_age_seconds:.1f}s")
        return age

    def require_protection(self) -> dict:
        if self.evidence_type != "native_protection":
            raise EvidenceError(
                f"expected native_protection evidence, got "
                f"{self.evidence_type!r}")
        facts = self.facts
        if not facts["stop_loss_accepted"]:
            raise EvidenceError(
                "native protection requires a broker-ACCEPTED stop "
                "loss in the payload")
        return facts

    def require_flat(self) -> dict:
        if self.evidence_type != "reconciliation":
            raise EvidenceError(
                f"expected reconciliation evidence, got "
                f"{self.evidence_type!r}")
        facts = self.facts
        if facts["positions_total"] != 0 or \
                facts["orders_total"] != 0:
            raise EvidenceError(
                f"not flat: {facts['positions_total']} positions and "
                f"{facts['orders_total']} orders in the payload")
        return facts

    @property
    def is_flat(self) -> bool:
        facts = self.facts
        return (facts.get("positions_total") == 0
                and facts.get("orders_total") == 0)
