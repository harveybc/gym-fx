"""Typed direct-evidence envelopes (order @…D2/E2).

A string is not evidence. Custody may only consume IMMUTABLE VALIDATED
envelopes that bind venue, account, symbol, position id, observation
time, source/evidence id, a digest of the RAW payload and a maximum
age. The digest is recomputed from the raw payload at claim/finish
time, so an arbitrary digest cannot authorize anything.

Refusals (typed, never coerced): strings where booleans/counts are
required, bool-as-count, NaN, missing facts, stale facts, identity
mismatch and fabricated digests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from app.session_exposure import (SessionEvidenceError, require_count,
                                  require_identity, require_real,
                                  require_utc)


class EvidenceError(SessionEvidenceError):
    """The direct-evidence envelope is unusable — typed refusal."""


def payload_digest(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        raise EvidenceError("raw payload must be a mapping")
    return hashlib.sha256(json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise EvidenceError(
            f"{name}: a strict bool is required, got "
            f"{type(value).__name__} {value!r}")
    return value


def _strict_zero(name: str, value: Any) -> int:
    count = require_count(name, value, minimum=0)
    if count != 0:
        raise EvidenceError(f"{name}: must be exactly 0, got {count}")
    return count


@dataclass(frozen=True)
class DirectEvidence:
    """Base envelope: identity + freshness + raw-payload binding."""

    venue: str
    account_fingerprint: str
    symbol: str
    position_identity: str
    observed_at: datetime
    source: str
    evidence_id: str
    raw_payload: tuple          # canonical (key, value) pairs
    raw_digest: str
    max_age_seconds: float

    def __post_init__(self):
        for name in ("venue", "account_fingerprint", "symbol",
                     "position_identity", "source", "evidence_id",
                     "raw_digest"):
            require_identity(name, getattr(self, name))
        require_utc("observed_at", self.observed_at)
        require_real("max_age_seconds", self.max_age_seconds,
                     positive=True)
        if not isinstance(self.raw_payload, tuple):
            raise EvidenceError("raw_payload must be a tuple")
        recomputed = payload_digest(dict(self.raw_payload))
        if recomputed != self.raw_digest:
            raise EvidenceError(
                "raw_digest does not match the raw payload — a "
                "fabricated digest can never authorize")

    def verify_identity(self, *, venue: str, account_fingerprint: str,
                        symbol: str, position_identity: str) -> None:
        for label, expected, actual in (
                ("venue", venue, self.venue),
                ("account_fingerprint", account_fingerprint,
                 self.account_fingerprint),
                ("symbol", symbol, self.symbol),
                ("position_identity", position_identity,
                 self.position_identity)):
            if expected != actual:
                raise EvidenceError(
                    f"evidence {label} mismatch: expected "
                    f"{expected!r}, evidence carries {actual!r}")

    def verify_fresh(self, now: Any) -> float:
        moment = require_utc("now", now)
        age = (moment - self.observed_at).total_seconds()
        if age < 0:
            raise EvidenceError(
                "evidence observed in the future — refused")
        if age > self.max_age_seconds:
            raise EvidenceError(
                f"stale evidence: age {age:.1f}s exceeds "
                f"{self.max_age_seconds:.1f}s")
        return age

    def rehash(self) -> str:
        """Recompute the digest from the raw payload (E2: re-hash
        referenced evidence before claim/finish)."""
        return payload_digest(dict(self.raw_payload))


@dataclass(frozen=True)
class NativeProtectionEvidence(DirectEvidence):
    """Direct proof that the carried position keeps broker-accepted
    protection."""

    stop_loss_accepted: bool = False
    take_profit_accepted: bool = False

    def __post_init__(self):
        super().__post_init__()
        _strict_bool("stop_loss_accepted", self.stop_loss_accepted)
        _strict_bool("take_profit_accepted",
                     self.take_profit_accepted)
        if not self.stop_loss_accepted:
            raise EvidenceError(
                "native protection requires a broker-ACCEPTED stop "
                "loss")

    @staticmethod
    def build(*, venue: str, account_fingerprint: str, symbol: str,
              position_identity: str, observed_at: Any, source: str,
              evidence_id: str, raw_payload: Mapping[str, Any],
              max_age_seconds: float = 120.0,
              stop_loss_accepted: Any = None,
              take_profit_accepted: Any = None
              ) -> "NativeProtectionEvidence":
        payload = dict(raw_payload)
        return NativeProtectionEvidence(
            venue=venue, account_fingerprint=account_fingerprint,
            symbol=symbol, position_identity=position_identity,
            observed_at=require_utc("observed_at", observed_at),
            source=source, evidence_id=evidence_id,
            raw_payload=tuple(sorted(payload.items())),
            raw_digest=payload_digest(payload),
            max_age_seconds=max_age_seconds,
            stop_loss_accepted=stop_loss_accepted,
            take_profit_accepted=take_profit_accepted)


@dataclass(frozen=True)
class ReconciliationEvidence(DirectEvidence):
    """Direct venue proof of exposure state. ``completed`` requires
    STRICT integer zero positions AND zero pending orders."""

    positions_total: int = -1
    orders_total: int = -1

    def __post_init__(self):
        super().__post_init__()
        require_count("positions_total", self.positions_total,
                      minimum=0)
        require_count("orders_total", self.orders_total, minimum=0)

    @property
    def is_flat(self) -> bool:
        return self.positions_total == 0 and self.orders_total == 0

    def require_flat(self) -> None:
        _strict_zero("positions_total", self.positions_total)
        _strict_zero("orders_total", self.orders_total)

    @staticmethod
    def build(*, venue: str, account_fingerprint: str, symbol: str,
              position_identity: str, observed_at: Any, source: str,
              evidence_id: str, raw_payload: Mapping[str, Any],
              positions_total: Any, orders_total: Any,
              max_age_seconds: float = 120.0
              ) -> "ReconciliationEvidence":
        payload = dict(raw_payload)
        return ReconciliationEvidence(
            venue=venue, account_fingerprint=account_fingerprint,
            symbol=symbol, position_identity=position_identity,
            observed_at=require_utc("observed_at", observed_at),
            source=source, evidence_id=evidence_id,
            raw_payload=tuple(sorted(payload.items())),
            raw_digest=payload_digest(payload),
            max_age_seconds=max_age_seconds,
            positions_total=positions_total,
            orders_total=orders_total)
