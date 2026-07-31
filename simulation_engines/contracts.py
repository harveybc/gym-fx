"""Engine-neutral contracts for deterministic portfolio replays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal
from pathlib import Path
from typing import Any


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{field} must be decimal-compatible") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class ExecutionCostProfile:
    """Versioned execution assumptions shared by all simulation engines."""

    schema_version: str
    profile_id: str
    commission_rate_per_side: Decimal
    full_spread_rate: Decimal
    slippage_bps_per_side: Decimal
    latency_ms: int
    financing_enabled: bool
    intrabar_collision_policy: str
    limit_fill_policy: str
    margin_model: str
    enforce_margin_preflight: bool
    random_seed: int

    @property
    def slippage_rate_per_side(self) -> Decimal:
        return self.slippage_bps_per_side / Decimal("10000")

    @property
    def quote_adverse_rate_per_side(self) -> Decimal:
        """Synthetic quote displacement from mid used for OHLC-only inputs."""

        return self.full_spread_rate / Decimal("2") + self.slippage_rate_per_side

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExecutionCostProfile":
        required = {
            "schema_version",
            "profile_id",
            "commission_rate_per_side",
            "full_spread_rate",
            "slippage_bps_per_side",
            "latency_ms",
            "financing_enabled",
            "intrabar_collision_policy",
            "limit_fill_policy",
            "margin_model",
            "enforce_margin_preflight",
            "random_seed",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"execution cost profile missing fields: {missing}")
        if raw["schema_version"] != "execution_cost_profile.v1":
            raise ValueError("unsupported execution cost profile schema_version")

        profile = cls(
            schema_version=str(raw["schema_version"]),
            profile_id=str(raw["profile_id"]),
            commission_rate_per_side=_decimal(
                raw["commission_rate_per_side"], "commission_rate_per_side"
            ),
            full_spread_rate=_decimal(raw["full_spread_rate"], "full_spread_rate"),
            slippage_bps_per_side=_decimal(
                raw["slippage_bps_per_side"], "slippage_bps_per_side"
            ),
            latency_ms=int(raw["latency_ms"]),
            financing_enabled=bool(raw["financing_enabled"]),
            intrabar_collision_policy=str(raw["intrabar_collision_policy"]),
            limit_fill_policy=str(raw["limit_fill_policy"]),
            margin_model=str(raw["margin_model"]),
            enforce_margin_preflight=bool(raw["enforce_margin_preflight"]),
            random_seed=int(raw["random_seed"]),
        )
        for field in (
            "commission_rate_per_side",
            "full_spread_rate",
            "slippage_bps_per_side",
        ):
            if getattr(profile, field) < 0:
                raise ValueError(f"{field} cannot be negative")
        if profile.full_spread_rate >= 1:
            raise ValueError("full_spread_rate must be below 1")
        if profile.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if profile.intrabar_collision_policy not in {"worst_case", "adaptive", "ohlc"}:
            raise ValueError("unsupported intrabar_collision_policy")
        if profile.limit_fill_policy not in {"conservative", "touch", "cross"}:
            raise ValueError("unsupported limit_fill_policy")
        if profile.margin_model not in {"standard", "leveraged"}:
            raise ValueError("unsupported margin_model")
        return profile


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    venue: str
    base_currency: str
    quote_currency: str
    price_precision: int
    size_precision: int
    margin_init: Decimal
    margin_maint: Decimal
    min_quantity: Decimal = Decimal("1")
    lot_size: Decimal | None = None

    @property
    def instrument_id(self) -> str:
        return f"{self.symbol}.{self.venue}"


@dataclass(frozen=True)
class MarketFrame:
    instrument_id: str
    timeframe_minutes: int
    ts_event_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    execution_path: tuple[Decimal, ...] | None = None


@dataclass(frozen=True)
class EntryExecutionRequest:
    """Engine-neutral instructions for opening or changing a target position.

    ``expires_at_ns`` is the absolute UTC deadline for a pending limit or stop
    entry. A missing deadline means good-till-canceled. Market entries remain
    immediate and cannot carry a deadline.
    """

    order_type: str = "market"
    limit_price: Decimal | None = None
    trigger_price: Decimal | None = None
    expires_at_ns: int | None = None
    unfilled_fallback: str = "cancel"

    def __post_init__(self) -> None:
        order_type = str(self.order_type).strip().lower()
        if order_type not in {"market", "limit", "stop"}:
            raise ValueError("entry order_type must be market, limit, or stop")
        object.__setattr__(self, "order_type", order_type)

        if self.limit_price is not None:
            limit_price = _decimal(self.limit_price, "entry limit_price")
            if limit_price <= 0:
                raise ValueError("entry limit_price must be positive")
            object.__setattr__(self, "limit_price", limit_price)
        if self.trigger_price is not None:
            trigger_price = _decimal(self.trigger_price, "entry trigger_price")
            if trigger_price <= 0:
                raise ValueError("entry trigger_price must be positive")
            object.__setattr__(self, "trigger_price", trigger_price)
        if self.expires_at_ns is not None:
            expires_at_ns = int(self.expires_at_ns)
            if expires_at_ns <= 0:
                raise ValueError("entry expires_at_ns must be positive")
            object.__setattr__(self, "expires_at_ns", expires_at_ns)
        fallback = str(self.unfilled_fallback).strip().lower()
        if fallback not in {"cancel", "market"}:
            raise ValueError("entry unfilled_fallback must be cancel or market")
        object.__setattr__(self, "unfilled_fallback", fallback)

        if order_type == "market":
            if self.limit_price is not None or self.trigger_price is not None:
                raise ValueError("market entry cannot carry limit or trigger prices")
            if self.expires_at_ns is not None:
                raise ValueError("market entry cannot carry expires_at_ns")
            if fallback != "cancel":
                raise ValueError("market entry cannot carry an unfilled fallback")
        elif order_type == "limit":
            if self.limit_price is None:
                raise ValueError("limit entry requires limit_price")
            if self.trigger_price is not None:
                raise ValueError("limit entry cannot carry trigger_price")
        elif self.trigger_price is None:
            raise ValueError("stop entry requires trigger_price")
        elif self.limit_price is not None:
            raise ValueError("stop entry cannot carry limit_price")

    @classmethod
    def from_bar_ttl(
        cls,
        *,
        order_type: str,
        action_ts_ns: int,
        timeframe_minutes: int,
        valid_for_bars: int | None = None,
        limit_price: Any | None = None,
        trigger_price: Any | None = None,
        unfilled_fallback: str = "cancel",
    ) -> "EntryExecutionRequest":
        """Resolve a router's relative bar TTL into an absolute GTD deadline."""
        normalized_type = str(order_type).strip().lower()
        expires_at_ns = None
        if normalized_type != "market":
            if valid_for_bars is None:
                raise ValueError("pending entry requires valid_for_bars")
            bars = int(valid_for_bars)
            minutes = int(timeframe_minutes)
            if bars < 1:
                raise ValueError("valid_for_bars must be >= 1")
            if minutes < 1:
                raise ValueError("timeframe_minutes must be >= 1")
            expires_at_ns = int(action_ts_ns) + bars * minutes * 60 * 1_000_000_000
        return cls(
            order_type=normalized_type,
            limit_price=limit_price,
            trigger_price=trigger_price,
            expires_at_ns=expires_at_ns,
            unfilled_fallback=unfilled_fallback,
        )


@dataclass(frozen=True)
class TargetAction:
    instrument_id: str
    ts_event_ns: int
    target_units: Decimal
    action_id: str
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    entry_execution: EntryExecutionRequest = field(
        default_factory=EntryExecutionRequest
    )
    market_available: bool = True
    signal_valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.market_available, bool):
            raise ValueError("market_available must be a boolean")
        if not isinstance(self.signal_valid, bool):
            raise ValueError("signal_valid must be a boolean")
        if (
            self.entry_execution.expires_at_ns is not None
            and self.entry_execution.expires_at_ns <= self.ts_event_ns
        ):
            raise ValueError("entry expiration must be after the action timestamp")


def load_execution_cost_profile(path: str | Path) -> ExecutionCostProfile:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("execution cost profile must contain a JSON object")
    return ExecutionCostProfile.from_dict(raw)
