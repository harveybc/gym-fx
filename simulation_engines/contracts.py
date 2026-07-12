"""Engine-neutral contracts for deterministic portfolio replays."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
class TargetAction:
    instrument_id: str
    ts_event_ns: int
    target_units: Decimal
    action_id: str
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None


def load_execution_cost_profile(path: str | Path) -> ExecutionCostProfile:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("execution cost profile must contain a JSON object")
    return ExecutionCostProfile.from_dict(raw)
