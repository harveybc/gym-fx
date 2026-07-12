"""Deterministic multi-asset fixture and independent reconciliation oracle."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from datetime import datetime
from datetime import timezone

import pandas as pd

from simulation_engines.contracts import ExecutionCostProfile
from simulation_engines.contracts import InstrumentSpec
from simulation_engines.contracts import MarketFrame
from simulation_engines.contracts import TargetAction


NANOSECONDS_PER_MINUTE = 60_000_000_000
BAKEOFF_START_NS = 1_704_204_000_000_000_000  # 2024-01-02T14:00:00Z


def _ts(minutes: int) -> int:
    return BAKEOFF_START_NS + minutes * NANOSECONDS_PER_MINUTE


def build_multi_asset_fixture() -> tuple[
    list[InstrumentSpec],
    list[MarketFrame],
    list[TargetAction],
]:
    """Return a small asynchronous FX replay with netting and currency conversion."""

    instruments = [
        InstrumentSpec(
            symbol="EUR/USD",
            venue="SIM",
            base_currency="EUR",
            quote_currency="USD",
            price_precision=5,
            size_precision=0,
            margin_init=Decimal("0.05"),
            margin_maint=Decimal("0.025"),
            min_quantity=Decimal("1000"),
            lot_size=Decimal("1000"),
        ),
        InstrumentSpec(
            symbol="USD/JPY",
            venue="SIM",
            base_currency="USD",
            quote_currency="JPY",
            price_precision=3,
            size_precision=0,
            margin_init=Decimal("0.05"),
            margin_maint=Decimal("0.025"),
            min_quantity=Decimal("1000"),
            lot_size=Decimal("1000"),
        ),
    ]

    frames: list[MarketFrame] = []
    for minute, value in enumerate(
        ("1.10000", "1.10100", "1.10200", "1.10300", "1.10400", "1.10500"),
        start=1,
    ):
        close = Decimal(value)
        frames.append(
            MarketFrame(
                instrument_id="EUR/USD.SIM",
                timeframe_minutes=1,
                ts_event_ns=_ts(minute),
                open=close,
                high=close + Decimal("0.00030"),
                low=close - Decimal("0.00030"),
                close=close,
                volume=Decimal("1000000"),
            )
        )
    for minute, value in ((1, "145.000"), (6, "145.500")):
        close = Decimal(value)
        frames.append(
            MarketFrame(
                instrument_id="USD/JPY.SIM",
                timeframe_minutes=5,
                ts_event_ns=_ts(minute),
                open=close,
                high=close + Decimal("0.050"),
                low=close - Decimal("0.050"),
                close=close,
                volume=Decimal("1000000"),
            )
        )

    actions = [
        TargetAction("EUR/USD.SIM", _ts(1), Decimal("2000"), "eur-open-long"),
        TargetAction("EUR/USD.SIM", _ts(3), Decimal("1000"), "eur-partial-close"),
        TargetAction("EUR/USD.SIM", _ts(4), Decimal("-1000"), "eur-reverse-short"),
        TargetAction("EUR/USD.SIM", _ts(6), Decimal("0"), "eur-flatten"),
        TargetAction("USD/JPY.SIM", _ts(1), Decimal("1000"), "jpy-open-long"),
        TargetAction("USD/JPY.SIM", _ts(6), Decimal("0"), "jpy-flatten"),
    ]
    return instruments, frames, actions


def build_rollover_rate_fixture() -> pd.DataFrame:
    """Minimal monthly rates required by the fixture's FX currencies."""

    return pd.DataFrame(
        [
            {"LOCATION": "EA19", "TIME": "2024-01", "Value": 5.0},
            {"LOCATION": "USA", "TIME": "2024-01", "Value": 4.0},
            {"LOCATION": "JPN", "TIME": "2024-01", "Value": 0.1},
        ]
    )


def build_intrabar_collision_fixture() -> tuple[
    list[InstrumentSpec],
    list[MarketFrame],
    list[TargetAction],
]:
    instruments, _, _ = build_multi_asset_fixture()
    eurusd = [instruments[0]]
    first = Decimal("1.10000")
    collision_open = Decimal("1.10000")
    frames = [
        MarketFrame(
            "EUR/USD.SIM",
            1,
            _ts(1),
            first,
            first + Decimal("0.00010"),
            first - Decimal("0.00010"),
            first,
            Decimal("1000000"),
        ),
        MarketFrame(
            "EUR/USD.SIM",
            1,
            _ts(2),
            collision_open,
            Decimal("1.10300"),
            Decimal("1.09700"),
            Decimal("1.10200"),
            Decimal("1000000"),
            execution_path=(
                collision_open,
                Decimal("1.09700"),
                Decimal("1.10300"),
                Decimal("1.10200"),
            ),
        ),
    ]
    actions = [
        TargetAction(
            "EUR/USD.SIM",
            _ts(1),
            Decimal("1000"),
            "long-bracket",
            stop_loss_price=Decimal("1.09800"),
            take_profit_price=Decimal("1.10200"),
        )
    ]
    return eurusd, frames, actions


def build_margin_rejection_fixture() -> tuple[
    list[InstrumentSpec],
    list[MarketFrame],
    list[TargetAction],
]:
    instruments, frames, _ = build_multi_asset_fixture()
    return (
        [instruments[0]],
        [frame for frame in frames if frame.instrument_id == "EUR/USD.SIM"][:2],
        [TargetAction("EUR/USD.SIM", _ts(1), Decimal("10000000"), "oversized")],
    )


def build_financing_fixture() -> tuple[
    list[InstrumentSpec],
    list[MarketFrame],
    list[TargetAction],
]:
    instruments, _, _ = build_multi_asset_fixture()
    eurusd = [instruments[0]]
    times = (
        int(pd.Timestamp("2024-01-02T21:58:00Z").value),
        int(pd.Timestamp("2024-01-02T22:01:00Z").value),
        int(pd.Timestamp("2024-01-02T22:02:00Z").value),
    )
    frames = []
    for timestamp, value in zip(times, ("1.10000", "1.10000", "1.10000")):
        close = Decimal(value)
        frames.append(
            MarketFrame(
                "EUR/USD.SIM",
                1,
                timestamp,
                close,
                close + Decimal("0.00010"),
                close - Decimal("0.00010"),
                close,
                Decimal("1000000"),
            )
        )
    actions = [
        TargetAction("EUR/USD.SIM", times[0], Decimal("1000"), "overnight-open"),
        TargetAction("EUR/USD.SIM", times[2], Decimal("0"), "overnight-close"),
    ]
    return eurusd, frames, actions


def _conversion_rate(
    spec: InstrumentSpec,
    mid: Decimal,
    base_currency: str,
) -> Decimal:
    if spec.quote_currency == base_currency:
        return Decimal(1)
    if spec.base_currency == base_currency:
        return Decimal(1) / mid
    raise ValueError(
        f"fixture cannot convert {spec.quote_currency} to {base_currency} "
        f"using {spec.instrument_id}"
    )


def reconcile_fills(
    result: dict[str, Any],
    instrument_specs: list[InstrumentSpec],
    profile: ExecutionCostProfile,
    *,
    initial_cash: Decimal,
    base_currency: str = "USD",
) -> dict[str, Any]:
    """Independently reconcile the tiny fixture from immutable fill facts.

    This is test-oracle arithmetic only. It is not used by the environment and
    cannot become a competing production account ledger.
    """

    specs = {spec.instrument_id: spec for spec in instrument_specs}
    positions: dict[str, tuple[Decimal, Decimal]] = {}
    realized_base = Decimal(0)
    commission_base = Decimal(0)
    spread_drag_base = Decimal(0)
    slippage_drag_base = Decimal(0)

    fills = [event for event in result["events"] if event["event_type"] == "order_filled"]
    for fill in fills:
        instrument_id = fill["instrument_id"]
        spec = specs[instrument_id]
        mid = Decimal(fill["reference_mid"])
        conversion = _conversion_rate(spec, mid, base_currency)
        price = Decimal(fill["price"])
        quantity = Decimal(fill["quantity"])
        signed_fill = quantity if fill["side"] in {"BUY", "1"} else -quantity
        current_units, average_price = positions.get(
            instrument_id, (Decimal(0), Decimal(0))
        )

        if current_units == 0 or current_units * signed_fill > 0:
            new_units = current_units + signed_fill
            if current_units == 0:
                new_average = price
            else:
                new_average = (
                    abs(current_units) * average_price + abs(signed_fill) * price
                ) / abs(new_units)
        else:
            closing = min(abs(current_units), abs(signed_fill))
            quote_pnl = (
                closing * (price - average_price)
                if current_units > 0
                else closing * (average_price - price)
            )
            realized_base += quote_pnl * conversion
            new_units = current_units + signed_fill
            new_average = price if current_units * new_units < 0 else average_price
            if new_units == 0:
                new_average = Decimal(0)
        positions[instrument_id] = (new_units, new_average)

        commission = Decimal(fill["commission"])
        commission_base += commission * conversion
        spread_drag_base += (
            quantity * mid * profile.full_spread_rate / Decimal(2) * conversion
        )
        slippage_drag_base += (
            quantity * mid * profile.slippage_rate_per_side * conversion
        )

    expected_final = initial_cash + realized_base - commission_base
    return {
        "initial_cash": str(initial_cash),
        "realized_pnl_before_commission": str(realized_base),
        "commission": str(commission_base),
        "modeled_half_spread_fill_drag": str(spread_drag_base),
        "modeled_slippage_fill_drag": str(slippage_drag_base),
        "expected_final_balance": str(expected_final),
        "all_positions_flat": all(units == 0 for units, _ in positions.values()),
        "fill_count": len(fills),
    }


def export_execution_reports(
    result: dict[str, Any],
    instrument_specs: list[InstrumentSpec],
    profile: ExecutionCostProfile,
    *,
    base_currency: str = "USD",
) -> list[dict[str, Any]]:
    """Validate and serialize fill facts as trading-contracts reports."""

    try:
        from trading_contracts import ExecutionReport
        from trading_contracts import ProducerIdentity
        from simulation_engines.nautilus_adapter import NautilusReplayAdapter
    except ImportError as exc:
        raise RuntimeError(
            "trading-contracts must be installed to export canonical reports"
        ) from exc

    specs = {spec.instrument_id: spec for spec in instrument_specs}
    requested = {
        event["action_id"]: abs(Decimal(event["delta_units"]))
        for event in result["events"]
        if event["event_type"] == "target_requested"
    }
    reports = []
    for fill in result["events"]:
        if fill["event_type"] != "order_filled":
            continue
        spec = specs[fill["instrument_id"]]
        mid = Decimal(fill["reference_mid"])
        conversion = _conversion_rate(spec, mid, base_currency)
        quantity = Decimal(fill["quantity"])
        commission = Decimal(fill["commission"]) * conversion
        spread_cost = (
            quantity * mid * profile.full_spread_rate / Decimal(2) * conversion
        )
        slippage_cost = quantity * mid * profile.slippage_rate_per_side * conversion
        signed_quantity = quantity if fill["side"] in {"BUY", "1"} else -quantity
        action_id = fill["action_id"]
        report = ExecutionReport(
            object_id=f"nautilus-fill:{fill['client_order_id']}:{fill['sequence']}",
            as_of=datetime.fromtimestamp(
                fill["ts_event_ns"] / 1_000_000_000, tz=timezone.utc
            ),
            producer=ProducerIdentity(
                name="gym-fx-nautilus-adapter",
                version=NautilusReplayAdapter.ENGINE_VERSION,
            ),
            trace_id=result["result_hash"],
            order_intent_id=action_id,
            state="filled",
            requested_units=float(requested.get(action_id, quantity)),
            filled_units=float(signed_quantity),
            requested_price=float(mid),
            filled_price=float(fill["price"]),
            spread_cost=float(spread_cost),
            slippage_cost=float(slippage_cost),
            commission=float(commission),
            financing=0.0,
            conversion_cost=0.0,
            broker_ids={
                "client_order_id": fill["client_order_id"],
                "instrument_id": fill["instrument_id"],
                "cost_currency": base_currency,
            },
            latency_ms=float(profile.latency_ms),
        )
        reports.append(report.model_dump(mode="json"))
    return reports
