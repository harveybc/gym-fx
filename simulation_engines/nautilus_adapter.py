"""NautilusTrader adapter for deterministic target-position replays.

Nautilus owns fills, positions, account balances, margin, commissions and P&L.
This module only translates engine-neutral inputs and exports immutable facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from simulation_engines.contracts import ExecutionCostProfile
from simulation_engines.contracts import InstrumentSpec
from simulation_engines.contracts import MarketFrame
from simulation_engines.contracts import TargetAction


class NautilusUnavailableError(RuntimeError):
    pass


def require_nautilus() -> None:
    try:
        import nautilus_trader  # noqa: F401
    except ImportError as exc:
        raise NautilusUnavailableError(
            "NautilusTrader is optional. Install gym-fx[nautilus] in an isolated "
            "Python 3.12+ environment."
        ) from exc


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _money_parts(value: Any) -> tuple[str, str]:
    text = str(value)
    amount, _, currency = text.partition(" ")
    return amount, currency


def _build_instrument(spec: InstrumentSpec, profile: ExecutionCostProfile):
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.instruments import CurrencyPair
    from nautilus_trader.model.objects import Currency
    from nautilus_trader.model.objects import Money
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.objects import Quantity

    price_increment = Decimal(1).scaleb(-spec.price_precision)
    size_increment = Decimal(1).scaleb(-spec.size_precision)
    base = Currency.from_str(spec.base_currency)
    quote = Currency.from_str(spec.quote_currency)
    lot_size = Quantity.from_str(str(spec.lot_size)) if spec.lot_size is not None else None
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(spec.symbol), Venue(spec.venue)),
        raw_symbol=Symbol(spec.symbol),
        base_currency=base,
        quote_currency=quote,
        price_precision=spec.price_precision,
        size_precision=spec.size_precision,
        price_increment=Price.from_str(f"{price_increment:.{spec.price_precision}f}"),
        size_increment=Quantity.from_str(f"{size_increment:.{spec.size_precision}f}"),
        lot_size=lot_size,
        max_quantity=Quantity.from_str("100000000"),
        min_quantity=Quantity.from_str(str(spec.min_quantity)),
        max_notional=None,
        min_notional=Money(Decimal("1"), quote),
        max_price=None,
        min_price=None,
        margin_init=spec.margin_init,
        margin_maint=spec.margin_maint,
        maker_fee=profile.commission_rate_per_side,
        taker_fee=profile.commission_rate_per_side,
        ts_event=0,
        ts_init=0,
    )


def _to_nautilus_data(
    frames: list[MarketFrame],
    instruments: dict[str, Any],
    profile: ExecutionCostProfile,
) -> tuple[list[Any], dict[tuple[str, int], Decimal]]:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.data import QuoteTick

    output: list[Any] = []
    close_by_key: dict[tuple[str, int], Decimal] = {}
    adverse = profile.quote_adverse_rate_per_side
    for frame in frames:
        instrument = instruments[frame.instrument_id]
        bar_type = BarType.from_str(
            f"{frame.instrument_id}-{frame.timeframe_minutes}-MINUTE-MID-EXTERNAL"
        )
        execution_path = frame.execution_path or (frame.close,)
        for offset, mid in enumerate(execution_path, start=1):
            quote_ts = frame.ts_event_ns - len(execution_path) + offset - 1
            bid = mid * (Decimal(1) - adverse)
            ask = mid * (Decimal(1) + adverse)
            output.append(
                QuoteTick(
                    instrument_id=instrument.id,
                    bid_price=instrument.make_price(bid),
                    ask_price=instrument.make_price(ask),
                    bid_size=instrument.make_qty(frame.volume),
                    ask_size=instrument.make_qty(frame.volume),
                    ts_event=quote_ts,
                    ts_init=quote_ts,
                )
            )
            close_by_key[(frame.instrument_id, quote_ts)] = mid
        output.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(frame.open),
                high=instrument.make_price(frame.high),
                low=instrument.make_price(frame.low),
                close=instrument.make_price(frame.close),
                volume=instrument.make_qty(frame.volume),
                ts_event=frame.ts_event_ns,
                ts_init=frame.ts_event_ns,
            )
        )
        close_by_key[(frame.instrument_id, frame.ts_event_ns)] = frame.close
    return output, close_by_key


class _ScriptedTargetStrategy:
    """Factory namespace to avoid importing Nautilus when the adapter is unused."""

    @staticmethod
    def build(
        actions: list[TargetAction],
        bar_types: list[Any],
        profile: ExecutionCostProfile,
    ):
        from nautilus_trader.config import StrategyConfig
        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.enums import OrderType
        from nautilus_trader.model.enums import PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.trading.strategy import Strategy

        action_by_key = {(item.instrument_id, item.ts_event_ns): item for item in actions}

        class ScriptedTargetStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__(StrategyConfig(log_events=False, log_commands=False))
                self.current_units: dict[str, Decimal] = {}
                self.active_action_ids: dict[str, str] = {}
                self.events: list[dict[str, Any]] = []

            def on_start(self) -> None:
                for bar_type in bar_types:
                    self.subscribe_bars(bar_type)
                    self.subscribe_quote_ticks(bar_type.instrument_id)

            def on_bar(self, bar) -> None:
                instrument_key = str(bar.bar_type.instrument_id)
                action = action_by_key.get((instrument_key, int(bar.ts_event)))
                if action is None:
                    return
                current = self.current_units.get(instrument_key, Decimal(0))
                delta = action.target_units - current
                self.events.append(
                    {
                        "event_type": "target_requested",
                        "ts_event_ns": int(bar.ts_event),
                        "instrument_id": instrument_key,
                        "action_id": action.action_id,
                        "target_units": str(action.target_units),
                        "current_units": str(current),
                        "delta_units": str(delta),
                    }
                )
                self.active_action_ids[instrument_key] = action.action_id
                if delta == 0:
                    return
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                instrument_id = InstrumentId.from_str(instrument_key)
                instrument = self.cache.instrument(instrument_id)
                quantity = instrument.make_qty(abs(delta))
                if profile.enforce_margin_preflight:
                    opening_units = Decimal(0)
                    if current == 0 or current * delta > 0:
                        opening_units = abs(delta)
                    elif abs(delta) > abs(current):
                        opening_units = abs(delta) - abs(current)
                    if opening_units > 0:
                        account = self.cache.account_for_venue(instrument_id.venue)
                        required = account.calculate_margin_init(
                            instrument,
                            instrument.make_qty(opening_units),
                            instrument.make_price(bar.close),
                        )
                        free = account.balance_free(required.currency)
                        if free is None:
                            free = account.balance_free()
                        if free is None:
                            raise RuntimeError("margin preflight requires a free balance")
                        required_amount = required.as_decimal()
                        if required.currency != free.currency:
                            xrate = self.cache.get_xrate(
                                venue=instrument_id.venue,
                                from_currency=required.currency,
                                to_currency=free.currency,
                                price_type=PriceType.MID,
                            )
                            if xrate is None:
                                raise RuntimeError(
                                    "margin preflight could not resolve currency conversion"
                                )
                            required_amount *= Decimal(str(xrate))
                        if required_amount > free.as_decimal():
                            self.events.append(
                                {
                                    "event_type": "preflight_denied",
                                    "ts_event_ns": int(bar.ts_event),
                                    "instrument_id": instrument_key,
                                    "action_id": action.action_id,
                                    "reason": "CUM_MARGIN_EXCEEDS_FREE_BALANCE",
                                    "required_margin": str(required),
                                    "required_margin_in_free_currency": str(
                                        required_amount
                                    ),
                                    "free_balance": str(free),
                                }
                            )
                            return
                if (
                    current == 0
                    and action.stop_loss_price is not None
                    and action.take_profit_price is not None
                ):
                    order_list = self.order_factory.bracket(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                        entry_order_type=OrderType.MARKET,
                        sl_trigger_price=instrument.make_price(action.stop_loss_price),
                        tp_price=instrument.make_price(action.take_profit_price),
                        tp_post_only=False,
                    )
                    self.submit_order_list(order_list)
                else:
                    order = self.order_factory.market(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                    )
                    self.submit_order(order)

            def on_order_filled(self, event) -> None:
                instrument_key = str(event.instrument_id)
                signed = Decimal(str(event.last_qty))
                if str(event.order_side) in {"SELL", "2"}:
                    signed = -signed
                self.current_units[instrument_key] = (
                    self.current_units.get(instrument_key, Decimal(0)) + signed
                )
                commission_amount, commission_currency = _money_parts(event.commission)
                self.events.append(
                    {
                        "event_type": "order_filled",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": instrument_key,
                        "action_id": self.active_action_ids.get(
                            instrument_key, "unattributed"
                        ),
                        "client_order_id": str(event.client_order_id),
                        "side": str(event.order_side),
                        "quantity": str(event.last_qty),
                        "price": str(event.last_px),
                        "commission": commission_amount,
                        "commission_currency": commission_currency,
                        "position_units_after": str(self.current_units[instrument_key]),
                    }
                )
                if self.current_units[instrument_key] == 0:
                    self.active_action_ids.pop(instrument_key, None)

            def on_order_rejected(self, event) -> None:
                self.events.append(
                    {
                        "event_type": "order_rejected",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": str(event.instrument_id),
                        "client_order_id": str(event.client_order_id),
                        "reason": str(event.reason),
                    }
                )

            def on_order_denied(self, event) -> None:
                self.events.append(
                    {
                        "event_type": "order_denied",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": str(event.instrument_id),
                        "client_order_id": str(event.client_order_id),
                        "reason": str(event.reason),
                    }
                )

        return ScriptedTargetStrategy()


class NautilusReplayAdapter:
    """Run deterministic target-position scripts through NautilusTrader."""

    ENGINE_VERSION = "1.230.0"

    def __init__(self, profile: ExecutionCostProfile) -> None:
        require_nautilus()
        self.profile = profile

    def run(
        self,
        *,
        instrument_specs: list[InstrumentSpec],
        frames: list[MarketFrame],
        actions: list[TargetAction],
        initial_cash: Decimal = Decimal("100000"),
        base_currency: str = "USD",
        default_leverage: Decimal = Decimal("20"),
        financing_rate_data: Any | None = None,
    ) -> dict[str, Any]:
        import nautilus_trader
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.backtest.models import FillModel
        from nautilus_trader.backtest.models import LatencyModel
        from nautilus_trader.backtest.models import MakerTakerFeeModel
        from nautilus_trader.backtest.models import LeveragedMarginModel
        from nautilus_trader.backtest.models import StandardMarginModel
        from nautilus_trader.config import BacktestEngineConfig
        from nautilus_trader.config import LoggingConfig
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.enums import AccountType
        from nautilus_trader.model.enums import OmsType
        from nautilus_trader.model.identifiers import Venue
        from nautilus_trader.model.objects import Currency
        from nautilus_trader.model.objects import Money

        if nautilus_trader.__version__ != self.ENGINE_VERSION:
            raise RuntimeError(
                f"NautilusTrader {self.ENGINE_VERSION} is required, found "
                f"{nautilus_trader.__version__}"
            )
        modules = None
        if self.profile.financing_enabled:
            if financing_rate_data is None:
                raise ValueError(
                    "financing_rate_data is required when financing_enabled is true"
                )
            from nautilus_trader.backtest.config import FXRolloverInterestConfig
            from nautilus_trader.backtest.modules import FXRolloverInterestModule

            modules = [
                FXRolloverInterestModule(
                    FXRolloverInterestConfig(rate_data=financing_rate_data)
                )
            ]

        margin_model = (
            StandardMarginModel()
            if self.profile.margin_model == "standard"
            else LeveragedMarginModel()
        )

        venues = {spec.venue for spec in instrument_specs}
        if len(venues) != 1:
            raise ValueError("one replay currently requires a single shared-account venue")
        venue = Venue(next(iter(venues)))
        currency = Currency.from_str(base_currency)
        instruments = {
            spec.instrument_id: _build_instrument(spec, self.profile)
            for spec in instrument_specs
        }
        data, close_by_key = _to_nautilus_data(frames, instruments, self.profile)
        bar_types = sorted(
            {
                BarType.from_str(
                    f"{frame.instrument_id}-{frame.timeframe_minutes}-MINUTE-MID-EXTERNAL"
                )
                for frame in frames
            },
            key=str,
        )
        strategy = _ScriptedTargetStrategy.build(actions, bar_types, self.profile)
        engine = BacktestEngine(
            BacktestEngineConfig(
                logging=LoggingConfig(bypass_logging=True),
                run_analysis=True,
            )
        )
        try:
            engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                starting_balances=[Money(initial_cash, currency)],
                base_currency=currency,
                default_leverage=default_leverage,
                margin_model=margin_model,
                modules=modules,
                fill_model=FillModel(random_seed=self.profile.random_seed),
                fee_model=MakerTakerFeeModel(),
                latency_model=LatencyModel(
                    base_latency_nanos=self.profile.latency_ms * 1_000_000,
                ),
                bar_execution=False,
                trade_execution=False,
                use_random_ids=False,
                use_position_ids=True,
            )
            for instrument in instruments.values():
                engine.add_instrument(instrument)
            engine.add_data(data)
            engine.add_strategy(strategy)
            engine.run(run_config_id=self.profile.profile_id)

            native_result = engine.get_result()
            fills = [event for event in strategy.events if event["event_type"] == "order_filled"]
            for fill in fills:
                key = (fill["instrument_id"], fill["ts_event_ns"])
                mid = close_by_key.get(key)
                if mid is not None:
                    fill["reference_mid"] = str(mid)
            event_facts = []
            for sequence, event in enumerate(strategy.events):
                event_facts.append({"sequence": sequence, **event})
            deterministic_payload = {
                "engine": "nautilus_trader",
                "engine_version": nautilus_trader.__version__,
                "profile": asdict(self.profile),
                "events": event_facts,
                "summary": dict(native_result.summary),
            }
            return {
                **deterministic_payload,
                "event_hash": _stable_hash(event_facts),
                "result_hash": _stable_hash(deterministic_payload),
                "native": {
                    "iterations": native_result.iterations,
                    "total_events": native_result.total_events,
                    "total_orders": native_result.total_orders,
                    "total_positions": native_result.total_positions,
                },
            }
        finally:
            engine.dispose()
