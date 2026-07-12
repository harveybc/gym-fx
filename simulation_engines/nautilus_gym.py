"""Gymnasium bridge backed by the NautilusTrader event engine."""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

import pandas as pd

from app.bt_bridge import BTBridge
from app.env import GymFxEnv
from simulation_engines.contracts import InstrumentSpec
from simulation_engines.contracts import MarketFrame
from simulation_engines.contracts import load_execution_cost_profile
from simulation_engines.nautilus_adapter import _build_instrument
from simulation_engines.nautilus_adapter import _to_nautilus_data
from simulation_engines.nautilus_adapter import require_nautilus


def _timeframe_minutes(value: str) -> int:
    raw = str(value).strip().lower()
    if raw.startswith("m") and raw[1:].isdigit():
        return int(raw[1:])
    if raw.startswith("h") and raw[1:].isdigit():
        return int(raw[1:]) * 60
    if raw.endswith("m") and raw[:-1].isdigit():
        return int(raw[:-1])
    if raw.endswith("h") and raw[:-1].isdigit():
        return int(raw[:-1]) * 60
    raise ValueError(f"unsupported Nautilus timeframe {value!r}")


def _instrument_spec(config: dict[str, Any]) -> InstrumentSpec:
    raw = str(config.get("instrument", "EUR_USD")).replace("_", "/")
    if "/" not in raw:
        raise ValueError("Nautilus FX instrument must identify base and quote currencies")
    base, quote = raw.split("/", 1)
    price_precision = int(config.get("price_precision", 3 if quote == "JPY" else 5))
    return InstrumentSpec(
        symbol=f"{base}/{quote}",
        venue=str(config.get("simulation_venue", "SIM")),
        base_currency=base,
        quote_currency=quote,
        price_precision=price_precision,
        size_precision=int(config.get("size_precision", 0)),
        margin_init=Decimal(str(config.get("margin_init", "0.05"))),
        margin_maint=Decimal(str(config.get("margin_maint", "0.025"))),
        min_quantity=Decimal(str(config.get("min_quantity", "1"))),
        lot_size=Decimal(str(config.get("lot_size", "1"))),
    )


def _frames(dataframe: pd.DataFrame, config: dict[str, Any], spec: InstrumentSpec):
    date_column = str(config.get("date_column", "DATE_TIME"))
    price_column = str(config.get("price_column", "CLOSE"))
    timeframe = _timeframe_minutes(str(config.get("timeframe", "M1")))
    frames = []
    for row_index, row in dataframe.iterrows():
        raw_timestamp = row.get(date_column, row_index)
        timestamp = pd.Timestamp(raw_timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        close = Decimal(str(row[price_column]))

        def value(name: str, fallback: Decimal) -> Decimal:
            raw = row.get(name, row.get(name.lower(), fallback))
            return Decimal(str(raw))

        open_price = value("OPEN", close)
        high_price = value("HIGH", close)
        low_price = value("LOW", close)
        high_price = max(open_price, high_price, low_price, close)
        low_price = min(open_price, high_price, low_price, close)
        frames.append(
            MarketFrame(
                instrument_id=spec.instrument_id,
                timeframe_minutes=timeframe,
                ts_event_ns=int(timestamp.value),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close,
                volume=max(value("VOLUME", Decimal("1000000")), Decimal("1")),
            )
        )
    return frames


def _build_bridge_strategy(bridge, spec, bar_type, position_size, profile):
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.enums import PriceType
    from nautilus_trader.trading.strategy import Strategy

    class GymBridgeStrategy(Strategy):
        def __init__(self):
            super().__init__(StrategyConfig(log_events=False, log_commands=False))
            self.current_units = Decimal(0)

        def on_start(self):
            self.subscribe_bars(bar_type)
            self.subscribe_quote_ticks(bar_type.instrument_id)

        def on_bar(self, bar):
            if bridge.stop_requested:
                self.stop()
                return
            self._publish(bar)
            bridge.action_ready.wait()
            bridge.action_ready.clear()
            if bridge.stop_requested:
                self.stop()
                return
            action = int(bridge.action_slot)
            target = {
                0: self.current_units,
                1: Decimal(str(position_size)),
                2: -Decimal(str(position_size)),
                3: Decimal(0),
            }.get(action, self.current_units)
            delta = target - self.current_units
            if delta != 0:
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                instrument = self.cache.instrument(bar_type.instrument_id)
                if profile.enforce_margin_preflight:
                    opening_units = Decimal(0)
                    if self.current_units == 0 or self.current_units * delta > 0:
                        opening_units = abs(delta)
                    elif abs(delta) > abs(self.current_units):
                        opening_units = abs(delta) - abs(self.current_units)
                    if opening_units > 0:
                        account = self.cache.account_for_venue(
                            bar_type.instrument_id.venue
                        )
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
                                venue=bar_type.instrument_id.venue,
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
                            bridge.execution_diagnostics["nautilus_preflight_denied"] = (
                                bridge.execution_diagnostics.get(
                                    "nautilus_preflight_denied", 0
                                )
                                + 1
                            )
                            bridge.execution_diagnostics[
                                "nautilus_last_denial_reason"
                            ] = "CUM_MARGIN_EXCEEDS_FREE_BALANCE"
                            return
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=bar_type.instrument_id,
                        order_side=side,
                        quantity=instrument.make_qty(abs(delta)),
                    )
                )

        def on_order_filled(self, event):
            previous = self.current_units
            quantity = Decimal(str(event.last_qty))
            if str(event.order_side) in {"SELL", "2"}:
                quantity = -quantity
            self.current_units += quantity
            bridge.commission_paid += float(event.commission.as_decimal())
            bridge.last_trade_cost = float(event.commission.as_decimal())
            if previous != 0 and self.current_units == 0:
                bridge.trade_count += 1

        def on_order_denied(self, event):
            bridge.execution_diagnostics["nautilus_order_denied"] = (
                bridge.execution_diagnostics.get("nautilus_order_denied", 0) + 1
            )
            bridge.execution_diagnostics["nautilus_last_denial_reason"] = str(
                event.reason
            )

        def on_order_rejected(self, event):
            bridge.execution_diagnostics["nautilus_order_rejected"] = (
                bridge.execution_diagnostics.get("nautilus_order_rejected", 0) + 1
            )
            bridge.execution_diagnostics["nautilus_last_rejection_reason"] = str(
                event.reason
            )

        def on_stop(self):
            bridge.terminated = True
            bridge.obs_ready.set()

        def _publish(self, bar):
            equity = self.portfolio.equity(bar_type.instrument_id.venue)
            bridge.prev_equity = bridge.equity
            if isinstance(equity, dict):
                bridge.equity = sum(
                    float(value.as_decimal()) for value in equity.values()
                )
            elif equity is not None:
                bridge.equity = float(equity.as_decimal())
            bridge.position = 1 if self.current_units > 0 else (-1 if self.current_units < 0 else 0)
            bridge.price = float(bar.close)
            bridge.bar_index += 1
            bridge.last_trade_cost = 0.0
            bridge.obs_ready.set()

    return GymBridgeStrategy()


class NautilusGymFxEnv(GymFxEnv):
    """Single-cell compatibility bridge; portfolio-native observations follow later."""

    def __init__(self, *args, **kwargs):
        require_nautilus()
        super().__init__(*args, **kwargs)
        profile_path = self.config.get("execution_cost_profile")
        if not profile_path:
            raise ValueError("execution_cost_profile is required for Nautilus")
        self._nautilus_profile = load_execution_cost_profile(profile_path)
        self._nautilus_engine = None
        self._nautilus_result = None

    def reset(self, *, seed=None, options=None):
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

        super(GymFxEnv, self).reset(seed=seed)
        self._teardown_runner()
        self.bridge = BTBridge(initial_cash=self.initial_cash)
        self.bridge.reset(initial_cash=self.initial_cash, total_bars=self.total_bars)
        self._reset_action_diagnostics()

        spec = _instrument_spec(self.config)
        instrument = _build_instrument(spec, self._nautilus_profile)
        frames = _frames(self.dataframe, self.config, spec)
        data, _ = _to_nautilus_data(frames, {spec.instrument_id: instrument}, self._nautilus_profile)
        bar_type = BarType.from_str(
            f"{spec.instrument_id}-{frames[0].timeframe_minutes}-MINUTE-MID-EXTERNAL"
        )
        engine = BacktestEngine(
            BacktestEngineConfig(logging=LoggingConfig(bypass_logging=True), run_analysis=True)
        )
        currency = Currency.from_str(str(self.config.get("account_currency", "USD")))
        modules = None
        if self._nautilus_profile.financing_enabled:
            rate_path = self.config.get("financing_rate_data_file")
            if not rate_path:
                raise ValueError(
                    "financing_rate_data_file is required by the selected cost profile"
                )
            from nautilus_trader.backtest.config import FXRolloverInterestConfig
            from nautilus_trader.backtest.modules import FXRolloverInterestModule

            rate_data = pd.read_csv(rate_path)
            modules = [
                FXRolloverInterestModule(
                    FXRolloverInterestConfig(rate_data=rate_data)
                )
            ]
        engine.add_venue(
            venue=Venue(spec.venue),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(Decimal(str(self.initial_cash)), currency)],
            base_currency=currency,
            default_leverage=Decimal(str(self.config.get("leverage", "20"))),
            margin_model=(
                StandardMarginModel()
                if self._nautilus_profile.margin_model == "standard"
                else LeveragedMarginModel()
            ),
            modules=modules,
            fill_model=FillModel(random_seed=self._nautilus_profile.random_seed),
            fee_model=MakerTakerFeeModel(),
            latency_model=LatencyModel(
                base_latency_nanos=self._nautilus_profile.latency_ms * 1_000_000
            ),
            bar_execution=False,
            trade_execution=False,
            use_random_ids=False,
        )
        engine.add_instrument(instrument)
        engine.add_data(data)
        strategy = _build_bridge_strategy(
            self.bridge,
            spec,
            bar_type,
            self.position_size,
            self._nautilus_profile,
        )
        engine.add_strategy(strategy)
        self._nautilus_engine = engine
        self._strategy_instance = strategy
        self._runner = threading.Thread(
            target=self._run_nautilus,
            name="gym-fx-nautilus",
            daemon=True,
        )
        self._runner.start()
        self._wait_obs()
        return self._make_observation(), self._make_info()

    def _run_nautilus(self):
        try:
            self._nautilus_engine.run(
                run_config_id=self._nautilus_profile.profile_id
            )
            self._nautilus_result = self._nautilus_engine.get_result()
        finally:
            self.bridge.terminated = True
            self.bridge.obs_ready.set()

    def _teardown_runner(self):
        super()._teardown_runner()
        if self._nautilus_engine is not None:
            self._nautilus_engine.dispose()
        self._nautilus_engine = None

    def summary(self):
        summary = self.metrics_plugin.summarize(
            initial_cash=self.initial_cash,
            final_equity=self.bridge.equity if self.bridge else self.initial_cash,
            analyzers={},
            config=self.config,
        )
        summary["simulation_engine"] = "nautilus_trader"
        summary["execution_cost_profile"] = self._nautilus_profile.profile_id
        if self._nautilus_result is not None:
            summary["native_summary"] = dict(self._nautilus_result.summary)
        return summary
