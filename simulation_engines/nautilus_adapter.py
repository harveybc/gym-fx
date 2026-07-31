"""NautilusTrader adapter for deterministic target-position replays.

Nautilus owns fills, positions, account balances, margin, commissions and P&L.
This module only translates engine-neutral inputs and exports immutable facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from decimal import Decimal
from typing import Any

from simulation_engines.contracts import ExecutionCostProfile
from simulation_engines.contracts import EntryExecutionRequest
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


def _utc_datetime_from_ns(timestamp_ns: int) -> datetime:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanoseconds // 1_000
    )


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
        from nautilus_trader.model.enums import TimeInForce
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.trading.strategy import Strategy

        action_by_key = {(item.instrument_id, item.ts_event_ns): item for item in actions}
        if len(action_by_key) != len(actions):
            raise ValueError(
                "actions must be unique by instrument_id and ts_event_ns"
            )

        class ScriptedTargetStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__(StrategyConfig(log_events=False, log_commands=False))
                self.current_units: dict[str, Decimal] = {}
                self.active_entry_orders: dict[str, Any] = {}
                self.order_context: dict[str, dict[str, Any]] = {}
                self.order_objects: dict[str, Any] = {}
                self.entry_actions: dict[str, TargetAction] = {}
                self.deferred_protection: dict[str, TargetAction] = {}
                self.protection_siblings: dict[str, Any] = {}
                self.events: list[dict[str, Any]] = []

            def on_start(self) -> None:
                for bar_type in bar_types:
                    self.subscribe_bars(bar_type)
                    self.subscribe_quote_ticks(bar_type.instrument_id)

            def _entry_expiration(self, action: TargetAction):
                expires_at_ns = action.entry_execution.expires_at_ns
                if expires_at_ns is None:
                    return TimeInForce.GTC, None
                return (
                    TimeInForce.GTD,
                    _utc_datetime_from_ns(expires_at_ns),
                )

            def _remember_order(
                self,
                order,
                action: TargetAction,
                *,
                role: str,
                requested_order_type: str,
            ) -> None:
                client_order_id = str(order.client_order_id)
                self.order_objects[client_order_id] = order
                self.order_context[client_order_id] = {
                    "action_id": action.action_id,
                    "instrument_id": action.instrument_id,
                    "role": role,
                    "requested_order_type": requested_order_type,
                }
                if role == "entry":
                    self.entry_actions[client_order_id] = action

            def _remember_bracket(
                self,
                order_list,
                action: TargetAction,
            ) -> Any:
                entry_order, stop_loss_order, take_profit_order = order_list.orders
                self._remember_order(
                    entry_order,
                    action,
                    role="entry",
                    requested_order_type=action.entry_execution.order_type,
                )
                self._remember_order(
                    stop_loss_order,
                    action,
                    role="stop_loss",
                    requested_order_type="stop",
                )
                self._remember_order(
                    take_profit_order,
                    action,
                    role="take_profit",
                    requested_order_type="limit",
                )
                return entry_order

            def _clear_active_entry(self, client_order_id: str) -> None:
                context = self.order_context.get(client_order_id)
                if context is None or context["role"] != "entry":
                    return
                instrument_id = context["instrument_id"]
                active = self.active_entry_orders.get(instrument_id)
                if active is not None and str(active.client_order_id) == client_order_id:
                    self.active_entry_orders.pop(instrument_id, None)
                self.deferred_protection.pop(client_order_id, None)
                self.entry_actions.pop(client_order_id, None)

            def _cancel_previous_entry(self, instrument_key: str, ts_event_ns: int) -> None:
                pending = self.active_entry_orders.pop(instrument_key, None)
                if pending is None or not pending.is_open:
                    return
                client_order_id = str(pending.client_order_id)
                context = self.order_context.get(client_order_id, {})
                self.events.append(
                    {
                        "event_type": "entry_cancel_requested",
                        "ts_event_ns": ts_event_ns,
                        "instrument_id": instrument_key,
                        "action_id": context.get("action_id", "unattributed"),
                        "client_order_id": client_order_id,
                        "reason": "SUPERSEDED_BY_NEW_TARGET",
                    }
                )
                self.cancel_order(pending)

            def _record_entry_submission(
                self,
                action: TargetAction,
                order,
                quantity,
            ) -> None:
                request = action.entry_execution
                self.events.append(
                    {
                        "event_type": "entry_submitted",
                        "ts_event_ns": action.ts_event_ns,
                        "instrument_id": action.instrument_id,
                        "action_id": action.action_id,
                        "client_order_id": str(order.client_order_id),
                        "order_type": request.order_type,
                        "quantity": str(quantity),
                        "limit_price": (
                            None
                            if request.limit_price is None
                            else str(request.limit_price)
                        ),
                        "trigger_price": (
                            None
                            if request.trigger_price is None
                            else str(request.trigger_price)
                        ),
                        "expires_at_ns": request.expires_at_ns,
                    }
                )

            def _submit_protection_after_stop_fill(self, event, action) -> None:
                if (
                    action.stop_loss_price is None
                    or action.take_profit_price is None
                ):
                    return
                instrument = self.cache.instrument(event.instrument_id)
                exit_side = (
                    OrderSide.SELL
                    if str(event.order_side) in {"BUY", "1"}
                    else OrderSide.BUY
                )
                quantity = instrument.make_qty(Decimal(str(event.last_qty)))
                stop_loss = self.order_factory.stop_market(
                    instrument_id=event.instrument_id,
                    order_side=exit_side,
                    quantity=quantity,
                    trigger_price=instrument.make_price(action.stop_loss_price),
                    reduce_only=True,
                    tags=["STOP_LOSS", f"ACTION:{action.action_id}"],
                )
                take_profit = self.order_factory.limit(
                    instrument_id=event.instrument_id,
                    order_side=exit_side,
                    quantity=quantity,
                    price=instrument.make_price(action.take_profit_price),
                    post_only=False,
                    reduce_only=True,
                    tags=["TAKE_PROFIT", f"ACTION:{action.action_id}"],
                )
                self._remember_order(
                    stop_loss,
                    action,
                    role="stop_loss",
                    requested_order_type="stop",
                )
                self._remember_order(
                    take_profit,
                    action,
                    role="take_profit",
                    requested_order_type="limit",
                )
                self.protection_siblings[
                    str(stop_loss.client_order_id)
                ] = take_profit
                self.protection_siblings[
                    str(take_profit.client_order_id)
                ] = stop_loss
                self.submit_order_list(
                    self.order_factory.create_list([stop_loss, take_profit])
                )
                self.events.append(
                    {
                        "event_type": "protection_submitted",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": str(event.instrument_id),
                        "action_id": action.action_id,
                        "stop_loss_order_id": str(stop_loss.client_order_id),
                        "take_profit_order_id": str(take_profit.client_order_id),
                    }
                )

            def _cancel_protection_sibling(self, client_order_id: str) -> None:
                sibling = self.protection_siblings.pop(client_order_id, None)
                if sibling is None:
                    return
                sibling_id = str(sibling.client_order_id)
                self.protection_siblings.pop(sibling_id, None)
                if sibling.is_open:
                    self.cancel_order(sibling)

            def _submit_market_fallback(self, event, action: TargetAction) -> None:
                request = action.entry_execution
                if request.unfilled_fallback != "market":
                    return
                instrument_key = action.instrument_id
                current = self.current_units.get(instrument_key, Decimal(0))
                delta = action.target_units - current
                if delta == 0:
                    return
                instrument_id = InstrumentId.from_str(instrument_key)
                instrument = self.cache.instrument(instrument_id)
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                quantity = instrument.make_qty(abs(delta))
                fallback_action = replace(
                    action,
                    ts_event_ns=int(event.ts_event),
                    action_id=f"{action.action_id}:market_fallback",
                    entry_execution=EntryExecutionRequest(),
                )
                has_protection = (
                    current == 0
                    and action.stop_loss_price is not None
                    and action.take_profit_price is not None
                )
                if has_protection:
                    order_list = self.order_factory.bracket(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                        entry_order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC,
                        sl_trigger_price=instrument.make_price(
                            action.stop_loss_price
                        ),
                        tp_price=instrument.make_price(
                            action.take_profit_price
                        ),
                        tp_post_only=False,
                    )
                    fallback_order = self._remember_bracket(
                        order_list,
                        fallback_action,
                    )
                    self.submit_order_list(order_list)
                else:
                    fallback_order = self.order_factory.market(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                    )
                    self._remember_order(
                        fallback_order,
                        fallback_action,
                        role="entry",
                        requested_order_type="market_fallback",
                    )
                    self.submit_order(fallback_order)
                self.active_entry_orders[instrument_key] = fallback_order
                self.events.append(
                    {
                        "event_type": "entry_market_fallback_submitted",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": instrument_key,
                        "action_id": action.action_id,
                        "fallback_action_id": fallback_action.action_id,
                        "expired_client_order_id": str(event.client_order_id),
                        "client_order_id": str(fallback_order.client_order_id),
                        "quantity": str(quantity),
                    }
                )

            def on_bar(self, bar) -> None:
                instrument_key = str(bar.bar_type.instrument_id)
                action = action_by_key.get((instrument_key, int(bar.ts_event)))
                if action is None:
                    return
                self._cancel_previous_entry(instrument_key, int(bar.ts_event))
                current = self.current_units.get(instrument_key, Decimal(0))
                delta = action.target_units - current
                request = action.entry_execution
                self.events.append(
                    {
                        "event_type": "target_requested",
                        "ts_event_ns": int(bar.ts_event),
                        "instrument_id": instrument_key,
                        "action_id": action.action_id,
                        "target_units": str(action.target_units),
                        "current_units": str(current),
                        "delta_units": str(delta),
                        "entry_order_type": request.order_type,
                        "entry_limit_price": (
                            None
                            if request.limit_price is None
                            else str(request.limit_price)
                        ),
                        "entry_trigger_price": (
                            None
                            if request.trigger_price is None
                            else str(request.trigger_price)
                        ),
                        "entry_expires_at_ns": request.expires_at_ns,
                    }
                )
                rejection_reason = None
                if not action.market_available:
                    rejection_reason = "MARKET_UNAVAILABLE"
                elif not action.signal_valid:
                    rejection_reason = "STALE_OR_INVALID_SIGNAL"
                if rejection_reason is not None:
                    self.events.append(
                        {
                            "event_type": "intent_rejected",
                            "ts_event_ns": int(bar.ts_event),
                            "instrument_id": instrument_key,
                            "action_id": action.action_id,
                            "reason": rejection_reason,
                            "target_units": str(action.target_units),
                            "position_units_after": str(current),
                        }
                    )
                    return
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
                        preflight_price = (
                            request.limit_price
                            or request.trigger_price
                            or Decimal(str(bar.close))
                        )
                        required = account.calculate_margin_init(
                            instrument,
                            instrument.make_qty(opening_units),
                            instrument.make_price(preflight_price),
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
                has_protection = (
                    current == 0
                    and action.stop_loss_price is not None
                    and action.take_profit_price is not None
                )
                time_in_force, expire_time = self._entry_expiration(action)
                if has_protection and request.order_type in {"market", "limit"}:
                    bracket_kwargs = {
                        "instrument_id": instrument_id,
                        "order_side": side,
                        "quantity": quantity,
                        "entry_order_type": (
                            OrderType.MARKET
                            if request.order_type == "market"
                            else OrderType.LIMIT
                        ),
                        "time_in_force": time_in_force,
                        "expire_time": expire_time,
                        "sl_trigger_price": instrument.make_price(
                            action.stop_loss_price
                        ),
                        "tp_price": instrument.make_price(
                            action.take_profit_price
                        ),
                        "tp_post_only": False,
                    }
                    if request.order_type == "limit":
                        bracket_kwargs["entry_price"] = instrument.make_price(
                            request.limit_price
                        )
                    order_list = self.order_factory.bracket(
                        **bracket_kwargs
                    )
                    entry_order = self._remember_bracket(order_list, action)
                    self.submit_order_list(order_list)
                elif request.order_type == "market":
                    entry_order = self.order_factory.market(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                    )
                    self._remember_order(
                        entry_order,
                        action,
                        role="entry",
                        requested_order_type="market",
                    )
                    self.submit_order(entry_order)
                elif request.order_type == "limit":
                    entry_order = self.order_factory.limit(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                        price=instrument.make_price(request.limit_price),
                        time_in_force=time_in_force,
                        expire_time=expire_time,
                        post_only=False,
                    )
                    self._remember_order(
                        entry_order,
                        action,
                        role="entry",
                        requested_order_type="limit",
                    )
                    self.submit_order(entry_order)
                else:
                    entry_order = self.order_factory.stop_market(
                        instrument_id=instrument_id,
                        order_side=side,
                        quantity=quantity,
                        trigger_price=instrument.make_price(
                            request.trigger_price
                        ),
                        time_in_force=time_in_force,
                        expire_time=expire_time,
                    )
                    self._remember_order(
                        entry_order,
                        action,
                        role="entry",
                        requested_order_type="stop",
                    )
                    if has_protection:
                        self.deferred_protection[
                            str(entry_order.client_order_id)
                        ] = action
                    self.submit_order(entry_order)
                self.active_entry_orders[instrument_key] = entry_order
                self._record_entry_submission(action, entry_order, quantity)

            def on_order_filled(self, event) -> None:
                instrument_key = str(event.instrument_id)
                client_order_id = str(event.client_order_id)
                context = self.order_context.get(client_order_id, {})
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
                        "action_id": context.get("action_id", "unattributed"),
                        "client_order_id": client_order_id,
                        "order_role": context.get("role", "unattributed"),
                        "requested_order_type": context.get(
                            "requested_order_type", "unattributed"
                        ),
                        "side": str(event.order_side),
                        "quantity": str(event.last_qty),
                        "price": str(event.last_px),
                        "commission": commission_amount,
                        "commission_currency": commission_currency,
                        "position_units_after": str(self.current_units[instrument_key]),
                    }
                )
                role = context.get("role")
                if role == "entry":
                    action = self.deferred_protection.get(client_order_id)
                    if action is not None:
                        self._submit_protection_after_stop_fill(event, action)
                    order = self.order_objects.get(client_order_id)
                    if order is not None and order.is_closed:
                        self._clear_active_entry(client_order_id)
                elif role in {"stop_loss", "take_profit"}:
                    self._cancel_protection_sibling(client_order_id)

            def _record_terminal_order_event(self, event, event_type: str) -> None:
                client_order_id = str(event.client_order_id)
                context = self.order_context.get(client_order_id, {})
                self.events.append(
                    {
                        "event_type": event_type,
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": str(event.instrument_id),
                        "action_id": context.get("action_id", "unattributed"),
                        "client_order_id": client_order_id,
                        "order_role": context.get("role", "unattributed"),
                        "requested_order_type": context.get(
                            "requested_order_type", "unattributed"
                        ),
                    }
                )
                self._clear_active_entry(client_order_id)
                self._cancel_protection_sibling(client_order_id)

            def on_order_expired(self, event) -> None:
                action = self.entry_actions.get(str(event.client_order_id))
                self._record_terminal_order_event(event, "order_expired")
                if action is not None:
                    self._submit_market_fallback(event, action)

            def on_order_canceled(self, event) -> None:
                self._record_terminal_order_event(event, "order_canceled")

            def on_order_rejected(self, event) -> None:
                client_order_id = str(event.client_order_id)
                context = self.order_context.get(client_order_id, {})
                self.events.append(
                    {
                        "event_type": "order_rejected",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": str(event.instrument_id),
                        "action_id": context.get("action_id", "unattributed"),
                        "client_order_id": client_order_id,
                        "reason": str(event.reason),
                    }
                )
                self._clear_active_entry(client_order_id)

            def on_order_denied(self, event) -> None:
                client_order_id = str(event.client_order_id)
                context = self.order_context.get(client_order_id, {})
                self.events.append(
                    {
                        "event_type": "order_denied",
                        "ts_event_ns": int(event.ts_event),
                        "instrument_id": str(event.instrument_id),
                        "action_id": context.get("action_id", "unattributed"),
                        "client_order_id": client_order_id,
                        "reason": str(event.reason),
                    }
                )
                self._clear_active_entry(client_order_id)

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
