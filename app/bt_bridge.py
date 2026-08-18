"""
bt_bridge.py

Synchronization bridge that turns a backtrader Cerebro run into a step-driven
Gymnasium-style environment. The BTBridgeStrategy runs inside a background
thread driven by cerebro.run(); the GymFxEnv on the main thread submits
actions and waits for observations using two threading.Event primitives.

Flow per step:
  1. env.step(action) -> writes action into bridge.action_slot,
                         sets action_ready.
  2. BTBridgeStrategy.next() wakes, applies the action (buy/sell/close),
     updates bridge.state, sets obs_ready, and waits for the next action.
  3. env.step returns (obs, reward, terminated, truncated, info).

Termination / close semantics:
  - env.close() sets bridge.stop_requested = True and releases action_ready
    so the strategy thread exits cleanly.
  - If data is exhausted, the strategy calls env.mark_terminated() before the
    last notification so the main thread sees terminated=True.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import backtrader as bt


class BTBridge:
    """Shared mutable state between the env (main thread) and cerebro (worker)."""

    def __init__(self, initial_cash: float):
        self.action_ready = threading.Event()
        self.obs_ready = threading.Event()

        self.action_slot: int = 0
        self.raw_action_slot: float = 0.0
        self.stop_requested: bool = False
        self.terminated: bool = False

        # populated by the strategy on each bar
        self.equity: float = float(initial_cash)
        self.prev_equity: float = float(initial_cash)
        self.position: int = 0  # -1/0/1
        self.position_units: float = 0.0
        self.entry_price: float = 0.0
        self.holding_bars: int = 0
        self.position_open_bar_index: Optional[int] = None
        self.price: float = 0.0
        self.bar_index: int = 0
        self.total_bars: int = 0
        self.trade_count: int = 0
        self.commission_paid: float = 0.0
        self.last_trade_cost: float = 0.0
        self.execution_diagnostics: Dict[str, int] = {}
        # Solvency continuation ledgers (owner curriculum order, WP-C):
        # operational equity is the broker value above; economic equity is
        # operational equity MINUS accumulated recapitalization debt, so a
        # recapitalization can never manufacture performance.
        self.recapitalization_debt: float = 0.0
        self.recapitalization_count: int = 0
        self.would_margin_call_events: list = []
        self.termination_cause: Optional[str] = None

    def reset(self, initial_cash: float, total_bars: int) -> None:
        self.action_ready.clear()
        self.obs_ready.clear()
        self.action_slot = 0
        self.raw_action_slot = 0.0
        self.stop_requested = False
        self.terminated = False
        self.equity = float(initial_cash)
        self.prev_equity = float(initial_cash)
        self.position = 0
        # Signed position QUANTITY and live-order count (152). `position`
        # above stays a direction for backwards compatibility.
        self.position_units = 0.0
        self.entry_price = 0.0
        self.holding_bars = 0
        self.position_open_bar_index = None
        self.open_order_count = 0
        self.force_flat_request = False
        self.price = 0.0
        self.bar_index = 0
        self.total_bars = int(total_bars)
        self.trade_count = 0
        self.commission_paid = 0.0
        self.last_trade_cost = 0.0
        self.recapitalization_debt = 0.0
        self.recapitalization_count = 0
        self.would_margin_call_events = []
        self.termination_cause = None
        self.execution_diagnostics = {
            "entry_actions_seen": 0,
            "entry_orders_submitted": 0,
            "blocked_session_filter": 0,
            "blocked_atr_warmup": 0,
            "blocked_non_positive_atr": 0,
            "blocked_non_positive_size": 0,
            "blocked_non_positive_price": 0,
            "default_orders_submitted": 0,
            "plugin_apply_errors": 0,
            "protected_entry_rejections": 0,
            "protected_market_entries": 0,
            "protected_limit_entries": 0,
            "protected_stop_entries": 0,
            "protected_bracket_cancellations": 0,
            "risk_reducing_close_orders": 0,
            "event_context_no_trade_active_steps": 0,
            "event_context_action_overrides": 0,
            "event_context_blocked_entries": 0,
            "event_context_forced_flat_actions": 0,
            "event_context_forced_flat_orders": 0,
        }


class BTBridgeStrategy(bt.Strategy):
    """backtrader Strategy that yields control to the env on each bar.

    Parameters:
      bridge (BTBridge): shared state.
      position_size (float): units per order.
      strategy_plugin: optional object exposing `apply_action(bt_strategy, action, config)`
        that takes over order placement for SL/TP bracket logic. When None
        or the plugin lacks `apply_action`, the default buy/sell/close flow
        is used.
      config (dict): env config forwarded to the strategy_plugin.
    """

    params = (
        ("bridge", None),
        ("position_size", 1.0),
        ("min_equity", 100.0),
        ("strategy_plugin", None),
        ("config", None),
        # WP-C: "normal_realistic" terminates on breach exactly as before;
        # "easy_chronological_continuation" (train-only, enforced by the
        # env) records the would-be margin call, liquidates retaining the
        # full economic loss, recapitalizes ONLY operational capital as
        # debt, and continues the chronological episode.
        ("solvency_mode", "normal_realistic"),
        ("recap_target_equity", None),
    )

    def __init__(self) -> None:  # type: ignore[no-redef]
        self.bridge: BTBridge = self.p.bridge
        self._started: bool = False
        self._order_cost_accum: float = 0.0
        self._strategy_plugin = self.p.strategy_plugin
        self._plugin_apply = getattr(self._strategy_plugin, "apply_action", None) if self._strategy_plugin else None
        self._plugin_config = self.p.config or {}
        self._require_protected_entries = bool(
            self._plugin_config.get("require_protected_entries", False)
        )
        if self._require_protected_entries and not callable(self._plugin_apply):
            raise ValueError(
                "require_protected_entries=true requires a strategy plugin "
                "that implements apply_action()"
            )
        plugin_reset = getattr(self._strategy_plugin, "on_reset", None) if self._strategy_plugin else None
        if callable(plugin_reset):
            try:
                plugin_reset(self, self._plugin_config)
            except Exception:
                if self._require_protected_entries:
                    raise

    # --- backtrader lifecycle --------------------------------------------------
    def start(self) -> None:
        self.bridge.commission_paid = 0.0
        self.bridge.trade_count = 0

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed,):
            comm = float(getattr(order.executed, "comm", 0.0) or 0.0)
            self._order_cost_accum += comm
            self.bridge.commission_paid += comm
        plugin_notify = (
            getattr(self._strategy_plugin, "notify_order", None)
            if self._strategy_plugin
            else None
        )
        if callable(plugin_notify):
            plugin_notify(self, order, self._plugin_config)

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.bridge.trade_count += 1

    def next(self) -> None:
        # If the env requested stop, exit the run as quickly as possible.
        if self.bridge.stop_requested:
            self.env.runstop()
            return

        # First bar acts as a warmup so the env can see an initial observation
        # before any action is applied.
        if not self._started:
            self._started = True
            self._publish_obs()
            self.bridge.action_ready.wait()
            self.bridge.action_ready.clear()
            if self.bridge.stop_requested:
                self.env.runstop()
                return

        action = int(self.bridge.action_slot)
        self._apply_action(action)
        # Publish the state only after terminal conditions have been bound to
        # this real transition.  Otherwise the final bar wakes the Gym thread
        # as non-terminal and forces one synthetic action merely to discover
        # Backtrader's subsequent ``stop()`` callback.
        self._publish_obs(signal=False)

        if self._is_broke():
            if self.p.solvency_mode == "easy_chronological_continuation":
                self._continue_after_would_margin_call()
            else:
                self.bridge.termination_cause = "min_equity"
                self.bridge.terminated = True
        if (not self.bridge.terminated
                and self.bridge.bar_index >= self.bridge.total_bars):
            self.bridge.termination_cause = "data_end"
            self.bridge.terminated = True

        self.bridge.obs_ready.set()
        if self.bridge.terminated:
            self.env.runstop()
            return

        # wait for the next action from the env
        self.bridge.action_ready.wait()
        self.bridge.action_ready.clear()
        if self.bridge.stop_requested:
            self.env.runstop()
            return

    def stop(self) -> None:
        # Data exhausted: mark terminated and signal the env so it stops waiting.
        if self.bridge.termination_cause is None:
            self.bridge.termination_cause = (
                "external_stop" if self.bridge.stop_requested else "data_end"
            )
        self.bridge.terminated = True
        self.bridge.obs_ready.set()

    def _continue_after_would_margin_call(self) -> None:
        """WP-C easy mode: record the would-be margin call, liquidate
        retaining the FULL economic loss, recapitalize only operational
        capital (journaled as debt, never profit), and continue the
        chronological episode. Conservation is exact by construction:
        adding recap cash raises broker value and debt by the same
        amount, so economic equity (value - debt) is unchanged at the
        recap instant and thereafter tracks only real trading results."""
        equity_before = float(self.bridge.equity)
        event = {
            "cause": "would_margin_call",
            "bar_index": int(self.bridge.bar_index),
            "timestamp": self.data.datetime.datetime(0).isoformat(),
            "position": int(self.position.size),
            "equity_before": equity_before,
            "min_equity": float(self.p.min_equity),
        }
        # Liquidate: cancel resting orders, close any open position at the
        # next chronological bar. The realized loss stays in the ledgers.
        for order in list(self.broker.get_orders_open() or []):
            try:
                self.cancel(order)
            except Exception:
                pass
        if self.position.size != 0:
            self.close()
        target = float(
            self.p.recap_target_equity
            if self.p.recap_target_equity is not None
            else self.broker.startingcash
        )
        recap_amount = max(0.0, target - float(self.broker.getvalue()))
        if recap_amount > 0.0:
            self.broker.add_cash(recap_amount)
        self.bridge.recapitalization_debt += recap_amount
        self.bridge.recapitalization_count += 1
        event["recap_amount"] = recap_amount
        event["debt_total"] = float(self.bridge.recapitalization_debt)
        self.bridge.would_margin_call_events.append(event)
        # Republish so the env sees post-recap operational equity and the
        # updated debt in the SAME step that recorded the event.
        self.bridge.prev_equity = equity_before
        self.bridge.equity = float(self.broker.getvalue())

    # --- helpers ---------------------------------------------------------------
    def _apply_action(self, action: int) -> None:
        self._order_cost_accum = 0.0

        # AUD-F1-20260806-152: an operator/handover liquidation must
        # take the SAME path the margin-call liquidation uses — cancel
        # every resting order (including protective brackets) and close
        # the position with real configured costs — and must not be
        # intercepted by the strategy plugin.
        if getattr(self.bridge, "force_flat_request", False):
            for order in list(self.broker.get_orders_open() or []):
                try:
                    self.cancel(order)
                except Exception:
                    pass
            if self.position.size != 0:
                self.close()
                self.bridge.execution_diagnostics[
                    "handover_close_orders"] = (
                    self.bridge.execution_diagnostics.get(
                        "handover_close_orders", 0) + 1)
            return

        # Delegate to strategy plugin if it implements apply_action (SL/TP bracket logic).
        if callable(self._plugin_apply):
            try:
                self._plugin_apply(self, int(action), self._plugin_config)
                return
            except Exception:
                self.bridge.execution_diagnostics["plugin_apply_errors"] = (
                    self.bridge.execution_diagnostics.get("plugin_apply_errors", 0) + 1
                )
                if self._require_protected_entries:
                    self.bridge.execution_diagnostics["protected_entry_rejections"] = (
                        self.bridge.execution_diagnostics.get(
                            "protected_entry_rejections", 0
                        )
                        + int(action in (1, 2))
                    )
                    return

        if int(action) == 3:
            current_size = self.position.size
            if current_size != 0:
                self.close()
                self.bridge.execution_diagnostics["default_orders_submitted"] = (
                    self.bridge.execution_diagnostics.get("default_orders_submitted", 0) + 1
                )
                self.bridge.execution_diagnostics["risk_reducing_close_orders"] = (
                    self.bridge.execution_diagnostics.get(
                        "risk_reducing_close_orders", 0
                    )
                    + 1
                )
                self.bridge.execution_diagnostics["event_context_forced_flat_orders"] = (
                    self.bridge.execution_diagnostics.get("event_context_forced_flat_orders", 0) + 1
                )
            return

        current_size = self.position.size  # backtrader position size
        size = float(self.p.position_size)

        target_dir = {0: None, 1: +1, 2: -1}.get(action)
        if target_dir is None:
            # hold: no change
            return
        if self._require_protected_entries:
            self.bridge.execution_diagnostics["protected_entry_rejections"] = (
                self.bridge.execution_diagnostics.get(
                    "protected_entry_rejections", 0
                )
                + 1
            )
            return
        self.bridge.execution_diagnostics["entry_actions_seen"] = (
            self.bridge.execution_diagnostics.get("entry_actions_seen", 0) + 1
        )

        if target_dir == +1:
            if current_size < 0:
                self.close()
                self.buy(size=size)
                self.bridge.execution_diagnostics["default_orders_submitted"] = (
                    self.bridge.execution_diagnostics.get("default_orders_submitted", 0) + 2
                )
            elif current_size == 0:
                self.buy(size=size)
                self.bridge.execution_diagnostics["default_orders_submitted"] = (
                    self.bridge.execution_diagnostics.get("default_orders_submitted", 0) + 1
                )
        elif target_dir == -1:
            if current_size > 0:
                self.close()
                self.sell(size=size)
                self.bridge.execution_diagnostics["default_orders_submitted"] = (
                    self.bridge.execution_diagnostics.get("default_orders_submitted", 0) + 2
                )
            elif current_size == 0:
                self.sell(size=size)
                self.bridge.execution_diagnostics["default_orders_submitted"] = (
                    self.bridge.execution_diagnostics.get("default_orders_submitted", 0) + 1
                )

    def _publish_obs(self, *, signal: bool = True) -> None:
        broker = self.broker
        pos = self.position.size
        bar_index = int(len(self.data))
        previous_direction = int(self.bridge.position)
        current_direction = int(1 if pos > 0 else (-1 if pos < 0 else 0))
        self.bridge.prev_equity = self.bridge.equity
        self.bridge.equity = float(broker.getvalue())
        self.bridge.position = current_direction
        # AUD-F1-20260806-152: `position` is a DIRECTION. Downstream
        # accounting (handover close costs) needs the signed QUANTITY
        # and the live-order count, so publish both explicitly.
        self.bridge.position_units = float(pos)
        if current_direction == 0:
            self.bridge.entry_price = 0.0
            self.bridge.holding_bars = 0
            self.bridge.position_open_bar_index = None
        else:
            if (
                previous_direction != current_direction
                or self.bridge.position_open_bar_index is None
            ):
                self.bridge.position_open_bar_index = bar_index
            self.bridge.entry_price = float(self.position.price)
            self.bridge.holding_bars = max(
                0, bar_index - int(self.bridge.position_open_bar_index)
            )
        try:
            self.bridge.open_order_count = len(
                self.broker.get_orders_open() or [])
        except Exception:
            self.bridge.open_order_count = None
        self.bridge.price = float(self.data.close[0])
        self.bridge.bar_index = bar_index
        self.bridge.last_trade_cost = float(self._order_cost_accum)
        if signal:
            self.bridge.obs_ready.set()

    def _is_broke(self) -> bool:
        return self.bridge.equity <= float(self.p.min_equity)


def build_cerebro(
    *,
    bt_feed: bt.feeds.DataBase,
    broker: bt.brokers.BackBroker,
    bridge: BTBridge,
    position_size: float,
    min_equity: float,
    strategy_plugin: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    analyzers: Optional[Dict[str, Any]] = None,
) -> bt.Cerebro:
    """Factory that wires a cerebro with the bridge strategy, feed, broker and analyzers."""
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt_feed)
    cerebro.setbroker(broker)
    cerebro.addstrategy(
        BTBridgeStrategy,
        bridge=bridge,
        position_size=position_size,
        min_equity=min_equity,
        strategy_plugin=strategy_plugin,
        config=config or {},
        solvency_mode=(config or {}).get("solvency_mode",
                                         "normal_realistic"),
        recap_target_equity=(config or {}).get("recap_target_equity"),
    )
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
    if analyzers:
        for name, (klass, kwargs) in analyzers.items():
            cerebro.addanalyzer(klass, _name=name, **(kwargs or {}))
    return cerebro
