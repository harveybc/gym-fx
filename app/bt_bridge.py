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
        self.stop_requested: bool = False
        self.terminated: bool = False

        # populated by the strategy on each bar
        self.equity: float = float(initial_cash)
        self.prev_equity: float = float(initial_cash)
        self.position: int = 0  # -1/0/1
        self.price: float = 0.0
        self.bar_index: int = 0
        self.total_bars: int = 0
        self.trade_count: int = 0
        self.commission_paid: float = 0.0
        self.last_trade_cost: float = 0.0

    def reset(self, initial_cash: float, total_bars: int) -> None:
        self.action_ready.clear()
        self.obs_ready.clear()
        self.action_slot = 0
        self.stop_requested = False
        self.terminated = False
        self.equity = float(initial_cash)
        self.prev_equity = float(initial_cash)
        self.position = 0
        self.price = 0.0
        self.bar_index = 0
        self.total_bars = int(total_bars)
        self.trade_count = 0
        self.commission_paid = 0.0
        self.last_trade_cost = 0.0


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
    )

    def __init__(self) -> None:  # type: ignore[no-redef]
        self.bridge: BTBridge = self.p.bridge
        self._started: bool = False
        self._order_cost_accum: float = 0.0
        self._strategy_plugin = self.p.strategy_plugin
        self._plugin_apply = getattr(self._strategy_plugin, "apply_action", None) if self._strategy_plugin else None
        self._plugin_config = self.p.config or {}
        plugin_reset = getattr(self._strategy_plugin, "on_reset", None) if self._strategy_plugin else None
        if callable(plugin_reset):
            try:
                plugin_reset(self, self._plugin_config)
            except Exception:
                pass

    # --- backtrader lifecycle --------------------------------------------------
    def start(self) -> None:
        self.bridge.commission_paid = 0.0
        self.bridge.trade_count = 0

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed,):
            comm = float(getattr(order.executed, "comm", 0.0) or 0.0)
            self._order_cost_accum += comm
            self.bridge.commission_paid += comm

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
        self._publish_obs()

        if self._is_broke():
            self.bridge.terminated = True
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
        self.bridge.terminated = True
        self.bridge.obs_ready.set()

    # --- helpers ---------------------------------------------------------------
    def _apply_action(self, action: int) -> None:
        self._order_cost_accum = 0.0

        # Delegate to strategy plugin if it implements apply_action (SL/TP bracket logic).
        if callable(self._plugin_apply):
            try:
                self._plugin_apply(self, int(action), self._plugin_config)
                return
            except Exception:
                # Fall back to default flow if the plugin fails, so a broken
                # strategy plugin does not kill the env silently.
                pass

        current_size = self.position.size  # backtrader position size
        size = float(self.p.position_size)

        target_dir = {0: None, 1: +1, 2: -1}.get(action)
        if target_dir is None:
            # hold: no change
            return

        if target_dir == +1:
            if current_size < 0:
                self.close()
                self.buy(size=size)
            elif current_size == 0:
                self.buy(size=size)
        elif target_dir == -1:
            if current_size > 0:
                self.close()
                self.sell(size=size)
            elif current_size == 0:
                self.sell(size=size)

    def _publish_obs(self) -> None:
        broker = self.broker
        pos = self.position.size
        self.bridge.prev_equity = self.bridge.equity
        self.bridge.equity = float(broker.getvalue())
        self.bridge.position = int(1 if pos > 0 else (-1 if pos < 0 else 0))
        self.bridge.price = float(self.data.close[0])
        self.bridge.bar_index = int(len(self.data))
        self.bridge.last_trade_cost = float(self._order_cost_accum)
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

