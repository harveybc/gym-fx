"""Shared execution envelope (Screen B correction order C2, 2026-08-25).

ONE typed protection/sizing contract consumed by mechanical rule arms
(B0-B3) and the learned B4 alike. Declared semantics:

* GEOMETRY — per risk-increasing entry, native-style SL and TP:
  ``fixed_fraction`` mode anchors both at the PARENT FILL price
  (sl_fraction / tp_fraction of it); ``atr`` mode uses a causal Wilder
  ATR over ``atr_window`` completed bars times ``atr_sl_mult`` /
  ``atr_tp_mult``. Same-direction size increases inherit the original
  levels (declared); a reversal re-anchors at the new fill.
* COLLISION RULE — ``stop_first_pessimistic``: the protective STOP is
  submitted before the LIMIT and linked OCO, so a single H4 bar whose
  range touches both resolves to the STOP. Proven by adversarial test,
  not assumed.
* SIZING — ``portfolio_fraction``: units = equity_at_decision *
  min(leverage_cap, |raw_action|) / close_at_decision. Both inputs are
  strictly pre-fill (the order executes at the NEXT bar's open), so no
  same-bar information enters the position (C3).
* CLOSE TAXONOMY — every position exit appends
  ``{bar_index, reason, price}`` to ``bridge.close_events`` with reason
  in {envelope_close_sl, envelope_close_tp, policy_close,
  reversal_close}; data-end liquidation is recorded by the DRIVER when
  the run terminates with an open position (the strategy has no
  post-data hook) — declared, not silent.
"""
from __future__ import annotations

from typing import Any, Dict


class Plugin:
    plugin_params = {
        "envelope_mode": "fixed_fraction",   # or "atr"
        "sl_fraction": 0.05,
        "tp_fraction": 0.10,
        "atr_window": 14,
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 4.0,
        "collision_rule": "stop_first_pessimistic",
        "sizing_mode": "portfolio_fraction",
        "leverage_cap": 1.0,
        "min_units": 1e-9,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            for k, v in config.items():
                if k in self.plugin_params and v is not None:
                    self.params[k] = v
        self._pending_entry = None   # set at entry submit, consumed at fill
        self._children = []          # live protective orders
        self._entry_anchor = None    # fill price of the anchoring entry

    def set_params(self, **kwargs) -> None:
        """Bundle-loader contract: absorb known envelope keys; the
        nested execution_envelope block (resolved per call) wins."""
        for k, v in kwargs.items():
            if k in self.plugin_params and v is not None:
                self.params[k] = v

    # -- helpers -----------------------------------------------------------
    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        env = config.get("execution_envelope") or {}
        for k in self.plugin_params:
            if k in env and env[k] is not None:
                merged[k] = env[k]
            elif k in config and config[k] is not None:
                merged[k] = config[k]
        return merged

    @staticmethod
    def _events(s):
        if not hasattr(s.bridge, "close_events"):
            s.bridge.close_events = []
        return s.bridge.close_events

    def _cancel_children(self, s) -> None:
        for order in self._children:
            try:
                s.cancel(order)
            except Exception:
                pass
        self._children = []

    def _atr(self, s, window: int) -> float:
        trs = []
        for i in range(-window, 0):
            high = float(s.data.high[i])
            low = float(s.data.low[i])
            prev_close = float(s.data.close[i - 1])
            trs.append(max(high - low, abs(high - prev_close),
                           abs(low - prev_close)))
        return sum(trs) / len(trs)

    def _distances(self, s, p, anchor: float) -> tuple:
        if p["envelope_mode"] == "atr":
            atr = self._atr(s, int(p["atr_window"]))
            return atr * float(p["atr_sl_mult"]), atr * float(
                p["atr_tp_mult"])
        return anchor * float(p["sl_fraction"]), anchor * float(
            p["tp_fraction"])

    # -- BTBridgeStrategy contract ----------------------------------------
    def on_reset(self, s, config: Dict[str, Any]) -> None:
        self._pending_entry = None
        self._children = []
        self._entry_anchor = None

    def apply_action(self, s, action: int, config: Dict[str, Any]) -> None:
        p = self._resolve(config)
        raw = abs(float(getattr(s.bridge, "raw_action_slot", 0.0)))
        equity = float(s.broker.getvalue())
        price = float(s.data.close[0])
        pos = float(s.position.size)
        bar = int(len(s.data))

        if action == 0:
            return  # hold: envelope children keep working

        if action == 3:
            self._cancel_children(s)
            if pos != 0:
                s.close()
                self._events(s).append({"bar_index": bar,
                                        "reason": "policy_close",
                                        "price": price})
            self._entry_anchor = None
            return

        target_dir = +1 if action == 1 else -1
        frac = min(float(p["leverage_cap"]), raw)
        units = equity * frac / price if price > 0 else 0.0

        if units <= float(p["min_units"]):
            # directional signal with ~zero fraction = target flat
            self._cancel_children(s)
            if pos != 0:
                s.close()
                self._events(s).append({"bar_index": bar,
                                        "reason": "policy_close",
                                        "price": price})
            self._entry_anchor = None
            return

        if pos != 0 and (pos > 0) != (target_dir > 0):
            # reversal: close now, re-enter with a fresh envelope
            self._cancel_children(s)
            s.close()
            self._events(s).append({"bar_index": bar,
                                    "reason": "reversal_close",
                                    "price": price})
            self._entry_anchor = None
            self._pending_entry = {"dir": target_dir, "units": units,
                                   "params": p}
            if target_dir > 0:
                s.buy(size=units)
            else:
                s.sell(size=units)
            return

        if pos == 0:
            self._pending_entry = {"dir": target_dir, "units": units,
                                   "params": p}
            if target_dir > 0:
                s.buy(size=units)
            else:
                s.sell(size=units)
            return

        # same direction: rebalance units toward the new target; the
        # ORIGINAL envelope levels remain anchored (declared).
        target_signed = units * (1.0 if pos > 0 else -1.0)
        delta = target_signed - pos
        if abs(delta) > float(p["min_units"]):
            if delta > 0:
                s.buy(size=delta)
            else:
                s.sell(size=-delta)

    def notify_order(self, s, order, config: Dict[str, Any]) -> None:
        import backtrader as bt
        if order.status != order.Completed:
            return
        exec_price = float(order.executed.price)
        bar = int(len(s.data))
        # protective child filled -> envelope close
        if order in self._children:
            reason = ("envelope_close_sl"
                      if order.exectype == bt.Order.Stop
                      else "envelope_close_tp")
            self._events(s).append({"bar_index": bar, "reason": reason,
                                    "price": exec_price})
            self._children = []
            self._entry_anchor = None
            return
        pending = self._pending_entry
        if pending is None:
            return
        self._pending_entry = None
        p = pending["params"]
        units = pending["units"]
        direction = pending["dir"]
        self._entry_anchor = exec_price
        sl_d, tp_d = self._distances(s, p, exec_price)
        # COLLISION RULE stop_first_pessimistic: STOP first, LIMIT OCO'd
        if direction > 0:
            stop = s.sell(exectype=bt.Order.Stop,
                          price=exec_price - sl_d, size=units)
            limit = s.sell(exectype=bt.Order.Limit,
                           price=exec_price + tp_d, size=units, oco=stop)
        else:
            stop = s.buy(exectype=bt.Order.Stop,
                         price=exec_price + sl_d, size=units)
            limit = s.buy(exectype=bt.Order.Limit,
                          price=exec_price - tp_d, size=units, oco=stop)
        self._children = [stop, limit]
