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
        # Entry cost headroom: a 100%-of-equity entry is margin-rejected
        # once commission is due, so the effective fraction is scaled by
        # (1 - entry_cost_headroom). DECLARED constant, covers two sides
        # of the evidence-backed primary cost (~5.5 bp) with margin.
        "entry_cost_headroom": 0.002,
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
        self._entry_bar_check = None  # (bar_index, dir, sl, tp, units)
        self._parent = None           # in-flight parent order ref
        self._entry_fill_bar = None   # bar of the last parent fill

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

    def _submit_bracket(self, s, p, direction: int, units: float,
                        decision_price: float) -> None:
        """ONE logical entry lifecycle: parent (transmit=False), STOP
        child first, LIMIT child last (transmit=True), children active
        from the parent fill. Geometry anchors at the DECISION price
        (the parent fills at the next open; the entry-bar synthetic
        check covers that bar). Stop submitted first = collision rule."""
        import backtrader as bt
        sl_d, tp_d = self._distances(s, p, decision_price)
        if direction > 0:
            sl = decision_price - sl_d
            tp = decision_price + tp_d
            parent = s.buy(size=units, transmit=False)
            stop = s.sell(exectype=bt.Order.Stop, price=sl, size=units,
                          parent=parent, transmit=False)
            limit = s.sell(exectype=bt.Order.Limit, price=tp,
                           size=units, parent=parent, transmit=True)
        else:
            sl = decision_price + sl_d
            tp = decision_price - tp_d
            parent = s.sell(size=units, transmit=False)
            stop = s.buy(exectype=bt.Order.Stop, price=sl, size=units,
                         parent=parent, transmit=False)
            limit = s.buy(exectype=bt.Order.Limit, price=tp,
                          size=units, parent=parent, transmit=True)
        self._children = [stop, limit]
        self._parent = int(parent.ref)
        self._pending_entry = {"dir": direction, "units": units,
                               "params": p, "sl": sl, "tp": tp}

    def _submit_children(self, s, p, anchor: float, units: float,
                         direction: int) -> None:
        import backtrader as bt
        sl_d, tp_d = self._distances(s, p, anchor)
        if direction > 0:
            stop = s.sell(exectype=bt.Order.Stop,
                          price=anchor - sl_d, size=units)
            limit = s.sell(exectype=bt.Order.Limit,
                           price=anchor + tp_d, size=units, oco=stop)
        else:
            stop = s.buy(exectype=bt.Order.Stop,
                         price=anchor + sl_d, size=units)
            limit = s.buy(exectype=bt.Order.Limit,
                          price=anchor - tp_d, size=units, oco=stop)
        self._children = [stop, limit]

    def _resize_children(self, s, new_units: float) -> None:
        if not self._children:
            return
        import backtrader as bt
        old = self._children
        specs = [(o.exectype, float(o.created.price), o.isbuy())
                 for o in old]
        for o in old:
            try:
                s.cancel(o)
            except Exception:
                pass
        self._children = []
        stop = limit = None
        for exectype, price, isbuy in specs:
            fn = s.buy if isbuy else s.sell
            if exectype == bt.Order.Stop:
                stop = fn(exectype=bt.Order.Stop, price=price,
                          size=new_units)
        for exectype, price, isbuy in specs:
            fn = s.buy if isbuy else s.sell
            if exectype == bt.Order.Limit:
                limit = fn(exectype=bt.Order.Limit, price=price,
                           size=new_units, oco=stop)
        self._children = [o for o in (stop, limit) if o is not None]

    def _settle_position(self, s, fill_price: float) -> None:
        """Deterministic same-bar settlement: close the ENTIRE position
        at fill_price by direct broker accounting — position and cash
        change exactly once, commission charged, no follow-up order."""
        position = s.broker.getposition(s.data)
        size = float(position.size)
        if size == 0.0:
            return
        comminfo = s.broker.getcommissioninfo(s.data)
        commission = float(comminfo.getcommission(abs(size), fill_price))
        # closing proceeds: long sells at fill, short buys back at fill
        entry_price = float(getattr(position, "price", 0.0) or 0.0)
        gross = float(size) * (float(fill_price) - entry_price)
        s.broker.add_cash(size * fill_price - commission)
        position.update(-size, fill_price)
        s.bridge.commission_paid += commission
        # Steps-1-2 correction order 2026-08-28: a direct settlement
        # is a real closure that backtrader's trade lifecycle never
        # sees — recorded ECONOMICALLY COMPLETE in the ONE
        # authoritative stream (side/size/entry/exit/gross/costs/net),
        # idempotent per settlement bar.
        bar = int(getattr(s.bridge, "bar_index", 0))
        s.bridge.record_trade_close(
            source="envelope_direct_settlement",
            event_id=f"direct_{bar}",
            bar_index=bar,
            reason="entry_bar_settlement_at_level",
            side="long" if size > 0 else "short",
            size=abs(float(size)),
            entry_price=entry_price,
            exit_price=float(fill_price),
            gross_pnl=gross,
            costs=float(commission),
            net_pnl=gross - float(commission))

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
        self._entry_bar_check = None
        self._parent = None
        self._relevel_todo = None
        self._entry_fill_bar = None

    def apply_action(self, s, action: int, config: Dict[str, Any]) -> None:
        p = self._resolve(config)
        raw = abs(float(getattr(s.bridge, "raw_action_slot", 0.0)))
        equity = float(s.broker.getvalue())
        price = float(s.data.close[0])
        pos = float(s.position.size)
        bar = int(len(s.data))

        # WP1 (finding 324): protection is ATOMIC — parent and both
        # children are one bracket, children active from the parent
        # fill. Backtrader structurally cannot evaluate bracket children
        # against the ENTRY bar itself (proven by micro-repros incl.
        # cheat-on-open), so the entry bar is covered SYNTHETICALLY
        # here: its OHL is checked against the levels; a touch closes at
        # the NEXT bar's open (conservative fill, declared) under the
        # stop_first_pessimistic collision rule. Later bars fill AT the
        # level via the resting children. Zero unprotected bars.
        if self._entry_bar_check is not None and pos != 0:
            eb_bar, eb_dir, eb_sl, eb_tp, eb_units = self._entry_bar_check
            self._entry_bar_check = None
            bar_open = float(s.data.open[0])
            high = float(s.data.high[0])
            low = float(s.data.low[0])
            hit_sl = low <= eb_sl if eb_dir > 0 else high >= eb_sl
            hit_tp = high >= eb_tp if eb_dir > 0 else low <= eb_tp
            if hit_sl or hit_tp:
                # N1 (finding 329): settle IN the entry bar at the
                # EXECUTABLE level — never a next-open order. Stop wins
                # ambiguous collisions. Gap treatment mirrors real
                # order semantics: a stop fills at the worse of
                # (level, open); a limit fills at level or better.
                if hit_sl:
                    reason = "envelope_close_sl"
                    fill = (min(eb_sl, bar_open) if eb_dir > 0
                            else max(eb_sl, bar_open))
                else:
                    reason = "envelope_close_tp"
                    fill = (max(eb_tp, bar_open) if eb_dir > 0
                            else min(eb_tp, bar_open))
                self._cancel_children(s)
                self._settle_position(s, fill)
                self._events(s).append({
                    "bar_index": bar, "reason": reason, "price": fill,
                    "detail": "entry_bar_settlement_at_level"})
                self._entry_anchor = None
                return
        if getattr(self, "_relevel_todo", None) is not None and pos != 0:
            rl_dir, rl_sl, rl_tp, rl_units = self._relevel_todo
            self._relevel_todo = None
            # replace decision-anchored children with fill-anchored ones
            self._cancel_children(s)
            self._submit_children(s, self._resolve(config),
                                  self._entry_anchor, rl_units, rl_dir)
        if (pos != 0 and not self._children
                and self._pending_entry is None
                and self._entry_bar_check is None):
            # TYPED RUN FAILURE (order WP1): an unprotected open
            # position is never accepted evidence.
            s.bridge.envelope_run_failure = (
                f"unprotected_position_bar_{bar}")
            s.close()
            self._events(s).append({"bar_index": bar,
                                    "reason": "envelope_residual_sweep",
                                    "price": price})
            diag = s.bridge.execution_diagnostics
            diag["envelope_residual_sweeps"] = diag.get(
                "envelope_residual_sweeps", 0) + 1
            return

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
        frac = min(float(p["leverage_cap"]), raw) * (
            1.0 - float(p["entry_cost_headroom"]))
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
            if self._entry_fill_bar == bar:
                # DECLARED: a flip signal on the entry-FILL bar defers
                # one bar — the freshly submitted children are still
                # broker-Submitted and a cancel would silently no-op
                # (third backtrader constraint, proven by the bar-2707
                # double-stop trace), leaving a live stop against the
                # reversed position.
                diag = s.bridge.execution_diagnostics
                diag["reversal_deferred_entry_bar"] = diag.get(
                    "reversal_deferred_entry_bar", 0) + 1
                return
            # reversal: cancel old children and close (apply context —
            # cancels are honored here), THEN the fresh atomic bracket.
            self._cancel_children(s)
            self._entry_bar_check = None
            s.close()
            self._events(s).append({"bar_index": bar,
                                    "reason": "reversal_close",
                                    "price": price})
            self._entry_anchor = None
            self._submit_bracket(s, p, target_dir, units, price)
            return

        if pos == 0:
            if self._pending_entry is not None:
                if self._pending_entry["dir"] == target_dir:
                    return  # declared: same-direction while pending drops
                self._cancel_children(s)
                self._pending_entry = None
            self._submit_bracket(s, p, target_dir, units, price)
            return

        # same direction: ENTRY-ANCHORED sizing (declared). The
        # position was sized from equity and price AT ENTRY and is HELD
        # unchanged; re-sizing happens only at the next entry (after an
        # envelope fire, a policy close or a reversal). Equity-tracking
        # per-bar rebalancing is deliberately absent: it re-sizes every
        # bar (equity moves every bar), keeps the protective children
        # perpetually one bar stale and manufactures churn — the second
        # v2 run measured hundreds of residual sweeps under it.
        return

    def notify_order(self, s, order, config: Dict[str, Any]) -> None:
        import backtrader as bt
        if order.status in (order.Margin, order.Rejected):
            # NEVER silent (first v2 run: a margin-rejected 100% entry
            # produced a zero-trade arm with no trace)
            diag = s.bridge.execution_diagnostics
            diag["envelope_order_rejections"] = diag.get(
                "envelope_order_rejections", 0) + 1
            if self._pending_entry is not None:
                self._pending_entry = None
            return
        if order.status != order.Completed:
            return
        exec_price = float(order.executed.price)
        bar = int(len(s.data))

        # protective child filled -> envelope close
        if int(order.ref) in {int(c.ref) for c in self._children}:
            reason = ("envelope_close_sl"
                      if order.exectype == bt.Order.Stop
                      else "envelope_close_tp")
            self._events(s).append({"bar_index": bar, "reason": reason,
                                    "price": exec_price,
                                    "order_ref": int(order.ref)})
            self._children = []
            self._entry_anchor = None
            return
        pending = self._pending_entry
        if pending is None or int(order.ref) != self._parent:
            return
        self._pending_entry = None
        self._parent = None
        self._entry_anchor = exec_price
        self._entry_fill_bar = bar
        # N1: re-anchor geometry to the ACTUAL parent fill; the resting
        # children (decision-anchored) are re-leveled on the next apply
        # if the fill moved the anchor. For the entry bar itself the
        # settlement check below uses fill-anchored levels.
        p = pending["params"]
        sl_d, tp_d = self._distances(s, p, exec_price)
        if pending["dir"] > 0:
            pending["sl"] = exec_price - sl_d
            pending["tp"] = exec_price + tp_d
        else:
            pending["sl"] = exec_price + sl_d
            pending["tp"] = exec_price - tp_d
        self._relevel_todo = (pending["dir"], pending["sl"],
                              pending["tp"], pending["units"])
        self._events(s).append({
            "bar_index": bar, "reason": "entry_fill",
            "price": exec_price, "order_ref": int(order.ref),
            "children_refs": [int(c.ref) for c in self._children]})
        # the children are ALREADY live (atomic bracket); arm the
        # synthetic entry-bar check for THIS bar's OHL
        self._entry_bar_check = (bar, pending["dir"], pending["sl"],
                                 pending["tp"], pending["units"])
