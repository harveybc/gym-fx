"""
direct_atr_sltp.py

Strategy plugin that places bracket orders with SL/TP sized by rolling ATR.
    SL distance = k_sl * ATR(atr_period)
    TP distance = k_tp * ATR(atr_period)

The plugin maintains its own rolling True-Range buffer from the backtrader
data lines (no backtrader indicator needed; avoids minperiod coupling with
BTBridgeStrategy). Until the ATR buffer is warmed, entries are skipped so the
environment never emits a naked order without SL/TP brackets.

Action semantics: {0=hold, 1=long, 2=short, 3=close}. Action 3 is the
model-visible risk-reducing close used by the v2 target-exposure contract.

    Config keys:
    atr_period: int   — ATR window (default 14), GA-tunable
    k_sl: float       — SL = k_sl * ATR (default 2.0), GA-tunable
    k_tp: float       — TP = k_tp * ATR (default 3.0), GA-tunable
    position_size: float  — fallback flat units per order if rel_volume is None
    rel_volume: float | None  — fraction of cash to risk per order (Project 2
        heuristic default: 0.10). When set, size = clamp(cash * rel_volume *
        leverage, min_order_volume, max_order_volume) and overrides position_size.
    leverage: float   — broker leverage multiplier (default 1.0; Project 2 FX=100)
    min_order_volume: float
    max_order_volume: float
    sltp_risk_mode: fixed_atr | rel_volume_aware_atr | margin_aware_atr
        fixed_atr preserves the historical behavior exactly.
        rel_volume_aware_atr shrinks SL/TP ATR multiples as rel_volume rises
        above the baseline, while preserving the baseline point.
        margin_aware_atr additionally caps SL by max_planned_loss_fraction.
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Deque, Dict

import backtrader as bt


_AUDIT_PATH = os.environ.get("GYMFX_BRACKET_AUDIT")


def _audit_emit(rec: Dict[str, Any]) -> None:
    if not _AUDIT_PATH:
        return
    try:
        with open(_AUDIT_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class Plugin:
    plugin_params = {
        "atr_period": 14,
        "k_sl": 2.0,
        "k_tp": 3.0,
        "position_size": 1.0,
        # Project 2 heuristic-strategy sizing (direction_atr plugin defaults):
        # rel_volume=0.10, leverage=100, min=10_000, max=1_000_000.
        # Leave rel_volume=None to disable and use flat position_size.
        "rel_volume": None,
        "leverage": 1.0,
        "min_order_volume": 0.0,
        "max_order_volume": 1e12,
        # "fx_units": size = cash * rel_volume * leverage  (Project 2 FX default,
        #    assumes 1 unit ~= $1 notional, correct for EURUSD-class quotes).
        # "notional": size = cash * rel_volume * leverage / price  (correct for
        #    instruments whose price is the per-unit cost, e.g. BTC/ETH spot).
        "size_mode": "fx_units",
        # SL/TP distance clamps as fraction of price. Prevent degenerate
        # brackets when ATR is pathological (flash-crash bar, thin liquidity).
        # Defaults allow 0.1%..20% of price, which covers realistic FX/crypto
        # volatility bands. Set to None to disable a bound.
        "min_sltp_frac": 0.001,
        "max_sltp_frac": 0.20,
        # ----- Risk-aware SL/TP geometry ---------------------------------
        # fixed_atr keeps the historical baseline: SL=k_sl*ATR, TP=k_tp*ATR.
        # rel_volume_aware_atr preserves that baseline at baseline_rel_volume
        # and shrinks the effective ATR multiples as exposure approaches
        # max_risk_rel_volume. This lets experiments compare fixed vs
        # exposure-aware SL/TP without changing the old baseline.
        "sltp_risk_mode": "fixed_atr",
        "baseline_rel_volume": 0.05,
        "max_risk_rel_volume": 0.50,
        "rel_volume_sl_shrink_alpha": 0.35,
        "rel_volume_tp_shrink_alpha": 0.20,
        "min_k_sl": 1.0,
        "min_reward_risk_ratio": 1.0,
        # Optional equity fraction cap for the planned stop loss. For notional
        # sizing this caps SL distance to:
        #   price * max_planned_loss_fraction / (rel_volume * leverage)
        # Leave None to avoid changing fixed_atr baseline behavior.
        "max_planned_loss_fraction": None,
        # ----- Session/weekend filter (avoid weekend volatility) ----------
        # When `session_filter` is True, new entries are only allowed inside
        # the entry window [entry_dow_start@entry_hour_start ..
        # force_close_dow@force_close_hour). Outside that window the strategy
        # IGNORES long/short actions (treats them as hold). Once the bar
        # crosses into the force-close zone (>= force_close_dow@hour up to
        # entry_dow_start@hour of next week), any open position is flattened
        # immediately with a market close, regardless of agent action.
        # dow: Monday=0 ... Sunday=6 (Python datetime.weekday()).
        "session_filter": False,
        "entry_dow_start": 0,        # Monday
        "entry_hour_start": 12,      # 12:00 (mid-morning, 12h after typical open)
        "force_close_dow": 4,        # Friday
        "force_close_hour": 20,      # 20:00 — flatten everything from here
        # ----- Protected entry execution --------------------------------
        # Every risk-increasing order is a Backtrader bracket whose parent can
        # be market, limit, or stop. SL and TP children are always attached.
        "entry_order_mode": "adaptive",  # adaptive|market|limit|stop
        "full_spread_rate": 0.0,
        "market_urgency_threshold": 0.75,
        "market_max_spread_bps": 8.0,
        "stop_breakout_threshold": 0.65,
        "limit_offset_spread_multiple": 0.5,
        "limit_offset_atr_multiple": 0.05,
        "stop_offset_spread_multiple": 0.5,
        "stop_offset_atr_multiple": 0.05,
        "breakout_lookback": 12,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)
        self._tr_buffer: Deque[float] = deque()
        self._prev_close: float | None = None
        self._high_buffer: Deque[float] = deque()
        self._low_buffer: Deque[float] = deque()
        self._bracket_orders: list[Any] = []
        self._pending_side: int = 0

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    def decide_action(self, obs, info, step: int) -> int:
        return 0

    def on_reset(self, bt_strategy, config: Dict[str, Any]) -> None:
        self._tr_buffer = deque(maxlen=int(self._resolve(config)["atr_period"]))
        self._prev_close = None
        lookback = max(2, int(self._resolve(config)["breakout_lookback"]))
        self._high_buffer = deque(maxlen=lookback)
        self._low_buffer = deque(maxlen=lookback)
        self._bracket_orders = []
        self._pending_side = 0

    def notify_order(self, bt_strategy, order, config: Dict[str, Any]) -> None:
        del bt_strategy, order, config
        if self._bracket_orders and not any(
            bool(getattr(item, "alive", lambda: False)())
            for item in self._bracket_orders
        ):
            self._bracket_orders = []
            self._pending_side = 0

    # ------------------------------------------------------------------
    # BTBridgeStrategy contract
    # ------------------------------------------------------------------
    def apply_action(self, bt_strategy, action: int, config: Dict[str, Any]) -> None:
        p = self._resolve(config)
        period = int(p["atr_period"])
        size = self._compute_size(bt_strategy, p)
        diag = getattr(getattr(bt_strategy, "bridge", None), "execution_diagnostics", None)

        def inc(key: str, amount: int = 1) -> None:
            if isinstance(diag, dict):
                diag[key] = int(diag.get(key, 0)) + int(amount)

        high = float(bt_strategy.data.high[0])
        low = float(bt_strategy.data.low[0])
        close = float(bt_strategy.data.close[0])
        prior_high = max(self._high_buffer) if self._high_buffer else high
        prior_low = min(self._low_buffer) if self._low_buffer else low

        # Update ATR buffer with True Range
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        if self._tr_buffer.maxlen != period:
            self._tr_buffer = deque(self._tr_buffer, maxlen=period)
        self._tr_buffer.append(tr)
        lookback = max(2, int(p["breakout_lookback"]))
        if self._high_buffer.maxlen != lookback:
            self._high_buffer = deque(self._high_buffer, maxlen=lookback)
            self._low_buffer = deque(self._low_buffer, maxlen=lookback)
        self._high_buffer.append(high)
        self._low_buffer.append(low)

        if action == 3:
            forced = bool(getattr(bt_strategy, "force_flat_request", False))
            reason = "forced_flat" if forced else "model_early_close"
            self._flatten(bt_strategy, inc, reason=reason)
            if not forced:
                inc("model_early_close_actions")
            return

        # ---- Session/weekend filter -------------------------------------
        # Forcefully flatten positions outside the trading window and ignore
        # entry actions outside the entry window. This is enforced by the
        # env regardless of what the agent decides.
        in_entry_window, in_close_zone = self._session_state(bt_strategy, p)
        if in_close_zone and bt_strategy.position.size != 0:
            self._flatten(bt_strategy, inc, reason="session_force_close")
            _audit_emit({
                "kind": "session_force_close",
                "entry": close,
                "size": float(bt_strategy.position.size),
            })
            return  # do not open a new position on a force-close bar

        if action == 0:
            return

        inc("entry_actions_seen")

        # Block new entries outside the entry window.
        if p.get("session_filter") and not in_entry_window:
            inc("blocked_session_filter")
            return

        pos_size = bt_strategy.position.size
        requested_side = 1 if action == 1 else -1
        live_bracket = self._has_live_bracket()
        if live_bracket:
            current_side = 1 if pos_size > 0 else (-1 if pos_size < 0 else self._pending_side)
            if current_side == requested_side:
                inc("blocked_existing_protected_position")
                return
            self._flatten(bt_strategy, inc, reason="protected_reversal")
            return
        if pos_size != 0:
            current_side = 1 if pos_size > 0 else -1
            if current_side == requested_side:
                inc("blocked_existing_protected_position")
                return
            self._flatten(bt_strategy, inc, reason="position_reversal")
            return

        ready = len(self._tr_buffer) >= period
        atr = sum(self._tr_buffer) / len(self._tr_buffer) if self._tr_buffer else 0.0

        # Require a warmed ATR and a positive size, otherwise skip the trade
        # entirely rather than emit a naked (SL/TP-less) order. This guarantees
        # every filled order has both brackets attached.
        if not ready:
            inc("blocked_atr_warmup")
            return
        if atr <= 0.0:
            inc("blocked_non_positive_atr")
            return
        if size <= 0.0:
            inc("blocked_non_positive_size")
            return
        if close <= 0.0:
            inc("blocked_non_positive_price")
            return

        # Clamp SL/TP distances to sane fractions of price to prevent degenerate
        # brackets from pathological ATR spikes (flash crashes, thin bars).
        k_sl_eff, k_tp_eff = self._effective_sltp_multiples(p)
        sl_dist = k_sl_eff * atr
        tp_dist = k_tp_eff * atr
        max_loss = p.get("max_planned_loss_fraction")
        rel = p.get("rel_volume")
        if str(p.get("sltp_risk_mode", "fixed_atr")).lower() == "margin_aware_atr" and max_loss is not None:
            try:
                rel_f = max(0.0, float(rel or 0.0))
                leverage = max(1e-12, float(p.get("leverage", 1.0)))
                max_loss_f = max(0.0, float(max_loss))
            except (TypeError, ValueError):
                rel_f = 0.0
                leverage = 1.0
                max_loss_f = 0.0
            if rel_f > 0.0 and max_loss_f > 0.0:
                sl_dist = min(sl_dist, close * max_loss_f / (rel_f * leverage))
        min_frac = p.get("min_sltp_frac")
        max_frac = p.get("max_sltp_frac")
        if min_frac is not None:
            floor = float(min_frac) * close
            sl_dist = max(sl_dist, floor)
            tp_dist = max(tp_dist, floor)
        if max_frac is not None:
            ceil = float(max_frac) * close
            sl_dist = min(sl_dist, ceil)
            tp_dist = min(tp_dist, ceil)
        # Final safety: SL must stay above zero on shorts too (close + sl_dist
        # is always > 0 for long-stop; short TP = close - tp_dist must be > 0).
        if tp_dist >= close:
            tp_dist = close * 0.5

        breakout_score = self._breakout_score(
            action=action,
            close=close,
            prior_high=prior_high,
            prior_low=prior_low,
            atr=atr,
        )
        order_type, entry_price = self._route_entry(
            bt_strategy=bt_strategy,
            action=action,
            close=close,
            atr=atr,
            breakout_score=breakout_score,
            params=p,
        )
        protected_entry = close if entry_price is None else entry_price
        if action == 1:
            stop = protected_entry - sl_dist
            limit = protected_entry + tp_dist
            submit = bt_strategy.buy_bracket
        else:
            stop = protected_entry + sl_dist
            limit = protected_entry - tp_dist
            submit = bt_strategy.sell_bracket
        if min(protected_entry, stop, limit) <= 0.0:
            inc("blocked_invalid_bracket_geometry")
            return

        kwargs: Dict[str, Any] = {
            "size": size,
            "exectype": {
                "market": bt.Order.Market,
                "limit": bt.Order.Limit,
                "stop": bt.Order.Stop,
            }[order_type],
            "stopprice": stop,
            "stopexec": bt.Order.Stop,
            "limitprice": limit,
            "limitexec": bt.Order.Limit,
        }
        if entry_price is not None:
            kwargs["price"] = entry_price
        orders = submit(**kwargs)
        self._bracket_orders = list(orders or [])
        self._pending_side = requested_side
        inc("entry_orders_submitted")
        inc(f"protected_{order_type}_entries")
        _audit_emit({
            "kind": f"{'long' if action == 1 else 'short'}_{order_type}_bracket",
            "entry": protected_entry,
            "stop": stop,
            "limit": limit,
            "size": size,
            "atr": atr,
            "breakout_score": breakout_score,
            "k_sl_eff": k_sl_eff,
            "k_tp_eff": k_tp_eff,
            "sltp_risk_mode": p.get("sltp_risk_mode"),
        })

    def _has_live_bracket(self) -> bool:
        return any(
            bool(getattr(order, "alive", lambda: False)())
            for order in self._bracket_orders
        )

    def _flatten(self, bt_strategy, inc, *, reason: str) -> None:
        for order in self._bracket_orders:
            if bool(getattr(order, "alive", lambda: False)()):
                bt_strategy.cancel(order)
                inc("protected_bracket_cancellations")
        self._bracket_orders = []
        self._pending_side = 0
        if bt_strategy.position.size != 0:
            bt_strategy.close()
            inc("risk_reducing_close_orders")
            if reason == "forced_flat":
                inc("event_context_forced_flat_orders")
            if reason == "model_early_close":
                inc("model_early_close_orders")

    @staticmethod
    def _breakout_score(
        *,
        action: int,
        close: float,
        prior_high: float,
        prior_low: float,
        atr: float,
    ) -> float:
        if atr <= 0.0:
            return 0.0
        distance = (
            close - prior_high
            if action == 1
            else prior_low - close
        )
        return max(0.0, min(1.0, distance / atr))

    def _route_entry(
        self,
        *,
        bt_strategy,
        action: int,
        close: float,
        atr: float,
        breakout_score: float,
        params: Dict[str, Any],
    ) -> tuple[str, float | None]:
        mode = str(params["entry_order_mode"]).strip().lower()
        if mode not in {"adaptive", "market", "limit", "stop"}:
            raise ValueError(
                "entry_order_mode must be adaptive, market, limit, or stop"
            )
        raw_action = float(
            getattr(getattr(bt_strategy, "bridge", None), "raw_action_slot", 0.0)
            or 0.0
        )
        urgency = min(1.0, abs(raw_action))
        if urgency == 0.0:
            urgency = 1.0
        spread_rate = max(
            0.0,
            float(params.get("full_spread_rate") or 0.0),
        )
        spread_bps = spread_rate * 10_000.0
        if mode == "adaptive":
            if (
                urgency >= float(params["market_urgency_threshold"])
                and spread_bps <= float(params["market_max_spread_bps"])
            ):
                mode = "market"
            elif (
                breakout_score * urgency * urgency
                >= float(params["stop_breakout_threshold"])
            ):
                mode = "stop"
            else:
                mode = "limit"
        if mode == "market":
            return mode, None

        if mode == "limit":
            spread_multiple = float(params["limit_offset_spread_multiple"])
            atr_multiple = float(params["limit_offset_atr_multiple"])
            direction = -1.0 if action == 1 else 1.0
        else:
            spread_multiple = float(params["stop_offset_spread_multiple"])
            atr_multiple = float(params["stop_offset_atr_multiple"])
            direction = 1.0 if action == 1 else -1.0
        offset = max(
            close * 1e-8,
            close * spread_rate * spread_multiple,
            atr * atr_multiple,
        )
        return mode, close + direction * offset

    def _effective_sltp_multiples(self, p: Dict[str, Any]) -> tuple[float, float]:
        k_sl = max(0.0, float(p["k_sl"]))
        k_tp = max(0.0, float(p["k_tp"]))
        mode = str(p.get("sltp_risk_mode", "fixed_atr")).strip().lower()
        if mode not in {"rel_volume_aware_atr", "margin_aware_atr"}:
            return k_sl, k_tp

        try:
            rel = max(0.0, float(p.get("rel_volume") or 0.0))
            baseline = max(0.0, float(p.get("baseline_rel_volume", 0.05)))
            max_rel = max(baseline + 1e-12, float(p.get("max_risk_rel_volume", 0.50)))
            sl_alpha = min(max(float(p.get("rel_volume_sl_shrink_alpha", 0.35)), 0.0), 0.95)
            tp_alpha = min(max(float(p.get("rel_volume_tp_shrink_alpha", 0.20)), 0.0), 0.95)
            min_k_sl = max(0.0, float(p.get("min_k_sl", 1.0)))
            min_rr = max(0.0, float(p.get("min_reward_risk_ratio", 1.0)))
        except (TypeError, ValueError):
            return k_sl, max(k_tp, k_sl)

        if rel <= baseline:
            k_sl_eff = k_sl
            k_tp_eff = k_tp
        else:
            exposure_progress = min(1.0, max(0.0, (rel - baseline) / (max_rel - baseline)))
            k_sl_eff = max(min_k_sl, k_sl * (1.0 - sl_alpha * exposure_progress))
            k_tp_eff = k_tp * (1.0 - tp_alpha * exposure_progress)
        k_tp_eff = max(k_tp_eff, k_sl_eff * min_rr)
        return k_sl_eff, k_tp_eff

    def _compute_size(self, bt_strategy, p: Dict[str, Any]) -> float:
        rel = p.get("rel_volume")
        if rel is None:
            return float(p["position_size"])
        try:
            cash = float(bt_strategy.broker.getcash())
        except Exception:
            cash = float(p.get("position_size", 1.0))
        leverage = float(p.get("leverage", 1.0))
        min_vol = float(p.get("min_order_volume", 0.0))
        max_vol = float(p.get("max_order_volume", 1e12))
        mode = str(p.get("size_mode", "fx_units")).lower()
        if mode == "notional":
            try:
                price = float(bt_strategy.data.close[0])
            except Exception:
                price = 0.0
            raw = (cash * float(rel) * leverage) / price if price > 0 else 0.0
        else:
            raw = cash * float(rel) * leverage
        return max(min_vol, min(raw, max_vol))

    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        for k in self.plugin_params:
            if k in config and config[k] is not None:
                merged[k] = config[k]
        return merged

    def _session_state(self, bt_strategy, p: Dict[str, Any]) -> tuple[bool, bool]:
        """Return (in_entry_window, in_close_zone) for the current bar.

        Trading week (using Python weekday convention Mon=0 .. Sun=6):
            entry_window = [entry_dow_start@entry_hour_start ..
                            force_close_dow@force_close_hour)
            close_zone   = complement (forces flatten on every bar there)

        When `session_filter` is False, entries are unrestricted and
        nothing is force-closed (returns (True, False)).
        """
        if not p.get("session_filter"):
            return True, False
        try:
            dt = bt_strategy.data.datetime.datetime(0)
        except Exception:
            return True, False
        # Minute-of-week as a single comparable scalar.
        cur = dt.weekday() * 24 * 60 + dt.hour * 60 + dt.minute
        start = int(p["entry_dow_start"]) * 24 * 60 + int(p["entry_hour_start"]) * 60
        end = int(p["force_close_dow"]) * 24 * 60 + int(p["force_close_hour"]) * 60
        in_entry = (start <= cur < end)
        return in_entry, (not in_entry)

    # Exposed for the GA optimizer to enumerate tunable hyperparameters.
    def hparam_schema(self):
        return [
            ("atr_period", 7, 30, "int"),
            ("k_sl", 1.0, 4.0, "float"),
            ("k_tp", 1.5, 6.0, "float"),
        ]
