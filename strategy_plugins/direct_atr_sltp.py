"""
direct_atr_sltp.py

Strategy plugin that places bracket orders with SL/TP sized by rolling ATR.
    SL distance = k_sl * ATR(atr_period)
    TP distance = k_tp * ATR(atr_period)

The plugin maintains its own rolling True-Range buffer from the backtrader
data lines (no backtrader indicator needed — avoids minperiod coupling with
BTBridgeStrategy). Until the ATR buffer is warmed, orders are placed without
brackets (falls back to plain buy/sell) so the env doesn't stall.

Action semantics: {0=hold, 1=long, 2=short} — same as direct_fixed_sltp.

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
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict


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
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)
        self._tr_buffer: Deque[float] = deque()
        self._prev_close: float | None = None

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    def decide_action(self, obs, info, step: int) -> int:
        return 0

    def on_reset(self, bt_strategy, config: Dict[str, Any]) -> None:
        self._tr_buffer = deque(maxlen=int(self._resolve(config)["atr_period"]))
        self._prev_close = None

    # ------------------------------------------------------------------
    # BTBridgeStrategy contract
    # ------------------------------------------------------------------
    def apply_action(self, bt_strategy, action: int, config: Dict[str, Any]) -> None:
        p = self._resolve(config)
        period = int(p["atr_period"])
        k_sl = float(p["k_sl"])
        k_tp = float(p["k_tp"])
        size = self._compute_size(bt_strategy, p)

        high = float(bt_strategy.data.high[0])
        low = float(bt_strategy.data.low[0])
        close = float(bt_strategy.data.close[0])

        # Update ATR buffer with True Range
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        if self._tr_buffer.maxlen != period:
            self._tr_buffer = deque(self._tr_buffer, maxlen=period)
        self._tr_buffer.append(tr)

        if action == 0:
            return

        pos_size = bt_strategy.position.size
        ready = len(self._tr_buffer) >= period
        atr = sum(self._tr_buffer) / len(self._tr_buffer) if self._tr_buffer else 0.0

        # Require a warmed ATR and a positive size, otherwise skip the trade
        # entirely rather than emit a naked (SL/TP-less) order. This guarantees
        # every filled order has both brackets attached.
        if not ready or atr <= 0.0 or size <= 0.0 or close <= 0.0:
            return

        # Clamp SL/TP distances to sane fractions of price to prevent degenerate
        # brackets from pathological ATR spikes (flash crashes, thin bars).
        sl_dist = k_sl * atr
        tp_dist = k_tp * atr
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

        if action == 1:  # long
            if pos_size < 0:
                bt_strategy.close()
            if pos_size <= 0:
                stop = close - sl_dist
                limit = close + tp_dist
                bt_strategy.buy_bracket(size=size, stopprice=stop, limitprice=limit)
        elif action == 2:  # short
            if pos_size > 0:
                bt_strategy.close()
            if pos_size >= 0:
                stop = close + sl_dist
                limit = close - tp_dist
                bt_strategy.sell_bracket(size=size, stopprice=stop, limitprice=limit)

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

    # Exposed for the GA optimizer to enumerate tunable hyperparameters.
    def hparam_schema(self):
        return [
            ("atr_period", 7, 30, "int"),
            ("k_sl", 1.0, 4.0, "float"),
            ("k_tp", 1.5, 6.0, "float"),
        ]
