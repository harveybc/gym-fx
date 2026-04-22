"""
direct_fixed_sltp.py

Strategy plugin that places bracket orders with fixed-pip SL/TP around each
agent-directed entry. Expected action semantics: {0=hold, 1=long, 2=short}.

Contract hook (`apply_action`): called by BTBridgeStrategy with
    apply_action(bt_strategy, action, config)
Replaces the default buy/sell flow with buy_bracket / sell_bracket so the
broker auto-exits at SL or TP regardless of subsequent agent actions.

Config keys (read from plugin_params or env config, env wins):
    sl_pips: float   — stop-loss distance in pips (default 20)
    tp_pips: float   — take-profit distance in pips (default 40)
    pip_size: float  — pip unit in price terms (default 0.0001 for 4-digit FX)
    position_size: float
"""
from __future__ import annotations

from typing import Any, Dict


class Plugin:
    plugin_params = {
        "sl_pips": 20.0,
        "tp_pips": 40.0,
        "pip_size": 0.0001,
        "position_size": 1.0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    # Kept for backward compatibility with the diagnostic-driver contract.
    def decide_action(self, obs, info, step: int) -> int:
        return 0  # hold — this plugin is a bracket manager, not a driver

    def on_reset(self, bt_strategy, config: Dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # BTBridgeStrategy contract
    # ------------------------------------------------------------------
    def apply_action(self, bt_strategy, action: int, config: Dict[str, Any]) -> None:
        p = self._resolve(config)
        size = float(p["position_size"])
        pip = float(p["pip_size"])
        sl_pips = float(p["sl_pips"])
        tp_pips = float(p["tp_pips"])

        price = float(bt_strategy.data.close[0])
        pos_size = bt_strategy.position.size

        if action == 0:
            return  # hold — existing bracket (if any) manages the position

        if action == 1:  # long
            if pos_size < 0:
                bt_strategy.close()
            if pos_size <= 0:
                stop = price - sl_pips * pip
                limit = price + tp_pips * pip
                bt_strategy.buy_bracket(size=size, stopprice=stop, limitprice=limit)
        elif action == 2:  # short
            if pos_size > 0:
                bt_strategy.close()
            if pos_size >= 0:
                stop = price + sl_pips * pip
                limit = price - tp_pips * pip
                bt_strategy.sell_bracket(size=size, stopprice=stop, limitprice=limit)

    def _resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.params)
        for k in self.plugin_params:
            if k in config and config[k] is not None:
                merged[k] = config[k]
        return merged
