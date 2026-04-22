"""
dd_penalized_reward.py

Reward = pnl_norm - penalty_lambda * drawdown_norm.
Tracks peak equity to compute current drawdown.
"""
from __future__ import annotations

from typing import Any, Dict


class Plugin:
    plugin_params = {
        "penalty_lambda": 1.0,
        "initial_cash": 10000.0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._peak: float = 0.0
        self._last_step: int = -1
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)
        self._peak = 0.0
        self._last_step = -1

    def compute_reward(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        step: int,
        config: Dict[str, Any],
    ) -> float:
        if step <= self._last_step:
            self._peak = 0.0
        self._last_step = int(step)
        self._peak = max(self._peak, float(new_equity), float(prev_equity))

        initial_cash = float(config.get("initial_cash", self.params["initial_cash"])) or 1.0
        pnl_norm = (float(new_equity) - float(prev_equity)) / initial_cash
        dd_norm = (self._peak - float(new_equity)) / initial_cash if self._peak > 0 else 0.0
        lam = float(config.get("penalty_lambda", self.params["penalty_lambda"]))
        return pnl_norm - lam * dd_norm
