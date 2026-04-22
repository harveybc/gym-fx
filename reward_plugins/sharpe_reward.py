"""
sharpe_reward.py

Rolling risk-adjusted reward. Maintains a window of step returns
(new_equity - prev_equity) / initial_cash and returns the annualized
Sharpe ratio over that window each step. Warmup steps return 0.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict


class Plugin:
    plugin_params = {
        "window": 64,
        "annualization_factor": 252.0,
        "initial_cash": 10000.0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._buffer: Deque[float] = deque(maxlen=self.params["window"])
        self._last_step: int = -1
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)
        self._buffer = deque(maxlen=int(self.params["window"]))
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
            # reset detected
            self._buffer.clear()
        self._last_step = int(step)

        initial_cash = float(config.get("initial_cash", self.params["initial_cash"])) or 1.0
        r = (float(new_equity) - float(prev_equity)) / initial_cash
        self._buffer.append(r)
        if len(self._buffer) < 2:
            return 0.0
        mean = sum(self._buffer) / len(self._buffer)
        var = sum((x - mean) ** 2 for x in self._buffer) / (len(self._buffer) - 1)
        std = math.sqrt(var)
        if std <= 0:
            return 0.0
        ann = float(config.get("annualization_factor", self.params["annualization_factor"]))
        return (mean / std) * math.sqrt(ann)
