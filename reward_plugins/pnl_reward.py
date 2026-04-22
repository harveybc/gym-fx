"""
pnl_reward.py

Reward = (new_equity - prev_equity) / initial_cash * reward_scale.
Normalizing by initial_cash keeps reward magnitudes agent-friendly.
"""
from __future__ import annotations

from typing import Any, Dict


class Plugin:
    plugin_params = {
        "reward_scale": 1.0,
        "initial_cash": 10000.0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def compute_reward(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        step: int,
        config: Dict[str, Any],
    ) -> float:
        initial_cash = float(config.get("initial_cash", self.params["initial_cash"])) or 1.0
        scale = float(config.get("reward_scale", self.params["reward_scale"]))
        return (float(new_equity) - float(prev_equity)) / initial_cash * scale
