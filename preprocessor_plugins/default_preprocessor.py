"""
default_preprocessor.py

Builds observations for the agent. In addition to price features
(a sliding window of prices and returns), it includes the agent's own
state via the bridge_state dict provided by GymFxEnv:
  - position (-1, 0, +1)
  - equity normalized by initial_cash
  - unrealized PnL normalized by initial_cash
  - steps remaining normalized to [0, 1]
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class Plugin:
    plugin_params = {
        "window_size": 32,
        "price_column": "CLOSE",
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def make_observation(
        self,
        *,
        data: pd.DataFrame,
        step: int,
        bridge_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        window_size = int(config.get("window_size", self.params["window_size"]))
        price_col = config.get("price_column", self.params["price_column"])
        values = data[price_col].astype(float).to_numpy()

        left = max(0, step - window_size)
        window = values[left:step] if step > 0 else values[:0]
        if len(window) < window_size:
            fill = float(window[0]) if len(window) else float(values[0])
            pad = np.full(window_size - len(window), fill, dtype=float)
            window = np.concatenate([pad, window])
        returns = np.diff(window, prepend=window[0])

        initial_cash = float(bridge_state.get("initial_cash", 1.0) or 1.0)
        equity = float(bridge_state.get("equity", initial_cash))
        price = float(bridge_state.get("price", 0.0) or 0.0)
        position = int(bridge_state.get("position", 0))
        bar_index = int(bridge_state.get("bar_index", 0))
        total_bars = int(bridge_state.get("total_bars", 1) or 1)

        # Rough unrealized PnL proxy: position * (price - last_window_price) * position_size
        pos_size = float(config.get("position_size", 1.0))
        reference_price = float(window[-1]) if len(window) else price
        unrealized_pnl = position * (price - reference_price) * pos_size

        equity_norm = (equity - initial_cash) / initial_cash if initial_cash else 0.0
        pnl_norm = unrealized_pnl / initial_cash if initial_cash else 0.0
        remaining = max(0, total_bars - bar_index) / max(1, total_bars)

        return {
            "prices": window.astype(np.float32),
            "returns": returns.astype(np.float32),
            "position": np.array([float(position)], dtype=np.float32),
            "equity_norm": np.array([float(equity_norm)], dtype=np.float32),
            "unrealized_pnl_norm": np.array([float(pnl_norm)], dtype=np.float32),
            "steps_remaining_norm": np.array([float(remaining)], dtype=np.float32),
        }
