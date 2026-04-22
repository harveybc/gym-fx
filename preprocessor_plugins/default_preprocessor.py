from __future__ import annotations

import numpy as np


class Plugin:
    plugin_params = {
        "window_size": 32,
        "price_column": "CLOSE",
    }

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def make_observation(self, data, step, config):
        window_size = int(config.get("window_size", self.params["window_size"]))
        price_col = config.get("price_column", self.params["price_column"])
        values = data[price_col].astype(float).to_numpy()

        left = max(0, step - window_size)
        window = values[left:step]
        if len(window) < window_size:
            pad = np.full(window_size - len(window), window[0] if len(window) else 0.0)
            window = np.concatenate([pad, window])

        returns = np.diff(window, prepend=window[0])
        return {
            "prices": window.astype(float),
            "returns": returns.astype(float),
        }
