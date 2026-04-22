from __future__ import annotations


class Plugin:
    plugin_params = {
        "commission": 0.0,
        "slippage": 0.0,
    }

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def trade_cost(self, old_position, new_position, price, config):
        if old_position == new_position:
            return 0.0
        commission = float(config.get("commission", self.params["commission"]))
        slippage = float(config.get("slippage", self.params["slippage"]))
        return (commission + slippage) * float(price)
