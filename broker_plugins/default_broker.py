"""
default_broker.py

Configures a backtrader BackBroker (simulated sim broker) with commission,
slippage and initial cash. Supports long↔short flips naturally: backtrader
records a separate execution for the close and the open, so both
commissions are charged.

A future oanda_broker.py will implement the same build_bt_broker interface
using bt.stores.OandaStore for live trading.
"""
from __future__ import annotations

from typing import Any, Dict

import backtrader as bt


class Plugin:
    plugin_params = {
        "initial_cash": 10000.0,
        "commission": 0.0,      # fraction of notional (e.g. 0.00002 = 0.2 pips)
        "slippage_perc": 0.0,   # fraction of price applied per fill
        "leverage": 1.0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def build_bt_broker(self, config: Dict[str, Any]) -> bt.brokers.BackBroker:
        cash = float(config.get("initial_cash", self.params["initial_cash"]))
        commission = float(config.get("commission", self.params["commission"]))
        # Accept both 'slippage' (legacy config key) and 'slippage_perc'
        slip = float(
            config.get(
                "slippage_perc",
                config.get("slippage", self.params["slippage_perc"]),
            )
        )
        leverage = float(config.get("leverage", self.params["leverage"]))

        broker = bt.brokers.BackBroker()
        broker.setcash(cash)
        # Commission scaled by notional (price * size) since commtype=PERC
        broker.setcommission(commission=commission, leverage=leverage)
        if slip > 0:
            broker.set_slippage_perc(perc=slip, slip_open=True, slip_limit=True, slip_match=True)
        return broker
