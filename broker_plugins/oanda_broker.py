"""
oanda_broker.py

Live OANDA broker plugin. Gated behind the GYMFX_ENABLE_LIVE=1 env var
to prevent accidental real-money trading during testing/simulation.

Uses backtrader's bt.stores.OandaStore (v20 REST API). Requires
`requests` and an OANDA practice/live token. Configuration keys
(with env-var fallbacks):

  - oanda_token              (env: OANDA_TOKEN)
  - oanda_account_id         (env: OANDA_ACCOUNT_ID)
  - oanda_practice           bool, default True
  - oanda_instrument         e.g. "EUR_USD"

NOTE: This plugin is a stub. Smoke tests use default_broker. Turning it
on in a training run means real orders will be placed.
"""
from __future__ import annotations

import os
from typing import Any, Dict


class Plugin:
    plugin_params: Dict[str, Any] = {
        "oanda_token": None,
        "oanda_account_id": None,
        "oanda_practice": True,
        "oanda_instrument": "EUR_USD",
        "initial_cash": 10000.0,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def build_bt_broker(self, config: Dict[str, Any]):
        if os.environ.get("GYMFX_ENABLE_LIVE") != "1":
            raise RuntimeError(
                "oanda_broker is disabled. Set GYMFX_ENABLE_LIVE=1 to enable live trading."
            )
        try:
            import backtrader as bt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backtrader is required for oanda_broker") from exc

        token = config.get("oanda_token") or os.environ.get("OANDA_TOKEN")
        account = config.get("oanda_account_id") or os.environ.get("OANDA_ACCOUNT_ID")
        practice = bool(config.get("oanda_practice", self.params["oanda_practice"]))
        if not token or not account:
            raise ValueError("oanda_token and oanda_account_id are required for live trading")

        store = bt.stores.OandaStore(
            token=token,
            account=account,
            practice=practice,
        )
        return store.getbroker()
