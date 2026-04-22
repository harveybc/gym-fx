"""
default_strategy.py

Backtrader Strategy **overlay**. Not an agent, not a decision maker.
The agent's actions arrive via the env; this plugin exists so future
experiments can attach analyzers, SL/TP managers, trade loggers, or
risk overlays that plug into cerebro alongside BTBridgeStrategy.

For v0, it is a no-op that exposes `decide_action` so the env can also
be driven by a diagnostic CLI (buy_hold / random / flat) without an
agent present.
"""
from __future__ import annotations

import random
from typing import Any, Dict


class Plugin:
    plugin_params = {
        "driver_mode": "buy_hold",   # buy_hold | random | flat | replay
        "replay_actions_file": None,
        "seed": None,
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._replay_actions: list[int] = []
        self._rng = random.Random()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)
        seed = self.params.get("seed")
        if seed is not None:
            self._rng = random.Random(seed)
        replay = self.params.get("replay_actions_file")
        if replay:
            import csv
            with open(replay, "r", encoding="utf-8") as fh:
                self._replay_actions = [int(row.get("action", 0)) for row in csv.DictReader(fh)]

    def decide_action(self, obs: Dict[str, Any], info: Dict[str, Any], step: int) -> int:
        mode = self.params.get("driver_mode", "buy_hold")
        if mode == "random":
            return self._rng.choice([0, 1, 2])
        if mode == "flat":
            return 0
        if mode == "replay":
            if step < len(self._replay_actions):
                return self._replay_actions[step]
            return 0
        return 1 if step == 0 else 0
