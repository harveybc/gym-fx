from __future__ import annotations

import csv
import random


class Plugin:
    plugin_params = {
        "driver_mode": "buy_hold",  # random|buy_hold|flat|replay
        "replay_actions_file": None,
    }

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        self._replay_actions = []
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        replay_file = self.params.get("replay_actions_file")
        if replay_file:
            self._replay_actions = self._load_replay_actions(replay_file)

    def _load_replay_actions(self, path):
        actions = []
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    actions.append(int(row.get("action", 0)))
                except Exception:
                    actions.append(0)
        return actions

    def decide_action(self, obs, info, step):
        mode = self.params.get("driver_mode", "buy_hold")
        if mode == "random":
            return random.choice([0, 1, 2])
        if mode == "flat":
            return 0
        if mode == "replay":
            if step < len(self._replay_actions):
                return self._replay_actions[step]
            return 0
        # buy_hold default
        return 1 if step == 0 else 0
