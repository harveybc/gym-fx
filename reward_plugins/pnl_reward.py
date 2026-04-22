from __future__ import annotations


class Plugin:
    plugin_params = {
        "reward_scale": 1.0,
    }

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def compute_reward(self, prev_equity, new_equity, step, config):
        scale = float(config.get("reward_scale", self.params["reward_scale"]))
        return (float(new_equity) - float(prev_equity)) * scale
