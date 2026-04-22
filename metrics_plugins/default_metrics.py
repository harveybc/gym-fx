from __future__ import annotations

import math


class Plugin:
    plugin_params = {
        "annualization_factor": 252.0,
    }

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def summarize(self, equity_curve, rewards, actions, config):
        if not equity_curve:
            return {
                "steps": 0,
                "final_equity": None,
                "total_return": None,
                "max_drawdown": None,
                "sharpe": None,
            }

        start = float(equity_curve[0])
        end = float(equity_curve[-1])
        total_return = (end / start - 1.0) if start else 0.0

        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            peak = max(peak, v)
            if peak:
                dd = (peak - v) / peak
                max_dd = max(max_dd, dd)

        rets = []
        for i in range(1, len(equity_curve)):
            prev = float(equity_curve[i - 1])
            curr = float(equity_curve[i])
            rets.append((curr - prev) / prev if prev else 0.0)

        sharpe = None
        if rets:
            mean_r = sum(rets) / len(rets)
            var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, (len(rets) - 1))
            std_r = math.sqrt(var_r)
            if std_r > 0:
                ann = float(config.get("annualization_factor", self.params["annualization_factor"]))
                sharpe = (mean_r / std_r) * math.sqrt(ann)

        return {
            "steps": len(actions),
            "final_equity": end,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "sum_rewards": float(sum(rewards)) if rewards else 0.0,
            "actions_taken": {
                "hold": sum(1 for a in actions if a == 0),
                "long": sum(1 for a in actions if a == 1),
                "short": sum(1 for a in actions if a == 2),
            },
        }
