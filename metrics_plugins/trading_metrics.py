"""Trading metric plugin for the gym-fx simulation boundary.

This plugin extends the stable raw simulator summary with risk-adjusted values.
It is intentionally independent of DOIN: local agent-multi runs and DOIN
wrapped runs receive identical metric semantics from the same gym-fx plugin.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from .default_metrics import Plugin as DefaultMetrics


class Plugin(DefaultMetrics):
    """Add explicit, unit-safe RAP fields to the base simulator summary."""

    plugin_params: Dict[str, Any] = {
        "risk_lambda": 1.0,
        "metric_schema": "trading.metrics.v1",
    }

    def summarize(
        self,
        *,
        initial_cash: float,
        final_equity: float,
        analyzers: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = super().summarize(
            initial_cash=initial_cash,
            final_equity=final_equity,
            analyzers=analyzers,
            config=config,
        )
        drawdown_pct = _finite_or_zero(summary.get("max_drawdown_pct"))
        total_return = _finite_or_zero(summary.get("total_return"))
        risk_lambda = float(
            config.get(
                "risk_lambda",
                config.get("risk_penalty_lambda", self.params["risk_lambda"]),
            )
        )
        drawdown_fraction = max(0.0, drawdown_pct / 100.0)
        rap = total_return - risk_lambda * drawdown_fraction

        summary.update({
            "metric_schema": str(config.get("metric_schema", self.params["metric_schema"])),
            "max_drawdown_fraction": drawdown_fraction,
            "risk_penalty_lambda": risk_lambda,
            "risk_adjusted_total_return": rap,
            "rap": rap,
        })

        # Annualization is only emitted when the caller supplies the elapsed
        # evaluation period. Never infer a year from an arbitrary row count.
        years = config.get("evaluation_years")
        if years is not None and float(years) > 0:
            summary["annual_return"] = (1.0 + total_return) ** (1.0 / float(years)) - 1.0
            summary["annual_rap"] = rap / float(years)
        return summary


def _finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0
