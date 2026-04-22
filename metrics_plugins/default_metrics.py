"""
default_metrics.py

Summarizes performance from backtrader analyzer outputs plus final equity.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class Plugin:
    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def summarize(
        self,
        *,
        initial_cash: float,
        final_equity: float,
        analyzers: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        trades = analyzers.get("trades") or {}
        sharpe = analyzers.get("sharpe") or {}
        drawdown = analyzers.get("drawdown") or {}
        sqn = analyzers.get("sqn") or {}

        total_return = (float(final_equity) / float(initial_cash) - 1.0) if initial_cash else 0.0

        def _get(d: Any, *path: str, default: Any = None) -> Any:
            cur: Any = d
            for k in path:
                if cur is None:
                    return default
                if hasattr(cur, "get"):
                    cur = cur.get(k, None)
                else:
                    return default
            return cur if cur is not None else default

        return {
            "initial_cash": float(initial_cash),
            "final_equity": float(final_equity),
            "total_return": float(total_return),
            "max_drawdown_pct": _get(drawdown, "max", "drawdown"),
            "max_drawdown_money": _get(drawdown, "max", "moneydown"),
            "sharpe_ratio": _get(sharpe, "sharperatio"),
            "sqn": _get(sqn, "sqn"),
            "trades_total": _get(trades, "total", "total", default=0),
            "trades_won": _get(trades, "won", "total", default=0),
            "trades_lost": _get(trades, "lost", "total", default=0),
            "avg_trade_pnl": _get(trades, "pnl", "net", "average"),
        }
