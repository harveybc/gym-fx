from __future__ import annotations

import pytest

from metrics_plugins.trading_metrics import Plugin


def test_trading_metrics_adds_unit_safe_rap() -> None:
    plugin = Plugin()
    result = plugin.summarize(
        initial_cash=1000.0,
        final_equity=1100.0,
        analyzers={"drawdown": {"max": {"drawdown": 20.0}}},
        config={"risk_lambda": 0.5, "evaluation_years": 1},
    )
    assert result["total_return"] == pytest.approx(0.10)
    assert result["max_drawdown_fraction"] == pytest.approx(0.20)
    assert result["risk_adjusted_total_return"] == pytest.approx(0.0)
    assert result["annual_return"] == pytest.approx(0.10)
    assert result["annual_rap"] == pytest.approx(0.0)


def test_trading_metrics_does_not_invent_annual_period() -> None:
    plugin = Plugin()
    result = plugin.summarize(
        initial_cash=1000.0,
        final_equity=1100.0,
        analyzers={},
        config={},
    )
    assert "annual_return" not in result
    assert "annual_rap" not in result
    assert result["win_rate_pct"] is None
    assert result["net_profit"] == 100.0


def test_win_rate_counts_only_closed_trades():
    result = Plugin().summarize(initial_cash=1000, final_equity=1100,
        analyzers={"trades": {"total": {"total": 5, "closed": 4},
                              "won": {"total": 3}, "lost": {"total": 1}}}, config={})
    assert result["win_rate_pct"] == 75.0
    assert result["trades_closed"] == 4
