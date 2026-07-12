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
