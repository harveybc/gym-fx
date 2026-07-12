from decimal import Decimal

import pytest

from simulation_engines.contracts import ExecutionCostProfile


def test_execution_cost_profile_derives_total_adverse_quote_rate():
    profile = ExecutionCostProfile.from_dict(
        {
            "schema_version": "execution_cost_profile.v1",
            "profile_id": "test",
            "commission_rate_per_side": 0.0002,
            "full_spread_rate": 0.0004,
            "slippage_bps_per_side": 2.0,
            "latency_ms": 0,
            "financing_enabled": True,
            "intrabar_collision_policy": "worst_case",
            "limit_fill_policy": "conservative",
            "margin_model": "standard",
            "enforce_margin_preflight": True,
            "random_seed": 42,
        }
    )
    assert profile.slippage_rate_per_side == Decimal("0.0002")
    assert profile.quote_adverse_rate_per_side == Decimal("0.0004")


def test_execution_cost_profile_rejects_negative_costs():
    with pytest.raises(ValueError, match="cannot be negative"):
        ExecutionCostProfile.from_dict(
            {
                "schema_version": "execution_cost_profile.v1",
                "profile_id": "bad",
                "commission_rate_per_side": -0.1,
                "full_spread_rate": 0,
                "slippage_bps_per_side": 0,
                "latency_ms": 0,
                "financing_enabled": False,
                "intrabar_collision_policy": "ohlc",
                "limit_fill_policy": "touch",
                "margin_model": "standard",
                "enforce_margin_preflight": True,
                "random_seed": 1,
            }
        )
