from decimal import Decimal

import pytest

from simulation_engines.contracts import EntryExecutionRequest
from simulation_engines.contracts import ExecutionCostProfile
from simulation_engines.contracts import TargetAction


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


def test_target_action_preserves_legacy_market_entry_defaults():
    action = TargetAction(
        "EUR/USD.SIM",
        1_000,
        Decimal("1000"),
        "legacy-market",
    )
    assert action.entry_execution == EntryExecutionRequest()
    assert action.entry_execution.order_type == "market"


@pytest.mark.parametrize(
    ("request_factory", "message"),
    [
        (EntryExecutionRequest, None),
        (
            lambda: EntryExecutionRequest(order_type="limit"),
            "requires limit_price",
        ),
        (
            lambda: EntryExecutionRequest(order_type="stop"),
            "requires trigger_price",
        ),
        (
            lambda: EntryExecutionRequest(
                order_type="market",
                expires_at_ns=2_000,
            ),
            "market entry cannot carry expires_at_ns",
        ),
    ],
)
def test_entry_execution_request_validation(request_factory, message):
    if message is None:
        assert request_factory().order_type == "market"
    else:
        with pytest.raises(ValueError, match=message):
            request_factory()


def test_target_action_rejects_expired_entry_request():
    with pytest.raises(ValueError, match="expiration must be after"):
        TargetAction(
            "EUR/USD.SIM",
            1_000,
            Decimal("1000"),
            "expired",
            entry_execution=EntryExecutionRequest(
                order_type="limit",
                limit_price=Decimal("1.1"),
                expires_at_ns=1_000,
            ),
        )


def test_entry_execution_resolves_bar_ttl_and_market_fallback():
    request = EntryExecutionRequest.from_bar_ttl(
        order_type="stop",
        action_ts_ns=1_000,
        timeframe_minutes=60,
        valid_for_bars=3,
        trigger_price="1.25",
        unfilled_fallback="market",
    )

    assert request.trigger_price == Decimal("1.25")
    assert request.expires_at_ns == 1_000 + 3 * 60 * 60 * 1_000_000_000
    assert request.unfilled_fallback == "market"


def test_entry_execution_market_rejects_fallback():
    with pytest.raises(ValueError, match="cannot carry an unfilled fallback"):
        EntryExecutionRequest(
            order_type="market",
            unfilled_fallback="market",
        )
