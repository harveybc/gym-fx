from dataclasses import replace
from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from simulation_engines.bakeoff import BAKEOFF_START_NS
from simulation_engines.bakeoff import NANOSECONDS_PER_MINUTE
from simulation_engines.bakeoff import build_multi_asset_fixture
from simulation_engines.contracts import EntryExecutionRequest
from simulation_engines.contracts import MarketFrame
from simulation_engines.contracts import TargetAction
from simulation_engines.contracts import load_execution_cost_profile
from simulation_engines.nautilus_adapter import NautilusReplayAdapter


PROFILE = "examples/config/execution_cost_profiles/project3_pessimistic_v1.json"
INSTRUMENT_ID = "EUR/USD.SIM"


def _ts(minutes: int) -> int:
    return BAKEOFF_START_NS + minutes * NANOSECONDS_PER_MINUTE


def _frame(minute: int, price: str) -> MarketFrame:
    close = Decimal(price)
    return MarketFrame(
        instrument_id=INSTRUMENT_ID,
        timeframe_minutes=1,
        ts_event_ns=_ts(minute),
        open=close,
        high=close + Decimal("0.00020"),
        low=close - Decimal("0.00020"),
        close=close,
        volume=Decimal("1000000"),
    )


def _run(frames, action):
    profile = replace(
        load_execution_cost_profile(PROFILE),
        financing_enabled=False,
    )
    instruments, _, _ = build_multi_asset_fixture()
    actions = action if isinstance(action, list) else [action]
    return NautilusReplayAdapter(profile).run(
        instrument_specs=[instruments[0]],
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
    )


def _events(result, event_type):
    return [
        event for event in result["events"] if event["event_type"] == event_type
    ]


@pytest.mark.parametrize(
    ("target_units", "expected_side"),
    [
        (Decimal("1000"), {"BUY", "1"}),
        (Decimal("-1000"), {"SELL", "2"}),
    ],
)
def test_native_market_entries_support_long_and_short(target_units, expected_side):
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        target_units,
        f"market-{target_units}",
    )
    result = _run(
        [_frame(1, "1.10000"), _frame(2, "1.10050")],
        action,
    )

    fills = _events(result, "order_filled")
    assert len(fills) == 1
    assert fills[0]["side"] in expected_side
    assert fills[0]["requested_order_type"] == "market"
    assert _events(result, "entry_submitted")[0]["order_type"] == "market"


@pytest.mark.parametrize(
    ("target_units", "limit_price", "next_price", "expected_side"),
    [
        (Decimal("1000"), "1.09900", "1.09800", {"BUY", "1"}),
        (Decimal("-1000"), "1.10100", "1.10200", {"SELL", "2"}),
    ],
)
def test_native_limit_entries_support_long_and_short(
    target_units,
    limit_price,
    next_price,
    expected_side,
):
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        target_units,
        f"limit-{target_units}",
        entry_execution=EntryExecutionRequest(
            order_type="limit",
            limit_price=Decimal(limit_price),
            expires_at_ns=_ts(3),
        ),
    )
    result = _run(
        [_frame(1, "1.10000"), _frame(2, next_price), _frame(3, next_price)],
        action,
    )

    fills = _events(result, "order_filled")
    assert len(fills) == 1
    assert fills[0]["side"] in expected_side
    assert fills[0]["requested_order_type"] == "limit"
    assert _events(result, "entry_submitted")[0]["limit_price"] == limit_price


@pytest.mark.parametrize(
    ("target_units", "trigger_price", "next_price", "expected_side"),
    [
        (Decimal("1000"), "1.10100", "1.10200", {"BUY", "1"}),
        (Decimal("-1000"), "1.09900", "1.09800", {"SELL", "2"}),
    ],
)
def test_native_stop_entries_support_long_and_short(
    target_units,
    trigger_price,
    next_price,
    expected_side,
):
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        target_units,
        f"stop-{target_units}",
        entry_execution=EntryExecutionRequest(
            order_type="stop",
            trigger_price=Decimal(trigger_price),
            expires_at_ns=_ts(3),
        ),
    )
    result = _run(
        [_frame(1, "1.10000"), _frame(2, next_price), _frame(3, next_price)],
        action,
    )

    fills = _events(result, "order_filled")
    assert len(fills) == 1
    assert fills[0]["side"] in expected_side
    assert fills[0]["requested_order_type"] == "stop"
    assert _events(result, "entry_submitted")[0]["trigger_price"] == trigger_price


def test_non_market_entry_replay_is_deterministic():
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        Decimal("1000"),
        "deterministic-stop",
        entry_execution=EntryExecutionRequest(
            order_type="stop",
            trigger_price=Decimal("1.10100"),
            expires_at_ns=_ts(3),
        ),
    )
    frames = [
        _frame(1, "1.10000"),
        _frame(2, "1.10200"),
        _frame(3, "1.10200"),
    ]

    first = _run(frames, action)
    second = _run(frames, action)

    assert first["event_hash"] == second["event_hash"]
    assert first["result_hash"] == second["result_hash"]


def test_unfilled_limit_entry_expires_without_market_fallback():
    expires_at_ns = _ts(2) - NANOSECONDS_PER_MINUTE // 2
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        Decimal("1000"),
        "unfilled-limit",
        entry_execution=EntryExecutionRequest(
            order_type="limit",
            limit_price=Decimal("1.09000"),
            expires_at_ns=expires_at_ns,
        ),
    )
    result = _run(
        [
            _frame(1, "1.10000"),
            _frame(2, "1.10050"),
            _frame(3, "1.10100"),
        ],
        action,
    )

    assert _events(result, "order_filled") == []
    expirations = _events(result, "order_expired")
    assert len(expirations) == 1
    assert expirations[0]["action_id"] == "unfilled-limit"
    assert result["native"]["total_orders"] == 1
    assert result["summary"]["positions.open"] == "0"


def test_unfilled_limit_entry_can_fall_back_to_market():
    expires_at_ns = _ts(2) - NANOSECONDS_PER_MINUTE // 2
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        Decimal("1000"),
        "unfilled-limit-market-fallback",
        entry_execution=EntryExecutionRequest(
            order_type="limit",
            limit_price=Decimal("1.09000"),
            expires_at_ns=expires_at_ns,
            unfilled_fallback="market",
        ),
    )
    result = _run(
        [
            _frame(1, "1.10000"),
            _frame(2, "1.10050"),
            _frame(3, "1.10100"),
        ],
        action,
    )

    assert len(_events(result, "order_expired")) == 1
    assert len(_events(result, "entry_market_fallback_submitted")) == 1
    fills = _events(result, "order_filled")
    assert len(fills) == 1
    assert fills[0]["requested_order_type"] == "market_fallback"
    assert result["summary"]["positions.open"] == "1"


def test_new_flat_target_cancels_pending_entry_before_it_can_fill():
    actions = [
        TargetAction(
            INSTRUMENT_ID,
            _ts(1),
            Decimal("1000"),
            "pending-limit",
            entry_execution=EntryExecutionRequest(
                order_type="limit",
                limit_price=Decimal("1.09000"),
                expires_at_ns=_ts(4),
            ),
        ),
        TargetAction(
            INSTRUMENT_ID,
            _ts(2),
            Decimal("0"),
            "cancel-with-flat-target",
        ),
    ]
    result = _run(
        [
            _frame(1, "1.10000"),
            _frame(2, "1.10000"),
            _frame(3, "1.08900"),
            _frame(4, "1.08900"),
        ],
        actions,
    )

    assert _events(result, "order_filled") == []
    assert len(_events(result, "entry_cancel_requested")) == 1
    cancellations = _events(result, "order_canceled")
    assert len(cancellations) == 1
    assert cancellations[0]["action_id"] == "pending-limit"
    assert len(_events(result, "entry_submitted")) == 1
    assert result["summary"]["positions.open"] == "0"


def test_stop_entry_attaches_native_sl_tp_and_cancels_surviving_sibling():
    action = TargetAction(
        INSTRUMENT_ID,
        _ts(1),
        Decimal("1000"),
        "protected-stop",
        stop_loss_price=Decimal("1.09700"),
        take_profit_price=Decimal("1.10500"),
        entry_execution=EntryExecutionRequest(
            order_type="stop",
            trigger_price=Decimal("1.10100"),
            expires_at_ns=_ts(4),
        ),
    )
    result = _run(
        [
            _frame(1, "1.10000"),
            _frame(2, "1.10200"),
            _frame(3, "1.09600"),
            _frame(4, "1.09600"),
        ],
        action,
    )

    fills = _events(result, "order_filled")
    assert [fill["order_role"] for fill in fills] == ["entry", "stop_loss"]
    assert fills[0]["side"] in {"BUY", "1"}
    assert fills[1]["side"] in {"SELL", "2"}
    assert len(_events(result, "protection_submitted")) == 1
    canceled = _events(result, "order_canceled")
    assert len(canceled) == 1
    assert canceled[0]["order_role"] == "take_profit"
    assert result["native"]["total_orders"] == 3
    assert result["summary"]["positions.open"] == "0"
