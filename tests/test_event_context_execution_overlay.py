from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.env import GymFxEnv


def _env(*, no_trade: float, position: int, force_flat: bool = False) -> GymFxEnv:
    env = object.__new__(GymFxEnv)
    env.total_bars = 1
    env.dataframe = pd.DataFrame(
        {
            "DATE_TIME": ["2024-01-01 00:00:00"],
            "CLOSE": [100.0],
            "event_no_trade_window_active": [no_trade],
            "event_spread_stress_multiplier": [2.0],
            "event_slippage_stress_multiplier": [3.0],
        }
    )
    env.event_context_execution_overlay = True
    env.event_context_no_trade_column = "event_no_trade_window_active"
    env.event_context_no_trade_threshold = 0.5
    env.event_context_block_new_entries = True
    env.event_context_force_flat = force_flat
    env.event_context_spread_stress_column = "event_spread_stress_multiplier"
    env.event_context_slippage_stress_column = "event_slippage_stress_multiplier"
    env.bridge = SimpleNamespace(
        position=position,
        bar_index=0,
        execution_diagnostics={},
    )
    return env


def test_event_no_trade_overlay_blocks_new_entries_when_flat():
    env = _env(no_trade=1.0, position=0, force_flat=False)

    action, info = env._apply_event_context_overlay(1)

    assert action == 0
    assert info["event_context_action_before_overlay"] == 1
    assert info["event_context_action_after_overlay"] == 0
    assert info["event_context_blocked_entry"] is True
    assert info["event_context_action_overridden"] is True
    assert env.bridge.execution_diagnostics["event_context_blocked_entries"] == 1
    assert env.bridge.execution_diagnostics["event_context_action_overrides"] == 1


def test_event_no_trade_overlay_forces_flat_when_position_open():
    env = _env(no_trade=1.0, position=1, force_flat=True)

    action, info = env._apply_event_context_overlay(1)

    assert action == 3
    assert info["event_context_forced_flat"] is True
    assert info["event_context_position_before_overlay"] == 1
    assert env.bridge.execution_diagnostics["event_context_forced_flat_actions"] == 1


def test_event_no_trade_overlay_is_neutral_when_event_inactive():
    env = _env(no_trade=0.0, position=0, force_flat=True)

    action, info = env._apply_event_context_overlay(1)

    assert action == 1
    assert info["event_context_no_trade_active"] == 0.0
    assert info["event_context_action_overridden"] is False
    assert env.bridge.execution_diagnostics == {}
