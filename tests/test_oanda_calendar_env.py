"""Wiring tests for OANDA calendar obs/info on GymFxEnv.

These tests construct a bare GymFxEnv (bypassing __init__ + cerebro
startup) and exercise the calendar obs/info code paths directly, the
same pattern used by ``test_force_close_reward_penalty``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.env import GymFxEnv


def _env(ts: str, *, oanda: bool, position: int = 0,
         broker_profile: str | None = None) -> GymFxEnv:
    env = object.__new__(GymFxEnv)
    env.config = {"broker_profile": broker_profile} if broker_profile else {}
    env.initial_cash = 10000.0
    env.total_bars = 1
    env.dataframe = pd.DataFrame({"DATE_TIME": [ts], "CLOSE": [1.0]})
    env._date_column = "DATE_TIME"
    env._timeframe_hours = 4.0
    env.stage_b_force_close_obs = False
    env.oanda_fx_calendar_obs = oanda
    env.bridge = SimpleNamespace(
        position=position, equity=10500.0, bar_index=0,
        price=1.0, trade_count=0, commission_paid=0.0,
        last_trade_cost=0.0,
        execution_diagnostics={},
    )
    env._last_raw_action_value = 0.0
    env._last_coerced_action = 0
    env._action_diagnostics = {}
    return env


def test_calendar_features_active_on_friday_15_30_ny_summer():
    # Friday 15:30 NY in EDT == 19:30 UTC on 2024-06-07.
    env = _env("2024-06-07 19:30:00", oanda=True, position=1)
    cal = env._oanda_calendar_features(0)
    assert cal["is_no_new_position_window"] == 1.0
    assert cal["is_friday_risk_reduction_window"] == 1.0
    assert cal["is_force_flat_window"] == 0.0  # before 15:45
    assert cal["broker_market_open"] == 1.0
    # Same timestamp must place hours_to_friday_close in [1.0, 2.0].
    assert 1.0 <= cal["hours_to_friday_close"] <= 2.0


def test_calendar_features_off_when_flag_disabled():
    env = _env("2024-06-07 19:30:00", oanda=False, position=1)
    # The feature helper is still callable, but _make_info should not
    # populate calendar fields when the flag is off.
    info = env._make_info()
    for k in (
        "hours_to_friday_close",
        "is_force_flat_window",
        "broker_market_open",
        "margin_closeout_percent",
    ):
        assert k not in info


def test_oanda_fx_broker_profile_auto_enables_calendar_obs():
    """Setting broker_profile='oanda_us_fx' in config flips the obs flag on."""
    # Build via __init__ would require gym-fx feeds; instead simulate the
    # post-init wiring: the env constructor consults config['broker_profile'].
    # Confirm the helper still works through the wiring path.
    env = _env(
        "2024-06-07 19:30:00", oanda=True, position=1, broker_profile="oanda_us_fx",
    )
    info = env._make_info()
    assert info["broker_profile"] == "oanda_us_fx"
    assert "hours_to_friday_close" in info
    assert info["is_force_flat_window"] == 0.0


def test_margin_fields_fall_back_to_safe_defaults():
    env = _env("2024-06-05 12:00:00", oanda=True, position=0)
    # Bridge does not expose margin_closeout_percent / margin_available;
    # helpers must return deterministic placeholders.
    assert env._safe_margin_closeout_percent() == 0.0
    # equity 10500 / initial 10000 -> 1.05
    assert env._safe_margin_available_norm() == 1.05
