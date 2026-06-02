from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.env import GymFxEnv


def _env_at(ts: str, *, position: int = 1) -> GymFxEnv:
    env = object.__new__(GymFxEnv)
    env.stage_b_force_close_obs = True
    env.stage_b_force_close_reward_penalty = True
    env.force_close_exposure_penalty_coef = 0.0002
    env.force_close_exposure_penalty_window_hours = 4.0
    env.force_close_dow = 4
    env.force_close_hour = 20
    env.force_close_window_hours = 4
    env.monday_entry_window_hours = 4
    env._date_column = "DATE_TIME"
    env._timeframe_hours = 4.0
    env.dataframe = pd.DataFrame({"DATE_TIME": [ts]})
    env.bridge = SimpleNamespace(position=position)
    return env


def test_force_close_penalty_applies_before_friday_close():
    env = _env_at("2024-01-05 16:00:00", position=1)

    assert env._force_close_features(0)["hours_to_force_close"] == 4.0
    assert env._force_close_reward_penalty(0) == 0.0002


def test_force_close_penalty_applies_inside_force_close_zone():
    env = _env_at("2024-01-05 20:00:00", position=-1)

    assert env._force_close_features(0)["is_force_close_zone"] == 1.0
    assert env._force_close_reward_penalty(0) == 0.0002


def test_force_close_penalty_skips_flat_or_outside_window():
    assert _env_at("2024-01-05 16:00:00", position=0)._force_close_reward_penalty(0) == 0.0
    assert _env_at("2024-01-05 12:00:00", position=1)._force_close_reward_penalty(0) == 0.0


def test_force_close_penalty_is_config_gated():
    env = _env_at("2024-01-05 16:00:00", position=1)
    env.stage_b_force_close_reward_penalty = False
    assert env._force_close_reward_penalty(0) == 0.0

    env.stage_b_force_close_reward_penalty = True
    env.stage_b_force_close_obs = False
    assert env._force_close_reward_penalty(0) == 0.0
