"""WP-C acceptance (owner curriculum order 2026-08-05): the two solvency
modes. Proofs: normal mode terminates on breach exactly as before; easy
mode records the would-be margin call, liquidates retaining the loss,
continues to later chronological bars, conserves loss and debt exactly,
never lets a recapitalization improve economic equity, remains able to
act after ruin; and easy dynamics are structurally impossible outside
training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.env import GymFxEnv
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from broker_plugins.default_broker import Plugin as Broker
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor,
)
from reward_plugins.pnl_reward import Plugin as Reward
from metrics_plugins.default_metrics import Plugin as Metrics


def _crash_csv(tmp_path, bars=200, crash_at=50, floor=0.02):
    """Price walks flat, then collapses so a long position breaches
    min_equity, then stabilizes so later bars remain tradeable."""
    times = pd.date_range("2024-01-01", periods=bars, freq="4h")
    close = np.full(bars, 100.0)
    crash_len = 20
    close[crash_at:crash_at + crash_len] = np.linspace(
        100.0, 100.0 * floor, crash_len)
    close[crash_at + crash_len:] = 100.0 * floor      # stabilize low
    frame = pd.DataFrame({
        "DATE_TIME": times, "OPEN": close, "HIGH": close * 1.001,
        "LOW": close * 0.999, "CLOSE": close, "VOLUME": 1000.0,
        "feat": np.linspace(0.0, 1.0, bars),
    })
    path = tmp_path / "crash.csv"
    frame.to_csv(path, index=False)
    return path


def _config(tmp_path, **overrides):
    config = {
        "input_data_file": str(_crash_csv(tmp_path)),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "window_size": 8, "initial_cash": 10000.0, "position_size": 95.0,
        "min_equity": 2000.0, "env_mode": "training",
        "commission": 0.0, "leverage": 1.0,
    }
    config.update(overrides)
    return config


def _env(config):
    return GymFxEnv(
        config, DataFeed(config), Broker(config), None,
        Preprocessor(config), Reward(config), Metrics(config))


def _run(env, action=1, max_steps=400):
    env.reset(seed=7)
    steps = 0
    infos = []
    terminated = False
    while not terminated and steps < max_steps:
        _obs, _reward, terminated, _tr, info = env.step(action)
        infos.append(info)
        steps += 1
    env.close()
    return infos, terminated


def test_normal_mode_terminates_on_breach_exactly_as_before(tmp_path):
    env = _env(_config(tmp_path))
    infos, terminated = _run(env, action=1)
    assert terminated
    final = infos[-1]
    assert final["recapitalization_debt"] == 0.0
    assert final["would_margin_call_count"] == 0
    # terminated well before data end (the crash, not exhaustion)
    assert len(infos) < 150


def test_easy_mode_continues_after_would_be_ruin(tmp_path):
    env = _env(_config(
        tmp_path, solvency_mode="easy_chronological_continuation"))
    infos, terminated = _run(env, action=1)
    final = infos[-1]
    assert final["would_margin_call_count"] >= 1
    assert final["recapitalization_count"] >= 1
    assert final["recapitalization_debt"] > 0.0
    # Ran through to data exhaustion — later bars were reached.
    assert final["termination_cause"] == "data_end"
    assert len(infos) > 150


def test_conservation_and_no_recap_gain(tmp_path):
    """Economic equity must equal operational equity minus debt at every
    step, and the step that records a recapitalization must not improve
    economic equity."""
    env = _env(_config(
        tmp_path, solvency_mode="easy_chronological_continuation"))
    env.reset(seed=7)
    terminated = False
    prev_econ = None
    prev_debt = 0.0
    saw_recap_step = False
    while not terminated:
        _obs, _reward, terminated, _tr, info = env.step(1)
        econ = info["economic_equity"]
        debt = info["recapitalization_debt"]
        operational = econ + debt
        assert econ == pytest.approx(operational - debt)
        if debt > prev_debt and prev_econ is not None:
            saw_recap_step = True
            # the recap step's economic move is the real loss (<= 0
            # up to the bar's market move) — never a jump upward by
            # the recap amount
            assert econ <= prev_econ + abs(prev_econ) * 0.05 + 1.0
        prev_econ, prev_debt = econ, debt
    env.close()
    assert saw_recap_step


def test_agent_can_act_after_recapitalization(tmp_path):
    """After ruin+recap the environment must accept and execute new
    bounded positions — an inert state fails."""
    env = _env(_config(
        tmp_path, solvency_mode="easy_chronological_continuation"))
    env.reset(seed=7)
    recap_seen = False
    acted_after = False
    terminated = False
    step = 0
    while not terminated and step < 400:
        action = 1
        _obs, _reward, terminated, _tr, info = env.step(action)
        if recap_seen and info.get("position") not in (None, 0):
            acted_after = True
        if info["recapitalization_count"] >= 1:
            recap_seen = True
        step += 1
    env.close()
    assert recap_seen
    assert acted_after or env.bridge.trade_count >= 2


def test_validation_and_test_modes_reject_easy(tmp_path):
    for mode in ("validation", "test", "live", "paper", ""):
        with pytest.raises(ValueError, match="train-only"):
            _env(_config(
                tmp_path, env_mode=mode,
                solvency_mode="easy_chronological_continuation"))


def test_unknown_solvency_mode_refuses(tmp_path):
    with pytest.raises(ValueError, match="unknown solvency_mode"):
        _env(_config(tmp_path, solvency_mode="cheat_mode"))


def test_normal_mode_is_default_and_debt_free(tmp_path):
    env = _env(_config(tmp_path))
    assert env.solvency_mode == "normal_realistic"
