"""Real-engine acceptance for the explicit next-open transition contract."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shared_execution_envelope import _env


def environment(path, **kwargs):
    return _env(path, [100.0] * 40, execution_transition='next_open_v2',
                commission=0.001, **kwargs)


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_entry_and_cost_belong_to_the_action_step(tmp_path, strict, direction):
    env = environment(tmp_path, window_size=13, observation_stride_bars=2,
                      require_full_observation_window=strict)
    try:
        env.reset(seed=7)
        initial_bar = env.bridge.bar_index
        _, reward, done, _, _ = env.step([direction])
        assert env.bridge.bar_index == initial_bar + 1
        assert env.bridge.position_units * direction > 0
        cost = abs(env.bridge.position_units) * 100 * .001
        assert env.bridge.equity == pytest.approx(10000 - cost)
        assert reward == pytest.approx(-cost / 10000)
        assert env.bridge.last_trade_cost == pytest.approx(cost)
        assert not done
        _, hold_reward, _, _, _ = env.step([0.0])
        assert hold_reward == pytest.approx(0.0)
        assert env.bridge.last_trade_cost == pytest.approx(0.0)
    finally:
        env.close()


def test_next_open_not_current_close_and_no_future_price_access(tmp_path):
    closes = [100.0] * 40
    opens = closes.copy()
    opens[1] = 101.0
    highs = [max(o, c) + .1 for o, c in zip(opens, closes)]
    env = _env(tmp_path, closes, opens=opens, highs=highs,
               execution_transition='next_open_v2', commission=0,
               execution_envelope={'envelope_mode':'fixed_fraction', 'sl_fraction':.05,
                                   'tp_fraction':.1, 'leverage_cap':.5})
    try:
        env.reset(seed=7)
        env.step([.5])
        assert env.bridge.entry_price == 101.0
        assert abs(env.bridge.position_units) == pytest.approx(10000 * .5 * .998 / 100)
    finally:
        env.close()


def test_every_step_advances_once_and_final_step_terminates(tmp_path):
    env = environment(tmp_path)
    try:
        env.reset(seed=7)
        for bar in range(2, 41):
            _, _, done, truncated, _ = env.step([0.0])
            assert env.bridge.bar_index == bar
            assert done == (bar == 40)
            assert not truncated
        assert env.bridge.termination_cause == 'data_end'
    finally:
        env.close()


def test_future_perturbation_preserves_observations_and_transition_prefix(tmp_path):
    def replay(path, future):
        path.mkdir()
        env = _env(path, [100.0] * 30 + [future] * 10,
                   execution_transition='next_open_v2', max_entries_per_day=1)
        try:
            obs, _ = env.reset(seed=7)
            rows = [obs]
            outcomes = []
            for a in [1.0] + [0.0] * 9:
                obs, reward, term, trunc, _ = env.step([a])
                rows.append(obs)
                outcomes.append((reward, term, trunc, env.bridge.position_units, env.bridge.equity))
            return rows, outcomes
        finally:
            env.close()
    left, a = replay(tmp_path / 'left', 100.0)
    right, b = replay(tmp_path / 'right', 200.0)
    assert a == b
    for lrow, rrow in zip(left, right):
        for key in lrow:
            np.testing.assert_array_equal(lrow[key], rrow[key])


def test_entry_bar_stop_is_processed_before_transition_is_returned(tmp_path):
    lows = [99.95] * 40
    lows[1] = 94.0
    env = _env(tmp_path, [100.] * 40, lows=lows,
               execution_transition='next_open_v2', commission=.001)
    try:
        env.reset(seed=7)
        _, reward, _, _, _ = env.step([1.])
        assert env.bridge.bar_index == 2
        assert env.bridge.position_units == 0
        assert env.bridge.trade_count == 1
        assert reward < 0
        event = env.bridge.closed_trade_stream[0]
        assert event['bar_index'] == 2
    finally:
        env.close()


@pytest.mark.parametrize('direction', [1., -1.])
@pytest.mark.parametrize('leverage', [1., 5.])
def test_entry_bar_close_net_matches_equity_and_both_commissions(tmp_path, direction, leverage):
    highs, lows = [100.05] * 40, [99.95] * 40
    if direction > 0:
        lows[1] = 94.
    else:
        highs[1] = 106.
    env = _env(tmp_path, [100.] * 40, highs=highs, lows=lows,
               execution_transition='next_open_v2', commission=.001, leverage=leverage)
    try:
        env.reset(seed=7)
        env.step([direction])
        event = env.bridge.closed_trade_stream[0]
        assert event['costs'] == pytest.approx(env.bridge.commission_paid)
        assert event['net_pnl'] == pytest.approx(env.bridge.equity - 10000)
        assert env.bridge.last_trade_cost == pytest.approx(event['costs'])
        assert env.bridge.position_units == 0
    finally:
        env.close()


def test_execution_error_reaches_caller_instead_of_a_successful_terminal(tmp_path, monkeypatch):
    env = environment(tmp_path)
    def broken(*args):
        raise ValueError('TEST_EXECUTION_FAILURE')
    try:
        monkeypatch.setattr(env.strategy_plugin, 'apply_action', broken)
        env.reset(seed=7)
        with pytest.raises(RuntimeError, match='SIMULATOR_EXECUTION_FAILED') as exc:
            env.step([1.0])
        assert isinstance(exc.value.__cause__, ValueError)
    finally:
        env.close()


def test_reset_starts_a_fresh_next_open_transition(tmp_path):
    env = environment(tmp_path)
    try:
        for _ in range(2):
            env.reset(seed=7)
            assert env.bridge.bar_index == 1
            assert env.bridge.position_units == 0
            env.step([1.0])
            assert env.bridge.bar_index == 2
            assert env.bridge.position_units > 0
    finally:
        env.close()


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_business_12_hour_buffers_with_actual_calendar_gap(tmp_path, direction):
    from test_session_exposure_env import _env as session_env, _csv, policy
    close = pd.Timestamp('2024-01-05T22:00:00Z')
    reopen = pd.Timestamp('2024-01-07T22:00:00Z')
    stamps = pd.date_range('2024-01-01', '2024-01-09', freq='h', tz='UTC')
    stamps = stamps[(stamps < close) | (stamps >= reopen)].tz_localize(None)
    path = _csv(tmp_path, stamps=stamps)
    env = session_env(tmp_path, csv=path, intervals=[[str(close), str(reopen)]],
        execution_transition='next_open_v2', execution_max_delay_hours=1, timeframe='1h',
        session_exposure_policy=policy(wind_down_hours=24, forced_flatten_hours=12,
                                      reopen_min_hours=12), max_entries_per_day=1)
    try:
        env.reset(seed=7)
        had_position = False
        for _ in range(len(stamps)):
            _, _, done, _, _ = env.step([direction])
            now = pd.Timestamp(env.dataframe.index[env.bridge.bar_index - 1], tz='UTC')
            had_position |= bool(env.bridge.position_units)
            if close - pd.Timedelta(hours=12) <= now < reopen + pd.Timedelta(hours=12):
                assert env.bridge.position_units == 0, now
            assert not close <= now < reopen
            if done:
                break
        assert had_position
        assert env.summary()['max_entries_filled_in_day'] <= 1
    finally:
        env.close()
