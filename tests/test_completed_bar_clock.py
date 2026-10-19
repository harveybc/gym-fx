"""Completed H1 availability clocks through GymFxEnv and real Backtrader."""
import os
import json
from pathlib import Path

import pandas as pd
import pytest

from app.session_exposure import SessionDataContradictionError, SessionEvidenceError
from test_session_exposure_env import _env, policy


CLOSE = pd.Timestamp('2024-01-05T22:00Z')
REOPEN = pd.Timestamp('2024-01-07T22:00Z')


def environment(tmp_path, stamps, **kwargs):
    # All observations are the completed row's close, never future features.
    frame = pd.DataFrame({'DATE_TIME': stamps, 'OPEN': 1.1, 'HIGH': 1.1001,
                          'LOW': 1.0999, 'CLOSE': 1.1, 'VOLUME': 0.0})
    path = tmp_path / 'completed.csv'
    frame.to_csv(path, index=False, float_format='%.17g')
    return _env(tmp_path, csv=path, feature_columns=['CLOSE'],
                window_size=1,
                execution_transition='next_open_v2', execution_max_delay_hours=1,
                timestamp_semantics='completed_bar_available_at', timeframe='1h',
                intervals=[[str(CLOSE), str(REOPEN)]],
                session_exposure_policy=policy(wind_down_hours=24,
                    forced_flatten_hours=12, reopen_min_hours=12), **kwargs)


def test_last_completed_bar_at_close_is_valid_but_cannot_submit_for_reopen(tmp_path):
    stamps = pd.DatetimeIndex([CLOSE - pd.Timedelta('1h'), CLOSE,
                              REOPEN + pd.Timedelta('1h'), REOPEN + pd.Timedelta('2h')])
    env = environment(tmp_path, stamps)
    try:
        env.reset(seed=7)
        env.step([1.0])  # reaches last valid completed bar, ending at close
        _, _, _, _, info = env.step([1.0])
        assert info['session_state'] == 'EXPECTED_MARKET_CLOSED'
        assert env.bridge.position_units == 0
        assert not env.bridge.fill_events
        assert pd.Timestamp(info['action_time']) == CLOSE
        assert pd.Timestamp(info['observation_available_at']) == REOPEN + pd.Timedelta('1h')
    finally:
        env.close()


@pytest.mark.parametrize('direction', [1.0, -1.0])
@pytest.mark.parametrize('strategy', ['envelope', 'atr'])
def test_midnight_quota_counts_next_open_not_next_end_label(tmp_path, direction, strategy):
    # Winter NY midnight = 05Z. END 05Z executes OPEN 04Z, still Jan 1 NY.
    env = environment(tmp_path, pd.date_range('2024-01-02T04:00Z', periods=8, freq='h'),
                      session=False, max_entries_per_day=1,
                      continuous_action_contract='target_exposure_hysteresis_v2')
    if strategy == 'atr':
        from strategy_plugins.direct_atr_sltp import Plugin
        env.config.update(atr_period=1, entry_order_mode='market', position_size=1.0)
        env.strategy_plugin = Plugin(env.config)
    try:
        env.reset(seed=7)
        env.step([direction * 0.1])
        quota = env.strategy_plugin._daily_quota
        assert quota.fills == {'2024-01-01': 1}
        fill = env.bridge.fill_events[0]
        assert pd.Timestamp(fill['fill_time']) == pd.Timestamp('2024-01-02T04:00Z')
        assert pd.Timestamp(fill['engine_bar_time']) == pd.Timestamp('2024-01-02T05:00Z')
        env.step([0.0])  # close, not a second entry
        env.step([direction * 0.1])
        assert quota.fills == {'2024-01-01': 1, '2024-01-02': 1}
    finally:
        env.close()


@pytest.mark.parametrize('bad_end', [CLOSE + pd.Timedelta('1h'), REOPEN])
def test_completed_bar_overlapping_closure_is_still_refused(tmp_path, bad_end):
    with pytest.raises(SessionDataContradictionError, match='closure'):
        environment(tmp_path, [CLOSE - pd.Timedelta('1h'), CLOSE, bad_end])


@pytest.mark.parametrize('kw', [dict(timeframe='4h'), dict(timeframe=None),
                              dict(execution_transition='legacy_delayed_v1'),
                              dict(timestamp_semantics='typo')])
def test_invalid_clock_contract_refuses(tmp_path, kw):
    config = dict(timestamp_semantics='completed_bar_available_at', timeframe='1h',
                  execution_transition='next_open_v2')
    config.update(kw)
    from test_shared_execution_envelope import _env as basic_env
    with pytest.raises(ValueError, match='TIMESTAMP|TIMEFRAME|COMPLETED_BAR'):
        basic_env(tmp_path, [1.1] * 10, **config)


@pytest.mark.parametrize('remaining', ['position', 'order'])
def test_cannot_carry_position_or_order_across_closure(tmp_path, remaining):
    env = environment(tmp_path, [CLOSE, REOPEN + pd.Timedelta('1h'), REOPEN + pd.Timedelta('2h')])
    try:
        env.reset(seed=7)
        # Fault injection at the public bridge boundary; no broker fill may run.
        if remaining == 'position':
            env.bridge.position_units = 1
            env.bridge.position = 1
        else:
            env.bridge.open_order_count = 1
            env.bridge.open_order_inventory = ({'ref': 99, 'role': 'entry',
                'side': 'buy', 'size': 1.0},)
        with pytest.raises(SessionEvidenceError, match='CROSS_CLOSURE'):
            env.step([0.0])
        assert env.bridge.bar_index == 1
    finally:
        env.close()


@pytest.mark.parametrize('start,expected', [('2024-03-12T03:00Z', '2024-03-11'),
                                           ('2024-11-04T04:00Z', '2024-11-03')])
def test_dst_midnight_fill_business_date(tmp_path, start, expected):
    env = environment(tmp_path, pd.date_range(start, periods=8, freq='h'),
                      session=False, max_entries_per_day=1)
    try:
        env.reset(seed=7)
        env.step([0.1])
        assert env.strategy_plugin._daily_quota.fills == {expected: 1}
    finally:
        env.close()


def test_admitted_raw_ohlc_csv_bitwise_fidelity(tmp_path):
    source = os.environ.get('GYM_FX_ADMITTED_EURUSD_PARQUET')
    if not source:
        pytest.skip('set GYM_FX_ADMITTED_EURUSD_PARQUET to admitted raw parquet')
    raw = pd.read_parquet(source)
    frame = raw.rename(columns={'available_at': 'DATE_TIME', 'BidOpen': 'OPEN',
        'BidHigh': 'HIGH', 'BidLow': 'LOW', 'BidClose': 'CLOSE'})[
            ['DATE_TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE']]
    path = tmp_path / 'roundtrip.csv'
    frame.to_csv(path, index=False, float_format='%.17g')
    env = _env(tmp_path, csv=path, session=False, feature_columns=['CLOSE'],
               require_curated_rl_input=True)
    try:
        for column in ('OPEN', 'HIGH', 'LOW', 'CLOSE'):
            assert env.dataframe[column].to_numpy(dtype='float64').tobytes() == \
                frame[column].to_numpy(dtype='float64').tobytes(), column
    finally:
        env.close()


@pytest.mark.parametrize('end', ['2024-01-10', '2024-02-01'], ids=['week', 'month'])
@pytest.mark.parametrize('direction', [1.0, -1.0])
@pytest.mark.parametrize('assumed_spread', [False, True], ids=['no-spread', 'assumed-spread'])
def test_admitted_eurusd_real_marketdata_env(tmp_path, end, direction, assumed_spread):
    source = os.environ.get('GYM_FX_ADMITTED_EURUSD_PARQUET')
    if not source:
        pytest.skip('set GYM_FX_ADMITTED_EURUSD_PARQUET to admitted raw parquet')
    raw = pd.read_parquet(source)
    raw = raw[(raw.timestamp >= pd.Timestamp('2024-01-02', tz='UTC')) &
              (raw.timestamp < pd.Timestamp(end, tz='UTC'))].copy()
    assert (raw.available_at - raw.timestamp == pd.Timedelta('1h')).all()
    frame = raw.rename(columns={'available_at': 'DATE_TIME', 'BidOpen': 'OPEN',
        'BidHigh': 'HIGH', 'BidLow': 'LOW', 'BidClose': 'CLOSE'})[
            ['DATE_TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE']]
    if assumed_spread:
        frame['ASSUMED_SPREAD'] = 0.00010  # scenario assumption, NOT observed quotes
    path = tmp_path / 'actual.csv'
    frame.to_csv(path, index=False, float_format='%.17g')
    calendar_path = Path(os.environ.get('GYM_FX_CLOSURE_INTERVALS',
                                        str(Path(source).parent / 'CLOSURE_INTERVALS.json')))
    with calendar_path.open() as stream:
        intervals = json.load(stream)
    # Retain the preceding closure as well, including holiday reopen evidence.
    intervals = [[c, r] for c, r in intervals
                 if pd.Timestamp(r) >= pd.Timestamp('2024-01-01', tz='UTC')
                 and pd.Timestamp(c) <= pd.Timestamp(end, tz='UTC') + pd.Timedelta('7D')]
    env = _env(tmp_path, csv=path, feature_columns=['CLOSE'],
        session_symbol='EURUSD', intervals=intervals,
        execution_transition='next_open_v2', execution_max_delay_hours=1,
        timestamp_semantics='completed_bar_available_at', timeframe='1h',
        require_curated_rl_input=True,
        session_spread_column='ASSUMED_SPREAD' if assumed_spread else 'SPREAD',
        commission=0.00005,  # nonzero analytical cost assumption, not broker-verified
        max_entries_per_day=1, require_full_observation_window=True,
        window_size=13, observation_stride_bars=2,
        session_exposure_policy=policy(wind_down_hours=24,
            forced_flatten_hours=12, reopen_min_hours=12))
    try:
        env.reset(seed=7)
        for column in ('OPEN', 'HIGH', 'LOW', 'CLOSE'):
            assert env.dataframe[column].to_numpy(dtype='float64').tobytes() == \
                frame[column].to_numpy(dtype='float64').tobytes(), column
        boundary_steps = 0
        for _ in range(len(frame)):
            prior = env.bridge.bar_index
            _, _, done, _, info = env.step([direction])
            assert env.bridge.bar_index == prior + 1
            available = pd.Timestamp(env.dataframe.index[env.bridge.bar_index - 1])
            for c, r in intervals:
                c, r = pd.Timestamp(c), pd.Timestamp(r)
                if c - pd.Timedelta('12h') <= available <= r + pd.Timedelta('12h'):
                    assert env.bridge.position_units == 0, available
                if pd.Timestamp(info['action_time']) == c:
                    boundary_steps += 1
                    assert info['session_state'] == 'EXPECTED_MARKET_CLOSED'
            if done:
                break
        assert done and boundary_steps >= (1 if end == '2024-01-10' else 4)
        entries = [f for f in env.bridge.fill_events if f['role'] == 'entry']
        if assumed_spread:
            assert entries
            assert env.bridge.commission_paid > 0
        for fill in entries:
            bar = frame.iloc[fill['bar_index'] - 1]
            assert fill['price'] == bar.OPEN
            assert pd.Timestamp(fill['fill_time']) == bar.DATE_TIME - pd.Timedelta('1h')
            for c, r in intervals:
                assert not pd.Timestamp(c) <= pd.Timestamp(fill['fill_time']) < pd.Timestamp(r)
        quota = env.strategy_plugin._daily_quota
        assert max(quota.fills.values(), default=0) <= 1
        if not assumed_spread:
            # Raw parquet has no spread: never invent one to release blackout.
            assert not entries
            assert info['session_evidence_failed_closed'] or info['session_state'] == 'REOPEN_BLACKOUT'
    finally:
        env.close()
