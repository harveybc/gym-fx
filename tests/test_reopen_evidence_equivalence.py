"""Reopen evidence equals the full causal scan without repeated prefix work."""
import pytest

from app.session_exposure import ReopenEvidence
from test_session_exposure_env import _env


def reference(env, step, now):
    reopen = env._session_calendar.most_recent_reopen(now)
    if reopen is None:
        return None, {}
    count, streak, last = 0, 0, None
    for index in range(step + 1):
        stamp = env._session_now(index)
        if stamp is None or stamp <= reopen:
            continue
        if index >= step and env.execution_transition != 'next_open_v2':
            continue
        count += 1
        check = env._session_stability_check(index)
        streak = streak + 1 if check['passed'] else 0
        last = dict(bar=index, **check)
    return ReopenEvidence.build(closed_bars_since_reopen=count,
        stability_checks_passed=streak, hint_time_since_reopen_hours=None), {
        'session_reopen_closed_bars': count,
        'session_reopen_stability_streak': streak,
        'session_reopen_last_check': last}


@pytest.mark.parametrize('transition', ['legacy_delayed_v1', 'next_open_v2'])
@pytest.mark.parametrize('reverse', [False, True])
def test_every_prefix_matches_full_scan_including_out_of_order_calls(tmp_path, transition, reverse):
    env = _env(tmp_path, execution_transition=transition, execution_max_delay_hours=4)
    try:
        indices = range(env.total_bars)
        if reverse:
            indices = reversed(indices)
        for step in indices:
            now = env._session_now(step)
            assert env._session_reopen_evidence(step, now) == reference(env, step, now)
    finally:
        env.close()


def test_failed_last_check_does_not_recompute_earlier_stability(tmp_path, monkeypatch):
    env = _env(tmp_path, execution_transition='next_open_v2', execution_max_delay_hours=4)
    calls = []
    def fails(index):
        calls.append(index)
        return dict(passed=False, reasons=['engineered_failure'])
    monkeypatch.setattr(env, '_session_stability_check', fails)
    try:
        step = env.total_bars - 1
        _, info = env._session_reopen_evidence(step, env._session_now(step))
        assert info['session_reopen_closed_bars'] > 1
        assert info['session_reopen_stability_streak'] == 0
        assert calls == [step]
    finally:
        env.close()
