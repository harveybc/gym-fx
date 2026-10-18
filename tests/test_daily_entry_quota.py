from datetime import datetime

import pytest

from app.daily_entry_quota import DailyEntryQuota
from test_shared_execution_envelope import _env, _drive


def dt(value):
    return datetime.fromisoformat(value)


def test_fill_day_partial_replay_and_protection():
    q = DailyEntryQuota(1)
    q.submitted(dt("2024-03-12T03:00:00+00:00"), 7)  # March 11 New York
    assert not q.allows(dt("2024-03-12T03:30:00+00:00"))
    assert q.allows(dt("2024-03-12T04:00:00+00:00"))
    q.filled(dt("2024-03-12T05:00:00+00:00"), 7)
    q.filled(dt("2024-03-12T06:00:00+00:00"), 7)  # completed after partial
    q.filled(dt("2024-03-12T07:00:00+00:00"), 8)  # protective child
    assert q.fills == {"2024-03-12": 1}
    assert not q.allows(dt("2024-03-12T07:00:00+00:00"))


def test_dst_business_day_not_utc_day():
    q = DailyEntryQuota(1)
    q.submitted(dt("2024-11-03T04:30:00+00:00"), 1)
    assert not q.allows(dt("2024-11-04T04:30:00+00:00"))
    assert q.allows(dt("2024-11-04T05:00:00+00:00"))


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_configurable_caps_are_maxima_not_obligations(limit):
    q = DailyEntryQuota(limit)
    now = dt("2024-03-12T12:00:00+00:00")
    assert q.fills == {}  # zero entries is always legal
    for ref in range(limit):
        assert q.allows(now)
        q.submitted(now, ref)
        q.filled(now, ref)
    assert not q.allows(now)
    with pytest.raises(ValueError, match="LIMIT_EXCEEDED"):
        q.submitted(now, limit)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_bad_quota(limit):
    with pytest.raises(ValueError):
        DailyEntryQuota(limit)


def test_real_broker_daily_cap_and_reversal_closes(tmp_path):
    env = _env(tmp_path, [100.] * 40, max_entries_per_day=1)
    try:
        _, events = _drive(env, [0.8, 0.8, -0.8, -0.8] * 12)
        fills = [e for e in events if e["reason"] == "entry_fill"]
        assert len(fills) >= 2  # quota rolls over, not permanent abstention
        assert any(e["reason"] == "reversal_close" for e in events)
        quota = env.strategy_plugin._daily_quota
        assert max(quota.fills.values()) == 1
        assert max(quota.submissions.values()) == 1
        assert env.bridge.execution_diagnostics["blocked_daily_entry_limit"] > 0
        summary = env.summary()
        assert summary["filled_entries"] == sum(quota.fills.values())
        assert summary["max_entries_filled_in_day"] == 1
        assert summary["trades_closed"] == len(env.bridge.closed_trade_stream)
        assert summary["win_rate_pct"] == 0.0  # flat prices; not winning by rounding
        assert "daily_entry_budget_available" in env.observation_space.spaces
    finally:
        env.close()
