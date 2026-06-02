"""Tests for the OANDA America/New_York FX calendar helper.

These tests exercise the pure helper functions in
``app.oanda_calendar`` and prove the DST/Friday/daily-break policy is
implemented via timezone conversion, not a fixed UTC offset.
"""
from __future__ import annotations

import datetime as _dt

import pytest

from app.oanda_calendar import (
    CALENDAR_POLICY_ID,
    OANDA_FX_TIMEZONE,
    broker_market_open,
    compute_fx_calendar_features,
    is_broker_daily_break_near,
    is_force_flat_window,
    is_friday_risk_reduction_window,
    is_no_new_position_window,
    is_no_trade_window,
)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


NY = ZoneInfo(OANDA_FX_TIMEZONE)


def _ny(ts: str) -> _dt.datetime:
    """Build an NY-localized datetime from a naive 'YYYY-MM-DD HH:MM' string."""
    return _dt.datetime.fromisoformat(ts).replace(tzinfo=NY)


def test_policy_id_is_stable():
    assert CALENDAR_POLICY_ID == "oanda_us_fx_ny_v1"


# ----- DST-awareness ----------------------------------------------------------
def test_friday_close_uses_zoneinfo_not_fixed_utc_offset():
    # Friday 16:59 NY in EDT (summer): 20:59 UTC.
    summer_close_utc = _dt.datetime(2024, 6, 7, 20, 59, tzinfo=_dt.timezone.utc)
    feats = compute_fx_calendar_features(summer_close_utc, timeframe_hours=4)
    assert feats["hours_to_friday_close"] == pytest.approx(0.0, abs=1e-6)

    # Friday 16:59 NY in EST (winter): 21:59 UTC. Same calendar minute in NY
    # — proof the conversion handles DST instead of hard-coding -4 hours.
    winter_close_utc = _dt.datetime(2024, 12, 6, 21, 59, tzinfo=_dt.timezone.utc)
    feats = compute_fx_calendar_features(winter_close_utc, timeframe_hours=4)
    assert feats["hours_to_friday_close"] == pytest.approx(0.0, abs=1e-6)


def test_summer_utc_timestamp_one_hour_before_friday_close():
    # 19:59 UTC on 2024-06-07 == 15:59 NY (EDT).
    feats = compute_fx_calendar_features(
        _dt.datetime(2024, 6, 7, 19, 59, tzinfo=_dt.timezone.utc),
        timeframe_hours=4,
    )
    assert feats["hours_to_friday_close"] == pytest.approx(1.0, abs=1e-6)
    assert feats["is_force_flat_window"] == 1.0  # 15:45 <= 15:59 < 16:59


# ----- Friday windows ---------------------------------------------------------
def test_friday_no_new_position_window_starts_at_14_00_ny():
    assert is_no_new_position_window(_ny("2024-06-07 13:59")) is False
    assert is_no_new_position_window(_ny("2024-06-07 14:00")) is True
    assert is_no_new_position_window(_ny("2024-06-07 16:58")) is True
    assert is_no_new_position_window(_ny("2024-06-07 16:59")) is False


def test_friday_risk_reduction_window_starts_at_15_00_ny():
    assert is_friday_risk_reduction_window(_ny("2024-06-07 14:59")) is False
    assert is_friday_risk_reduction_window(_ny("2024-06-07 15:00")) is True
    assert is_friday_risk_reduction_window(_ny("2024-06-07 16:58")) is True
    # Saturday should never be inside the Friday window.
    assert is_friday_risk_reduction_window(_ny("2024-06-08 15:30")) is False


def test_friday_force_flat_window_starts_at_15_45_ny():
    assert is_force_flat_window(_ny("2024-06-07 15:44")) is False
    assert is_force_flat_window(_ny("2024-06-07 15:45")) is True
    assert is_force_flat_window(_ny("2024-06-07 16:58")) is True
    assert is_force_flat_window(_ny("2024-06-07 16:59")) is False  # closed


# ----- Daily break ------------------------------------------------------------
def test_daily_break_near_activates_around_1659_ny():
    # 30 minutes before the break.
    assert is_broker_daily_break_near(_ny("2024-06-05 16:29")) is False
    assert is_broker_daily_break_near(_ny("2024-06-05 16:30")) is True
    # Inside the break itself.
    assert is_broker_daily_break_near(_ny("2024-06-05 17:00")) is True
    # After the break window.
    assert is_broker_daily_break_near(_ny("2024-06-05 17:05")) is False


def test_no_trade_window_covers_1650_to_1710_ny():
    assert is_no_trade_window(_ny("2024-06-05 16:49")) is False
    assert is_no_trade_window(_ny("2024-06-05 16:50")) is True
    assert is_no_trade_window(_ny("2024-06-05 17:09")) is True
    assert is_no_trade_window(_ny("2024-06-05 17:10")) is False


# ----- Broker market open -----------------------------------------------------
def test_broker_closed_saturday_and_pre_sunday_open():
    assert broker_market_open(_ny("2024-06-08 12:00")) is False  # Saturday
    assert broker_market_open(_ny("2024-06-09 17:04")) is False  # Sun pre-open
    assert broker_market_open(_ny("2024-06-09 17:05")) is True   # Sun open


def test_broker_closed_during_daily_break():
    assert broker_market_open(_ny("2024-06-05 16:58")) is True   # before break
    assert broker_market_open(_ny("2024-06-05 16:59")) is False  # inside break
    assert broker_market_open(_ny("2024-06-05 17:04")) is False
    assert broker_market_open(_ny("2024-06-05 17:05")) is True   # after break


def test_broker_closed_at_friday_weekly_close():
    assert broker_market_open(_ny("2024-06-07 16:58")) is True
    assert broker_market_open(_ny("2024-06-07 16:59")) is False
    assert broker_market_open(_ny("2024-06-07 23:00")) is False


# ----- Feature dict completeness ---------------------------------------------
def test_feature_dict_keys_complete_and_bars_scale_with_timeframe():
    feats = compute_fx_calendar_features(
        _dt.datetime(2024, 6, 7, 19, 30, tzinfo=_dt.timezone.utc),  # Fri 15:30 NY
        timeframe_hours=4,
    )
    expected_keys = {
        "hours_to_fx_daily_break",
        "bars_to_fx_daily_break",
        "hours_to_friday_close",
        "bars_to_friday_close",
        "is_friday_risk_reduction_window",
        "is_no_new_position_window",
        "is_force_flat_window",
        "is_broker_daily_break_near",
        "broker_market_open",
        "is_no_trade_window",
    }
    assert expected_keys.issubset(feats.keys())
    assert feats["is_friday_risk_reduction_window"] == 1.0
    assert feats["is_no_new_position_window"] == 1.0
    assert feats["is_force_flat_window"] == 0.0  # 15:30 < 15:45
    assert feats["bars_to_friday_close"] == pytest.approx(
        feats["hours_to_friday_close"] / 4.0
    )


def test_unparseable_timestamp_returns_neutral_features():
    feats = compute_fx_calendar_features("not a timestamp", timeframe_hours=4)
    for v in feats.values():
        assert v == 0.0
