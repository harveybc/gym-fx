"""OANDA FX calendar helper — DST-aware America/New_York policy.

Pure functions. No env coupling. Used by GymFxEnv (when
`oanda_fx_calendar_obs` is enabled in the config) and by Stage B trace
metadata emission to populate broker-policy observation fields.

Policy times (per FINRA/OANDA memo PROJECT3_FINRA_OANDA_TRADE_FREQUENCY_POLICY_MEMO.md):

- FX weekly open: Sunday 17:05 New York.
- FX weekly close: Friday 16:59 New York.
- Daily FX break: 16:59-17:05 New York.
- Project no-trade window: 16:50-17:10 New York.
- Friday no-new-position cutoff: 14:00 New York.
- Friday risk-reduction window begins: 15:00 New York.
- Friday force-flat deadline: 15:45 New York.
- Last-exit safety cutoff: 15:55 New York.

All times resolved via IANA `America/New_York` (DST-aware).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Mapping, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore

OANDA_FX_TIMEZONE = "America/New_York"
CALENDAR_POLICY_ID = "oanda_us_fx_ny_v1"

# Minute-of-day constants (New York local time).
WEEKLY_OPEN_DOW = 6   # Sunday (Mon=0..Sun=6)
WEEKLY_OPEN_HM = (17, 5)
WEEKLY_CLOSE_DOW = 4  # Friday
WEEKLY_CLOSE_HM = (16, 59)
DAILY_BREAK_START_HM = (16, 59)
DAILY_BREAK_END_HM = (17, 5)
NO_TRADE_WINDOW_START_HM = (16, 50)
NO_TRADE_WINDOW_END_HM = (17, 10)
FRIDAY_NO_NEW_POSITION_HM = (14, 0)
FRIDAY_RISK_REDUCTION_HM = (15, 0)
FRIDAY_FORCE_FLAT_HM = (15, 45)
FRIDAY_LAST_EXIT_HM = (15, 55)
BROKER_DAILY_BREAK_NEAR_MINUTES = 30  # within 30 min of 16:59 NY

_NY = ZoneInfo(OANDA_FX_TIMEZONE)


def _to_ny(ts: Any) -> Optional[_dt.datetime]:
    """Coerce any timestamp-like value into an aware NY datetime.

    Naive inputs are treated as UTC (matches Stage B trace convention).
    Returns None if the value cannot be parsed.
    """
    if ts is None:
        return None
    if isinstance(ts, _dt.datetime):
        dt = ts
    else:
        s = str(ts).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = _dt.datetime.fromisoformat(s.replace("T", " "))
        except ValueError:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
            ):
                try:
                    dt = _dt.datetime.strptime(s[: len(fmt) + 6], fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_NY)


def _minute_of_day(dt: _dt.datetime) -> int:
    return dt.hour * 60 + dt.minute


def _hm_minutes(hm) -> int:
    return hm[0] * 60 + hm[1]


def _next_friday_close(now_ny: _dt.datetime) -> _dt.datetime:
    """Next Friday 16:59 NY at or after ``now_ny`` (wraps to next week if past)."""
    days_ahead = (WEEKLY_CLOSE_DOW - now_ny.weekday()) % 7
    candidate = now_ny.replace(
        hour=WEEKLY_CLOSE_HM[0],
        minute=WEEKLY_CLOSE_HM[1],
        second=0,
        microsecond=0,
    ) + _dt.timedelta(days=days_ahead)
    if candidate < now_ny:
        candidate += _dt.timedelta(days=7)
    return candidate


def _next_daily_break(now_ny: _dt.datetime) -> _dt.datetime:
    """Next 16:59 NY (any weekday) at or after ``now_ny``."""
    today = now_ny.replace(
        hour=DAILY_BREAK_START_HM[0],
        minute=DAILY_BREAK_START_HM[1],
        second=0,
        microsecond=0,
    )
    if today <= now_ny:
        today += _dt.timedelta(days=1)
    return today


def is_no_new_position_window(dt_ny: _dt.datetime) -> bool:
    """True from Friday 14:00 NY through weekly close."""
    if dt_ny.weekday() != WEEKLY_CLOSE_DOW:
        return False
    return _minute_of_day(dt_ny) >= _hm_minutes(FRIDAY_NO_NEW_POSITION_HM) and \
        _minute_of_day(dt_ny) < _hm_minutes(WEEKLY_CLOSE_HM)


def is_friday_risk_reduction_window(dt_ny: _dt.datetime) -> bool:
    """True from Friday 15:00 NY through weekly close."""
    if dt_ny.weekday() != WEEKLY_CLOSE_DOW:
        return False
    return _minute_of_day(dt_ny) >= _hm_minutes(FRIDAY_RISK_REDUCTION_HM) and \
        _minute_of_day(dt_ny) < _hm_minutes(WEEKLY_CLOSE_HM)


def is_force_flat_window(dt_ny: _dt.datetime) -> bool:
    """True from Friday 15:45 NY through weekly close (positions must be flat)."""
    if dt_ny.weekday() != WEEKLY_CLOSE_DOW:
        return False
    return _minute_of_day(dt_ny) >= _hm_minutes(FRIDAY_FORCE_FLAT_HM) and \
        _minute_of_day(dt_ny) < _hm_minutes(WEEKLY_CLOSE_HM)


def is_broker_daily_break_near(dt_ny: _dt.datetime, *, near_minutes: int = BROKER_DAILY_BREAK_NEAR_MINUTES) -> bool:
    """True within ``near_minutes`` before, or inside, the 16:59-17:05 NY break.

    The weekly close itself counts as inside the break (Friday 16:59 is
    both the end of the trading week and the daily-break start).
    """
    mod = _minute_of_day(dt_ny)
    start = _hm_minutes(DAILY_BREAK_START_HM)
    end = _hm_minutes(DAILY_BREAK_END_HM)
    if start <= mod < end:
        return True
    return start - near_minutes < mod < start


def is_no_trade_window(dt_ny: _dt.datetime) -> bool:
    """Project no-trade window: 16:50-17:10 NY (covers the FX break)."""
    mod = _minute_of_day(dt_ny)
    return _hm_minutes(NO_TRADE_WINDOW_START_HM) <= mod < _hm_minutes(NO_TRADE_WINDOW_END_HM)


def broker_market_open(dt_ny: _dt.datetime) -> bool:
    """True when FX is tradeable: between Sun 17:05 NY and Fri 16:59 NY,
    excluding the daily 16:59-17:05 NY break.
    """
    mod = _minute_of_day(dt_ny)
    dow = dt_ny.weekday()
    # Closed: Saturday entirely.
    if dow == 5:
        return False
    # Sunday: closed until 17:05.
    if dow == WEEKLY_OPEN_DOW:
        return mod >= _hm_minutes(WEEKLY_OPEN_HM)
    # Friday: closed at/after 16:59.
    if dow == WEEKLY_CLOSE_DOW and mod >= _hm_minutes(WEEKLY_CLOSE_HM):
        return False
    # Daily break 16:59-17:05 (Mon-Thu).
    if _hm_minutes(DAILY_BREAK_START_HM) <= mod < _hm_minutes(DAILY_BREAK_END_HM):
        return False
    return True


def compute_fx_calendar_features(
    ts: Any,
    *,
    timeframe_hours: float = 4.0,
) -> Dict[str, float]:
    """Return the OANDA-FX calendar observation/info field dict.

    Keys returned (always populated, fall back to 0.0 on parse failure
    so a downstream env never crashes mid-rollout):

    ``hours_to_fx_daily_break``,
    ``bars_to_fx_daily_break``,
    ``hours_to_friday_close``,
    ``bars_to_friday_close``,
    ``is_friday_risk_reduction_window``,
    ``is_no_new_position_window``,
    ``is_force_flat_window``,
    ``is_broker_daily_break_near``,
    ``broker_market_open``,
    ``is_no_trade_window``.
    """
    neutral = {
        "hours_to_fx_daily_break": 0.0,
        "bars_to_fx_daily_break": 0.0,
        "hours_to_friday_close": 0.0,
        "bars_to_friday_close": 0.0,
        "is_friday_risk_reduction_window": 0.0,
        "is_no_new_position_window": 0.0,
        "is_force_flat_window": 0.0,
        "is_broker_daily_break_near": 0.0,
        "broker_market_open": 0.0,
        "is_no_trade_window": 0.0,
    }
    dt_ny = _to_ny(ts)
    if dt_ny is None:
        return neutral

    tf_h = max(float(timeframe_hours or 0.0), 1e-9)

    hours_to_break = (_next_daily_break(dt_ny) - dt_ny).total_seconds() / 3600.0
    hours_to_close = (_next_friday_close(dt_ny) - dt_ny).total_seconds() / 3600.0

    return {
        "hours_to_fx_daily_break": float(max(hours_to_break, 0.0)),
        "bars_to_fx_daily_break": float(max(hours_to_break, 0.0) / tf_h),
        "hours_to_friday_close": float(max(hours_to_close, 0.0)),
        "bars_to_friday_close": float(max(hours_to_close, 0.0) / tf_h),
        "is_friday_risk_reduction_window": 1.0 if is_friday_risk_reduction_window(dt_ny) else 0.0,
        "is_no_new_position_window": 1.0 if is_no_new_position_window(dt_ny) else 0.0,
        "is_force_flat_window": 1.0 if is_force_flat_window(dt_ny) else 0.0,
        "is_broker_daily_break_near": 1.0 if is_broker_daily_break_near(dt_ny) else 0.0,
        "broker_market_open": 1.0 if broker_market_open(dt_ny) else 0.0,
        "is_no_trade_window": 1.0 if is_no_trade_window(dt_ny) else 0.0,
    }


def resolve_broker_metadata(config: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """Pull broker/policy metadata fields from a run config.

    None values are kept so downstream evidence emission can distinguish
    "absent" from a synthesised default.
    """
    return {
        "broker_profile": config.get("broker_profile"),
        "market_type": config.get("market_type"),
        "trade_rate_band_id": config.get("trade_rate_band_id"),
        "calendar_policy_id": config.get("calendar_policy_id"),
    }
