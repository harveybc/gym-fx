"""Work plan 42 state-machine tests (order @45c49003 WP1/WP2):
transitions, Tuesday stale-feed vs expected weekend, holiday,
contradictory/missing session evidence, long/short/pending paths,
forced-flatten reconciliation, restart determinism."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.session_exposure import (  # noqa: E402
    ExposureFacts, SessionEvidence, SessionPolicyError,
    overlay_action, reconciliation_gate, session_state,
    validate_policy, watchdog_state)

UTC = timezone.utc


def policy(**overrides):
    base = {
        "enabled": True,
        "session_source": "venue_symbol_sessions_v1",
        "wind_down_hours": 36,
        "forced_flatten_hours": 4,
        "cancel_pending_on_wind_down": True,
        "allow_risk_increase_during_wind_down": False,
        "reopen_min_hours": 4,
        "reopen_min_closed_bars": 1,
        "stability_consecutive_checks": 3,
        "max_spread_relative_to_baseline": 2.0,
        "max_gap_sigma": 3.0,
        "max_realized_vol_relative_to_baseline": 2.0,
        "carried_position_recovery":
            "protected_opportunistic_then_forced",
        "holiday_policy": "same_as_weekly",
        "calendar_identity": "a" * 16,
    }
    base.update(overrides)
    return validate_policy(base)


FRIDAY_CLOSE = datetime(2026, 8, 28, 17, 0, tzinfo=UTC)
SUNDAY_REOPEN = datetime(2026, 8, 30, 21, 0, tzinfo=UTC)
CLOSURES = [(FRIDAY_CLOSE, SUNDAY_REOPEN)]


def evidence(now, **kw):
    return SessionEvidence(now=now, closures=CLOSURES, **kw)


class TestConfigContract:
    def test_flatten_must_precede_closure_inside_wind_down(self):
        with pytest.raises(SessionPolicyError, match="forced flatten"):
            policy(forced_flatten_hours=40)

    def test_risk_increase_with_flat_demand_is_invalid(self):
        with pytest.raises(SessionPolicyError, match="invalid"):
            policy(allow_risk_increase_during_wind_down=True)

    def test_positive_integer_stability_counts(self):
        with pytest.raises(SessionPolicyError, match="positive"):
            policy(stability_consecutive_checks=0)
        with pytest.raises(SessionPolicyError, match="positive"):
            policy(reopen_min_closed_bars=True)

    def test_unknown_and_missing_fields_refuse(self):
        with pytest.raises(SessionPolicyError, match="unknown"):
            validate_policy({**policy(), "extra": 1})
        with pytest.raises(SessionPolicyError, match="missing"):
            validate_policy({"enabled": True})


class TestStateMachine:
    def test_normal_far_from_closure(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=100)))
        assert block["state"] == "NORMAL_TRADING"
        assert block["time_to_next_close_hours"] == 100.0

    def test_wind_down_window(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=20)))
        assert block["state"] == "WIND_DOWN"
        assert block["wind_down"] and not block["forced_flatten"]

    def test_forced_flatten_window(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=2)))
        assert block["state"] == "FORCED_FLATTEN"
        assert block["forced_flatten"]

    def test_expected_market_closed(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE + timedelta(hours=30)))
        assert block["state"] == "EXPECTED_MARKET_CLOSED"

    def test_reopen_blackout_until_stability(self):
        after = SUNDAY_REOPEN + timedelta(hours=1)
        block = session_state(policy(), SessionEvidence(
            now=after, closures=CLOSURES,
            time_since_reopen_hours=1.0,
            closed_bars_since_reopen=0,
            stability_checks_passed=0))
        assert block["state"] == "REOPEN_BLACKOUT"
        # both minimum time/bars AND stability must pass to exit
        block2 = session_state(policy(), SessionEvidence(
            now=SUNDAY_REOPEN + timedelta(hours=5),
            closures=[(FRIDAY_CLOSE + timedelta(days=7),
                       SUNDAY_REOPEN + timedelta(days=7))],
            time_since_reopen_hours=5.0,
            closed_bars_since_reopen=2,
            stability_checks_passed=3))
        assert block2["state"] == "NORMAL_TRADING"

    def test_missing_session_evidence_fails_closed(self):
        block = session_state(policy(), SessionEvidence(
            now=FRIDAY_CLOSE - timedelta(hours=100),
            closures=[], evidence_ok=False))
        assert block["state"] == "WIND_DOWN"
        assert block["evidence_failed_closed"] is True

    def test_disabled_policy_passes_through(self):
        block = session_state(policy(enabled=False), evidence(
            FRIDAY_CLOSE - timedelta(hours=1)))
        assert block["state"] == "NORMAL_TRADING"
        assert block["policy_enabled"] is False

    def test_restart_determinism_same_inputs_same_state(self):
        a = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=20)))
        b = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=20)))
        assert a == b


class TestOverlay:
    def test_wind_down_masks_entries_and_cancels_pendings(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=20)))
        decision = overlay_action(
            policy(), block,
            ExposureFacts(open_position=False, pending_orders=2),
            raw_action=0.7)
        assert decision["overlay"] == "masked_risk_increase"
        assert decision["final_action"] == 0.0
        assert decision["cancel_pending"] is True
        assert decision["raw_model_action"] == 0.7  # recorded raw

    def test_wind_down_allows_model_close_long_and_short(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=20)))
        for side_action in (0.0, -0.0):
            decision = overlay_action(
                policy(), block,
                ExposureFacts(open_position=True),
                raw_action=side_action)
            assert decision["overlay"] == "pass_through"

    def test_forced_flatten_closes_open_position(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=2)))
        decision = overlay_action(
            policy(), block, ExposureFacts(open_position=True,
                                           pending_orders=1),
            raw_action=0.9)
        assert decision["overlay"] == "forced_close"
        assert decision["final_action"] == "CLOSE"
        assert decision["cancel_pending"] is True

    def test_closed_market_no_actionable_step(self):
        block = session_state(policy(), evidence(
            FRIDAY_CLOSE + timedelta(hours=10)))
        decision = overlay_action(policy(), block, ExposureFacts(),
                                  raw_action=0.5)
        assert decision["overlay"] == "no_actionable_step"
        assert decision["final_action"] is None

    def test_blackout_masks_entries_only(self):
        block = {"state": "REOPEN_BLACKOUT", "evidence_ok": True}
        entry = overlay_action(policy(), block, ExposureFacts(),
                               raw_action=1.0)
        assert entry["overlay"] == "masked_entry_during_blackout"
        carried = overlay_action(
            policy(), block, ExposureFacts(open_position=True),
            raw_action=0.0)
        assert carried["overlay"] == "pass_through"


class TestReconciliationAndWatchdog:
    def test_flatten_success_requires_fresh_zero_zero(self):
        ok = reconciliation_gate(0, 0, evidence_age_seconds=10)
        assert ok["flat_confirmed"] and ok["incident"] is None
        stale = reconciliation_gate(0, 0, evidence_age_seconds=999)
        assert not stale["flat_confirmed"]
        assert "stale" in stale["incident"]
        exposed = reconciliation_gate(1, 0, evidence_age_seconds=5)
        assert "exposure remains" in exposed["incident"]

    def test_expected_weekend_vs_tuesday_stale_feed(self):
        closed = session_state(policy(), evidence(
            FRIDAY_CLOSE + timedelta(hours=30)))
        assert watchdog_state(closed, bars_fresh=False,
                              terminal_connected=True,
                              exposure=ExposureFacts()) == \
            "EXPECTED_MARKET_CLOSED"
        tuesday = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=80)))
        assert watchdog_state(tuesday, bars_fresh=False,
                              terminal_connected=True,
                              exposure=ExposureFacts()) == \
            "FEED_STALE_DURING_OPEN_WINDOW"

    def test_closure_never_suppresses_terminal_or_exposure(self):
        closed = session_state(policy(), evidence(
            FRIDAY_CLOSE + timedelta(hours=30)))
        assert watchdog_state(closed, bars_fresh=False,
                              terminal_connected=False,
                              exposure=ExposureFacts()) == \
            "TERMINAL_DISCONNECTED"
        wind = session_state(policy(), evidence(
            FRIDAY_CLOSE - timedelta(hours=2)))
        assert watchdog_state(wind, bars_fresh=True,
                              terminal_connected=True,
                              exposure=ExposureFacts(
                                  open_position=True)) == \
            "WIND_DOWN_EXPOSURE_PRESENT"

    def test_holiday_uses_same_machine(self):
        holiday = [(datetime(2026, 9, 7, 13, 0, tzinfo=UTC),
                    datetime(2026, 9, 8, 13, 0, tzinfo=UTC))]
        block = session_state(policy(), SessionEvidence(
            now=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
            closures=holiday))
        assert block["state"] == "FORCED_FLATTEN"
