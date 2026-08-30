"""Work plan 42 state machine — corrected under order @e303e386.

Every counterexample the auditor reproduced against gym-fx@bec4d1a is
a PERMANENT regression here (marked AUDIT-PRE)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.session_exposure import (  # noqa: E402
    CarriedPositionMigration, ExposureFacts, MigrationLedger,
    ReopenEvidence, SessionCalendar, SessionEvidenceError,
    SessionPolicyError, classify_action, overlay_action,
    reconciliation_gate, session_state, validate_policy,
    watchdog_state)

UTC = timezone.utc
CLOSE = datetime(2026, 8, 28, 17, 0, tzinfo=UTC)
REOPEN = datetime(2026, 8, 30, 21, 0, tzinfo=UTC)
NEXT_CLOSE = datetime(2026, 9, 4, 17, 0, tzinfo=UTC)
NEXT_REOPEN = datetime(2026, 9, 6, 21, 0, tzinfo=UTC)


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
        "calendar_identity": "cal-digest-abc",
    }
    base.update(overrides)
    return validate_policy(base)


def calendar(intervals=None, digest="cal-digest-abc",
             symbol="ETHUSD"):
    return SessionCalendar.build(
        venue="mt5_demo", account_fingerprint="fp-1234",
        symbol=symbol, calendar_digest=digest,
        intervals=intervals or [(CLOSE, REOPEN),
                                (NEXT_CLOSE, NEXT_REOPEN)])


def fresh_reopen(bars=2, checks=3, hint=None):
    return ReopenEvidence.build(closed_bars_since_reopen=bars,
                                stability_checks_passed=checks,
                                hint_time_since_reopen_hours=hint)


def flat():
    return ExposureFacts.build(signed_exposure=0.0)


def long_position(size=1.0, pending=0):
    return ExposureFacts.build(signed_exposure=size,
                               pending_orders=pending)


def short_position(size=1.0):
    return ExposureFacts.build(signed_exposure=-size)


def migration_record(symbol="ETHUSD", venue="mt5_demo",
                     account="fp-1234", position="pos-1",
                     protected=True, closure=None):
    return CarriedPositionMigration.build(
        migration_id="mig-2026-08-28-eth", venue=venue,
        account_fingerprint=account, symbol=symbol,
        position_identity=position,
        opened_before=CLOSE - timedelta(hours=6),
        covers_closure_started_at=closure or CLOSE,
        native_protection_confirmed=protected)


# ============ FINAL-AUDIT bypasses (order @8acade57) ============ #

class TestFinalAuditBypasses:
    """The four bypasses reproduced against gym-fx@9915138."""

    def test_f1_direct_constructors_cannot_build_invalid(self):
        """WAS: ExposureFacts(nan, 'long', None, -9, -4, 'bad')
        constructed successfully."""
        with pytest.raises((SessionPolicyError,
                            SessionEvidenceError)):
            ExposureFacts(float("nan"), "long", None, -9, -4, "bad")
        with pytest.raises((SessionPolicyError,
                            SessionEvidenceError)):
            ExposureFacts(1.0, "short", None, 0.0, 0,
                          "target_exposure_v2")
        with pytest.raises((SessionPolicyError,
                            SessionEvidenceError)):
            ReopenEvidence(-1, 3, None)
        with pytest.raises((SessionPolicyError,
                            SessionEvidenceError)):
            SessionCalendar("mt5_demo", "fp", "ETHUSD", "d",
                            ((REOPEN, CLOSE),))
        with pytest.raises((SessionPolicyError,
                            SessionEvidenceError)):
            CarriedPositionMigration(
                "m", "v", "a", "s", "p", CLOSE,
                CLOSE - timedelta(hours=1), True)

    def test_f2_mismatched_calendar_identity_refuses(self):
        """WAS: a policy declaring cal-A accepted a cal-DIFFERENT
        calendar and published the latter as authoritative."""
        with pytest.raises(SessionEvidenceError, match="mismatch"):
            session_state(policy(), now=CLOSE - timedelta(hours=20),
                          calendar=calendar(digest="cal-DIFFERENT"),
                          reopen_evidence=fresh_reopen())

    def test_f2_cross_symbol_and_account_refuse(self):
        with pytest.raises(SessionEvidenceError, match="symbol"):
            session_state(policy(), now=CLOSE - timedelta(hours=20),
                          calendar=calendar(),
                          reopen_evidence=fresh_reopen(),
                          expected_symbol="BTCUSD")
        with pytest.raises(SessionEvidenceError, match="account"):
            session_state(policy(), now=CLOSE - timedelta(hours=20),
                          calendar=calendar(),
                          reopen_evidence=fresh_reopen(),
                          expected_account_fingerprint="fp-OTHER")

    def test_f3_disabled_cancellation_is_unmaterializable(self):
        """WAS: enabled policy with cancel_pending_on_wind_down=false
        left a pending long entry alive through wind-down."""
        with pytest.raises(SessionPolicyError, match="invalid while"):
            policy(cancel_pending_on_wind_down=False)

    def test_f3_entry_orders_cancelled_brackets_preserved(self):
        block = session_state(policy(),
                              now=CLOSE - timedelta(hours=20),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        # one pending ENTRY + one protective bracket
        exposure = ExposureFacts.build(
            signed_exposure=1.0, pending_entry_side="long",
            pending_entry_size=0.5, pending_orders=2,
            protective_orders=1)
        assert exposure.entry_orders == 1
        decision = overlay_action(policy(), block, exposure,
                                  raw_action=1.0)
        assert decision["cancel_pending"] is True
        assert decision["cancel_scope"] == "pending_entry_orders_only"
        # a position protected ONLY by brackets cancels nothing
        protected_only = ExposureFacts.build(
            signed_exposure=1.0, pending_orders=2,
            protective_orders=2)
        assert protected_only.entry_orders == 0
        quiet = overlay_action(policy(), block, protected_only,
                               raw_action=1.0)
        assert quiet["cancel_pending"] is False

    def test_f4_migration_is_one_use(self):
        """WAS: the same record returned RECOVERY_ACTIVE repeatedly
        for the same closure and for other symbols."""
        closed = session_state(policy(),
                               now=CLOSE + timedelta(hours=30),
                               calendar=calendar())
        ledger = MigrationLedger()
        migration = migration_record()
        first = watchdog_state(
            closed, bars_fresh=False, terminal_connected=True,
            exposure=short_position(), carried_migration=migration,
            migration_ledger=ledger, position_identity="pos-1",
            now=CLOSE + timedelta(hours=30))
        assert first == "CARRIED_POSITION_RECOVERY_ACTIVE"
        assert ledger.is_consumed(migration.migration_id)

    def test_f4_wrong_symbol_account_position_refuse(self):
        closed = session_state(policy(),
                               now=CLOSE + timedelta(hours=30),
                               calendar=calendar())
        for record, position in (
                (migration_record(symbol="BTCUSD"), "pos-1"),
                (migration_record(account="fp-OTHER"), "pos-1"),
                (migration_record(), "pos-OTHER"),
                (migration_record(protected=False), "pos-1")):
            assert watchdog_state(
                closed, bars_fresh=False, terminal_connected=True,
                exposure=short_position(), carried_migration=record,
                migration_ledger=MigrationLedger(),
                position_identity=position,
                now=CLOSE + timedelta(hours=30)) == \
                "UNEXPECTED_EXPOSURE_DURING_CLOSURE"

    def test_f4_ledger_refuses_reuse_across_closures(self):
        ledger = MigrationLedger()
        ledger.consume("mig-1", CLOSE.isoformat())
        with pytest.raises(SessionEvidenceError, match="one-use"):
            ledger.consume("mig-1", NEXT_CLOSE.isoformat())


# ================= AUDIT-PRE counterexamples ==================== #

class TestAuditPreCounterexamples:
    """The six facts reproduced against gym-fx@bec4d1a."""

    def test_pre1_string_numeric_now_refuses(self):
        with pytest.raises(SessionPolicyError, match="finite real"):
            policy(wind_down_hours="36")

    def test_pre2_nan_now_refuses(self):
        with pytest.raises(SessionPolicyError, match="non-finite"):
            policy(max_gap_sigma=float("nan"))
        with pytest.raises(SessionPolicyError, match="non-finite"):
            policy(max_spread_relative_to_baseline=float("inf"))

    def test_pre3_open_position_reversal_is_masked(self):
        """WAS: raw_action=-1.0 with an open position passed through
        WIND_DOWN untouched."""
        block = session_state(policy(), now=CLOSE - timedelta(hours=20),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        assert block["state"] == "WIND_DOWN"
        decision = overlay_action(policy(), block, long_position(1.0),
                                  raw_action=-1.0)
        assert decision["mapped_action"]["kind"] == "reversal"
        assert decision["overlay"] == "masked_risk_increase"
        assert decision["final_action"] == 1.0  # holds, never flips

    def test_pre4_closed_market_with_exposure_is_not_healthy(self):
        """WAS: one position + one pending order during closure
        reported only EXPECTED_MARKET_CLOSED."""
        block = session_state(policy(), now=CLOSE + timedelta(hours=30),
                              calendar=calendar())
        assert block["state"] == "EXPECTED_MARKET_CLOSED"
        assert watchdog_state(
            block, bars_fresh=False, terminal_connected=True,
            exposure=long_position(1.0, pending=1)) == \
            "UNEXPECTED_EXPOSURE_DURING_CLOSURE"

    def test_pre5_one_hour_after_reopen_is_blackout(self):
        """WAS: NORMAL_TRADING one hour after reopen when the adapter
        hint was absent."""
        block = session_state(policy(),
                              now=REOPEN + timedelta(hours=1),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        assert block["state"] == "REOPEN_BLACKOUT"
        assert block["time_since_reopen_hours"] == 1.0
        # and with NO reopen evidence at all it still fails closed
        missing = session_state(policy(),
                                now=REOPEN + timedelta(hours=1),
                                calendar=calendar(),
                                reopen_evidence=None)
        assert missing["state"] == "REOPEN_BLACKOUT"
        assert missing["reopen_evidence_missing"] is True

    def test_pre6_unavailable_reconciliation_is_typed(self):
        """WAS: raw TypeError as policy behavior."""
        with pytest.raises(SessionPolicyError, match="integer count"):
            reconciliation_gate(None, 0, evidence_age_seconds=5)
        with pytest.raises(SessionPolicyError, match="integer count"):
            reconciliation_gate(0, True, evidence_age_seconds=5)
        with pytest.raises(SessionPolicyError, match="finite real"):
            reconciliation_gate(0, 0, evidence_age_seconds="5")


# ================= C1 exposure classification =================== #

class TestSignedExposure:
    def test_long_enlargement_and_reduction_and_close(self):
        pos = long_position(1.0)
        assert classify_action(1.5, pos)["kind"] == "enlargement"
        assert classify_action(1.5, pos)["risk_increasing"] is True
        assert classify_action(0.5, pos)["kind"] == "reduction"
        assert classify_action(0.5, pos)["risk_increasing"] is False
        assert classify_action(0.0, pos)["kind"] == "close"
        assert classify_action(1.0, pos)["kind"] == "hold"

    def test_short_mirror_cases(self):
        pos = short_position(1.0)
        assert classify_action(-1.5, pos)["kind"] == "enlargement"
        assert classify_action(-0.5, pos)["kind"] == "reduction"
        assert classify_action(0.7, pos)["kind"] == "reversal"
        assert classify_action(0.7, pos)["risk_increasing"] is True

    def test_entry_from_flat_is_risk_increasing(self):
        assert classify_action(0.3, flat())["kind"] == \
            "entry_from_flat"
        assert classify_action(0.0, flat())["kind"] == "hold_flat"

    def test_ambiguous_and_nonfinite_actions_refuse(self):
        for bad in (None, "0.5", True, float("nan"), float("inf")):
            with pytest.raises(SessionEvidenceError):
                classify_action(bad, flat())

    def test_side_contradicting_exposure_refuses(self):
        with pytest.raises(SessionEvidenceError, match="contradicts"):
            ExposureFacts.build(signed_exposure=1.0, side="short")

    def test_pending_entry_requires_size_and_order(self):
        with pytest.raises(SessionEvidenceError, match="zero size"):
            ExposureFacts.build(signed_exposure=0.0,
                                pending_entry_side="long",
                                pending_entry_size=0.0,
                                pending_orders=1)
        with pytest.raises(SessionEvidenceError,
                           match="zero pending orders"):
            ExposureFacts.build(signed_exposure=0.0,
                                pending_entry_side="long",
                                pending_entry_size=0.5,
                                pending_orders=0)

    def test_partial_fill_exposure_is_representable(self):
        partial = ExposureFacts.build(signed_exposure=0.4,
                                      pending_entry_side="long",
                                      pending_entry_size=0.6,
                                      pending_orders=1)
        assert partial.side == "long"
        assert classify_action(0.4, partial)["kind"] == "hold"
        assert classify_action(1.0, partial)["kind"] == "enlargement"


# ================= C2 calendar-derived state ==================== #

class TestCalendarAuthority:
    def test_intervals_must_be_ordered_utc_nonoverlapping(self):
        with pytest.raises(SessionEvidenceError, match="precede"):
            calendar([(REOPEN, CLOSE)])
        with pytest.raises(SessionEvidenceError, match="aware"):
            calendar([(CLOSE.replace(tzinfo=None), REOPEN)])
        with pytest.raises(SessionEvidenceError, match="overlap"):
            calendar([(CLOSE, REOPEN),
                      (CLOSE + timedelta(hours=1),
                       REOPEN + timedelta(hours=5))])

    def test_server_time_converted_to_utc(self):
        offset = timezone(timedelta(hours=3))
        cal = calendar([(CLOSE.astimezone(offset),
                         REOPEN.astimezone(offset))])
        assert cal.intervals[0][0] == CLOSE
        assert cal.intervals[0][1].tzinfo == UTC

    def test_exact_boundaries(self):
        cal = calendar()
        at_close = session_state(policy(), now=CLOSE, calendar=cal)
        assert at_close["state"] == "EXPECTED_MARKET_CLOSED"
        at_reopen = session_state(policy(), now=REOPEN, calendar=cal,
                                  reopen_evidence=fresh_reopen())
        assert at_reopen["state"] == "REOPEN_BLACKOUT"

    def test_blackout_exits_only_after_all_predicates(self):
        late = REOPEN + timedelta(hours=5)
        assert session_state(policy(), now=late, calendar=calendar(),
                             reopen_evidence=fresh_reopen(bars=0)
                             )["state"] == "REOPEN_BLACKOUT"
        assert session_state(policy(), now=late, calendar=calendar(),
                             reopen_evidence=fresh_reopen(checks=1)
                             )["state"] == "REOPEN_BLACKOUT"
        assert session_state(policy(), now=late, calendar=calendar(),
                             reopen_evidence=fresh_reopen()
                             )["state"] == "NORMAL_TRADING"

    def test_adapter_hint_cannot_authorize_trading(self):
        """A lying hint (claiming 100h since reopen) does not shorten
        the blackout: authority is the bound interval."""
        block = session_state(
            policy(), now=REOPEN + timedelta(hours=1),
            calendar=calendar(),
            reopen_evidence=fresh_reopen(hint=100.0))
        assert block["state"] == "REOPEN_BLACKOUT"
        assert block["adapter_hint_disagrees"] is True

    def test_holiday_adjacency_uses_same_machine(self):
        holiday = (datetime(2026, 9, 7, 13, 0, tzinfo=UTC),
                   datetime(2026, 9, 8, 13, 0, tzinfo=UTC))
        cal = calendar([(CLOSE, REOPEN), (NEXT_CLOSE, NEXT_REOPEN),
                        holiday])
        block = session_state(policy(),
                              now=holiday[0] - timedelta(hours=2),
                              calendar=cal,
                              reopen_evidence=fresh_reopen())
        assert block["state"] == "FORCED_FLATTEN"

    def test_stale_calendar_absent_evidence_fails_closed(self):
        block = session_state(policy(), now=CLOSE - timedelta(days=5),
                              calendar=None)
        assert block["state"] == "WIND_DOWN"
        assert block["evidence_failed_closed"] is True

    def test_restart_determinism(self):
        args = dict(now=CLOSE - timedelta(hours=20),
                    calendar=calendar(),
                    reopen_evidence=fresh_reopen())
        assert session_state(policy(), **args) == \
            session_state(policy(), **args)


# ================= C3 watchdog precedence ======================= #

class TestWatchdogPrecedence:
    def _closed(self):
        return session_state(policy(), now=CLOSE + timedelta(hours=30),
                             calendar=calendar())

    def test_closure_suppresses_only_bar_staleness(self):
        assert watchdog_state(self._closed(), bars_fresh=False,
                              terminal_connected=True,
                              exposure=flat()) == \
            "EXPECTED_MARKET_CLOSED"

    def test_terminal_account_bracket_take_precedence(self):
        closed = self._closed()
        assert watchdog_state(closed, bars_fresh=False,
                              terminal_connected=False,
                              exposure=flat()) == \
            "TERMINAL_DISCONNECTED"
        assert watchdog_state(closed, bars_fresh=False,
                              terminal_connected=True,
                              exposure=flat(),
                              account_ok=False) == \
            "ACCOUNT_OR_BRACKET_FAULT"
        assert watchdog_state(closed, bars_fresh=False,
                              terminal_connected=True,
                              exposure=flat(),
                              brackets_ok=False) == \
            "ACCOUNT_OR_BRACKET_FAULT"

    def test_flatten_incident_takes_precedence(self):
        block = session_state(policy(),
                              now=CLOSE - timedelta(hours=2),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        assert watchdog_state(
            block, bars_fresh=True, terminal_connected=True,
            exposure=long_position(),
            flatten_incident="FORCED_FLATTEN_FAILED: exposure remains"
        ) == "FORCED_FLATTEN_FAILED"

    def test_carried_position_recovery_is_one_use_and_scoped(self):
        closed = self._closed()
        migration = migration_record()
        ledger = MigrationLedger()
        assert watchdog_state(
            closed, bars_fresh=False, terminal_connected=True,
            exposure=short_position(), carried_migration=migration,
            migration_ledger=ledger, position_identity="pos-1",
            now=CLOSE + timedelta(hours=30)) == \
            "CARRIED_POSITION_RECOVERY_ACTIVE"
        # a FUTURE closure is NOT normalized by the same record
        future_block = session_state(
            policy(), now=NEXT_CLOSE + timedelta(hours=5),
            calendar=calendar())
        assert watchdog_state(
            future_block, bars_fresh=False, terminal_connected=True,
            exposure=short_position(), carried_migration=migration,
            migration_ledger=ledger, position_identity="pos-1",
            now=NEXT_CLOSE + timedelta(hours=5)) == \
            "UNEXPECTED_EXPOSURE_DURING_CLOSURE"

    def test_open_window_staleness_still_alerts(self):
        block = session_state(policy(), now=CLOSE - timedelta(hours=80),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        assert watchdog_state(block, bars_fresh=False,
                              terminal_connected=True,
                              exposure=flat()) == \
            "FEED_STALE_DURING_OPEN_WINDOW"


# ================= overlay + reconciliation ===================== #

class TestOverlayAndReconciliation:
    def test_forced_flatten_closes_and_cancels(self):
        block = session_state(policy(), now=CLOSE - timedelta(hours=2),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        decision = overlay_action(policy(), block,
                                  long_position(1.0, pending=1),
                                  raw_action=0.9)
        assert decision["overlay"] == "forced_close"
        assert decision["final_action"] == "CLOSE"
        assert decision["cancel_pending"] is True

    def test_closed_market_produces_no_actionable_step(self):
        block = session_state(policy(), now=CLOSE + timedelta(hours=10),
                              calendar=calendar())
        decision = overlay_action(policy(), block, flat(),
                                  raw_action=0.5)
        assert decision["overlay"] == "no_actionable_step"
        assert decision["final_action"] is None
        assert decision["mapped_action"] is None

    def test_reduction_survives_wind_down(self):
        block = session_state(policy(), now=CLOSE - timedelta(hours=20),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        decision = overlay_action(policy(), block, long_position(1.0),
                                  raw_action=0.3)
        assert decision["overlay"] == "pass_through"
        assert decision["final_action"] == 0.3

    def test_reconciliation_requires_fresh_zero_zero(self):
        assert reconciliation_gate(
            0, 0, evidence_age_seconds=10)["flat_confirmed"] is True
        assert "stale" in reconciliation_gate(
            0, 0, evidence_age_seconds=999)["incident"]
        assert "exposure remains" in reconciliation_gate(
            1, 0, evidence_age_seconds=5)["incident"]

    def test_raw_mapped_overlay_final_are_separate_facts(self):
        block = session_state(policy(), now=CLOSE - timedelta(hours=20),
                              calendar=calendar(),
                              reopen_evidence=fresh_reopen())
        decision = overlay_action(policy(), block, flat(),
                                  raw_action=0.8)
        assert decision["raw_model_action"] == 0.8
        assert decision["mapped_action"]["kind"] == "entry_from_flat"
        assert decision["overlay"] == "masked_risk_increase"
        assert decision["final_action"] == 0.0
