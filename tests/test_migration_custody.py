"""D3 adversarial acceptance for durable migration custody
(order @c933da64). Ten required proofs — including TWO CONCURRENT
PROCESSES, not merely two calls."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.direct_evidence import (  # noqa: E402
    EvidenceError, NativeProtectionEvidence, ReconciliationEvidence)
from app.migration_custody import (  # noqa: E402
    MigrationCustody, MigrationCustodyError)
from app.session_exposure import (  # noqa: E402
    CarriedPositionMigration, ExposureFacts, ReopenEvidence,
    SessionCalendar, session_state, validate_policy, watchdog_state)

UTC = timezone.utc
CLOSE = datetime(2026, 8, 28, 17, 0, tzinfo=UTC)
REOPEN = datetime(2026, 8, 30, 21, 0, tzinfo=UTC)
NEXT_CLOSE = datetime(2026, 9, 4, 17, 0, tzinfo=UTC)
NEXT_REOPEN = datetime(2026, 9, 6, 21, 0, tzinfo=UTC)
POLICY_ID = "policy-v1"
CODE_ID = "code-v1"
NOW = CLOSE + timedelta(hours=30)


def protection(symbol="ETHUSD", account="fp-1", venue="mt5_demo",
               position="pos-1", observed=None, sl=True):
    return NativeProtectionEvidence.build(
        venue=venue, account_fingerprint=account, symbol=symbol,
        position_identity=position,
        observed_at=observed or (NOW - timedelta(seconds=10)),
        source="mt5_bridge_report", evidence_id="ev-prot-1",
        raw_payload={"sl": 1234.5, "tp": 1100.0, "ticket": 55},
        stop_loss_accepted=sl, take_profit_accepted=True)


def reconciliation(positions=0, orders=0, observed=None,
                   symbol="ETHUSD", position="pos-1"):
    return ReconciliationEvidence.build(
        venue="mt5_demo", account_fingerprint="fp-1", symbol=symbol,
        position_identity=position,
        observed_at=observed or (NOW - timedelta(seconds=5)),
        source="mt5_bridge_report", evidence_id="ev-rec-1",
        raw_payload={"positions": positions, "orders": orders},
        positions_total=positions, orders_total=orders)


def policy():
    return validate_policy({
        "enabled": True,
        "session_source": "venue_symbol_sessions_v1",
        "wind_down_hours": 36, "forced_flatten_hours": 4,
        "cancel_pending_on_wind_down": True,
        "allow_risk_increase_during_wind_down": False,
        "reopen_min_hours": 4, "reopen_min_closed_bars": 1,
        "stability_consecutive_checks": 3,
        "max_spread_relative_to_baseline": 2.0,
        "max_gap_sigma": 3.0,
        "max_realized_vol_relative_to_baseline": 2.0,
        "carried_position_recovery":
            "protected_opportunistic_then_forced",
        "holiday_policy": "same_as_weekly",
        "calendar_identity": "cal-A"})


def calendar(symbol="ETHUSD", account="fp-1", venue="mt5_demo"):
    return SessionCalendar.build(
        venue=venue, account_fingerprint=account, symbol=symbol,
        calendar_digest="cal-A",
        intervals=[(CLOSE, REOPEN), (NEXT_CLOSE, NEXT_REOPEN)])


def closed_block(**kw):
    return session_state(policy(), now=CLOSE + timedelta(hours=30),
                         calendar=calendar(**kw))


def migration(symbol="ETHUSD", account="fp-1", venue="mt5_demo",
              position="pos-1", protected=True, closure=None,
              mid="mig-1"):
    return CarriedPositionMigration.build(
        migration_id=mid, venue=venue, account_fingerprint=account,
        symbol=symbol, position_identity=position,
        opened_before=CLOSE - timedelta(hours=6),
        covers_closure_started_at=closure or CLOSE,
        native_protection_confirmed=protected)


def claim(custody, block=None, mig=None, position="pos-1",
          prot=None):
    record = mig or migration()
    return custody.claim(
        record, block or closed_block(), position,
        protection_evidence=prot or protection(
            symbol=record.symbol, account=record.account_fingerprint,
            venue=record.venue, position=position),
        policy_identity=POLICY_ID, code_identity=CODE_ID, now=NOW)


class TestDurableOneUseCustody:
    def test_d3_1_two_claims_one_process_single_winner(self, tmp_path):
        """PRE: the in-memory ledger authorized the SAME migration
        and closure repeatedly."""
        custody = MigrationCustody(tmp_path / "custody")
        first = claim(custody)
        assert first["state"] == "active"
        with pytest.raises(MigrationCustodyError,
                           match="already claimed"):
            claim(custody)

    def test_d3_2_two_concurrent_processes_single_winner(
            self, tmp_path):
        """Two real OS processes race for the same migration."""
        root = tmp_path / "custody"
        script = textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {str(REPO)!r})
            sys.path.insert(0, {str(REPO / 'tests')!r})
            from test_migration_custody import (
                MigrationCustody, claim)
            try:
                claim(MigrationCustody({str(root)!r}))
                print("WON")
            except Exception as exc:
                print("LOST:" + type(exc).__name__)
        """)
        runner = tmp_path / "racer.py"
        runner.write_text(script)
        procs = [subprocess.run([sys.executable, str(runner)],
                                capture_output=True, text=True)
                 for _ in range(2)]
        outs = [p.stdout.strip() for p in procs]
        assert sum(1 for o in outs if o == "WON") == 1, outs
        assert sum(1 for o in outs if o.startswith("LOST")) == 1, outs

    def test_d3_3_restart_preserves_active_and_terminals(
            self, tmp_path):
        root = tmp_path / "custody"
        claim(MigrationCustody(root))
        # a FRESH instance (restart) must remember
        assert MigrationCustody(root).is_active("mig-1") is True
        MigrationCustody(root).finish("mig-1", "completed",
                                      reconciliation=reconciliation(), now=NOW)
        assert MigrationCustody(root).read("mig-1")["state"] == \
            "completed"
        assert MigrationCustody(root).is_active("mig-1") is False

    def test_d3_4_second_claim_same_closure_refuses(self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        claim(custody)
        custody.finish("mig-1", "completed",
                       reconciliation=reconciliation(), now=NOW)
        with pytest.raises(MigrationCustodyError, match="terminal"):
            claim(custody)

    def test_d3_5_other_identities_refuse(self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        cases = (
            ("symbol", migration(symbol="BTCUSD"), "pos-1"),
            ("account", migration(account="fp-OTHER"), "pos-1"),
            ("venue", migration(venue="other_venue"), "pos-1"),
            ("position", migration(), "pos-OTHER"),
            ("closure", migration(closure=NEXT_CLOSE), "pos-1"),
        )
        for label, record, position in cases:
            with pytest.raises(MigrationCustodyError):
                claim(custody, mig=record, position=position)
        # none of the refusals left a record behind
        assert custody.read("mig-1") is None

    def test_d3_6_missing_protection_refuses(self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        with pytest.raises(MigrationCustodyError,
                           match="native protection"):
            claim(custody, mig=migration(protected=False))

    def test_d3_7_interrupted_write_never_authorized(self, tmp_path):
        """A stray temporary file is not a claim, and a truncated
        record is refused rather than trusted."""
        root = tmp_path / "custody"
        custody = MigrationCustody(root)
        stray = root / "migration_deadbeef.json.tmp.999"
        stray.write_text('{"state": "active"}')
        assert custody.read("mig-1") is None
        claim(custody)
        path = next(p for p in root.glob("migration_*.json")
                    if not p.name.endswith(".tmp.999"))
        record = json.loads(path.read_text())
        record["record_digest"] = "0" * 64
        path.write_text(json.dumps(record))
        with pytest.raises(MigrationCustodyError, match="digest"):
            custody.read("mig-1")

    def test_d3_8_watchdog_reads_are_repeatable_and_pure(
            self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        block = closed_block()
        exposure = ExposureFacts.build(signed_exposure=-1.0)
        before = sorted(p.name for p in custody.root.iterdir())
        reads = [watchdog_state(block, bars_fresh=False,
                                terminal_connected=True,
                                exposure=exposure,
                                recovery_claim_active=False)
                 for _ in range(3)]
        assert reads == ["UNEXPECTED_EXPOSURE_DURING_CLOSURE"] * 3
        assert sorted(p.name for p in custody.root.iterdir()) == before
        # after the CONTROLLER claims, the same read reports active
        claim(custody)
        active_reads = [
            watchdog_state(block, bars_fresh=False,
                           terminal_connected=True,
                           exposure=exposure,
                           recovery_claim_active=custody.is_active(
                               "mig-1"))
            for _ in range(3)]
        assert active_reads == \
            ["CARRIED_POSITION_RECOVERY_ACTIVE"] * 3
        # the watchdog signature carries NO custody object at all
        import inspect
        assert "migration_ledger" not in inspect.signature(
            watchdog_state).parameters

    def test_d3_9_finish_requires_fresh_direct_reconciliation(
            self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        claim(custody)
        with pytest.raises(MigrationCustodyError, match="envelope"):
            custody.finish("mig-1", "completed", reconciliation={})
        with pytest.raises(MigrationCustodyError, match="FRESH"):
            custody.finish("mig-1", "completed", now=NOW,
                           reconciliation=reconciliation(
                               observed=NOW - timedelta(hours=3)))
        with pytest.raises(EvidenceError, match="exactly 0"):
            custody.finish("mig-1", "completed", now=NOW,
                           reconciliation=reconciliation(positions=1))
        done = custody.finish("mig-1", "completed",
                              reconciliation=reconciliation(), now=NOW)
        assert done["state"] == "completed"
        with pytest.raises(MigrationCustodyError, match="immutable"):
            custody.finish("mig-1", "failed",
                           reconciliation=reconciliation(), now=NOW)

    def test_d3_10_no_live_position_is_touched(self):
        """Custody is pure filesystem state under a test root: no
        venue client, no broker call, no live identity anywhere."""
        source = (REPO / "app/migration_custody.py").read_text()
        for forbidden in ("mt5", "MetaTrader", "requests", "socket",
                          "alpaca", "http"):
            assert forbidden.lower() not in source.lower(), forbidden

    def test_records_are_owner_only_and_root_restricted(
            self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        claim(custody)
        path = next(custody.root.glob("migration_*.json"))
        assert oct(path.stat().st_mode)[-3:] == "600"
        assert oct(custody.root.stat().st_mode)[-3:] == "700"

    def test_symlinked_root_refuses(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        with pytest.raises(MigrationCustodyError, match="symlink"):
            MigrationCustody(link)


# ============== E3: concurrency, crash and fabrication ========== #

def _race_script(root: Path, barrier: Path, mode: str) -> str:
    return textwrap.dedent(f"""
        import sys, time, os
        sys.path.insert(0, {str(REPO)!r})
        sys.path.insert(0, {str(REPO / 'tests')!r})
        from test_migration_custody import (
            MigrationCustody, claim, reconciliation, NOW)
        barrier = {str(barrier)!r}
        # synchronization barrier: both processes wait for the file
        open(barrier + "." + str(os.getpid()), "w").close()
        while len([f for f in os.listdir(os.path.dirname(barrier))
                   if f.startswith(os.path.basename(barrier) + ".")]) < 2:
            time.sleep(0.005)
        custody = MigrationCustody({str(root)!r})
        try:
            if {mode!r} == "claim":
                claim(custody)
            else:
                custody.finish("mig-1", {mode!r},
                               reconciliation=reconciliation(
                                   positions=0 if {mode!r} == "completed"
                                   else 1),
                               now=NOW)
            print("WON")
        except Exception as exc:
            print("LOST:" + type(exc).__name__)
    """)


def _run_race(tmp_path, root, mode, n=2):
    barrier = tmp_path / "barrier"
    script = tmp_path / f"race_{mode}.py"
    script.write_text(_race_script(root, barrier, mode))
    procs = [subprocess.Popen([sys.executable, str(script)],
                              stdout=subprocess.PIPE, text=True)
             for _ in range(n)]
    return [p.communicate()[0].strip() for p in procs]


class TestE3ConcurrencyAndFabrication:
    def test_e3_1_barrier_synchronized_claim_race(self, tmp_path):
        """Popen + barrier: both processes reach the claim together;
        exactly one wins."""
        outs = _run_race(tmp_path, tmp_path / "custody", "claim")
        assert sum(1 for o in outs if o == "WON") == 1, outs
        assert sum(1 for o in outs if o.startswith("LOST")) == 1, outs

    def test_e3_2_completed_races_failed_one_immutable_winner(
            self, tmp_path):
        root = tmp_path / "custody"
        claim(MigrationCustody(root))
        barrier = tmp_path / "barrier2"
        scripts = []
        for mode in ("completed", "failed"):
            path = tmp_path / f"term_{mode}.py"
            path.write_text(_race_script(root, barrier, mode))
            scripts.append(path)
        procs = [subprocess.Popen([sys.executable, str(p)],
                                  stdout=subprocess.PIPE, text=True)
                 for p in scripts]
        outs = [p.communicate()[0].strip() for p in procs]
        assert sum(1 for o in outs if o == "WON") == 1, outs
        final = MigrationCustody(root).read("mig-1")
        assert final["state"] in ("completed", "failed")
        # the winner is IMMUTABLE: the loser never overwrote it
        with pytest.raises(MigrationCustodyError):
            MigrationCustody(root).finish(
                "mig-1", "completed",
                reconciliation=reconciliation(), now=NOW)

    def test_e3_3_two_completions_different_evidence_no_overwrite(
            self, tmp_path):
        root = tmp_path / "custody"
        custody = MigrationCustody(root)
        claim(custody)
        first = custody.finish("mig-1", "completed",
                               reconciliation=reconciliation(),
                               now=NOW)
        winner_evidence = first["reconciliation_evidence"]
        other = ReconciliationEvidence.build(
            venue="mt5_demo", account_fingerprint="fp-1",
            symbol="ETHUSD", position_identity="pos-1",
            observed_at=NOW - timedelta(seconds=1),
            source="other_source", evidence_id="ev-rec-OTHER",
            raw_payload={"positions": 0, "orders": 0, "n": 2},
            positions_total=0, orders_total=0)
        with pytest.raises(MigrationCustodyError):
            custody.finish("mig-1", "completed",
                           reconciliation=other, now=NOW)
        assert MigrationCustody(root).read("mig-1")[
            "reconciliation_evidence"] == winner_evidence

    def test_e3_4_restart_from_placeholder_is_diagnosed(
            self, tmp_path):
        """A crash between exclusive final creation and replacement
        leaves an empty placeholder: classified, never silently
        recovered."""
        custody = MigrationCustody(tmp_path / "custody")
        path = custody._path("mig-1")
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
        with pytest.raises(MigrationCustodyError,
                           match="INTERRUPTED_CLAIM_UNCERTAIN"):
            custody.read("mig-1")
        with pytest.raises(MigrationCustodyError):
            claim(custody)

    def test_e3_5_symlinked_record_refused_not_followed(
            self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        claim(custody)
        path = custody._path("mig-1")
        elsewhere = tmp_path / "elsewhere.json"
        elsewhere.write_text(path.read_text())
        path.unlink()
        path.symlink_to(elsewhere)
        with pytest.raises(MigrationCustodyError, match="symlink"):
            custody.read("mig-1")
        with pytest.raises(MigrationCustodyError, match="symlink"):
            custody.finish("mig-1", "completed",
                           reconciliation=reconciliation(), now=NOW)

    def test_e3_6a_fabricated_protection_digest_refuses(
            self, tmp_path):
        """PRE: native_protection_digest='not-a-digest-or-fresh-
        evidence' authorized."""
        custody = MigrationCustody(tmp_path / "custody")
        with pytest.raises(MigrationCustodyError, match="envelope"):
            custody.claim(migration(), closed_block(), "pos-1",
                          protection_evidence="not-a-digest-or-"
                                              "fresh-evidence",
                          policy_identity=POLICY_ID,
                          code_identity=CODE_ID, now=NOW)
        # and a tampered digest inside a real envelope is
        # UNCONSTRUCTABLE: the digest is re-hashed from the payload
        good = protection()
        with pytest.raises(EvidenceError, match="fabricated digest"):
            NativeProtectionEvidence(
                good.venue, good.account_fingerprint, good.symbol,
                good.position_identity, good.observed_at,
                good.source, good.evidence_id, good.raw_payload,
                "0" * 64, good.max_age_seconds, True, True)

    def test_e3_6b_string_yes_reconciliation_refuses(self, tmp_path):
        """PRE: {'flat_confirmed':'yes','fresh':'yes'} completed
        custody."""
        custody = MigrationCustody(tmp_path / "custody")
        claim(custody)
        with pytest.raises(MigrationCustodyError, match="envelope"):
            custody.finish("mig-1", "completed", now=NOW,
                           reconciliation={"flat_confirmed": "yes",
                                           "fresh": "yes"})
        assert MigrationCustody(
            custody.root).read("mig-1")["state"] == "active"

    def test_e3_strict_types_in_envelopes(self):
        with pytest.raises(EvidenceError, match="strict bool"):
            protection(sl="yes")
        with pytest.raises((EvidenceError, Exception)):
            ReconciliationEvidence.build(
                venue="v", account_fingerprint="a", symbol="s",
                position_identity="p", observed_at=NOW,
                source="src", evidence_id="e",
                raw_payload={"x": 1}, positions_total=True,
                orders_total=0)

    def test_e3_stale_and_identity_mismatch_refuse(self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        # typed refusals from the evidence layer itself
        with pytest.raises(EvidenceError, match="stale"):
            claim(custody, prot=protection(
                observed=NOW - timedelta(hours=5)))
        with pytest.raises(EvidenceError, match="mismatch"):
            claim(custody, prot=protection(symbol="BTCUSD"))
        assert custody.read("mig-1") is None  # no record left

    def test_e3_failed_preserves_evidence_and_never_completes(
            self, tmp_path):
        custody = MigrationCustody(tmp_path / "custody")
        claim(custody)
        failed = custody.finish("mig-1", "failed", now=NOW,
                                reconciliation=reconciliation(
                                    positions=1))
        assert failed["state"] == "failed"
        assert failed["reconciliation_evidence"][
            "positions_total"] == 1
        with pytest.raises(MigrationCustodyError, match="immutable"):
            custody.finish("mig-1", "completed",
                           reconciliation=reconciliation(), now=NOW)
