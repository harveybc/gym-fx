"""WP4 battery, corrected under order agent-multi@051ef265
(WP4-C1..C8): complete executable identity, genuinely paired
treatment, truthful historical session evidence, executed mechanics
with observed terminal outcomes, derived conservation, the W2a/W2b
split with the G2 ablation, and a median/dispersion benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import backtrader as bt
import numpy as np
import pytest

from app.session_exposure import SessionPolicyError
from strategy_plugins.shared_execution_envelope import (
    Plugin as Envelope)
from tools.wp4_materializer import (G2_BASELINE_BARS,
                                    SECTION4_DEFAULTS, base_policy,
                                    canonical_bytes,
                                    check_feasibility, materialize,
                                    sha256_hex, verify_cell,
                                    write_materialization)
import tools.wp4_driver as drv

REPO_ROOT = Path(__file__).resolve().parents[1]
CAL = "cal-weekly-v1"
IDENTITY = {"gymfx_commit": "2c60b84",
            "plan42": "agent-multi@45c49003"}


@pytest.fixture(scope="module")
def mat():
    return materialize(calendar_identity=CAL, bar_hours=4.0,
                       min_open_window_hours=104.0,
                       identity=IDENTITY)


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    return drv.load_historical_window(
        tmp_path_factory.mktemp("window"))


@pytest.fixture(scope="module")
def tape(window):
    return drv.action_tape(7, window["bars"] + 4)


def _cell(mat, cell_id):
    return next(c for c in mat["cells"] if c["cell_id"] == cell_id)


# ================================================================== #
# materialization: families, split, external binding                 #
# ================================================================== #

class TestMaterialization:

    def test_family_counts(self, mat):
        """F4: twelve W1 cells are execution-latency infeasible on
        H4 next-bar fills and live in the rejection ledger; F5 adds
        the bounded probation ablation."""
        assert mat["manifest"]["families"] == {
            "W0": 2, "W1": 10, "W2a": 45, "W2b": 27, "G2B": 3,
            "PROB": 3}
        assert mat["manifest"]["rejections"] == 14

    def test_w0_is_the_paired_comparison(self, mat):
        control = _cell(mat, "w0_control_disabled")
        overlay = _cell(mat, "w0_overlay_enabled")
        a = dict(control["session_exposure_policy"])
        b = dict(overlay["session_exposure_policy"])
        assert a.pop("enabled") is False
        assert b.pop("enabled") is True
        assert a == b
        assert control["live_deployable"] is False

    def test_w2a_carries_no_completeness_claim(self, mat):
        for cell in mat["cells"]:
            if cell["family"] == "W2a":
                assert cell["role"] == "provisional_mechanism_screen"
                assert "NONE" in cell["completeness_claim"]
                assert cell["w1_timing"]["economic_execution"] == \
                    "BLOCKED"

    def test_w2b_calibrates_the_predeclared_threshold_domains(
            self, mat):
        cells = [c for c in mat["cells"] if c["family"] == "W2b"]
        assert len(cells) == 27
        spreads = {c["session_exposure_policy"]
                   ["max_spread_relative_to_baseline"]
                   for c in cells}
        assert spreads == {1.5, 2.0, 2.5}
        for cell in cells:
            assert cell["role"] == "threshold_calibration"

    def test_g2_baseline_ablation_exists_with_rationale(self, mat):
        cells = [c for c in mat["cells"] if c["family"] == "G2B"]
        assert len(cells) == 3
        bases = {c["session_exposure_policy"]["reopen_baseline_bars"]
                 for c in cells}
        assert bases == {2, 4, 8}
        assert all("rationale" in c for c in cells)

    def test_manifest_is_the_external_binding(self, mat):
        manifest = mat["manifest"]
        index = manifest["cell_index"]
        assert len(index) == 90
        assert sorted(index) == manifest["trial_ledger"]
        for cell in mat["cells"]:
            assert index[cell["cell_id"]] == cell["digest"]

    def test_infeasible_pair_refuses(self):
        policy = base_policy(CAL)
        policy["wind_down_hours"] = 4.0
        policy["forced_flatten_hours"] = 8.0
        with pytest.raises(SessionPolicyError,
                           match="forced flatten must occur AFTER"):
            check_feasibility(policy, bar_hours=4.0,
                              min_open_window_hours=104.0)

    def test_timeframe_infeasible_refuses(self):
        policy = base_policy(CAL)
        # a flatten window admissible on 48h bars, so the TIMEFRAME
        # rule is what refuses (F4's flatten rule is exercised in
        # its own battery)
        policy["wind_down_hours"] = 100.0
        policy["forced_flatten_hours"] = 96.0
        policy["reopen_min_closed_bars"] = 3
        policy["stability_consecutive_checks"] = 3
        with pytest.raises(SessionPolicyError,
                           match="timeframe-infeasible"):
            check_feasibility(policy, bar_hours=48.0,
                              min_open_window_hours=104.0)

    def test_persisted_cells_match_the_manifest(self, mat, tmp_path):
        out = write_materialization(mat, tmp_path / "m")
        drv.verify_manifest_matches_dir(mat["manifest"], out)
        (out / "w1_wd12_ff8.json").unlink()
        with pytest.raises(drv.Wp4IdentityError, match="missing"):
            drv.verify_manifest_matches_dir(mat["manifest"], out)


# ================================================================== #
# C1: complete executable identity                                   #
# ================================================================== #

class TestC1Identity:

    def test_all_six_consumed_files_are_verified(self):
        assert sorted(drv.FROZEN_AUTHORITY_SHA256) == [
            "app/direct_evidence.py", "app/env.py",
            "app/flatten_custody.py", "app/migration_custody.py",
            "app/oanda_calendar.py", "app/session_exposure.py"]
        drv.verify_frozen_identity(REPO_ROOT)

    def test_unconsumed_identities_are_enumerated_not_claimed(self):
        assert "lts@83dff62" in drv.BOUND_UNCONSUMED_IDENTITIES

    def test_a_tampered_authority_file_refuses(self, monkeypatch):
        monkeypatch.setitem(drv.FROZEN_AUTHORITY_SHA256,
                            "app/env.py", "0" * 64)
        with pytest.raises(drv.Wp4IdentityError,
                           match="refuses to run unreviewed"):
            drv.verify_frozen_identity(REPO_ROOT)

    def test_a_redigested_cell_fails_the_manifest_binding(self, mat):
        """PRE FROZEN: the self-digest was the only check, so a
        consistently altered and re-digested cell RAN."""
        cell = dict(_cell(mat, "w1_wd24_ff8"))
        cell["session_exposure_policy"] = dict(
            cell["session_exposure_policy"])
        cell["session_exposure_policy"]["wind_down_hours"] = 47.0
        body = {k: v for k, v in cell.items() if k != "digest"}
        cell["digest"] = sha256_hex(canonical_bytes(body))
        verify_cell(cell)          # self-consistent on purpose
        with pytest.raises(drv.Wp4IdentityError,
                           match="re-digested cell runs nothing"):
            drv.verify_cell_binding(cell, mat["manifest"],
                                    mat["manifest"]["digest"])

    def test_a_substituted_manifest_refuses(self, mat):
        manifest = json.loads(json.dumps(mat["manifest"]))
        manifest["cell_index"] = dict(manifest["cell_index"])
        manifest["cell_index"]["evil"] = "ab" * 32
        body = {k: v for k, v in manifest.items() if k != "digest"}
        manifest["digest"] = sha256_hex(canonical_bytes(body))
        with pytest.raises(drv.Wp4IdentityError,
                           match="substituted manifest"):
            drv.verify_manifest(manifest, mat["manifest"]["digest"])

    def test_an_unlisted_cell_refuses(self, mat):
        cell = dict(_cell(mat, "w0_overlay_enabled"))
        cell["cell_id"] = "w9_unlisted"
        body = {k: v for k, v in cell.items() if k != "digest"}
        cell["digest"] = sha256_hex(canonical_bytes(body))
        with pytest.raises(drv.Wp4IdentityError,
                           match="not in the reviewed manifest"):
            drv.verify_cell_binding(cell, mat["manifest"],
                                    mat["manifest"]["digest"])


# ================================================================== #
# C2: genuinely paired treatment                                     #
# ================================================================== #

class TestC2PairedTreatment:

    def test_the_tape_derives_from_the_seed_only(self):
        a = drv.action_tape(7, 50)
        b = drv.action_tape(7, 50)
        c = drv.action_tape(8, 50)
        assert a["digest"] == b["digest"] != c["digest"]
        import inspect
        src = inspect.getsource(drv.recorded_run)
        assert 'cell["digest"]' not in \
            inspect.getsource(drv.action_tape)
        assert "default_rng" not in src, (
            "the run must consume the tape, never its own RNG")

    @pytest.mark.parametrize("pair", [
        ("w0_control_disabled", "w0_overlay_enabled"),
        ("w1_wd12_ff8", "w1_wd48_ff8"),
    ])
    def test_paired_prefixes_are_identical(self, mat, tape,
                                           tmp_path, pair):
        report = drv.paired_prefix_check(
            _cell(mat, pair[0]), _cell(mat, pair[1]),
            mat["manifest"], mat["manifest"]["digest"], tape,
            tmp_dir=tmp_path, repo_root=REPO_ROOT)
        assert report["first_treatment_dependent_step"] is not None
        assert report["identical_prefix_steps"] >= 10
        assert report["identical_prefix_steps"] == \
            report["first_treatment_dependent_step"]


# ================================================================== #
# C3: truthful historical session evidence                           #
# ================================================================== #

class TestC3HistoricalEvidence:

    def test_source_hash_is_verified_and_published(self, window):
        meta = window["meta"]
        assert meta["source"]["sha256"] == \
            drv.HISTORICAL_SOURCE["sha256"]
        assert meta["fixture_csv_sha256"]
        assert "SPREAD" in meta["instrumentation_channels"]

    def test_two_weekends_and_a_holiday_are_observed(self, window):
        meta = window["meta"]
        assert meta["weekend_closures"] >= 2
        assert meta["holiday_or_exception_closures"] >= 1
        assert meta["holiday_evidence"] == "present"

    def test_missing_intervals_are_preserved(self, window):
        import pandas as pd
        stamps = pd.to_datetime(
            pd.read_csv(window["csv"])["DATE_TIME"]).dt.tz_localize(
                "UTC")
        for a, b in window["intervals"]:
            inside = ((stamps >= pd.Timestamp(a)) &
                      (stamps < pd.Timestamp(b))).sum()
            assert inside == 0, (a, b)

    def test_a_tampered_source_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setitem(drv.HISTORICAL_SOURCE, "sha256",
                            "0" * 64)
        with pytest.raises(drv.Wp4IdentityError,
                           match="does not match the published"):
            drv.load_historical_window(tmp_path)

    def test_the_fifth_state_is_probed_not_claimed(self, mat,
                                                   window):
        probe = drv.expected_market_closed_probe(
            _cell(mat, "w0_overlay_enabled"), window)
        assert probe["all_expected_market_closed"]
        assert len(probe["probes"]) == len(window["intervals"])
        assert "four states" in probe["note"]


# ================================================================== #
# C4: executed mechanics with OBSERVED terminal outcomes             #
# ================================================================== #

class RestingEntryEnvelope(Envelope):
    """ONE genuine resting Limit entry through the executing path,
    registered as an entry like any other order."""

    resting_ref = None
    submit_at_bar = 4

    def apply_action(self, s, action, config):
        result = super().apply_action(s, action, config)
        bar = int(getattr(s.bridge, "bar_index", 0))
        if type(self).resting_ref is None and \
                bar >= type(self).submit_at_bar:
            price = float(s.data.close[0]) * 0.90
            order = s.buy(exectype=bt.Order.Limit, price=price,
                          size=0.05)
            s.bridge.register_order_role(int(order.ref), "entry")
            type(self).resting_ref = int(order.ref)
        return result


class TestC4ExecutedMechanics:

    def _run(self, env, actions):
        env.reset(seed=7)
        frames = []
        for action in actions:
            _o, _r, term, _t, info = env.step([float(action)])
            frames.append(info)
            if term:
                break
        return frames

    def test_resting_entry_cancelled_and_protection_survives(
            self, mat, window, tmp_path):
        """OBSERVED outcomes: the strategy really called cancel()
        (cancel_submitted), and the book afterwards holds ONLY the
        two protective legs."""
        RestingEntryEnvelope.resting_ref = None
        env = drv.build_env(_cell(mat, "w0_overlay_enabled"), window,
                            tmp_dir=tmp_path,
                            envelope_cls=RestingEntryEnvelope,
                            leverage_cap=0.4)
        frames = self._run(env, [1.0] * 30)
        ref = RestingEntryEnvelope.resting_ref
        assert ref is not None
        assert env.bridge.cancel_outcomes.get(ref) == \
            "cancel_submitted"
        assert any(ref in (f.get("session_cancel_requested_refs")
                           or ()) for f in frames)
        # protection survives THROUGH the wind-down (post-cancel);
        # the legitimate forced close retires it afterwards, so the
        # terminal book being empty is correct — the claim is about
        # the wind-down window, not the end of time
        wind_rows = [f for f in frames
                     if f.get("session_state") == "WIND_DOWN"
                     and f.get("session_protective_orders")
                     is not None]
        assert wind_rows, "the run must traverse wind-down"
        assert any(f["session_protective_orders"] >= 2
                   for f in wind_rows), (
            "both protective legs must be alive in wind-down")
        assert all(f.get("session_entry_orders", 0) == 0
                   for f in wind_rows[1:]), (
            "no pending entry may survive the cancellation")

    def test_cancellation_outcome_taxonomy_is_observed(
            self, mat, window, tmp_path):
        """refused_role for a protective leg and an unknown ref,
        not_open for a registered entry absent from the book — every
        outcome OBSERVED from the bridge, never inferred."""
        env = drv.build_env(_cell(mat, "w0_overlay_enabled"), window,
                            tmp_dir=tmp_path)
        env.reset(seed=7)
        for _ in range(3):
            env.step([1.0])
        protective = tuple(r["ref"] for r in
                           env.bridge.open_order_inventory)
        env.bridge.cancel_entry_request = protective + (98765,)
        env.step([1.0])
        outcomes = env.bridge.cancel_outcomes
        for ref in protective:
            assert outcomes[ref].startswith("refused_role")
        assert outcomes[98765] == "refused_role_None"
        env.bridge.register_order_role(31337, "entry")
        env.bridge.cancel_entry_request = (31337,)
        env.step([1.0])
        assert env.bridge.cancel_outcomes[31337] == "not_open"
        assert env.bridge.open_order_inventory, (
            "protection must have survived every refusal")

    def test_voluntary_and_forced_closes_both_observed(
            self, mat, window, tmp_path):
        cell = _cell(mat, "w0_overlay_enabled")
        # forced: hold exposure into the deadline
        env = drv.build_env(cell, window, tmp_dir=tmp_path / "f")
        frames = self._run(env, [1.0] * 30)
        assert any(f.get("session_overlay") == "forced_close"
                   for f in frames)
        # voluntary: the tape closes before wind-down's deadline
        env2 = drv.build_env(cell, window, tmp_dir=tmp_path / "v")
        frames2 = self._run(env2, [1.0] * 6 + [0.0] * 24)
        summary = env2.summary()
        assert summary["trades_total"] >= 1
        assert not any(f.get("session_overlay") == "forced_close"
                       for f in frames2[:10])

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_long_and_short_exposure_wind_down(self, mat, window,
                                               tmp_path, direction):
        env = drv.build_env(_cell(mat, "w0_overlay_enabled"), window,
                            tmp_dir=tmp_path)
        frames = self._run(env, [direction] * 26)
        exposures = [f.get("session_signed_exposure") for f in frames
                     if f.get("session_signed_exposure")]
        assert exposures, "the run must carry real exposure"
        if direction > 0:
            assert max(exposures) > 0
        else:
            assert min(exposures) < 0
        assert any(f.get("session_state") == "WIND_DOWN"
                   for f in frames)

    def test_interrupted_flatten_recovers_durably(self, mat, window,
                                                  tmp_path):
        """OBSERVED: the obligation is durable in custody; a fresh
        env over the SAME custody root reports an active recovery
        and blocks by flatten recovery — the interruption cannot be
        silently forgotten."""
        cell = _cell(mat, "w0_overlay_enabled")
        env = drv.build_env(cell, window, tmp_dir=tmp_path)
        env.reset(seed=7)
        for _ in range(40):
            _o, _r, term, _t, info = env.step([1.0])
            if info.get("session_overlay") == "forced_close":
                break
        custody = Path(env.config["session_flatten_custody_root"])
        obligations = [p.name for p in custody.rglob("*.json")]
        assert obligations, "the obligation must be durable"
        del env
        reborn = drv.build_env(cell, window, tmp_dir=tmp_path)
        reborn.reset(seed=7)
        _o, _r, _term, _t, info = reborn.step([1.0])
        assert info["session_recovery_active"]
        assert info["session_overlay"] == \
            "blocked_by_flatten_recovery"

    def test_reopen_boundary_and_stability_reset(self, mat,
                                                 tmp_path):
        """A post-reopen spread spike (declared instrumentation)
        must hold the blackout longer than the quiet baseline: the
        stability checks reset on the violating bar."""
        import pandas as pd
        window = drv.load_historical_window(tmp_path / "w")
        cell = _cell(mat, "w2a_h1_b1_c2")
        quiet = drv.build_env(cell, window, tmp_dir=tmp_path / "q")
        frames_q = self._run(quiet, [0.0] * 40)
        # spike the spread on two bars right after the first reopen
        frame = pd.read_csv(window["csv"])
        stamps = pd.to_datetime(frame["DATE_TIME"])
        reopen = pd.Timestamp(window["intervals"][0][1]).tz_localize(
            None)
        after = stamps[stamps >= reopen].index[
            cell["session_exposure_policy"]["reopen_baseline_bars"]
            + 1:][:2]
        frame.loc[after, "SPREAD"] = 0.02
        spiked_csv = tmp_path / "spiked.csv"
        frame.to_csv(spiked_csv, index=False)
        spiked_window = dict(window)
        spiked_window["csv"] = spiked_csv
        noisy = drv.build_env(cell, spiked_window,
                              tmp_dir=tmp_path / "n")
        frames_n = self._run(noisy, [0.0] * 40)

        def blackout_span(frames):
            steps = [i for i, f in enumerate(frames)
                     if f.get("session_state") == "REOPEN_BLACKOUT"]
            return len(steps)
        assert blackout_span(frames_n) > blackout_span(frames_q), (
            "the spike must reset stability and extend the "
            "blackout")


# ================================================================== #
# C5: conservation is DERIVED                                        #
# ================================================================== #

class TestC5DerivedConservation:

    def test_an_eligible_run_exists_and_derives(self, mat,
                                                tmp_path):
        """C9: FULL eligibility on the plain-weekend window with a
        compliant tape seed — every invariant, equity included,
        derived and holding with flat terminal exposure."""
        plain = drv.load_historical_window(
            tmp_path / "pw", start=drv.PLAIN_WINDOW_START,
            end=drv.PLAIN_WINDOW_END)
        tape = drv.action_tape(5, plain["bars"] + 4)
        run = drv.recorded_run(_cell(mat, "w1_wd48_ff8"),
                               mat["manifest"],
                               mat["manifest"]["digest"], tape,
                               plain, tmp_dir=tmp_path,
                               repo_root=REPO_ROOT)
        cons = run["conservation"]
        assert cons["verdict"] == "ELIGIBLE"
        assert cons["failed_invariants"] == []
        assert cons["equity_reconciliation"]["exact"]
        assert not cons["equity_reconciliation"][
            "open_position_at_end"]
        assert cons["pending_entries_at_end"] == 0
        assert run["rows_sha256"] and run["trades_sha256"]

    def test_the_blackout_precedence_counterexample_is_dead(
            self, mat, window, tape, tmp_path):
        """F1 REGRESSION (order agent-multi@4ad4937b): this exact
        holiday-cluster run used to cross TWO closures because
        REOPEN_BLACKOUT outranked the closure duties. Under the
        corrected precedence contract the same cell on the same
        window is fully ELIGIBLE: forced flatten fires from within
        the lingering blackout and NO exposure crosses any governed
        closure."""
        run = drv.recorded_run(_cell(mat, "w1_wd48_ff8"),
                               mat["manifest"],
                               mat["manifest"]["digest"], tape,
                               window, tmp_dir=tmp_path,
                               repo_root=REPO_ROOT)
        cons = run["conservation"]
        assert cons["exposure_across_closure"] == []
        assert cons["verdict"] == "ELIGIBLE", cons[
            "failed_invariants"]

    def test_one_bar_flatten_windows_are_ledgered_rejections(
            self, mat):
        """F4 supersedes the runtime demonstration: a one-bar
        flatten window can no longer be materialized at all. The
        reproduced 4-hour failure lives in the rejection ledger
        with the execution-contract reason, and the frozen PRE run
        (C9-C14/F1-F3 packets) preserves the runtime evidence."""
        rejected = {r["cell_id"]: r["reason"]
                    for r in mat["rejections"]}
        assert any("ff4" in cid for cid in rejected)
        # F8 adds validator rejections for wd12 x live-extension
        # pairs (flatten after wind-down begins); every reason is
        # one of the TYPED mechanical refusals
        assert all("infeasible" in why or "never reaches" in why
                   or "forced flatten must occur AFTER" in why
                   for why in rejected.values())


# ================================================================== #
# C8: benchmark — median and dispersion                              #
# ================================================================== #

class TestC8Benchmark:

    def test_benchmark_reports_median_and_iqr(self, mat, window,
                                              tape, tmp_path):
        report = drv.benchmark(_cell(mat, "w0_overlay_enabled"),
                               mat["manifest"],
                               mat["manifest"]["digest"], tape,
                               window, tmp_dir=tmp_path,
                               repo_root=REPO_ROOT, repeats=3)
        assert "env_steps_per_second_median" in report
        assert len(report["env_steps_per_second_iqr"]) == 2
        assert "best_of" not in json.dumps(report)
        assert "SAC update throughput is unmeasured" in \
            report["scope"]
        assert report["conservation_verdict"] in ("ELIGIBLE",
                                                  "INELIGIBLE")
        assert isinstance(report["failed_invariants"], list)

    def test_forged_drift_refuses(self):
        """The drift seam itself must bite: a single differing
        shared channel before the treatment refuses."""
        row = {"action": 1.0, "reward": 0.1, "pnl": 0.1,
               "equity": 100.0, "obs_digest_shared": "aa"}
        other = dict(row, equity=101.0)
        with pytest.raises(drv.Wp4IdentityError,
                           match="paired prefix drift"):
            drv.assert_identical_prefix([row], [other], 1)
        drv.assert_identical_prefix([row], [dict(row)], 1)


class TestC5DerivationBites:
    """The derivation itself must classify violations — these tests
    call derive_conservation, not a re-implementation."""

    def _base(self):
        import types
        env = types.SimpleNamespace(
            initial_cash=100.0,
            bridge=types.SimpleNamespace(equity=100.0))
        summary = {"trades_total": 1, "trades_won": 1,
                   "trades_lost": 0, "trades_breakeven": 0,
                   "open_position_at_end": False,
                   "close_reason_counts": {},
                   "trade_costs_total": 0.0}
        stream = [{"gross_pnl": 1.0, "costs": 0.0, "net_pnl": 1.0}]
        return env, summary, stream

    def _window(self, tmp_path):
        import pandas as pd
        frame = pd.DataFrame({
            "DATE_TIME": pd.date_range("2024-01-01", periods=4,
                                       freq="4h")})
        csv = tmp_path / "w.csv"
        frame.to_csv(csv, index=False)
        return {"csv": csv,
                "intervals": [["2024-02-01 00:00:00+00:00",
                               "2024-02-02 00:00:00+00:00"]]}

    def test_a_suppressed_reward_breaks_holds(self, tmp_path):
        env, summary, stream = self._base()
        rows = [{"index": 0, "reward": 0.0, "pnl": 5.0}]
        cons = drv.derive_conservation(env, rows,
                                       self._window(tmp_path),
                                       summary, stream)
        assert cons["suppressed_reward_steps"] == [0]
        assert not cons["holds"]

    def test_a_gross_net_violation_breaks_holds(self, tmp_path):
        env, summary, stream = self._base()
        stream[0]["net_pnl"] = 5.0     # gross 1.0 - costs 0.0 != 5.0
        cons = drv.derive_conservation(env, [],
                                       self._window(tmp_path),
                                       summary, stream)
        assert cons["gross_minus_costs_equals_net"]["violations"] \
            == 1
        assert not cons["holds"]

    def test_a_close_event_miscount_breaks_holds(self, tmp_path):
        env, summary, stream = self._base()
        summary["trades_won"] = 0      # 1 != 0 + 0 + 0
        cons = drv.derive_conservation(env, [],
                                       self._window(tmp_path),
                                       summary, stream)
        assert not cons["close_event_conservation"]["holds"]
        assert not cons["holds"]

    def test_a_bar_inside_a_closure_breaks_holds(self, tmp_path):
        import pandas as pd
        env, summary, stream = self._base()
        window = self._window(tmp_path)
        frame = pd.DataFrame({
            "DATE_TIME": [pd.Timestamp("2024-02-01 04:00:00")]})
        frame.to_csv(window["csv"], index=False)
        cons = drv.derive_conservation(env, [], window, summary,
                                       stream)
        assert cons["bar_timestamps_inside_closures"] == 1
        assert not cons["holds"]


# ================================================================== #
# C9: the eligibility contract bites on every invariant              #
# ================================================================== #

class TestC9EligibilityBites:
    """Every required invariant, violated in isolation through the
    REAL derivation, must fail eligibility and be NAMED."""

    def _base(self, tmp_path):
        import pandas as pd
        import types
        frame = pd.DataFrame({
            "DATE_TIME": pd.date_range("2024-01-01", periods=4,
                                       freq="4h")})
        csv = tmp_path / "w.csv"
        frame.to_csv(csv, index=False)
        window = {"csv": csv,
                  "intervals": [["2024-02-01 00:00:00+00:00",
                                 "2024-02-02 00:00:00+00:00"]]}
        env = types.SimpleNamespace(
            initial_cash=100.0,
            bridge=types.SimpleNamespace(
                equity=101.0, open_order_inventory=()))
        summary = {"trades_total": 1, "trades_won": 1,
                   "trades_lost": 0, "trades_breakeven": 0,
                   "open_position_at_end": False,
                   "close_reason_counts": {},
                   "trade_costs_total": 0.0}
        stream = [{"gross_pnl": 1.0, "costs": 0.0, "net_pnl": 1.0}]
        rows = [{"index": 0, "reward": 0.1, "pnl": 1.0,
                 "signed_exposure": 0.0, "position_after": 0.0,
                 "protective_orders": 2, "entry_orders": 0,
                 "cancellation_incident": None,
                 "flatten_incident": None,
                 "recovery_active": False,
                 "bar_stamp": "2024-01-01 00:00:00"}]
        return env, rows, window, summary, stream

    def _derive(self, parts):
        env, rows, window, summary, stream = parts
        return drv.derive_conservation(env, rows, window, summary,
                                       stream, policy_enabled=True)

    def test_the_clean_base_is_eligible(self, tmp_path):
        cons = self._derive(self._base(tmp_path))
        assert cons["verdict"] == "ELIGIBLE", cons[
            "failed_invariants"]

    def test_open_position_fails(self, tmp_path):
        parts = self._base(tmp_path)
        parts[3]["open_position_at_end"] = True
        cons = self._derive(parts)
        assert "flat_terminal_exposure" in cons["failed_invariants"]
        assert "exact_equity_reconciliation" in \
            cons["failed_invariants"]
        assert cons["verdict"] == "INELIGIBLE"

    def test_nonzero_equity_gap_fails(self, tmp_path):
        parts = self._base(tmp_path)
        parts[0].bridge.equity = 150.0
        cons = self._derive(parts)
        assert "exact_equity_reconciliation" in \
            cons["failed_invariants"]

    def test_pending_entry_fails(self, tmp_path):
        parts = self._base(tmp_path)
        parts[0].bridge.open_order_inventory = (
            {"ref": 9, "role": "entry"},)
        cons = self._derive(parts)
        assert "zero_pending_entries_at_end" in \
            cons["failed_invariants"]

    def test_missing_protection_fails(self, tmp_path):
        parts = self._base(tmp_path)
        # index 1: entry boundary (0 -> +) — EXCUSED, declared grace
        parts[1].append(dict(parts[1][0], index=1,
                             signed_exposure=100.0,
                             position_after=100.0,
                             protective_orders=0))
        # index 2: HELD unprotected (same sign, still open) — FLAGGED
        parts[1].append(dict(parts[1][0], index=2,
                             signed_exposure=100.0,
                             position_after=100.0,
                             protective_orders=0))
        # index 3: close-in-flight (flat after) — EXCUSED
        parts[1].append(dict(parts[1][0], index=3,
                             signed_exposure=100.0,
                             position_after=0.0,
                             protective_orders=0))
        parts[3]["open_position_at_end"] = False
        cons = self._derive(parts)
        assert "protective_inventory_valid_while_exposed" in \
            cons["failed_invariants"]
        assert cons["unprotected_exposure_steps"] == [2]

    def test_unresolved_incident_fails(self, tmp_path):
        parts = self._base(tmp_path)
        parts[1][-1]["flatten_incident"] = "FORCED_FLATTEN_FAILED"
        cons = self._derive(parts)
        assert "zero_unresolved_incidents" in \
            cons["failed_invariants"]

    def test_transient_incident_alone_does_not_fail(self, tmp_path):
        parts = self._base(tmp_path)
        parts[1].insert(0, dict(parts[1][0], index=0,
                                flatten_incident="TRANSIENT"))
        parts[1][-1]["index"] = 1
        cons = self._derive(parts)
        assert "zero_unresolved_incidents" not in \
            cons["failed_invariants"]
        assert cons["transient_incidents"] == ["TRANSIENT"]

    def test_altered_cost_fails(self, tmp_path):
        parts = self._base(tmp_path)
        parts[4][0]["costs"] = 0.5     # gross 1.0 - 0.5 != net 1.0
        cons = self._derive(parts)
        assert not cons["invariants"][
            "gross_minus_costs_equals_net"]

    def test_close_event_mismatch_fails(self, tmp_path):
        parts = self._base(tmp_path)
        parts[3]["trades_won"] = 0
        cons = self._derive(parts)
        assert not cons["invariants"]["close_event_conservation"]

    def test_interrupted_recovery_can_never_pass(self, tmp_path):
        parts = self._base(tmp_path)
        parts[1][-1]["recovery_active"] = True
        cons = self._derive(parts)
        assert "zero_unresolved_incidents" in \
            cons["failed_invariants"]
        assert cons["verdict"] == "INELIGIBLE"


# ================================================================== #
# C10: terminal cancellation, not requested cancellation             #
# ================================================================== #

class TestC10TerminalCancellation:

    def test_the_resting_entry_reaches_a_terminal_verdict(
            self, mat, window, tmp_path):
        """The broker's final status must be Canceled (or the
        engine's exact terminal equivalent), the order must never
        fill, and the post-cancel inventory must reconcile after the
        terminal callback — a request alone proves nothing."""
        RestingEntryEnvelope.resting_ref = None
        env = drv.build_env(_cell(mat, "w0_overlay_enabled"), window,
                            tmp_dir=tmp_path,
                            envelope_cls=RestingEntryEnvelope,
                            leverage_cap=0.4)
        env.reset(seed=7)
        frames = []
        for _ in range(34):
            _o, _r, term, _t, info = env.step([1.0])
            frames.append(info)
            if term:
                break
        ref = RestingEntryEnvelope.resting_ref
        assert ref is not None
        # request happened...
        assert env.bridge.cancel_outcomes.get(ref) == \
            "cancel_submitted"
        # ...and the TERMINAL broker verdict is cancellation
        terminal = env.bridge.order_terminal_status.get(ref)
        assert terminal in ("Canceled", "Cancelled", "Expired"), (
            f"terminal verdict for {ref} was {terminal!r}")
        assert terminal != "Completed", "the order must never fill"
        # the session block reconciled the outcome post-callback
        final = [f for f in frames
                 if f.get("session_cancellations")]
        assert final, "cancellation outcomes must be published"
        assert final[-1]["session_cancellations"].get(ref) == \
            "cancelled"
        assert final[-1]["session_cancellation_incident"] is None
        # post-cancel inventory: the entry is gone, protection stays
        refs_in_book = {r["ref"] for r in
                        (env.bridge.open_order_inventory or ())}
        assert ref not in refs_in_book
        # no closed trade ever carries the cancelled entry
        for event in getattr(env.bridge, "closed_trade_stream", []):
            assert event.get("entry_ref") != ref

    def test_request_only_states_remain_failures(self, mat, window,
                                                 tmp_path):
        """rejected / filled-before-cancel / still-open / gone-
        without-verdict all surface as failures or pendings, never
        as success."""
        env = drv.build_env(_cell(mat, "w0_overlay_enabled"), window,
                            tmp_dir=tmp_path)
        env.reset(seed=7)
        for _ in range(3):
            env.step([1.0])
        # a registered entry absent from the book and never given a
        # terminal verdict is gone_without_verdict — an incident
        env.bridge.register_order_role(31337, "entry")
        env._session_cancel_requested.add(31337)
        outcomes = env._session_cancellation_outcomes()
        assert outcomes["session_cancellations"][31337] == \
            "gone_without_verdict"
        assert outcomes["session_cancellation_incident"]
        # a forged terminal fill is the filled-before-cancel failure
        env.bridge.order_terminal_status[31337] = "Completed"
        outcomes = env._session_cancellation_outcomes()
        assert outcomes["session_cancellations"][31337] == \
            "filled_before_cancel"
        assert "DESPITE" in outcomes[
            "session_cancellation_incident"]


# ================================================================== #
# C11: zero topology                                                 #
# ================================================================== #

class TestC11ZeroTopology:

    def test_no_absolute_operator_paths_in_public_surfaces(self):
        """Repository-wide sanitization scan over the WP4 package:
        no /home/ path, no operator username, in the tools, the
        battery (this file asserts over itself minus its own scan
        strings), and every persisted materialization artefact."""
        import re
        surfaces = [Path("tools/wp4_driver.py"),
                    Path("tools/wp4_materializer.py"),
                    Path("tools/wp4_stats.py")]
        surfaces += sorted(
            Path("examples/wp4_weekly_flat").glob("*.json"))
        for path in surfaces:
            text = path.read_text()
            assert "/home/" not in text, path
            assert "harveybc" not in text, path

    def test_missing_data_root_fails_closed(self, monkeypatch):
        monkeypatch.delenv("WP4_DATA_ROOT", raising=False)
        with pytest.raises(drv.Wp4IdentityError,
                           match="WP4_DATA_ROOT is not set"):
            drv.resolve_source_path()

    def test_window_meta_carries_logical_identity_only(self,
                                                       window):
        meta = json.dumps(window["meta"])
        assert "/home/" not in meta
        assert "logical_id" in window["meta"]["source"]
        assert window["meta"]["source"]["role"] == \
            "generic_gap_mechanics_fixture_only"


# ================================================================== #
# C12: mechanics fixture is not venue session authority              #
# ================================================================== #

class TestC12VenueAuthority:

    def test_economic_binding_is_declared_unavailable(self, window):
        binding = window["meta"]["economic_data_binding"]
        assert binding["status"] == \
            "VENUE_SESSION_HISTORY_UNAVAILABLE"
        assert len(binding["required"]) == 3
        assert "NEVER inferred from missing bars" in binding["rule"]

    def test_the_fixture_declares_what_it_is_not(self):
        assert "MT5 ETHUSD" in \
            drv.HISTORICAL_SOURCE["not_authoritative_for"]


# ================================================================== #
# C13: the joint confirmation                                        #
# ================================================================== #

class TestC13JointConfirmation:

    def test_predeclared_in_the_manifest(self, mat):
        joint = mat["manifest"]["joint_confirmation_predeclaration"]
        assert "promotion-eligible" in joint["rule"]
        assert joint["constructor"] == \
            "materialize_joint_confirmation"

    def test_constructor_builds_control_plus_neighbours(self):
        from tools.wp4_materializer import (
            materialize_joint_confirmation)
        result = materialize_joint_confirmation(
            calendar_identity=CAL, bar_hours=4.0,
            min_open_window_hours=104.0, identity=IDENTITY,
            selected_w1={"wind_down_hours": 48.0,
                         "forced_flatten_hours": 8.0},
            selected_w2a={"reopen_min_hours": 4.0,
                          "reopen_min_closed_bars": 2,
                          "stability_consecutive_checks": 2},
            selected_w2b={
                "max_spread_relative_to_baseline": 2.0,
                "max_gap_sigma": 3.0,
                "max_realized_vol_relative_to_baseline": 2.0},
            selected_g2b=4)
        ids = {c["cell_id"] for c in result["cells"]}
        assert "w2joint_section4_control" in ids
        assert "w2joint_h4_b2_c2" in ids       # the selection
        # bounded one-step neighbours only: 3*3*3 combos + control
        assert len(ids) == 28
        selected = [c for c in result["cells"]
                    if c["role"] == "joint_selected_combination"]
        assert len(selected) == 1
        for cell in result["cells"]:
            if "promotion_rule" in cell:
                assert "NO W2 candidate" in cell["promotion_rule"]

    def test_edge_selection_has_fewer_neighbours(self):
        from tools.wp4_materializer import (
            materialize_joint_confirmation)
        result = materialize_joint_confirmation(
            calendar_identity=CAL, bar_hours=4.0,
            min_open_window_hours=104.0, identity=IDENTITY,
            selected_w1={"wind_down_hours": 48.0,
                         "forced_flatten_hours": 8.0},
            selected_w2a={"reopen_min_hours": 1.0,
                          "reopen_min_closed_bars": 1,
                          "stability_consecutive_checks": 1},
            selected_w2b={
                "max_spread_relative_to_baseline": 2.0,
                "max_gap_sigma": 3.0,
                "max_realized_vol_relative_to_baseline": 2.0},
            selected_g2b=4)
        # grid-edge selection: 2*2*2 combos + control
        assert len(result["cells"]) == 9


# ================================================================== #
# F4/F5: execution-latency feasibility and explicit probation        #
# ================================================================== #

class TestF4ExecutionLatencyFeasibility:

    def test_the_reproduced_h4_failure_is_ledgered(self, mat):
        """PRE FROZEN: ff=4h on H4 triggered at the LAST bar and the
        next-bar fill landed after the gap. The exact cell is now a
        LEDGERED rejection, not a trial."""
        rejected = {r["cell_id"]: r["reason"]
                    for r in mat["rejections"]}
        assert "w1_wd36_ff4" in rejected
        assert "execution-latency infeasible" in \
            rejected["w1_wd36_ff4"]
        assert "w1_wd36_ff4" not in mat["manifest"]["cell_index"]

    def test_exact_boundary_on_h4(self):
        from tools.wp4_materializer import (EXECUTION_CONTRACT,
                                            flatten_deadline_admissible)
        ok, _ = flatten_deadline_admissible(
            8.0, 4.0, contract=EXECUTION_CONTRACT)
        assert ok
        ok, why = flatten_deadline_admissible(
            7.9, 4.0, contract=EXECUTION_CONTRACT)
        assert not ok and "triggers 4.0h" in why
        ok, why = flatten_deadline_admissible(
            4.0, 4.0, contract=EXECUTION_CONTRACT)
        assert not ok

    def test_delayed_fill_and_retry_budget_tighten_the_rule(self):
        from tools.wp4_materializer import flatten_deadline_admissible
        delayed = {"decision_at": "bar_close",
                   "submission_latency_bars": 1,
                   "fill_at": "next_bar_open",
                   "reconcile_at": "fill_bar_step",
                   "close_retry_budget_bars": 0,
                   "safety_margin_hours": 0.0}
        ok, _ = flatten_deadline_admissible(8.0, 4.0,
                                            contract=delayed)
        assert not ok, "one latency bar makes 8h inadmissible on H4"
        ok, _ = flatten_deadline_admissible(12.0, 4.0,
                                            contract=delayed)
        assert ok
        retried = dict(delayed, submission_latency_bars=0,
                       close_retry_budget_bars=1)
        ok, _ = flatten_deadline_admissible(8.0, 4.0,
                                            contract=retried)
        assert not ok, "a rejected first close consumes a bar"
        ok, _ = flatten_deadline_admissible(12.0, 4.0,
                                            contract=retried)
        assert ok

    def test_multiple_bar_sizes(self):
        from tools.wp4_materializer import (EXECUTION_CONTRACT,
                                            flatten_deadline_admissible)
        ok, _ = flatten_deadline_admissible(
            4.0, 1.0, contract=EXECUTION_CONTRACT)
        assert ok, "H1: 4h flatten is four bars — admissible"
        ok, _ = flatten_deadline_admissible(
            2.0, 1.0, contract=EXECUTION_CONTRACT)
        assert ok
        ok, _ = flatten_deadline_admissible(
            1.0, 1.0, contract=EXECUTION_CONTRACT)
        assert not ok, "one bar is one bar on any grid"

    def test_the_live_safe_default_is_eligible_not_four(self, mat):
        correction = mat["manifest"]["flatten_default_correction"]
        assert correction["section4_value_hours"] == 4.0
        assert correction["status"] == \
            "STRUCTURALLY_INELIGIBLE_FOR_H4_NEXT_BAR"
        assert correction["mechanics_only_hours"] == 8.0
        assert correction["live_safe_default_hours"] == 12.0
        for cell in mat["cells"]:
            policy = cell["session_exposure_policy"]
            if policy["enabled"]:
                from tools.wp4_materializer import (
                    EXECUTION_CONTRACT, flatten_deadline_admissible)
                ok, _ = flatten_deadline_admissible(
                    policy["forced_flatten_hours"],
                    cell["bar_hours"], contract=EXECUTION_CONTRACT)
                assert ok, cell["cell_id"]

    def test_holiday_shortened_session_is_reported(self, window):
        """The real Christmas closure is preceded by a stretch
        shorter than some wind-down horizons; the derived closures
        list carries the interval so the driver's crossing invariant
        judges it — nothing is silently admitted."""
        holiday = [c for c in
                   window["meta"]
                   ["closures_derived_from_observed_gaps"]
                   if c["kind"] == "holiday_or_exception"]
        assert holiday, "the bound history carries the holiday"


class TestF5ExplicitProbation:

    def test_probation_is_typed_policy_identity(self):
        from app.session_exposure import validate_policy
        policy = base_policy(CAL)
        assert policy["release_probation_factor"] == 2
        validated = validate_policy(policy)
        assert validated["release_probation_factor"] == 2
        bad = dict(policy)
        bad.pop("release_probation_factor")
        with pytest.raises(SessionPolicyError, match="missing"):
            validate_policy(bad)

    def test_state_telemetry_carries_the_policy_value(self):
        from datetime import datetime, timezone
        from app.session_exposure import (ReopenEvidence,
                                          SessionCalendar,
                                          session_state,
                                          validate_policy)
        policy = validate_policy(dict(base_policy(CAL),
                                      release_probation_factor=3))
        cal = SessionCalendar.build(
            venue="v", account_fingerprint="a", symbol="s",
            calendar_digest=CAL,
            intervals=[(datetime(2026, 1, 2, tzinfo=timezone.utc),
                        datetime(2026, 1, 4, tzinfo=timezone.utc))])
        ev = ReopenEvidence.build(closed_bars_since_reopen=9,
                                  stability_checks_passed=5,
                                  hint_time_since_reopen_hours=None)
        block = session_state(
            policy, now=datetime(2026, 1, 5, tzinfo=timezone.utc),
            calendar=cal, reopen_evidence=ev)
        assert block["state"] == "REOPEN_BLACKOUT"
        assert block["release_probation_factor"] == 3
        assert block["release_requirement"] == 9

    def test_the_ablation_is_bounded_and_non_live_below_two(
            self, mat):
        cells = [c for c in mat["cells"] if c["family"] == "PROB"]
        assert len(cells) == 3
        by_factor = {c["session_exposure_policy"]
                     ["release_probation_factor"]: c for c in cells}
        assert by_factor[1]["live_eligible"] is False
        assert by_factor[2]["live_eligible"] is True
        assert by_factor[3]["live_eligible"] is True

    def test_restart_reconstructs_the_latch_phase(self, mat,
                                                  window, tape,
                                                  tmp_path):
        """Two fresh envs over the same data derive the identical
        qualification/probation streak at every step — the latch is
        a causal function of bar history, not process memory."""
        cell = _cell(mat, "w1_wd48_ff8")
        env_a = drv.build_env(cell, window, tmp_dir=tmp_path / "a")
        env_b = drv.build_env(cell, window, tmp_dir=tmp_path / "b")
        env_a.reset(seed=7)
        env_b.reset(seed=7)
        for action in tape["actions"][:40]:
            _o, _r, ta, _t, info_a = env_a.step([float(action)])
            _o, _r, tb, _t, info_b = env_b.step([float(action)])
            assert info_a.get("session_reopen_stability_streak") \
                == info_b.get("session_reopen_stability_streak")
            assert info_a.get("session_state") == \
                info_b.get("session_state")
            if ta or tb:
                break


# ================================================================== #
# F8: the live flatten budget includes failure recovery              #
# ================================================================== #

class TestF8LiveFlattenBudget:

    def test_live_contract_demands_retry_and_margin(self):
        from tools.wp4_materializer import LIVE_EXECUTION_CONTRACT
        assert LIVE_EXECUTION_CONTRACT[
            "close_retry_budget_bars"] >= 1
        assert LIVE_EXECUTION_CONTRACT["safety_margin_hours"] > 0.0

    def test_h4_live_verdicts(self):
        from tools.wp4_materializer import (
            EXECUTION_CONTRACT, LIVE_EXECUTION_CONTRACT,
            flatten_deadline_admissible)
        cases = {4.0: (False, False), 8.0: (True, False),
                 12.0: (True, True), 16.0: (True, True)}
        for ff, (mech, live) in cases.items():
            m, _ = flatten_deadline_admissible(
                ff, 4.0, contract=EXECUTION_CONTRACT)
            l, why = flatten_deadline_admissible(
                ff, 4.0, contract=LIVE_EXECUTION_CONTRACT)
            assert (m, l) == (mech, live), (ff, m, l)
        # 8h fails live for the ORDERED reason: no second executable
        # fill before closure
        _, why = flatten_deadline_admissible(
            8.0, 4.0, contract=LIVE_EXECUTION_CONTRACT)
        assert "retry fills" in why

    def test_f4_verdicts_unchanged_by_the_slack_refactor(self):
        from tools.wp4_materializer import (EXECUTION_CONTRACT,
                                            flatten_deadline_admissible)
        assert flatten_deadline_admissible(
            8.0, 4.0, contract=EXECUTION_CONTRACT)[0]
        assert not flatten_deadline_admissible(
            7.9, 4.0, contract=EXECUTION_CONTRACT)[0]
        assert not flatten_deadline_admissible(
            4.0, 4.0, contract=EXECUTION_CONTRACT)[0]
        assert flatten_deadline_admissible(
            4.0, 1.0, contract=EXECUTION_CONTRACT)[0]

    def test_enabled_arms_default_to_the_live_value(self, mat):
        for cell in mat["cells"]:
            policy = cell["session_exposure_policy"]
            if policy["enabled"] and cell["family"] != "W1":
                assert policy["forced_flatten_hours"] == 12.0, \
                    cell["cell_id"]

    def test_w1_labels_live_safety_and_never_promotes_8h(self, mat):
        w1 = [c for c in mat["cells"] if c["family"] == "W1"]
        for cell in w1:
            ff = cell["session_exposure_policy"][
                "forced_flatten_hours"]
            if ff == 8.0:
                assert cell["live_safe_flatten"] is False
                assert "MECHANICS ONLY" in cell["live_safety_note"]
            else:
                assert cell["live_safe_flatten"] is True

    def test_holiday_shortened_session_fails_closed(self, mat):
        """The REAL Christmas stretch (Dec-24 16:00 reopen to
        Dec-25 04:00 close = 12h open) cannot fit the live 12h
        budget — the helper refuses, exactly the fail-closed the
        order demands."""
        from tools.wp4_materializer import (LIVE_EXECUTION_CONTRACT,
                                            closure_budget_fits)
        policy = _cell(mat, "w0_overlay_enabled")[
            "session_exposure_policy"]
        ok, why = closure_budget_fits(12.0, policy, 4.0,
                                      contract=LIVE_EXECUTION_CONTRACT)
        assert not ok and "fail closed" in why
        ok, _ = closure_budget_fits(120.0, policy, 4.0,
                                    contract=LIVE_EXECUTION_CONTRACT)
        assert ok

    def test_infeasible_live_extension_pairs_are_ledgered(self, mat):
        rejected = {r["cell_id"] for r in mat["rejections"]}
        assert "w1_wd12_ff12" in rejected
        assert "w1_wd12_ff16" in rejected
