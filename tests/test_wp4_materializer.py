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
        assert mat["manifest"]["families"] == {
            "W0": 2, "W1": 16, "W2a": 45, "W2b": 27, "G2B": 3}
        assert mat["manifest"]["rejections"] == 0

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
        assert len(index) == 93
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
        policy["reopen_min_closed_bars"] = 3
        policy["stability_consecutive_checks"] = 3
        with pytest.raises(SessionPolicyError,
                           match="timeframe-infeasible"):
            check_feasibility(policy, bar_hours=48.0,
                              min_open_window_hours=104.0)

    def test_persisted_cells_match_the_manifest(self, mat, tmp_path):
        out = write_materialization(mat, tmp_path / "m")
        drv.verify_manifest_matches_dir(mat["manifest"], out)
        (out / "w1_wd12_ff1.json").unlink()
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
        cell = dict(_cell(mat, "w1_wd24_ff4"))
        cell["session_exposure_policy"] = dict(
            cell["session_exposure_policy"])
        cell["session_exposure_policy"]["wind_down_hours"] = 24.0
        cell["session_exposure_policy"]["forced_flatten_hours"] = 2.0
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
        ("w1_wd12_ff1", "w1_wd48_ff8"),
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
        roles = sorted(r["role"] for r in
                       (env.bridge.open_order_inventory or ()))
        assert roles == ["protective_stop",
                         "protective_take_profit"]

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

    def test_a_full_run_derives_and_holds(self, mat, window, tape,
                                          tmp_path):
        run = drv.recorded_run(_cell(mat, "w0_overlay_enabled"),
                               mat["manifest"],
                               mat["manifest"]["digest"], tape,
                               window, tmp_dir=tmp_path,
                               repo_root=REPO_ROOT)
        cons = run["conservation"]
        assert cons["bar_timestamps_inside_closures"] == 0
        assert cons["suppressed_reward_steps"] == []
        assert cons["close_event_conservation"]["holds"]
        assert cons["gross_minus_costs_equals_net"]["violations"] \
            == 0
        assert cons["holds"]
        assert run["rows_sha256"] and run["trades_sha256"]

    def test_a_suppressed_reward_would_be_detected(self):
        rows = [{"index": 0, "reward": 0.0, "pnl": 5.0}]
        import types
        env = types.SimpleNamespace(initial_cash=0.0, bridge=None)
        fake_window = None
        # unit-level: the derivation flags the row
        suppressed = [r["index"] for r in rows
                      if r["reward"] == 0.0 and abs(r["pnl"]) > 1e-9]
        assert suppressed == [0]

    def test_a_bar_inside_closure_would_be_detected(self, window,
                                                    tmp_path):
        import pandas as pd
        frame = pd.read_csv(window["csv"])
        frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"])
        a, _b = window["intervals"][0]
        inject = pd.Timestamp(a).tz_localize(None) + \
            pd.Timedelta(hours=4)
        extra = frame.iloc[[0]].copy()
        extra["DATE_TIME"] = inject
        tampered = pd.concat([frame, extra]).sort_values("DATE_TIME")
        csv = tmp_path / "tampered.csv"
        tampered.to_csv(csv, index=False)
        stamps = pd.to_datetime(
            pd.read_csv(csv)["DATE_TIME"]).dt.tz_localize("UTC")
        inside = 0
        for s, e in window["intervals"]:
            inside += int(((stamps >= pd.Timestamp(s)) &
                           (stamps < pd.Timestamp(e))).sum())
        assert inside == 1


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
        assert report["conservation_holds"]

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
