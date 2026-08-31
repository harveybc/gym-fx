"""WP4 materialization and driver battery (orders agent-multi@
ab5ce68d + label resolution agent-multi@2ab28bab).

Schema/identity/dry-run tests with no model effects, plus the bounded
mechanics smokes the resolution names: both W0 arms, W1 boundary and
default representatives including an infeasible refusal, W2 boundary
and default representatives including timeframe-infeasible and
missing-evidence refusals.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.session_exposure import SessionPolicyError
from tools.wp4_materializer import (G2_BASELINE_BARS,
                                    SECTION4_DEFAULTS, base_policy,
                                    canonical_bytes,
                                    check_feasibility, materialize,
                                    sha256_hex, verify_cell,
                                    write_materialization)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAL = "cal-weekly-v1"
IDENTITY = {"gymfx_commit": "2c60b84",
            "plan42": "agent-multi@45c49003"}


def _materialize():
    return materialize(calendar_identity=CAL, bar_hours=4.0,
                       min_open_window_hours=104.0,
                       identity=IDENTITY)


class TestMaterialization:

    def test_family_counts_match_plan_42(self):
        result = _materialize()
        families = result["manifest"]["families"]
        assert families == {"W0": 2, "W1": 16, "W2": 45}
        assert result["manifest"]["rejections"] == 0

    def test_w0_is_the_paired_comparison(self):
        cells = {c["cell_id"]: c for c in _materialize()["cells"]
                 if c["family"] == "W0"}
        control = cells["w0_control_disabled"]
        overlay = cells["w0_overlay_enabled"]
        assert control["session_exposure_policy"]["enabled"] is False
        assert control["live_deployable"] is False
        assert overlay["session_exposure_policy"]["enabled"] is True
        # everything except the single factor is identical
        a = dict(control["session_exposure_policy"])
        b = dict(overlay["session_exposure_policy"])
        a.pop("enabled"), b.pop("enabled")
        assert a == b

    def test_w1_changes_only_timing(self):
        cells = [c for c in _materialize()["cells"]
                 if c["family"] == "W1"]
        assert len(cells) == 16
        for cell in cells:
            policy = cell["session_exposure_policy"]
            for key, value in base_policy(CAL).items():
                if key in ("wind_down_hours",
                           "forced_flatten_hours"):
                    continue
                assert policy[key] == value, (cell["cell_id"], key)
            assert policy["forced_flatten_hours"] < \
                policy["wind_down_hours"]

    def test_w2_changes_only_reopen_and_is_economically_blocked(
            self):
        cells = [c for c in _materialize()["cells"]
                 if c["family"] == "W2"]
        assert len(cells) == 45
        for cell in cells:
            policy = cell["session_exposure_policy"]
            for key, value in base_policy(CAL).items():
                if key in ("reopen_min_hours",
                           "reopen_min_closed_bars",
                           "stability_consecutive_checks"):
                    continue
                assert policy[key] == value, (cell["cell_id"], key)
            timing = cell["w1_timing"]
            assert timing["status"] == "pending_w1_selection"
            assert timing["economic_execution"] == "BLOCKED"

    def test_cells_persist_with_digests_before_execution(
            self, tmp_path):
        out = write_materialization(_materialize(), tmp_path / "m")
        cells = sorted(out.glob("w*_*.json"))
        assert len(cells) == 63
        for path in cells:
            verify_cell(json.loads(path.read_text()))
        assert (out / "rejection_ledger.json").is_file()
        assert (out / "manifest.json").is_file()

    def test_an_altered_cell_refuses(self, tmp_path):
        result = _materialize()
        cell = dict(result["cells"][0])
        cell["session_exposure_policy"] = dict(
            cell["session_exposure_policy"])
        cell["session_exposure_policy"]["wind_down_hours"] = 999.0
        with pytest.raises(SessionPolicyError,
                           match="digest mismatch"):
            verify_cell(cell)

    def test_infeasible_pair_is_ledgered_not_launched(self):
        """A wind-down/flatten pair the ACCEPTED validator refuses
        (forced flatten after wind-down) lands in the rejection
        ledger with the typed reason."""
        policy = base_policy(CAL)
        policy["wind_down_hours"] = 4.0
        policy["forced_flatten_hours"] = 8.0
        with pytest.raises(SessionPolicyError,
                           match="forced flatten must occur AFTER"):
            check_feasibility(policy, bar_hours=4.0,
                              min_open_window_hours=104.0)

    def test_timeframe_infeasible_reopen_refuses(self):
        """On 48-hour bars the reopen gate could never exit inside
        the weekly open window — the mechanical rule refuses."""
        policy = base_policy(CAL)
        policy["reopen_min_closed_bars"] = 3
        policy["stability_consecutive_checks"] = 3
        with pytest.raises(SessionPolicyError,
                           match="timeframe-infeasible"):
            check_feasibility(policy, bar_hours=48.0,
                              min_open_window_hours=104.0)

    def test_wind_down_covering_the_open_window_refuses(self):
        policy = base_policy(CAL)
        policy["wind_down_hours"] = 120.0
        with pytest.raises(SessionPolicyError,
                           match="never leave wind-down"):
            check_feasibility(policy, bar_hours=4.0,
                              min_open_window_hours=104.0)

    def test_g2_baselines_are_declared_frozen(self):
        manifest = _materialize()["manifest"]
        assert manifest["g2_baseline_bars_frozen_not_in_plan_s4"] \
            == G2_BASELINE_BARS
        assert "reopen_baseline_bars" not in SECTION4_DEFAULTS


class TestDriver:

    def _cell(self, cell_id):
        return next(c for c in _materialize()["cells"]
                    if c["cell_id"] == cell_id)

    def test_identity_check_refuses_tampered_authority(
            self, tmp_path, monkeypatch):
        import tools.wp4_driver as drv
        monkeypatch.setitem(drv.FROZEN_AUTHORITY_SHA256,
                            "app/session_exposure.py", "0" * 64)
        with pytest.raises(drv.Wp4IdentityError,
                           match="refuses to run unreviewed"):
            drv.mechanics_smoke(self._cell("w0_overlay_enabled"),
                                tmp_dir=tmp_path,
                                repo_root=REPO_ROOT)

    def test_missing_calendar_evidence_fails_closed(self, tmp_path):
        """The accepted fail-closed contract: with the calendar
        intervals absent, every step declares
        session_evidence_failed_closed, the state degrades to
        WIND_DOWN, and no entry is admitted — never a neutral
        default."""
        import tools.wp4_driver as drv
        cell = self._cell("w2_h4_b1_c3")
        window = drv.gap_window(cell, tmp_dir=tmp_path)
        env = drv.build_env(cell, window, tmp_dir=tmp_path,
                            calendar_intervals=None)
        env.reset(seed=7)
        infos = []
        for _ in range(6):
            _obs, _r, term, _tr, info = env.step([1.0])
            infos.append(info)
            if term:
                break
        assert all(i["session_evidence_failed_closed"]
                   for i in infos)
        assert all(i["session_state"] == "WIND_DOWN"
                   for i in infos)
        assert all(i["session_signed_exposure"] == 0.0
                   for i in infos), "no entry may be admitted"

    @pytest.mark.parametrize("cell_id", [
        "w0_control_disabled", "w0_overlay_enabled"])
    def test_w0_smokes(self, tmp_path, cell_id):
        import tools.wp4_driver as drv
        result = drv.mechanics_smoke(self._cell(cell_id),
                                     tmp_dir=tmp_path,
                                     repo_root=REPO_ROOT)
        assert result["steps"] > 0
        assert result["no_bar_inside_closure"]
        if cell_id == "w0_overlay_enabled":
            counts = result["session_state_counts"]
            assert counts, "the overlay arm must publish states"
            assert any(s in counts for s in
                       ("WIND_DOWN", "FORCED_FLATTEN",
                        "REOPEN_BLACKOUT"))
        else:
            assert result["session_state_counts"] == {}

    @pytest.mark.parametrize("cell_id", [
        "w1_wd12_ff1", "w1_wd48_ff8", "w1_wd36_ff4"])
    def test_w1_boundary_and_default_smokes(self, tmp_path,
                                            cell_id):
        import tools.wp4_driver as drv
        result = drv.mechanics_smoke(self._cell(cell_id),
                                     tmp_dir=tmp_path,
                                     repo_root=REPO_ROOT)
        counts = result["session_state_counts"]
        assert counts.get("WIND_DOWN") or \
            counts.get("FORCED_FLATTEN"), counts

    @pytest.mark.parametrize("cell_id", [
        "w2_h1_b1_c1", "w2_h12_b3_c3", "w2_h4_b1_c3"])
    def test_w2_boundary_and_default_smokes(self, tmp_path,
                                            cell_id):
        import tools.wp4_driver as drv
        result = drv.mechanics_smoke(self._cell(cell_id),
                                     tmp_dir=tmp_path,
                                     repo_root=REPO_ROOT)
        assert result["session_state_counts"].get("REOPEN_BLACKOUT")

    def test_replay_is_deterministic(self, tmp_path):
        import tools.wp4_driver as drv
        assert drv.replay_is_deterministic(
            self._cell("w0_overlay_enabled"), tmp_dir=tmp_path,
            repo_root=REPO_ROOT)
