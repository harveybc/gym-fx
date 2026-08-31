"""WP4 cell materializer for work plan 42 (order agent-multi@ab5ce68d,
label resolution agent-multi@2ab28bab: PLAN_42_PREVAILS).

Materializes the three experiment FAMILIES exactly as plan 42 §7
defines them — W0 paired overlay comparison, W1 wind-down timing
grid, W2 reopen calibration grid — with one factor changed at a time
and NOTHING trained. Every candidate policy passes through the
ACCEPTED gym-fx validator (`app.session_exposure.validate_policy`);
this module never reinterprets the policy. Mechanically infeasible
candidates land in a rejection ledger with the typed reason, never in
the cell list.

W2 economic execution is BLOCKED by construction: its wind-down/
flatten timing must come from a W1 selection made on fit/calibration
evidence under a predeclared rule, and no such selection exists yet.
Each W2 cell records that status explicitly. Mechanics smokes may
run W2 cells using the section-4 default timing, which is predeclared
in plan 42 itself — that use carries ZERO economic authority and is
stamped on the cell.

Frozen constants that plan 42 §4 does not name are declared here,
never silently: the G2 baseline windows (`reopen_baseline_bars`,
`reopen_gap_sigma_bars`, `reopen_realized_vol_bars`) are frozen at
the accepted-suite value of 4 fully closed bars each.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.session_exposure import SessionPolicyError, validate_policy

PLAN42_SHA256 = ("26cfe85b15acbf286be8560669217f6ee5a6cf58311c161f"
                 "2f48d5eb431bef2d")

# plan 42 §4, verbatim numbers
SECTION4_DEFAULTS = {
    "enabled": True,
    "session_source": "venue_symbol_sessions_v1",
    "wind_down_hours": 36.0,
    "forced_flatten_hours": 4.0,
    "cancel_pending_on_wind_down": True,
    "allow_risk_increase_during_wind_down": False,
    "reopen_min_hours": 4.0,
    "reopen_min_closed_bars": 1,
    "stability_consecutive_checks": 3,
    "max_spread_relative_to_baseline": 2.0,
    "max_gap_sigma": 3.0,
    "max_realized_vol_relative_to_baseline": 2.0,
    "carried_position_recovery":
        "protected_opportunistic_then_forced",
    "holiday_policy": "same_as_weekly",
}

# WP4-frozen, NOT from plan §4: the G2 baseline windows, at the
# accepted-suite value. Declared openly here and in the manifest.
G2_BASELINE_BARS = {
    "reopen_baseline_bars": 4,
    "reopen_gap_sigma_bars": 4,
    "reopen_realized_vol_bars": 4,
}

# plan 42 §7, predeclared grids — no post-result additions
W1_WIND_DOWN_HOURS = (12.0, 24.0, 36.0, 48.0)
W1_FORCED_FLATTEN_HOURS = (1.0, 2.0, 4.0, 8.0)
W2_REOPEN_MIN_HOURS = (1.0, 2.0, 4.0, 8.0, 12.0)
W2_REOPEN_MIN_CLOSED_BARS = (1, 2, 3)
W2_STABILITY_CHECKS = (1, 2, 3)

# WP4-C6 (order agent-multi@051ef265): W2 is SPLIT. W2a screens
# hours/bars/checks with the spread/gap/volatility thresholds frozen
# PROVISIONALLY at the section-4 values — it carries NO claim that W2
# is complete. W2b calibrates the thresholds over these predeclared
# bounded domains, and its trials enter the multiplicity ledger.
W2B_SPREAD_DOMAIN = (1.5, 2.0, 2.5)
W2B_GAP_SIGMA_DOMAIN = (2.0, 3.0, 4.0)
W2B_VOL_DOMAIN = (1.5, 2.0, 2.5)

# WP4-C6: the G2 baseline windows are treatment-bearing. Mechanical
# rationale for the provisional value 4: the smallest window is 2
# (the validator's floor, one degree of freedom for a dispersion
# estimate); 4 closed bars give three degrees of freedom while still
# fitting inside the shortest post-reopen stretch of the bound
# history at the 4h timeframe. That rationale does NOT establish
# optimality, so the value is also ABLATED over this bounded domain.
G2_BASELINE_ABLATION = (2, 4, 8)


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True,
                      separators=(",", ":"),
                      ensure_ascii=True).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def base_policy(calendar_identity: str) -> dict:
    policy = dict(SECTION4_DEFAULTS)
    policy["calendar_identity"] = calendar_identity
    policy.update(G2_BASELINE_BARS)
    return policy


def timeframe_exit_hours(policy: dict, bar_hours: float) -> float:
    """The mechanical worst-case hours the reopen gate needs before
    it can possibly exit: the baseline window must fill, the minimum
    closed bars must form, the stability checks must run on further
    closed bars, and the minimum hours must also elapse."""
    bars_needed = (policy["reopen_baseline_bars"]
                   + policy["reopen_min_closed_bars"]
                   + policy["stability_consecutive_checks"])
    return max(float(policy["reopen_min_hours"]),
               bars_needed * float(bar_hours))


def check_feasibility(policy: dict, *, bar_hours: float,
                      min_open_window_hours: float) -> dict:
    """ACCEPTED validator first, then the mechanical timeframe rule.
    Returns the validated policy or raises SessionPolicyError with
    the typed reason — the caller ledgers it, never launches it."""
    validated = validate_policy(policy)
    exit_hours = timeframe_exit_hours(validated, bar_hours)
    if exit_hours >= min_open_window_hours:
        raise SessionPolicyError(
            f"timeframe-infeasible: the reopen gate needs up to "
            f"{exit_hours}h to exit on {bar_hours}h bars but the "
            f"shortest open window is {min_open_window_hours}h — "
            "the blackout could never exit before the next closure")
    if validated["enabled"] and \
            float(validated["wind_down_hours"]) >= \
            min_open_window_hours:
        raise SessionPolicyError(
            f"timeframe-infeasible: wind_down_hours "
            f"{validated['wind_down_hours']} covers the whole open "
            f"window of {min_open_window_hours}h — the session "
            "would never leave wind-down")
    return validated


def materialize(*, calendar_identity: str, bar_hours: float,
                min_open_window_hours: float,
                identity: dict) -> dict:
    """Build every W0/W1/W2 cell and the rejection ledger. Nothing
    is trained and nothing economic is selected."""
    cells, rejections = [], []

    def admit(family: str, cell_id: str, policy: dict,
              extra: dict | None = None) -> None:
        try:
            validated = check_feasibility(
                policy, bar_hours=bar_hours,
                min_open_window_hours=min_open_window_hours)
        except SessionPolicyError as exc:
            rejections.append({"family": family, "cell_id": cell_id,
                               "candidate": policy,
                               "reason": str(exc)})
            return
        cell = {
            "cell_id": cell_id,
            "family": family,
            "session_exposure_policy": validated,
            "bar_hours": bar_hours,
            "min_open_window_hours": min_open_window_hours,
            "identity": identity,
        }
        if extra:
            cell.update(extra)
        cell["digest"] = sha256_hex(canonical_bytes(cell))
        cells.append(cell)

    # -- W0: paired overlay comparison (plan 42 §7 W0) --------------
    disabled = base_policy(calendar_identity)
    disabled["enabled"] = False
    admit("W0", "w0_control_disabled", disabled, {
        "role": "diagnostic_control",
        "live_deployable": False,
        "note": "diagnostic only; owner policy requires flat "
                "weekends, so this arm can never deploy live"})
    admit("W0", "w0_overlay_enabled", base_policy(calendar_identity),
          {"role": "full_accepted_overlay_at_section4_defaults"})

    # -- W1: wind-down timing family (plan 42 §7 W1) ----------------
    for wind in W1_WIND_DOWN_HOURS:
        for flatten in W1_FORCED_FLATTEN_HOURS:
            policy = base_policy(calendar_identity)
            policy["wind_down_hours"] = wind
            policy["forced_flatten_hours"] = flatten
            admit("W1",
                  f"w1_wd{wind:g}_ff{flatten:g}", policy,
                  {"reopen_policy": "frozen_at_section4_defaults"})

    # -- W2a: reopen mechanism screen (plan 42 §7 W2, C6 split) -----
    W2_TIMING_BLOCK = {
        "status": "pending_w1_selection",
        "economic_execution": "BLOCKED",
        "rule": "W2 timing must be a W1 selection frozen from "
                "fit/calibration evidence under a predeclared "
                "rule; no such selection exists",
        "mechanics_smoke_timing":
            "section4_defaults_zero_economic_authority",
    }
    for hours in W2_REOPEN_MIN_HOURS:
        for bars in W2_REOPEN_MIN_CLOSED_BARS:
            for checks in W2_STABILITY_CHECKS:
                policy = base_policy(calendar_identity)
                policy["reopen_min_hours"] = hours
                policy["reopen_min_closed_bars"] = bars
                policy["stability_consecutive_checks"] = checks
                admit("W2a",
                      f"w2a_h{hours:g}_b{bars}_c{checks}", policy, {
                          "w1_timing": dict(W2_TIMING_BLOCK),
                          "role": "provisional_mechanism_screen",
                          "completeness_claim": "NONE — the spread/"
                              "gap/volatility thresholds are frozen "
                              "PROVISIONALLY at section-4 values; "
                              "W2b calibrates them and W2 is not "
                              "complete until it does",
                          "thresholds":
                              "provisionally_frozen_at_section4"})

    # -- W2b: threshold calibration (C6, predeclared domains) -------
    for spread in W2B_SPREAD_DOMAIN:
        for gap in W2B_GAP_SIGMA_DOMAIN:
            for vol in W2B_VOL_DOMAIN:
                policy = base_policy(calendar_identity)
                policy["max_spread_relative_to_baseline"] = spread
                policy["max_gap_sigma"] = gap
                policy["max_realized_vol_relative_to_baseline"] = vol
                admit("W2b",
                      f"w2b_s{spread:g}_g{gap:g}_v{vol:g}", policy, {
                          "w1_timing": dict(W2_TIMING_BLOCK),
                          "role": "threshold_calibration",
                          "gated_on": "the frozen W1 timing AND the "
                                      "W2a mechanism screen",
                          "reopen_counts":
                              "frozen_at_section4_defaults"})

    # -- G2B: baseline-window ablation (C6) -------------------------
    for baseline in G2_BASELINE_ABLATION:
        policy = base_policy(calendar_identity)
        policy["reopen_baseline_bars"] = baseline
        policy["reopen_gap_sigma_bars"] = baseline
        policy["reopen_realized_vol_bars"] = baseline
        admit("G2B", f"g2b_base{baseline}", policy, {
            "w1_timing": dict(W2_TIMING_BLOCK),
            "role": "g2_baseline_window_ablation",
            "rationale": "the provisional value 4 has a mechanical "
                         "rationale (three dispersion degrees of "
                         "freedom inside the shortest post-reopen "
                         "stretch) but no optimality claim; this "
                         "bounded ablation carries the burden"})

    manifest = {
        "schema": "gymfx.wp4.materialization.v2",
        "plan42_sha256": PLAN42_SHA256,
        "identity": identity,
        "bar_hours": bar_hours,
        "min_open_window_hours": min_open_window_hours,
        "g2_baseline_bars_provisional": G2_BASELINE_BARS,
        "cells": len(cells),
        "rejections": len(rejections),
        "families": {
            family: sum(1 for c in cells if c["family"] == family)
            for family in ("W0", "W1", "W2a", "W2b", "G2B")},
        # C1: the manifest, not the cell, is the external binding —
        # every cell id and digest is enumerated here, and the
        # manifest digest is what a reviewed dispatch binds
        "cell_index": {c["cell_id"]: c["digest"] for c in cells},
        # C7: every attempted cell is a recorded trial
        "trial_ledger": sorted(c["cell_id"] for c in cells),
    }
    manifest["digest"] = sha256_hex(canonical_bytes(manifest))
    return {"manifest": manifest, "cells": cells,
            "rejections": rejections}


def write_materialization(result: dict, out_dir: Path) -> Path:
    """Persist the effective configs and their digests BEFORE any
    execution, plus the rejection ledger."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in result["cells"]:
        path = out_dir / f"{cell['cell_id']}.json"
        path.write_text(json.dumps(cell, indent=1, sort_keys=True))
    (out_dir / "rejection_ledger.json").write_text(
        json.dumps(result["rejections"], indent=1, sort_keys=True))
    (out_dir / "manifest.json").write_text(
        json.dumps(result["manifest"], indent=1, sort_keys=True))
    return out_dir


def verify_cell(cell: dict) -> dict:
    """Digest-verify a persisted cell before it may drive anything."""
    body = {k: v for k, v in cell.items() if k != "digest"}
    if cell.get("digest") != sha256_hex(canonical_bytes(body)):
        raise SessionPolicyError(
            f"cell {cell.get('cell_id')!r}: digest mismatch — an "
            "altered cell drives nothing")
    revalidated = validate_policy(dict(
        cell["session_exposure_policy"]))
    if revalidated != cell["session_exposure_policy"]:
        raise SessionPolicyError(
            f"cell {cell.get('cell_id')!r}: policy no longer "
            "revalidates identically — refused")
    return cell
