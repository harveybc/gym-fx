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
    # F5: the probation factor is explicit policy identity; 2 is the
    # mandatory provisional minimum for any live-capable arm
    "release_probation_factor": 2,
}

# WP4-frozen, NOT from plan §4: the G2 baseline windows, at the
# accepted-suite value. Declared openly here and in the manifest.
G2_BASELINE_BARS = {
    "reopen_baseline_bars": 4,
    "reopen_gap_sigma_bars": 4,
    "reopen_realized_vol_bars": 4,
}

# F4 (order agent-multi@a678fd55): the EXECUTION CONTRACT that
# feasibility must respect. A forced-flatten window is admissible
# only if the worst-case permitted close can be FILLED and
# RECONCILED before the closure under this contract. The H4
# next-bar-fill contract is what the accepted simulator implements;
# the reproduced 4-hour failure is ledgered by these numbers.
EXECUTION_CONTRACT = {
    "decision_at": "bar_close",
    "submission_latency_bars": 0,
    "fill_at": "next_bar_open",
    "reconcile_at": "fill_bar_step",
    "close_retry_budget_bars": 0,
    "safety_margin_hours": 0.0,
}

# F8 (order agent-multi@22218df1): the LIVE contract. Mechanics
# admissibility (retry 0, margin 0) is NOT live safety: one rejected
# or delayed close would leave no second executable fill before the
# closure. The live contract demands ONE full retry opportunity
# after an observed rejection/non-fill, reconciliation before the
# closure, and a POSITIVE safety margin tied to the venue boundary.
# All latencies are measured in bars.
LIVE_EXECUTION_CONTRACT = {
    "decision_at": "bar_close",
    "submission_latency_bars": 0,
    "fill_at": "next_bar_open",
    "reconcile_at": "fill_bar_step",
    "close_retry_budget_bars": 1,
    "safety_margin_hours": 1.0,
}

# F8: the mechanical extension of the predeclared flatten domain —
# the smallest grid values that can satisfy the live contract on H4
# (12h: first fill, observed verdict, retry fill and reconciliation
# all before closure with the margin; 16h adds one bar of headroom).
W1_FORCED_FLATTEN_LIVE_EXTENSION = (12.0, 16.0)


def flatten_deadline_admissible(ff_hours: float, bar_hours: float,
                                *, contract: dict) -> tuple:
    """Mechanical admissibility of a forced-flatten window.

    The flatten first triggers at the largest bar-grid multiple of
    hours-to-close that is <= ff_hours. The worst-case sequence is:
    trigger decision, submission (+latency bars), first fill at the
    next bar open, an observed verdict, and up to
    close_retry_budget_bars retry fills — the FINAL fill bar must
    open strictly before the closure and leave at least the safety
    margin between that fill and the closure boundary. Everything is
    derived, nothing is tuned. (F8 refactor of the F4 rule to the
    slack form; every F4 verdict is unchanged — the suite proves
    it — and retry/margin now participate correctly.)"""
    import math
    trigger = math.floor(float(ff_hours) / float(bar_hours)) \
        * float(bar_hours)
    if trigger <= 0.0:
        return False, (
            f"flatten window {ff_hours}h never reaches a decision "
            f"bar on the {bar_hours}h grid — the close could never "
            "even be submitted before the closure")
    fills = (1 + contract["submission_latency_bars"]
             + contract["close_retry_budget_bars"])
    slack = trigger - fills * float(bar_hours)
    margin = float(contract["safety_margin_hours"])
    if slack <= 0.0 or slack < margin:
        return False, (
            f"execution-latency infeasible: the flatten first "
            f"triggers {trigger}h before close on the {bar_hours}h "
            f"grid; after {contract['submission_latency_bars']} "
            f"latency bars, the first fill and "
            f"{contract['close_retry_budget_bars']} retry fills, "
            f"the final fill lands {slack}h before the closure and "
            f"the contract demands a positive margin of {margin}h "
            "— the worst-case close cannot fill and reconcile "
            "before the closure")
    return True, None


def closure_budget_fits(open_window_hours: float, policy: dict,
                        bar_hours: float, *,
                        contract: dict) -> tuple:
    """F8: a holiday-shortened session that cannot fit the flatten
    budget FAILS CLOSED — the trigger bar must exist inside the open
    stretch before the closure, or exposure there can never be
    flattened in time."""
    import math
    trigger = math.floor(
        float(policy["forced_flatten_hours"]) / float(bar_hours)) \
        * float(bar_hours)
    if float(open_window_hours) <= trigger:
        return False, (
            f"holiday-shortened session: the open stretch is "
            f"{open_window_hours}h but the flatten budget needs the "
            f"trigger bar {trigger}h before the closure — the "
            "budget cannot fit and entries there must fail closed")
    return True, None


def corrected_flatten_default(bar_hours: float, *,
                              contract: dict) -> float:
    """The live-safe flatten default MUST come from an eligible
    value: the smallest value of the predeclared grid PLUS its
    mechanical live extension that is admissible under the given
    contract. It may not silently remain the section-4 four hours.
    F8: enabled arms default under the LIVE contract (H4 -> 12h);
    8h stays admissible for mechanics only."""
    domain = sorted(set(W1_FORCED_FLATTEN_HOURS)
                    | set(W1_FORCED_FLATTEN_LIVE_EXTENSION))
    for candidate in domain:
        ok, _reason = flatten_deadline_admissible(
            candidate, bar_hours, contract=contract)
        if ok:
            return candidate
    raise SessionPolicyError(
        f"no predeclared flatten value is admissible on "
        f"{bar_hours}h bars under the execution contract")


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


def base_policy(calendar_identity: str, *,
                bar_hours: float = 4.0) -> dict:
    policy = dict(SECTION4_DEFAULTS)
    policy["calendar_identity"] = calendar_identity
    policy.update(G2_BASELINE_BARS)
    # F4/F8: the section-4 forced_flatten_hours=4 default is
    # structurally ineligible for H4 under next-bar fills, and the
    # 8h mechanics value is NOT live-safe (no retry, no margin).
    # Every enabled arm defaults to the smallest LIVE-safe value
    # (H4 -> 12h) and the correction is ledgered on the manifest.
    policy["forced_flatten_hours"] = corrected_flatten_default(
        bar_hours, contract=LIVE_EXECUTION_CONTRACT)
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
    """ACCEPTED validator first, then the mechanical timeframe rule
    and the F4 execution-latency rule. Returns the validated policy
    or raises SessionPolicyError with the typed reason — the caller
    ledgers it, never launches it."""
    validated = validate_policy(policy)
    if validated["enabled"]:
        ok, reason = flatten_deadline_admissible(
            validated["forced_flatten_hours"], bar_hours,
            contract=EXECUTION_CONTRACT)
        if not ok:
            raise SessionPolicyError(reason)
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

    # -- W1: wind-down timing family (plan 42 §7 W1 + F8 live
    # extension). Every admitted cell is labelled with its LIVE
    # safety under the live contract; mechanics-only values stay
    # runnable but can never be called live-safe.
    for wind in W1_WIND_DOWN_HOURS:
        for flatten in (tuple(W1_FORCED_FLATTEN_HOURS)
                        + tuple(W1_FORCED_FLATTEN_LIVE_EXTENSION)):
            policy = base_policy(calendar_identity)
            policy["wind_down_hours"] = wind
            policy["forced_flatten_hours"] = flatten
            live_ok, live_why = flatten_deadline_admissible(
                flatten, bar_hours,
                contract=LIVE_EXECUTION_CONTRACT)
            admit("W1",
                  f"w1_wd{wind:g}_ff{flatten:g}", policy,
                  {"reopen_policy": "frozen_at_section4_defaults",
                   "live_safe_flatten": bool(live_ok),
                   "live_safety_note": (
                       "satisfies the LIVE contract (retry + "
                       "margin)" if live_ok else
                       f"MECHANICS ONLY — {live_why}")})

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

    # -- PROB: probation-factor ablation (F5, non-live) -------------
    for factor in (1, 2, 3):
        policy = base_policy(calendar_identity)
        policy["release_probation_factor"] = factor
        release_bars = (policy["stability_consecutive_checks"]
                        * factor)
        admit("PROB", f"prob_factor{factor}", policy, {
            "w1_timing": dict(W2_TIMING_BLOCK),
            "role": "release_probation_ablation",
            "live_eligible": factor >= 2 and release_bars >= 2,
            "live_note": ("factor 2 is the mandatory provisional "
                          "minimum for any live-capable arm; no arm "
                          "with a one-bar release window is "
                          "live-eligible" if factor < 2 or
                          release_bars < 2 else
                          "live-capable at the provisional minimum "
                          "or above")})

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
        "execution_contract": EXECUTION_CONTRACT,
        "live_execution_contract": LIVE_EXECUTION_CONTRACT,
        "w1_flatten_live_extension":
            list(W1_FORCED_FLATTEN_LIVE_EXTENSION),
        "flatten_default_correction": {
            "section4_value_hours": 4.0,
            "status": "STRUCTURALLY_INELIGIBLE_FOR_H4_NEXT_BAR",
            "mechanics_only_hours": corrected_flatten_default(
                bar_hours, contract=EXECUTION_CONTRACT),
            "live_safe_default_hours": corrected_flatten_default(
                bar_hours, contract=LIVE_EXECUTION_CONTRACT),
            "rule": "no value is called live-safe unless the first "
                    "attempt, terminal verdict, retry fill and "
                    "reconciliation all fit before the closure "
                    "with a positive margin; the plan-42 section-4 "
                    "text is amended in the same return"},
        "cells": len(cells),
        "rejections": len(rejections),
        "families": {
            family: sum(1 for c in cells if c["family"] == family)
            for family in ("W0", "W1", "W2a", "W2b", "G2B",
                           "PROB")},
        # C1: the manifest, not the cell, is the external binding —
        # every cell id and digest is enumerated here, and the
        # manifest digest is what a reviewed dispatch binds
        "cell_index": {c["cell_id"]: c["digest"] for c in cells},
        # C7: every attempted cell is a recorded trial
        "trial_ledger": sorted(c["cell_id"] for c in cells),
        # C13: the joint confirmation is PREDECLARED here — its
        # cells are constructible only once W1/W2a/W2b/G2B
        # selections exist, and they enter this same ledger
        "joint_confirmation_predeclaration": {
            "constructor": "materialize_joint_confirmation",
            "contents": "selected combination + section-4 control "
                        "+ bounded one-step neighbours of the "
                        "selected W2a coordinates inside the "
                        "predeclared grids",
            "rule": "no W2 candidate is promotion-eligible before "
                    "the joint confirmation on untouched data; "
                    "every joint cell enters the multiplicity "
                    "ledger",
        },
    }
    manifest["digest"] = sha256_hex(canonical_bytes(manifest))
    return {"manifest": manifest, "cells": cells,
            "rejections": rejections}


def materialize_joint_confirmation(*, calendar_identity: str,
                                   bar_hours: float,
                                   min_open_window_hours: float,
                                   identity: dict,
                                   selected_w1: dict,
                                   selected_w2a: dict,
                                   selected_w2b: dict,
                                   selected_g2b: int) -> dict:
    """WP4-C13: the predeclared JOINT confirmation. Independent
    coordinate screens (W2a, W2b, G2B) do not test interactions, so
    no W2 candidate is promotion-eligible before this joint set is
    evaluated on untouched data: the selected combination, the
    section-4 control, and every bounded one-step neighbour of the
    selected combination inside the predeclared grids. Every cell
    here enters the multiplicity ledger like any other trial."""
    def neighbours(value, grid):
        grid = sorted(grid)
        index = grid.index(value)
        out = {value}
        if index > 0:
            out.add(grid[index - 1])
        if index + 1 < len(grid):
            out.add(grid[index + 1])
        return sorted(out)

    cells, rejections = [], []

    def admit(cell_id, policy, extra):
        try:
            validated = check_feasibility(
                policy, bar_hours=bar_hours,
                min_open_window_hours=min_open_window_hours)
        except SessionPolicyError as exc:
            rejections.append({"family": "W2JOINT",
                               "cell_id": cell_id,
                               "reason": str(exc)})
            return
        cell = {"cell_id": cell_id, "family": "W2JOINT",
                "session_exposure_policy": validated,
                "bar_hours": bar_hours,
                "min_open_window_hours": min_open_window_hours,
                "identity": identity, **extra}
        cell["digest"] = sha256_hex(canonical_bytes(cell))
        cells.append(cell)

    # section-4 control
    admit("w2joint_section4_control", base_policy(calendar_identity),
          {"role": "joint_confirmation_control"})
    # the selected combination and its bounded one-step neighbours
    combos = set()
    for hours in neighbours(selected_w2a["reopen_min_hours"],
                            W2_REOPEN_MIN_HOURS):
        for bars in neighbours(selected_w2a["reopen_min_closed_bars"],
                               W2_REOPEN_MIN_CLOSED_BARS):
            for checks in neighbours(
                    selected_w2a["stability_consecutive_checks"],
                    W2_STABILITY_CHECKS):
                combos.add((hours, bars, checks))
    for hours, bars, checks in sorted(combos):
        policy = base_policy(calendar_identity)
        policy["wind_down_hours"] = selected_w1["wind_down_hours"]
        policy["forced_flatten_hours"] =             selected_w1["forced_flatten_hours"]
        policy["reopen_min_hours"] = hours
        policy["reopen_min_closed_bars"] = bars
        policy["stability_consecutive_checks"] = checks
        policy["max_spread_relative_to_baseline"] =             selected_w2b["max_spread_relative_to_baseline"]
        policy["max_gap_sigma"] = selected_w2b["max_gap_sigma"]
        policy["max_realized_vol_relative_to_baseline"] =             selected_w2b["max_realized_vol_relative_to_baseline"]
        policy["reopen_baseline_bars"] = selected_g2b
        policy["reopen_gap_sigma_bars"] = selected_g2b
        policy["reopen_realized_vol_bars"] = selected_g2b
        selected = (hours == selected_w2a["reopen_min_hours"] and
                    bars == selected_w2a["reopen_min_closed_bars"]
                    and checks == selected_w2a[
                        "stability_consecutive_checks"])
        admit(f"w2joint_h{hours:g}_b{bars}_c{checks}", policy, {
            "role": ("joint_selected_combination" if selected
                     else "joint_bounded_neighbour"),
            "promotion_rule": "NO W2 candidate is promotion-"
                              "eligible before this joint set is "
                              "confirmed on untouched data"})
    return {"cells": cells, "rejections": rejections}


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
