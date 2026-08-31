"""WP4-C14: the EXECUTABLE statistical protocol (order
agent-multi@c4b0ecf9), materializing the v2 predeclaration as a
validator/aggregator with typed refusals. Nothing here trains or
selects economically — it is the machine that will judge economic
runs when they are separately authorized.

Contract (predeclared, agent-multi@58269487, now executable):
- the economic unit is the paired CLOSURE-WEEK, never the seed;
- every cell of a family must present the identical closure-week set
  under the identical seed set (missing pairs and duplicates refuse);
- roles are fit/calibration/decision and never mix: a winner is
  never judged on data that chose it;
- per-week paired differentials are aggregated across seeds FIRST
  (hierarchical), and seed dispersion is reported separately;
- inference is a stationary (Politis-Romano) block bootstrap over
  closure weeks with a deterministic seed;
- minimum 30 eligible closure weeks, else INCONCLUSIVE;
- a selection needs the bootstrap 90% CI half-width to be at most
  half the point estimate, else INCONCLUSIVE;
- closure compliance is a HARD gate for enabled arms;
- the family procedure is an SPA-style max-statistic stationary
  bootstrap against the declared benchmark; Holm is descriptive
  only and is not implemented as a selector;
- the one-SE tie rule uses the closure-week bootstrap SE, never a
  seed-level SE;
- every attempted cell must appear in the trial ledger, or the
  family refuses evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

MIN_ELIGIBLE_WEEKS = 30
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_BLOCK_WEEKS = 4
CI_LEVEL = 0.90
ROLES = ("fit", "calibration", "decision")


class StatsProtocolError(ValueError):
    """A typed refusal from the executable protocol."""


@dataclass(frozen=True)
class WeekRecord:
    """One cell x seed x closure-week observation."""

    closure_week_id: str
    cell_id: str
    seed: int
    role: str
    net_return_after_costs: float
    closure_compliant: bool

    def __post_init__(self):
        if not isinstance(self.closure_week_id, str) or \
                not self.closure_week_id:
            raise StatsProtocolError("closure_week_id required")
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise StatsProtocolError("cell_id required")
        if isinstance(self.seed, bool) or \
                not isinstance(self.seed, int):
            raise StatsProtocolError("seed must be an int")
        if self.role not in ROLES:
            raise StatsProtocolError(
                f"role {self.role!r} is not one of {ROLES}")
        value = self.net_return_after_costs
        if not isinstance(value, float) or not np.isfinite(value):
            raise StatsProtocolError(
                "net_return_after_costs must be a finite float")
        if not isinstance(self.closure_compliant, bool):
            raise StatsProtocolError(
                "closure_compliant must be a bool")


def validate_dataset(records: Sequence[WeekRecord], *,
                     role: str) -> dict:
    """Exact paired support: every cell must present the identical
    (closure_week, seed) grid for the given role. Duplicates,
    missing pairs and foreign roles refuse."""
    if role not in ROLES:
        raise StatsProtocolError(f"unknown role {role!r}")
    scoped = [r for r in records if r.role == role]
    if not scoped:
        raise StatsProtocolError(f"no records for role {role!r}")
    seen = set()
    for record in scoped:
        key = (record.cell_id, record.closure_week_id, record.seed)
        if key in seen:
            raise StatsProtocolError(
                f"duplicate week record {key} — refused")
        seen.add(key)
    cells = sorted({r.cell_id for r in scoped})
    grids = {}
    for cell in cells:
        grids[cell] = {(r.closure_week_id, r.seed)
                       for r in scoped if r.cell_id == cell}
    reference = grids[cells[0]]
    for cell in cells[1:]:
        if grids[cell] != reference:
            missing = sorted(reference - grids[cell])[:3]
            extra = sorted(grids[cell] - reference)[:3]
            raise StatsProtocolError(
                f"cell {cell!r} does not present the identical "
                f"paired support — missing {missing}, extra "
                f"{extra}; unpaired evaluation refused")
    weeks = sorted({w for w, _s in reference})
    seeds = sorted({s for _w, s in reference})
    return {"cells": cells, "weeks": weeks, "seeds": seeds,
            "records": scoped}


def compliance_gate(records: Sequence[WeekRecord],
                    cell_id: str) -> dict:
    """HARD gate: any non-compliant closure week makes the cell
    ineligible for selection."""
    bad = sorted({r.closure_week_id for r in records
                  if r.cell_id == cell_id
                  and not r.closure_compliant})
    return {"cell_id": cell_id, "non_compliant_weeks": bad,
            "eligible": not bad}


def weekly_differentials(dataset: dict, cell_id: str,
                         benchmark_id: str) -> dict:
    """Per-closure-week paired differentials, seeds aggregated
    HIERARCHICALLY first. Seed dispersion is reported separately
    and never enters the week-level series."""
    for name in (cell_id, benchmark_id):
        if name not in dataset["cells"]:
            raise StatsProtocolError(f"unknown cell {name!r}")
    by_key = {}
    for record in dataset["records"]:
        by_key[(record.cell_id, record.closure_week_id,
                record.seed)] = record.net_return_after_costs
    weekly = []
    seed_spread = []
    for week in dataset["weeks"]:
        per_seed = [by_key[(cell_id, week, seed)]
                    - by_key[(benchmark_id, week, seed)]
                    for seed in dataset["seeds"]]
        weekly.append(float(np.mean(per_seed)))
        seed_spread.append(float(np.std(per_seed, ddof=1))
                           if len(per_seed) > 1 else 0.0)
    return {"weeks": dataset["weeks"],
            "weekly_differentials": weekly,
            "n_weeks": len(weekly),
            "seed_dispersion_by_week": seed_spread,
            "seed_dispersion_mean": float(np.mean(seed_spread)),
            "unit": "paired closure-week (seeds aggregated "
                    "hierarchically; NEVER treated as weeks)"}


def stationary_bootstrap_indices(n: int, resamples: int,
                                 expected_block: float,
                                 seed: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths,
    circular wrap, deterministic seed."""
    if n < 2:
        raise StatsProtocolError(
            "stationary bootstrap needs at least 2 observations")
    rng = np.random.default_rng(int(seed))
    p = 1.0 / float(expected_block)
    starts = rng.integers(0, n, size=(resamples, n))
    new_block = rng.random(size=(resamples, n)) < p
    new_block[:, 0] = True
    indices = np.zeros((resamples, n), dtype=np.int64)
    for column in range(n):
        indices[:, column] = np.where(
            new_block[:, column], starts[:, column],
            (indices[:, column - 1] + 1) % n)
    return indices


def bootstrap_mean(diffs: Sequence[float], *, seed: int,
                   resamples: int = BOOTSTRAP_RESAMPLES,
                   expected_block: float = EXPECTED_BLOCK_WEEKS
                   ) -> dict:
    values = np.asarray(diffs, dtype=float)
    indices = stationary_bootstrap_indices(
        len(values), resamples, expected_block, seed)
    means = values[indices].mean(axis=1)
    low, high = np.percentile(
        means, [100 * (1 - CI_LEVEL) / 2,
                100 * (1 + CI_LEVEL) / 2])
    return {"point": float(values.mean()),
            "se": float(means.std(ddof=1)),
            "ci90": [float(low), float(high)],
            "resamples": resamples,
            "expected_block_weeks": expected_block,
            "bootstrap_seed": int(seed)}


def evaluate_cell(dataset: dict, cell_id: str, benchmark_id: str, *,
                  bootstrap_seed: int) -> dict:
    """One cell versus the benchmark under the full contract:
    compliance gate, minimum support, CI precision — INCONCLUSIVE
    paths instead of forced winners."""
    gate = compliance_gate(dataset["records"], cell_id)
    if not gate["eligible"]:
        return {"cell_id": cell_id, "verdict": "INELIGIBLE",
                "reason": "closure compliance gate",
                "non_compliant_weeks": gate["non_compliant_weeks"]}
    diffs = weekly_differentials(dataset, cell_id, benchmark_id)
    if diffs["n_weeks"] < MIN_ELIGIBLE_WEEKS:
        return {"cell_id": cell_id, "verdict": "INCONCLUSIVE",
                "reason": f"only {diffs['n_weeks']} eligible closure "
                          f"weeks; {MIN_ELIGIBLE_WEEKS} required"}
    boot = bootstrap_mean(diffs["weekly_differentials"],
                          seed=bootstrap_seed)
    half_width = (boot["ci90"][1] - boot["ci90"][0]) / 2.0
    if abs(boot["point"]) < 1e-12 or \
            half_width > 0.5 * abs(boot["point"]):
        verdict = "INCONCLUSIVE"
        reason = ("CI precision rule: half-width "
                  f"{half_width:.6g} exceeds half of "
                  f"|{boot['point']:.6g}|")
    else:
        verdict = "CONCLUSIVE"
        reason = None
    return {"cell_id": cell_id, "verdict": verdict,
            "reason": reason, "bootstrap": boot,
            "seed_dispersion_mean": diffs["seed_dispersion_mean"],
            "n_weeks": diffs["n_weeks"]}


def spa_family(dataset: dict, benchmark_id: str, *,
               trial_ledger: Sequence[str],
               bootstrap_seed: int) -> dict:
    """SPA-style family check: the max studentized mean differential
    over the WHOLE family versus its stationary-bootstrap null. The
    family is every ledger cell — an omitted trial refuses."""
    family = sorted(c for c in dataset["cells"]
                    if c != benchmark_id)
    ledger = set(trial_ledger)
    missing = sorted(set(family) - ledger)
    if missing:
        raise StatsProtocolError(
            f"cells missing from the trial ledger: {missing} — "
            "family evaluation refused")
    series = {}
    for cell in family:
        gate = compliance_gate(dataset["records"], cell)
        if gate["eligible"]:
            series[cell] = np.asarray(weekly_differentials(
                dataset, cell, benchmark_id)
                ["weekly_differentials"])
    if not series:
        return {"verdict": "NO_ELIGIBLE_CELLS"}
    n = len(next(iter(series.values())))
    if n < MIN_ELIGIBLE_WEEKS:
        return {"verdict": "INCONCLUSIVE",
                "reason": f"{n} weeks < {MIN_ELIGIBLE_WEEKS}"}
    indices = stationary_bootstrap_indices(
        n, BOOTSTRAP_RESAMPLES, EXPECTED_BLOCK_WEEKS,
        bootstrap_seed)
    observed, null_max = {}, np.full(BOOTSTRAP_RESAMPLES, -np.inf)
    for cell, values in series.items():
        se = values.std(ddof=1) / np.sqrt(n) or 1e-12
        observed[cell] = float(values.mean() / se)
        boot = values[indices].mean(axis=1)
        centred = (boot - values.mean()) / se
        null_max = np.maximum(null_max, centred)
    best_cell = max(observed, key=observed.get)
    best_stat = observed[best_cell]
    p_value = float((null_max >= best_stat).mean())
    return {"verdict": "EVALUATED", "best_cell": best_cell,
            "max_statistic": best_stat, "spa_p_value": p_value,
            "eligible_cells": sorted(series),
            "bootstrap_seed": int(bootstrap_seed),
            "note": "Holm versus the default remains descriptive "
                    "only and never selects"}


def select_winner(dataset_calibration: dict, benchmark_id: str, *,
                  trial_ledger: Sequence[str], bootstrap_seed: int,
                  tie_prefer) -> dict:
    """Selection on calibration ONLY. The winner is frozen and must
    be judged later on an untouched decision dataset; judging it on
    its own selection data refuses (see judge_frozen_winner)."""
    spa = spa_family(dataset_calibration, benchmark_id,
                     trial_ledger=trial_ledger,
                     bootstrap_seed=bootstrap_seed)
    if spa["verdict"] != "EVALUATED":
        return {"verdict": spa.get("verdict"),
                "reason": spa.get("reason")}
    evaluations = {}
    for cell in spa["eligible_cells"]:
        evaluations[cell] = evaluate_cell(
            dataset_calibration, cell, benchmark_id,
            bootstrap_seed=bootstrap_seed)
    conclusive = {c: e for c, e in evaluations.items()
                  if e["verdict"] == "CONCLUSIVE"
                  and e["bootstrap"]["point"] > 0}
    if not conclusive:
        return {"verdict": "INCONCLUSIVE",
                "reason": "no cell is conclusively better than the "
                          "benchmark under the precision rule",
                "evaluations": evaluations, "spa": spa}
    best = max(conclusive.values(),
               key=lambda e: e["bootstrap"]["point"])
    ties = [e for e in conclusive.values()
            if abs(e["bootstrap"]["point"]
                   - best["bootstrap"]["point"])
            <= best["bootstrap"]["se"]]
    winner = max(ties, key=lambda e: tie_prefer(e["cell_id"]))
    return {"verdict": "SELECTED",
            "winner": winner["cell_id"],
            "selection_role": "calibration",
            "tie_group": sorted(e["cell_id"] for e in ties),
            "tie_rule": "within one closure-week bootstrap SE; "
                        "preference by the predeclared "
                        "conservative order",
            "spa": spa, "evaluations": evaluations,
            "frozen": True}


def judge_frozen_winner(selection: dict, dataset_decision: dict,
                        benchmark_id: str, *,
                        selection_dataset: dict,
                        bootstrap_seed: int) -> dict:
    """The frozen winner is judged ONLY on untouched decision data.
    Reusing any selection week refuses."""
    if selection.get("verdict") != "SELECTED":
        raise StatsProtocolError(
            "no frozen winner to judge — selection was "
            f"{selection.get('verdict')!r}")
    overlap = set(selection_dataset["weeks"]) & \
        set(dataset_decision["weeks"])
    if overlap:
        raise StatsProtocolError(
            f"decision data reuses selection weeks {sorted(overlap)[:3]}"
            " — a winner may never be judged on the data that "
            "chose it")
    return evaluate_cell(dataset_decision, selection["winner"],
                         benchmark_id,
                         bootstrap_seed=bootstrap_seed)
