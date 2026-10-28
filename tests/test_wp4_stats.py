"""WP4-C14 battery: the statistical protocol is EXECUTABLE and
refuses every abuse the order names — reference formulas, block
resampling, selection leakage, duplicate weeks, missing pairs, seeds
treated as weeks, trial omission and winner reuse on its selection
data. Holm remains descriptive only (nothing here implements it as a
selector)."""
from __future__ import annotations

import numpy as np
import pytest

from tools.wp4_stats import (MIN_ELIGIBLE_WEEKS, StatsProtocolError,
                             WeekRecord, bootstrap_mean,
                             compliance_gate, evaluate_cell,
                             judge_frozen_winner, select_winner,
                             spa_family,
                             stationary_bootstrap_indices,
                             validate_dataset,
                             weekly_differentials)


def _records(cells, weeks, seeds, *, role="calibration",
             value=None, noncompliant=()):
    records = []
    for cell_index, cell in enumerate(cells):
        for week_index, week in enumerate(weeks):
            for seed in seeds:
                if value is not None:
                    net = value(cell_index, week_index, seed)
                else:
                    net = 0.01 * cell_index + 0.001 * week_index
                records.append(WeekRecord(
                    closure_week_id=week, cell_id=cell, seed=seed,
                    role=role,
                    net_return_after_costs=float(net),
                    closure_compliant=(cell, week)
                    not in noncompliant))
    return records


WEEKS = [f"2024-W{w:02d}" for w in range(1, 41)]
SEEDS = [1, 2, 3]


class TestSchemaAndSupport:

    def test_typed_records_refuse_malformed_values(self):
        with pytest.raises(StatsProtocolError, match="finite"):
            WeekRecord("w1", "a", 1, "fit", float("nan"), True)
        with pytest.raises(StatsProtocolError, match="role"):
            WeekRecord("w1", "a", 1, "test", 0.0, True)
        with pytest.raises(StatsProtocolError, match="seed"):
            WeekRecord("w1", "a", True, "fit", 0.0, True)

    def test_duplicate_weeks_refuse(self):
        records = _records(["a", "b"], WEEKS[:2], SEEDS)
        records.append(records[0])
        with pytest.raises(StatsProtocolError, match="duplicate"):
            validate_dataset(records, role="calibration")

    def test_missing_pairs_refuse(self):
        records = _records(["a", "b"], WEEKS[:3], SEEDS)
        records = [r for r in records
                   if not (r.cell_id == "b"
                           and r.closure_week_id == WEEKS[0]
                           and r.seed == 1)]
        with pytest.raises(StatsProtocolError,
                           match="identical paired support"):
            validate_dataset(records, role="calibration")

    def test_role_separation_is_enforced(self):
        records = _records(["a", "b"], WEEKS[:2], SEEDS,
                           role="fit")
        with pytest.raises(StatsProtocolError, match="no records"):
            validate_dataset(records, role="decision")


class TestReferenceFormulas:

    def test_weekly_differentials_match_hand_computation(self):
        # cell 'b' beats 'a' by exactly 0.01 for every week/seed
        records = _records(
            ["a", "b"], WEEKS[:4], SEEDS,
            value=lambda c, w, s: 0.01 * c + 0.001 * w)
        dataset = validate_dataset(records, role="calibration")
        diffs = weekly_differentials(dataset, "b", "a")
        assert diffs["weekly_differentials"] == \
            pytest.approx([0.01] * 4)
        assert diffs["seed_dispersion_mean"] == pytest.approx(0.0)

    def test_seeds_are_never_treated_as_weeks(self):
        """3 seeds x 4 weeks must yield n_weeks == 4, not 12 — the
        hierarchical aggregation collapses seeds FIRST."""
        records = _records(["a", "b"], WEEKS[:4], SEEDS)
        dataset = validate_dataset(records, role="calibration")
        diffs = weekly_differentials(dataset, "b", "a")
        assert diffs["n_weeks"] == 4
        assert len(diffs["weekly_differentials"]) == 4
        assert "NEVER treated as weeks" in diffs["unit"]

    def test_seed_dispersion_is_reported_separately(self):
        records = _records(
            ["a", "b"], WEEKS[:4], SEEDS,
            value=lambda c, w, s: 0.01 * c + 0.005 * c * s)
        dataset = validate_dataset(records, role="calibration")
        diffs = weekly_differentials(dataset, "b", "a")
        assert diffs["seed_dispersion_mean"] > 0.0


class TestBlockBootstrap:

    def test_deterministic_under_its_seed(self):
        a = stationary_bootstrap_indices(20, 50, 4.0, seed=9)
        b = stationary_bootstrap_indices(20, 50, 4.0, seed=9)
        c = stationary_bootstrap_indices(20, 50, 4.0, seed=10)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_blocks_are_contiguous_runs(self):
        indices = stationary_bootstrap_indices(50, 200, 5.0, seed=3)
        # within a resample, consecutive indices either wrap-advance
        # by one (same block) or jump (new block); advancing must
        # dominate at expected block length 5
        advancing = ((indices[:, 1:] - indices[:, :-1]) % 50 == 1)
        assert advancing.mean() > 0.6

    def test_bootstrap_mean_recovers_the_point(self):
        rng = np.random.default_rng(4)
        diffs = list(rng.normal(0.02, 0.01, size=60))
        boot = bootstrap_mean(diffs, seed=11)
        assert boot["point"] == pytest.approx(np.mean(diffs))
        assert boot["ci90"][0] < boot["point"] < boot["ci90"][1]


class TestGatesAndVerdicts:

    def test_minimum_weeks_yields_inconclusive(self):
        records = _records(["a", "b"], WEEKS[:10], SEEDS,
                           value=lambda c, w, s: 0.05 * c)
        dataset = validate_dataset(records, role="calibration")
        result = evaluate_cell(dataset, "b", "a", bootstrap_seed=1)
        assert result["verdict"] == "INCONCLUSIVE"
        assert str(MIN_ELIGIBLE_WEEKS) in result["reason"]

    def test_wide_ci_yields_inconclusive(self):
        rng = np.random.default_rng(8)
        records = _records(
            ["a", "b"], WEEKS, SEEDS,
            value=lambda c, w, s:
                float(rng.normal(0.0001 * c, 0.5)))
        dataset = validate_dataset(records, role="calibration")
        result = evaluate_cell(dataset, "b", "a", bootstrap_seed=1)
        assert result["verdict"] == "INCONCLUSIVE"
        assert "precision" in result["reason"]

    def test_compliance_is_a_hard_gate(self):
        records = _records(["a", "b"], WEEKS, SEEDS,
                           value=lambda c, w, s: 0.05 * c,
                           noncompliant={("b", WEEKS[3])})
        dataset = validate_dataset(records, role="calibration")
        result = evaluate_cell(dataset, "b", "a", bootstrap_seed=1)
        assert result["verdict"] == "INELIGIBLE"
        assert compliance_gate(records, "b")["non_compliant_weeks"] \
            == [WEEKS[3]]

    def test_a_clean_strong_effect_is_conclusive(self):
        rng = np.random.default_rng(2)
        records = _records(
            ["a", "b"], WEEKS, SEEDS,
            value=lambda c, w, s:
                float(0.05 * c + rng.normal(0, 0.005)))
        dataset = validate_dataset(records, role="calibration")
        result = evaluate_cell(dataset, "b", "a", bootstrap_seed=1)
        assert result["verdict"] == "CONCLUSIVE"
        assert result["bootstrap"]["point"] > 0.0


class TestFamilyAndSelection:

    def _dataset(self, role="calibration", weeks=None):
        rng = np.random.default_rng(6)
        offsets = {"a": 0.0, "b": 0.05, "c": 0.02, "d": -0.01}
        cells = list(offsets)
        weeks = weeks or WEEKS

        def value(c, w, s):
            return float(offsets[cells[c]]
                         + rng.normal(0, 0.004))
        records = _records(cells, weeks, SEEDS, role=role,
                           value=value)
        return validate_dataset(records, role=role)

    def test_trial_omission_refuses(self):
        dataset = self._dataset()
        with pytest.raises(StatsProtocolError,
                           match="missing from the trial ledger"):
            spa_family(dataset, "a", trial_ledger=["b", "c"],
                       bootstrap_seed=1)

    def test_spa_names_the_best_and_a_p_value(self):
        dataset = self._dataset()
        result = spa_family(dataset, "a",
                            trial_ledger=["b", "c", "d"],
                            bootstrap_seed=1)
        assert result["verdict"] == "EVALUATED"
        assert result["best_cell"] == "b"
        assert 0.0 <= result["spa_p_value"] <= 1.0
        assert "descriptive only" in result["note"]

    def test_selection_freezes_a_winner_with_the_tie_rule(self):
        dataset = self._dataset()
        selection = select_winner(
            dataset, "a", trial_ledger=["b", "c", "d"],
            bootstrap_seed=1,
            tie_prefer=lambda cell_id: cell_id)
        assert selection["verdict"] == "SELECTED"
        assert selection["winner"] == "b"
        assert selection["frozen"] is True

    def test_winner_reuse_on_selection_data_refuses(self):
        dataset = self._dataset()
        selection = select_winner(
            dataset, "a", trial_ledger=["b", "c", "d"],
            bootstrap_seed=1, tie_prefer=lambda c: c)
        with pytest.raises(StatsProtocolError,
                           match="never be judged on the data"):
            judge_frozen_winner(selection, dataset, "a",
                                selection_dataset=dataset,
                                bootstrap_seed=2)

    def test_frozen_winner_judged_on_untouched_weeks(self):
        selection_data = self._dataset()
        decision_weeks = [f"2025-W{w:02d}" for w in range(1, 41)]
        decision_data = self._dataset(role="decision",
                                      weeks=decision_weeks)
        selection = select_winner(
            selection_data, "a", trial_ledger=["b", "c", "d"],
            bootstrap_seed=1, tie_prefer=lambda c: c)
        verdict = judge_frozen_winner(
            selection, decision_data, "a",
            selection_dataset=selection_data, bootstrap_seed=2)
        assert verdict["cell_id"] == "b"
        assert verdict["verdict"] in ("CONCLUSIVE", "INCONCLUSIVE")
