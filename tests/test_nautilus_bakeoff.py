from decimal import Decimal
from dataclasses import replace

import pytest

pytest.importorskip("nautilus_trader")

from simulation_engines.bakeoff import build_multi_asset_fixture
from simulation_engines.bakeoff import reconcile_fills
from simulation_engines.bakeoff import build_intrabar_collision_fixture
from simulation_engines.bakeoff import build_margin_rejection_fixture
from simulation_engines.bakeoff import build_rollover_rate_fixture
from simulation_engines.bakeoff import build_financing_fixture
from simulation_engines.bakeoff import export_execution_reports
from simulation_engines.contracts import load_execution_cost_profile
from simulation_engines.nautilus_adapter import NautilusReplayAdapter


PROFILE = "examples/config/execution_cost_profiles/project3_pessimistic_v1.json"


def _run():
    profile = load_execution_cost_profile(PROFILE)
    instruments, frames, actions = build_multi_asset_fixture()
    result = NautilusReplayAdapter(profile).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
        financing_rate_data=build_rollover_rate_fixture(),
    )
    return profile, instruments, result


def test_nautilus_multi_asset_replay_is_deterministic_and_flat():
    _, _, first = _run()
    _, _, second = _run()
    assert first["result_hash"] == second["result_hash"]
    assert first["event_hash"] == second["event_hash"]
    assert first["native"]["total_orders"] == 6
    assert first["summary"]["positions.open"] == "0"


def test_nautilus_account_reconciles_to_independent_fill_oracle():
    profile, instruments, result = _run()
    reconciliation = reconcile_fills(
        result,
        instruments,
        profile,
        initial_cash=Decimal("100000"),
    )
    native_balance = Decimal(result["summary"]["account.SIM.balance.USD.total"].split()[0])
    expected = Decimal(reconciliation["expected_final_balance"])
    assert reconciliation["all_positions_flat"] is True
    assert reconciliation["fill_count"] == 6
    assert abs(native_balance - expected) <= Decimal("0.02")
    reports = export_execution_reports(result, instruments, profile)
    assert len(reports) == 6
    assert all(report["schema_version"] == "execution_report.v1" for report in reports)
    assert all(report["broker_ids"]["cost_currency"] == "USD" for report in reports)


def test_nautilus_worst_case_intrabar_path_hits_stop_before_take_profit():
    profile = load_execution_cost_profile(PROFILE)
    instruments, frames, actions = build_intrabar_collision_fixture()
    result = NautilusReplayAdapter(profile).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
        financing_rate_data=build_rollover_rate_fixture(),
    )
    fills = [event for event in result["events"] if event["event_type"] == "order_filled"]
    assert len(fills) == 2
    assert fills[0]["side"] in {"BUY", "1"}
    assert fills[1]["side"] in {"SELL", "2"}
    assert Decimal(fills[1]["price"]) < Decimal("1.10000")
    assert result["summary"]["positions.open"] == "0"


def test_nautilus_standard_margin_rejects_oversized_target():
    profile = load_execution_cost_profile(PROFILE)
    instruments, frames, actions = build_margin_rejection_fixture()
    result = NautilusReplayAdapter(profile).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("10000"),
        financing_rate_data=build_rollover_rate_fixture(),
    )
    types = [event["event_type"] for event in result["events"]]
    assert "preflight_denied" in types
    assert "order_filled" not in types
    assert result["summary"]["account.SIM.balance.USD.total"] == "10000.00 USD"


def test_nautilus_fx_rollover_changes_account_balance_at_boundary():
    financed = load_execution_cost_profile(PROFILE)
    unfinanced = replace(financed, financing_enabled=False)
    instruments, frames, actions = build_financing_fixture()
    with_financing = NautilusReplayAdapter(financed).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
        financing_rate_data=build_rollover_rate_fixture(),
    )
    without_financing = NautilusReplayAdapter(unfinanced).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
    )
    financed_balance = Decimal(
        with_financing["summary"]["account.SIM.balance.USD.total"].split()[0]
    )
    unfinanced_balance = Decimal(
        without_financing["summary"]["account.SIM.balance.USD.total"].split()[0]
    )
    assert financed_balance < unfinanced_balance
    assert with_financing["summary"]["account.SIM.event_count"] > without_financing["summary"]["account.SIM.event_count"]


def test_future_market_mutation_cannot_change_earlier_fill_facts():
    profile = load_execution_cost_profile(PROFILE)
    instruments, frames, actions = build_multi_asset_fixture()
    cutoff = max(frame.ts_event_ns for frame in frames)
    baseline = NautilusReplayAdapter(profile).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
        financing_rate_data=build_rollover_rate_fixture(),
    )
    mutated_frames = [
        replace(
            frame,
            open=frame.open * Decimal("5"),
            high=frame.high * Decimal("5"),
            low=frame.low * Decimal("5"),
            close=frame.close * Decimal("5"),
        )
        if frame.ts_event_ns == cutoff
        else frame
        for frame in frames
    ]
    mutated = NautilusReplayAdapter(profile).run(
        instrument_specs=instruments,
        frames=mutated_frames,
        actions=actions,
        initial_cash=Decimal("100000"),
        financing_rate_data=build_rollover_rate_fixture(),
    )
    baseline_prefix = [event for event in baseline["events"] if event["ts_event_ns"] < cutoff]
    mutated_prefix = [event for event in mutated["events"] if event["ts_event_ns"] < cutoff]
    assert baseline_prefix == mutated_prefix
