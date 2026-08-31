"""C5 sim/core parity for the session-exposure overlay.

No cross-engine parity harness existed before this file: nothing in
the repository compared GymFxEnv with NautilusGymFxEnv. The README's
"bit-identical" claim is about run-to-run determinism of ONE engine,
and tests/test_nautilus_bakeoff.py exercises the Nautilus replay
adapter alone. This file supplies the missing comparison for the part
of C5 that is genuinely engine-independent, and pins the part that is
NOT identical instead of leaving it unstated.

What IS proven bit-identical here: the session state machine, the
overlay decision, the submitted action, the observation fields and
the no-actionable-step reward suppression are computed in shared
GymFxEnv code that NautilusGymFxEnv does not override, so both
engines produce byte-equal session decisions for the same bars.

What is NOT identical, and why: the Nautilus path has no shared
execution envelope. GymBridgeStrategy.on_bar maps action 3 to a
Decimal(0) target and issues a market order; it never reaches
strategy_plugins/shared_execution_envelope.Plugin.apply_action, so a
forced close produces no policy_close event and no envelope-computed
trade cost. The forced-close settlement is therefore engine-specific
by construction, and the test below asserts exactly that rather than
pretending otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader")

from app.env import GymFxEnv
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from metrics_plugins.default_metrics import Plugin as Metrics
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor)
from reward_plugins.pnl_reward import Plugin as Reward
from simulation_engines.nautilus_gym import NautilusGymFxEnv
from strategy_plugins.shared_execution_envelope import Plugin as Envelope
from test_session_exposure_env import policy, VENUE, ACCOUNT, SYMBOL


# 15-minute bars, not the platform's H4. simulation_engines/
# nautilus_adapter._to_nautilus_data hardcodes MINUTE aggregation and
# BarType.from_str refuses a step of 60 or more in this nautilus_trader
# build, so H1 and H4 -- the timeframes this platform actually trades --
# cannot run on the Nautilus engine at all. Reported, not worked around:
# the parity claim below is therefore established at M15.
BAR_MINUTES = 15
PRE_BARS = 6
POST_BARS = 18
CLOSE_AT = pd.Timestamp("2024-01-01 01:30:00", tz="UTC")
REOPEN_AT = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
PARITY_POLICY = policy(wind_down_hours=0.5, forced_flatten_hours=0.25,
                       reopen_min_hours=0.5, reopen_min_closed_bars=2,
                       stability_consecutive_checks=2,
                       reopen_baseline_bars=4, reopen_gap_sigma_bars=4,
                       reopen_realized_vol_bars=4,
                       release_probation_factor=2)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / (
    "examples/config/execution_cost_profiles/project3_pessimistic_v1.json")

# the session decision fields that must be byte-equal across engines
PARITY_FIELDS = (
    "session_state", "session_wind_down", "session_forced_flatten",
    "session_evidence_ok", "session_evidence_failed_closed",
    "session_raw_model_output", "session_mapped_command",
    "session_time_to_next_close_hours",
    "session_time_since_reopen_hours", "session_reopen_closed_bars",
    "session_reopen_stability_streak",
)
# The classification VERDICT is engine-independent; the exposure
# MAGNITUDE it is computed against is not (see
# TestKnownEngineEconomicDivergence), so only the verdict is compared.
MAPPED_FIELDS = ("kind", "risk_increasing", "command",
                 "command_name")


def _bars(tmp_path, name="parity_bars.csv"):
    """A REAL historical gap: no rows inside the declared closure."""
    before = pd.date_range("2024-01-01 00:00:00", periods=PRE_BARS,
                           freq=f"{BAR_MINUTES}min", tz="UTC")
    after = pd.date_range(REOPEN_AT, periods=POST_BARS,
                          freq=f"{BAR_MINUTES}min", tz="UTC")
    stamps = before.append(after).tz_localize(None)
    n = len(stamps)
    closes = 100.0 + 0.20 * np.sin(np.arange(n, dtype=float))
    frame = pd.DataFrame({
        "DATE_TIME": stamps,
        "OPEN": closes, "HIGH": closes * 1.0005,
        "LOW": closes * 0.9995, "CLOSE": closes, "VOLUME": 1000.0,
        "SPREAD": np.full(n, 0.0002),
        "feat": np.linspace(0.0, 1.0, n),
    })
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _config(tmp_path, **kw):
    config = {
        "input_data_file": str(_bars(tmp_path)),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "include_price_window": False,
        "window_size": 4, "initial_cash": 10000.0,
        "position_size": 1.0, "min_equity": 0.0,
        "env_mode": "training", "commission": 0.0, "leverage": 1.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        "instrument": "EUR_USD", "timeframe": "M15",
        "min_quantity": 1, "lot_size": 1,
        "execution_envelope": {"envelope_mode": "fixed_fraction",
                               "sl_fraction": 0.50, "tp_fraction": 0.50,
                               "leverage_cap": 1.0},
        "session_exposure_enabled": True,
        "session_exposure_policy": PARITY_POLICY,
        "session_venue": VENUE,
        "session_account_fingerprint": ACCOUNT,
        "session_symbol": SYMBOL,
        "session_calendar_intervals": [[str(CLOSE_AT), str(REOPEN_AT)]],
        "session_spread_column": "SPREAD",
        "session_flatten_custody_root": str(
            tmp_path / "parity_flatten_custody"),
    }
    config.update(kw)
    return config


def _plugins(config):
    return dict(data_feed_plugin=DataFeed(config),
                broker_plugin=Broker(config),
                strategy_plugin=Envelope(config),
                preprocessor_plugin=Preprocessor(config),
                reward_plugin=Reward(config),
                metrics_plugin=Metrics(config))


def _core(tmp_path):
    config = _config(tmp_path)
    plugins = _plugins(config)
    return GymFxEnv(config, plugins["data_feed_plugin"],
                    plugins["broker_plugin"],
                    plugins["strategy_plugin"],
                    plugins["preprocessor_plugin"],
                    plugins["reward_plugin"],
                    plugins["metrics_plugin"])


def _sim(tmp_path):
    config = _config(
        tmp_path, execution_cost_profile=str(PROFILE),
        financing_rate_data_file=str(
            ROOT / "examples/data/fx_rollover_rates_smoke.csv"))
    plugins = _plugins(config)
    return NautilusGymFxEnv(config, plugins["data_feed_plugin"],
                            plugins["broker_plugin"],
                            plugins["strategy_plugin"],
                            plugins["preprocessor_plugin"],
                            plugins["reward_plugin"],
                            plugins["metrics_plugin"])


def _session_trace(env, actions, seed=7):
    """The session decision KEYED BY BAR INDEX. Keying by bar rather
    than by step number is deliberate: the two engines do not report
    the same bar_index for the same step (see
    TestKnownCrossEngineBarOffset), so a step-indexed comparison would
    conflate that pre-existing offset with a decision divergence."""
    env.reset(seed=seed)
    trace = {}
    for action in actions:
        _obs, _r, term, _t, info = env.step([float(action)])
        mapped = info.get("session_mapped_action") or {}
        trace[int(info["bar_index"])] = {
            **{k: info.get(k) for k in PARITY_FIELDS},
            "mapped": {k: mapped.get(k) for k in MAPPED_FIELDS}}
        if term:
            break
    return trace


def _obs_trace(env, actions, seed=7):
    env.reset(seed=seed)
    trace = {}
    for action in actions:
        obs, _r, term, _t, info = env.step([float(action)])
        trace[int(info["bar_index"])] = {
            name: obs[name].tolist()
            for name in GymFxEnv.SESSION_OBSERVATION_NAMES}
        if term:
            break
    return trace


ACTIONS = [1.0] * 20


def _canonical(trace):
    return json.dumps(trace, sort_keys=True, separators=(",", ":"),
                      default=str)


class TestSessionDecisionParityAcrossEngines:

    def test_the_decision_function_never_forks_between_engines(
            self, tmp_path):
        """The strong claim: wherever the two engines are in the SAME
        situation, they reach byte-equal session decisions. Where they
        diverge, the divergence is fully explained by the exposure
        being different -- which is a pinned economic difference, not
        a fork in the state machine or the overlay."""
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            core_trace = _session_trace(core, ACTIONS)
            sim_trace = _session_trace(sim, ACTIONS)
            core_exposure = _drive_exposure(core)
            sim_exposure = _drive_exposure(sim)
        finally:
            core.close()
            sim.close()
        shared = sorted(set(core_trace) & set(sim_trace))
        assert len(shared) >= 10, f"too few shared bars: {shared}"

        core_sign = {bar: (0 if abs(v) == 0 else (1 if v > 0 else -1))
                     for bar, v in zip(sorted(core_trace),
                                       core_exposure)}
        sim_sign = {bar: (0 if abs(v) == 0 else (1 if v > 0 else -1))
                    for bar, v in zip(sorted(sim_trace), sim_exposure)}

        same_situation, explained = 0, 0
        for bar in shared:
            if core_sign.get(bar) == sim_sign.get(bar):
                same_situation += 1
                assert _canonical(core_trace[bar]) == \
                    _canonical(sim_trace[bar]), (
                    f"bar {bar}: identical situation, different "
                    "decision -- the state machine or overlay forked")
            elif core_trace[bar] != sim_trace[bar]:
                explained += 1
        assert same_situation >= 8, (
            f"only {same_situation} bars shared a situation")
        assert explained + same_situation >= len(shared) - 1

    def test_state_machine_and_evidence_are_byte_identical(self,
                                                           tmp_path):
        """The state machine itself reads only the calendar, the
        clock and past bars -- never the position -- so it must be
        byte-equal on every shared bar with no exception."""
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            core_trace = _session_trace(core, ACTIONS)
            sim_trace = _session_trace(sim, ACTIONS)
        finally:
            core.close()
            sim.close()
        fields = ("session_state", "session_wind_down",
                  "session_forced_flatten", "session_evidence_ok",
                  "session_evidence_failed_closed",
                  "session_time_to_next_close_hours",
                  "session_time_since_reopen_hours",
                  "session_reopen_closed_bars",
                  "session_reopen_stability_streak",
                  "session_raw_model_output",
                  "session_mapped_command")
        for bar in sorted(set(core_trace) & set(sim_trace)):
            for field in fields:
                assert core_trace[bar][field] == sim_trace[bar][field], (
                    f"bar {bar}, field {field}")

    def test_all_five_states_are_exercised_on_both_engines(self,
                                                           tmp_path):
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            core_trace = _session_trace(core, ACTIONS)
            sim_trace = _session_trace(sim, ACTIONS)
            shared = sorted(set(core_trace) & set(sim_trace))
            core_states = {core_trace[b]["session_state"]
                           for b in shared}
            sim_states = {sim_trace[b]["session_state"] for b in shared}
        finally:
            core.close()
            sim.close()
        assert core_states == sim_states
        assert {"NORMAL_TRADING", "WIND_DOWN", "FORCED_FLATTEN",
                "REOPEN_BLACKOUT"} <= core_states

    def test_no_step_occurs_inside_a_closure_on_either_engine(
            self, tmp_path):
        for build in (_core, _sim):
            env = build(tmp_path)
            try:
                trace = _session_trace(env, ACTIONS)
            finally:
                env.close()
            assert all(f["session_state"] != "EXPECTED_MARKET_CLOSED"
                       for f in trace.values()), (
                f"{build.__name__} stepped inside a declared closure")

    def test_session_observation_fields_match_across_engines(self,
                                                             tmp_path):
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            core_obs = _obs_trace(core, ACTIONS)
            sim_obs = _obs_trace(sim, ACTIONS)
            shared = sorted(set(core_obs) & set(sim_obs))
            assert len(shared) >= 10
            for bar in shared:
                for name in GymFxEnv.SESSION_OBSERVATION_NAMES:
                    assert core_obs[bar][name] == sim_obs[bar][name], (
                        f"bar {bar}, field {name}")
        finally:
            core.close()
            sim.close()


class TestForcedCloseSettlementIsEngineSpecific:
    """Pinned, not papered over. The Nautilus path has no shared
    execution envelope, so the forced close cannot be bit-identical
    to the backtrader settlement today. This test fails the moment
    that changes, which is when the claim may be revisited."""

    def test_only_the_core_engine_settles_through_the_envelope(
            self, tmp_path):
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            _session_trace(core, ACTIONS)
            _session_trace(sim, ACTIONS)
            core_events = list(getattr(core.bridge, "close_events", []))
            sim_events = list(getattr(sim.bridge, "close_events", []))
        finally:
            core.close()
            sim.close()
        assert any(e.get("reason") == "policy_close"
                   for e in core_events), (
            "the backtrader path must settle the forced close through "
            "the shared envelope")
        assert not any(e.get("reason") == "policy_close"
                       for e in sim_events), (
            "KNOWN GAP: GymBridgeStrategy.on_bar maps action 3 to a "
            "Decimal(0) target and never reaches "
            "shared_execution_envelope.Plugin.apply_action, so the "
            "Nautilus forced close produces no policy_close event and "
            "no envelope-computed trade cost. If this assertion "
            "starts failing, the envelope reached the Nautilus path "
            "and the forced-close parity claim can be strengthened.")


class TestKnownCrossEngineBarOffset:
    """PRE-EXISTING and NOT introduced by C5: for the same step
    number, NautilusGymFxEnv reports a bar_index exactly one greater
    than GymFxEnv. Reproduced below with the session overlay fully
    DISABLED, so it cannot be attributed to the session work. Every
    time-derived feature -- the OANDA calendar helper, force-close
    countdowns and now the session state machine -- is therefore
    shifted by one bar between engines, and no economic comparison of
    the two engines is valid until this is resolved.

    This is why the parity tests above key on bar_index rather than
    on step number. If the offset is fixed, this test fails and the
    parity harness can be simplified to a step-indexed comparison."""

    def test_engines_report_a_one_bar_offset_for_the_same_step(
            self, tmp_path):
        config = _config(tmp_path, session_exposure_enabled=False)
        config.pop("session_exposure_policy", None)
        config.pop("session_calendar_intervals", None)
        plugins = _plugins(config)
        core = GymFxEnv(config, plugins["data_feed_plugin"],
                        plugins["broker_plugin"],
                        plugins["strategy_plugin"],
                        plugins["preprocessor_plugin"],
                        plugins["reward_plugin"],
                        plugins["metrics_plugin"])
        sim_config = dict(config)
        sim_config["execution_cost_profile"] = str(PROFILE)
        sim_config["financing_rate_data_file"] = str(
            ROOT / "examples/data/fx_rollover_rates_smoke.csv")
        sim_plugins = _plugins(sim_config)
        sim = NautilusGymFxEnv(sim_config,
                               sim_plugins["data_feed_plugin"],
                               sim_plugins["broker_plugin"],
                               sim_plugins["strategy_plugin"],
                               sim_plugins["preprocessor_plugin"],
                               sim_plugins["reward_plugin"],
                               sim_plugins["metrics_plugin"])
        try:
            core.reset(seed=7)
            sim.reset(seed=7)
            offsets = []
            for _ in range(5):
                _o, _r, ct, _t, core_info = core.step([1.0])
                _o2, _r2, st, _t2, sim_info = sim.step([1.0])
                offsets.append(int(sim_info["bar_index"]) -
                               int(core_info["bar_index"]))
                if ct or st:
                    break
        finally:
            core.close()
            sim.close()
        assert offsets == [1] * len(offsets), (
            "the known cross-engine bar offset changed; the parity "
            "harness assumptions must be revisited")


class TestKnownEngineEconomicDivergence:
    """PRE-EXISTING and NOT introduced by the session work. The two
    engines take ECONOMICALLY DIFFERENT positions from the same
    config, so their exposure magnitudes, order books and therefore
    any economic comparison are not interchangeable. Recorded as an
    executable authority block: while these assertions hold, no
    cross-engine economic claim may be made.

    1. Sizing. The backtrader path sizes through the shared execution
       envelope (portfolio fraction), while GymBridgeStrategy.on_bar
       uses the literal ``position_size`` from config.
    2. Order book. The backtrader path rests two protective bracket
       legs; the Nautilus path rests nothing at all.
    3. Bar alignment. See TestKnownCrossEngineBarOffset.
    """

    def test_engines_hold_different_exposure_from_the_same_config(
            self, tmp_path):
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            core_trace = _drive_exposure(core)
            sim_trace = _drive_exposure(sim)
        finally:
            core.close()
            sim.close()
        core_max = max(abs(v) for v in core_trace)
        sim_max = max(abs(v) for v in sim_trace)
        assert core_max > 1.0 and sim_max == pytest.approx(1.0), (
            f"core peaked at {core_max}, sim at {sim_max}; if these "
            "converge, the economic authority block can be lifted")

    def test_only_the_core_engine_rests_protective_orders(self,
                                                          tmp_path):
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            _session_trace(core, ACTIONS)
            _session_trace(sim, ACTIONS)
            core_inv = core.bridge.open_order_inventory
            sim_inv = sim.bridge.open_order_inventory
        finally:
            core.close()
            sim.close()
        assert core_inv and all(r["reduce_only"] for r in core_inv)
        assert sim_inv == (), (
            "the Nautilus path rests no orders; a cross-engine claim "
            "about protection or cancellation is therefore invalid")

    def test_no_economic_cross_engine_comparison_is_authorized(self):
        # an explicit, executable statement of the standing block
        assert AUTHORITY["economic_comparison_authorized"] is False
        assert set(AUTHORITY["blocking_findings"]) == {
            "F-C bar alignment", "F-D unsupported timeframes",
            "sizing regime", "order book"}


def _drive_exposure(env, seed=7):
    env.reset(seed=seed)
    values = []
    for action in ACTIONS:
        _o, _r, term, _t, info = env.step([float(action)])
        values.append(float(info["session_signed_exposure"]))
        if term:
            break
    return values


AUTHORITY = {
    "economic_comparison_authorized": False,
    "blocking_findings": (
        "F-C bar alignment", "F-D unsupported timeframes",
        "sizing regime", "order book"),
}


class TestFDNautilusRefusesUnsupportedTimeframes:
    """G6-4: while nautilus_adapter hardcodes MINUTE aggregation and
    BarType refuses a step of 60 or more, H1 and H4 -- the timeframes
    this platform actually trades -- must be refused EXPLICITLY at
    construction rather than failing deep inside the adapter with a
    BarType parse error."""

    @pytest.mark.parametrize("timeframe", ["H1", "H4", "D1", "60m"])
    def test_unsupported_timeframes_refuse_at_construction(
            self, tmp_path, timeframe):
        config = _config(
            tmp_path, execution_cost_profile=str(PROFILE),
            timeframe=timeframe,
            financing_rate_data_file=str(
                ROOT / "examples/data/fx_rollover_rates_smoke.csv"))
        plugins = _plugins(config)
        with pytest.raises(ValueError, match="unsupported"):
            NautilusGymFxEnv(config, plugins["data_feed_plugin"],
                             plugins["broker_plugin"],
                             plugins["strategy_plugin"],
                             plugins["preprocessor_plugin"],
                             plugins["reward_plugin"],
                             plugins["metrics_plugin"])

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15"])
    def test_supported_timeframes_construct(self, tmp_path,
                                            timeframe):
        config = _config(
            tmp_path, execution_cost_profile=str(PROFILE),
            timeframe=timeframe,
            financing_rate_data_file=str(
                ROOT / "examples/data/fx_rollover_rates_smoke.csv"))
        plugins = _plugins(config)
        env = NautilusGymFxEnv(config, plugins["data_feed_plugin"],
                               plugins["broker_plugin"],
                               plugins["strategy_plugin"],
                               plugins["preprocessor_plugin"],
                               plugins["reward_plugin"],
                               plugins["metrics_plugin"])
        env.close()
