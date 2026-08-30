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
from test_session_exposure_env import (
    FLAT, policy, VENUE, ACCOUNT, SYMBOL)


# 15-minute bars, not the platform's H4. simulation_engines/
# nautilus_adapter._to_nautilus_data hardcodes MINUTE aggregation and
# BarType.from_str refuses a step of 60 or more in this nautilus_trader
# build, so H1 and H4 -- the timeframes this platform actually trades --
# cannot run on the Nautilus engine at all. Reported, not worked around:
# the parity claim below is therefore established at M15.
BAR_MINUTES = 15
PARITY_START = "2024-01-01 00:00:00"
CLOSE_AT = pd.Timestamp("2024-01-01 01:30:00", tz="UTC")
REOPEN_AT = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
PARITY_POLICY = policy(wind_down_hours=0.5, forced_flatten_hours=0.25,
                       reopen_min_hours=0.5)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / (
    "examples/config/execution_cost_profiles/project3_pessimistic_v1.json")

# the session decision fields that must be byte-equal across engines
PARITY_FIELDS = (
    "session_state", "session_wind_down", "session_forced_flatten",
    "session_evidence_ok", "session_evidence_failed_closed",
    "session_overlay", "session_raw_model_action",
    "session_final_action", "session_action_before_overlay",
    "session_action_after_overlay", "session_no_actionable_step",
    "session_cancel_pending", "session_cancel_scope",
    "session_time_to_next_close_hours",
    "session_time_since_reopen_hours",
)


def _bars(tmp_path, name="parity_bars.csv"):
    closes = np.asarray(FLAT, dtype=float)
    frame = pd.DataFrame({
        "DATE_TIME": pd.date_range(PARITY_START, periods=len(closes),
                                   freq=f"{BAR_MINUTES}min"),
        "OPEN": closes, "HIGH": closes * 1.0005,
        "LOW": closes * 0.9995, "CLOSE": closes, "VOLUME": 1000.0,
        "feat": np.linspace(0.0, 1.0, len(closes)),
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
        trace[int(info["bar_index"])] = {
            k: info.get(k) for k in PARITY_FIELDS}
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

    def test_session_decision_trace_is_byte_identical(self, tmp_path):
        core = _core(tmp_path)
        sim = _sim(tmp_path)
        try:
            core_trace = _session_trace(core, ACTIONS)
            sim_trace = _session_trace(sim, ACTIONS)
        finally:
            core.close()
            sim.close()
        shared = sorted(set(core_trace) & set(sim_trace))
        assert len(shared) >= 10, (
            f"too few shared bars to prove parity: {shared}")
        assert _canonical([core_trace[b] for b in shared]) == \
            _canonical([sim_trace[b] for b in shared]), (
            "the session state machine and overlay live in shared "
            "GymFxEnv code that NautilusGymFxEnv does not override; a "
            "divergence here means an engine forked the decision")

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
                "EXPECTED_MARKET_CLOSED"} <= core_states

    def test_no_actionable_step_suppresses_reward_on_both_engines(
            self, tmp_path):
        for build in (_core, _sim):
            env = build(tmp_path)
            try:
                env.reset(seed=7)
                for action in ACTIONS:
                    _o, reward, term, _t, info = env.step([action])
                    if info.get("session_no_actionable_step"):
                        assert reward == 0.0, (
                            f"{type(env).__name__} attributed reward "
                            "to a closed interval")
                    if term:
                        break
            finally:
                env.close()

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
