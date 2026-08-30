"""C5 corrected: the weekly session-exposure state machine through the
REAL GymFxEnv path, under Musashi's G1-G5 correction order.

The fixture uses a HISTORICAL TIMESTAMP GAP, not synthesized weekend
bars: the CSV simply has no rows between the declared close and the
declared reopen, exactly as real venue data behaves. A tradable bar
inside a declared closure is now a typed refusal, so no reward is ever
zeroed on a fabricated step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.env import GymFxEnv
from app.session_exposure import (
    DISCRETE_COMMANDS, ExposureFacts, SessionCalendar,
    SessionDataContradictionError, SessionEvidenceError,
    classify_discrete_command)
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from metrics_plugins.default_metrics import Plugin as Metrics
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor)
from reward_plugins.pnl_reward import Plugin as Reward
from strategy_plugins.shared_execution_envelope import Plugin as Envelope


VENUE = "mt5_demo"
ACCOUNT = "fp-1"
SYMBOL = "ETHUSD"
CALENDAR_DIGEST = "cal-weekly-v1"

# Monday 00:00 .. Monday 20:00 inclusive (6 bars), then the venue is
# closed for a full day, then bars resume. The closed interval has NO
# ROWS AT ALL -- that is what a real historical gap looks like.
CLOSE_AT = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
REOPEN_AT = pd.Timestamp("2024-01-03 00:00:00", tz="UTC")
PRE_BARS = 6
POST_BARS = 18
BAR_HOURS = 4


def policy(**kw):
    base = {
        "enabled": True,
        "session_source": "venue_symbol_sessions_v1",
        "wind_down_hours": 8.0,
        "forced_flatten_hours": 4.0,
        "cancel_pending_on_wind_down": True,
        "allow_risk_increase_during_wind_down": False,
        "reopen_min_hours": 8.0,
        "reopen_min_closed_bars": 2,
        "stability_consecutive_checks": 2,
        "max_spread_relative_to_baseline": 3.0,
        "max_gap_sigma": 6.0,
        "max_realized_vol_relative_to_baseline": 4.0,
        "carried_position_recovery":
            "protected_opportunistic_then_forced",
        "holiday_policy": "same_as_weekly",
        "calendar_identity": CALENDAR_DIGEST,
        "reopen_baseline_bars": 4,
        "reopen_gap_sigma_bars": 4,
        "reopen_realized_vol_bars": 4,
    }
    base.update(kw)
    return base


def _stamps(pre=PRE_BARS, post=POST_BARS):
    """Real venue timestamps: a contiguous pre-close block, then a GAP
    across the whole closure, then a contiguous post-reopen block."""
    before = pd.date_range("2024-01-01 00:00:00", periods=pre,
                           freq=f"{BAR_HOURS}h", tz="UTC")
    after = pd.date_range(REOPEN_AT, periods=post,
                          freq=f"{BAR_HOURS}h", tz="UTC")
    return before.append(after).tz_localize(None)


def _csv(tmp_path, stamps=None, spread=None, name="bars.csv",
         stable=True):
    stamps = _stamps() if stamps is None else stamps
    n = len(stamps)
    # a small deterministic wiggle: a flat series has zero variance,
    # which makes gap sigma and realized volatility UNAVAILABLE and
    # therefore never passes a stability check
    closes = 100.0 + 0.20 * np.sin(np.arange(n, dtype=float))
    if not stable:
        closes = closes.copy()
        closes[PRE_BARS + 2:] *= 1.15      # a violent post-reopen jump
    if spread is None:
        spread = np.full(n, 0.0002)
    frame = pd.DataFrame({
        "DATE_TIME": stamps,
        "OPEN": closes, "HIGH": closes * 1.0005,
        "LOW": closes * 0.9995, "CLOSE": closes, "VOLUME": 1000.0,
        "SPREAD": spread,
        "feat": np.linspace(0.0, 1.0, n),
    })
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _env(tmp_path, *, session=True, intervals="default",
         spread_column="SPREAD", csv=None, **kw):
    config = {
        "input_data_file": str(csv or _csv(tmp_path)),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "include_price_window": False,
        "window_size": 4, "initial_cash": 10000.0,
        "position_size": 1.0, "min_equity": 0.0,
        "env_mode": "training", "commission": 0.0, "leverage": 1.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        "execution_envelope": {"envelope_mode": "fixed_fraction",
                               "sl_fraction": 0.50, "tp_fraction": 0.50,
                               "leverage_cap": 1.0},
    }
    if session:
        config.update({
            "session_exposure_enabled": True,
            "session_exposure_policy": policy(),
            "session_venue": VENUE,
            "session_account_fingerprint": ACCOUNT,
            "session_symbol": SYMBOL,
            "session_spread_column": spread_column,
        })
        if intervals == "default":
            config["session_calendar_intervals"] = [
                [str(CLOSE_AT), str(REOPEN_AT)]]
        elif intervals is not None:
            config["session_calendar_intervals"] = intervals
    config.update(kw)
    return GymFxEnv(config, DataFeed(config), Broker(config),
                    Envelope(config), Preprocessor(config),
                    Reward(config), Metrics(config))


def _drive(env, actions, seed=7):
    obs, _info = env.reset(seed=seed)
    frames = []
    for a in actions:
        obs, reward, term, trunc, info = env.step([float(a)])
        frames.append({"obs": obs, "reward": reward,
                       "terminated": term, "info": info,
                       "submitted": int(env.bridge.action_slot)})
        if term:
            break
    return frames


LONG = [1.0] * 24
SHORT = [-1.0] * 24


# =================================================================== #
# G3: real missing-session semantics                                  #
# =================================================================== #

class TestG3MissingSessionSemantics:

    def test_no_bar_exists_inside_the_declared_closure(self, tmp_path):
        env = _env(tmp_path)
        stamps = [env._session_now(i)
                  for i in range(len(env.dataframe))]
        inside = [s for s in stamps
                  if s is not None and CLOSE_AT <= s < REOPEN_AT]
        assert inside == [], (
            "work plan 42 prohibits synthesized tradable bars inside a "
            f"closure; found {inside}")

    def test_no_step_ever_reports_expected_market_closed(self,
                                                         tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG)
        assert all(f["info"]["session_state"] !=
                   "EXPECTED_MARKET_CLOSED" for f in frames), (
            "with a real gap the simulator performs no step inside the "
            "closure, so the closed state is never observed")

    def test_a_fabricated_bar_inside_a_closure_is_refused(self,
                                                          tmp_path):
        # regular 4h bars straight through the declared closure --
        # exactly the fixture Musashi rejected
        fabricated = pd.date_range("2024-01-01 00:00:00", periods=24,
                                   freq="4h")
        env = _env(tmp_path, csv=_csv(tmp_path, stamps=fabricated,
                                      name="fabricated.csv"))
        with pytest.raises(SessionDataContradictionError,
                           match="inside a declared session closure"):
            _drive(env, LONG)

    def test_reward_is_never_zeroed_to_conceal_an_economic_change(
            self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG)
        assert not any(f["info"].get("session_no_actionable_step")
                       for f in frames)
        for frame in frames:
            info = frame["info"]
            if info["reward"] == 0.0:
                assert info["pnl"] == pytest.approx(0.0, abs=1e-9), (
                    "a zero reward must mean no economic change, not a "
                    "suppressed one")

    def test_the_bar_after_the_gap_jumps_straight_to_reopen_context(
            self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG)
        states = [f["info"]["session_state"] for f in frames]
        forced = states.index("FORCED_FLATTEN")
        assert states[forced + 1] == "REOPEN_BLACKOUT", (
            "the very next STEP after the pre-close flatten is the "
            "first post-reopen bar")
        before = env._session_now(
            frames[forced]["info"]["session_decision_bar_index"])
        after = env._session_now(
            frames[forced + 1]["info"]["session_decision_bar_index"])
        assert (after - before).total_seconds() / 3600.0 > BAR_HOURS, (
            "the two consecutive steps must straddle a real gap")


# =================================================================== #
# G1: one explicit action contract                                    #
# =================================================================== #

class TestG1ActionContract:

    def test_three_values_are_carried_separately(self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, [-1.0] * 6)
        for frame in frames:
            info = frame["info"]
            assert info["session_raw_model_output"] == -1.0, (
                "the ORIGINAL continuous model output, not the "
                "coerced command id")
            assert info["session_mapped_command"] == 2
            assert isinstance(info["session_signed_exposure"], float)
            assert info["session_mapped_action"]["command"] == 2
            assert info["session_mapped_action"]["command_name"] == \
                "short"

    def test_short_while_short_is_a_hold_not_a_reversal(self,
                                                        tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, SHORT[:4])
        holding = [f for f in frames
                   if f["info"]["session_signed_exposure"] < 0]
        assert holding
        for frame in holding:
            mapped = frame["info"]["session_mapped_action"]
            assert mapped["kind"] == "hold", (
                "feeding command id 2 to a target-value classifier "
                "reported an unchanged short as a reversal")
            assert mapped["risk_increasing"] is False

    def test_masked_reversal_submits_hold_not_the_masked_command(
            self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
        masked = [f for f in frames
                  if f["info"]["session_overlay"] ==
                  "masked_risk_increase"]
        assert masked, "the fixture must mask a reversal"
        for frame in masked:
            info = frame["info"]
            assert info["session_mapped_action"]["kind"] == "reversal"
            assert info["session_final_action"] == 0, (
                "a masked reversal must submit HOLD")
            assert frame["submitted"] == 0, (
                "the PLUGIN must receive hold, not the masked command")
            assert info["session_signed_exposure"] <= 0.0

    def test_masked_entry_from_flat_submits_hold(self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, [0.0] * 4 + [1.0] * 2)
        masked = [f for f in frames
                  if f["info"]["session_overlay"] in
                  ("masked_risk_increase",
                   "masked_entry_during_blackout")]
        assert masked
        for frame in masked:
            assert frame["submitted"] == 0
            assert frame["info"]["position"] == 0, (
                "the masked entry must have no economic effect")

    def test_close_and_hold_are_never_masked(self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        for frame in frames:
            mapped = frame["info"]["session_mapped_action"]
            if mapped["kind"] in ("hold", "hold_flat", "close"):
                assert mapped["risk_increasing"] is False
                assert frame["info"]["session_overlay"] in (
                    "pass_through", "forced_close")

    def test_forced_close_submits_the_close_command(self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG)
        forced = [f for f in frames
                  if f["info"]["session_overlay"] == "forced_close"]
        assert forced
        assert forced[0]["submitted"] == 3
        assert forced[0]["info"]["session_final_action"] == 3

    def test_discrete_mode_carries_the_same_contract(self, tmp_path):
        env = _env(tmp_path, action_space_mode="discrete")
        env.reset(seed=7)
        commands = (2, 2, 2, 2, 1, 1)
        frames = []
        for command in commands:
            _o, _r, term, _t, info = env.step(command)
            frames.append(info)
            if term:
                break
        for info, command in zip(frames, commands):
            assert info["session_raw_model_output"] == float(command)
            assert info["session_mapped_command"] == command
        masked = [i for i in frames
                  if i["session_overlay"] == "masked_risk_increase"]
        assert masked
        assert all(i["session_final_action"] == 0 for i in masked)

    def test_enlargement_and_reduction_are_not_expressible(self,
                                                           tmp_path):
        # both continuous contracts collapse to the same discrete
        # command domain, so a fixed-size env cannot express a partial
        # size change at all. Asserted, not assumed.
        assert set(DISCRETE_COMMANDS) == {0, 1, 2, 3}
        env = _env(tmp_path)
        kinds = {f["info"]["session_mapped_action"]["kind"]
                 for f in _drive(env, LONG)}
        kinds |= {f["info"]["session_mapped_action"]["kind"]
                  for f in _drive(env, SHORT)}
        assert "enlargement" not in kinds
        assert "reduction" not in kinds

    def test_unknown_command_refuses(self):
        exposure = ExposureFacts.build(
            signed_exposure=0.0, action_mapping="discrete_command_v1")
        for bad in (7, -1, True, 1.0, None, "1"):
            with pytest.raises(SessionEvidenceError):
                classify_discrete_command(bad, exposure)


# =================================================================== #
# G2: causal reopen evidence                                          #
# =================================================================== #

class TestG2CausalReopenEvidence:

    def test_blackout_exits_exactly_when_evidence_accumulates(
            self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG)
        states = [f["info"]["session_state"] for f in frames]
        assert "REOPEN_BLACKOUT" in states
        first_blackout = states.index("REOPEN_BLACKOUT")
        tail = states[first_blackout:]
        assert "NORMAL_TRADING" in tail, (
            "the blackout must EXIT; failing closed forever is not a "
            "reopen policy")
        exit_at = first_blackout + tail.index("NORMAL_TRADING")
        prior = frames[exit_at - 1]["info"]
        first = frames[exit_at]["info"]
        assert prior["session_reopen_stability_streak"] < 2
        assert first["session_reopen_stability_streak"] >= 2
        assert first["session_reopen_closed_bars"] >= 2
        assert first["session_time_since_reopen_hours"] >= 8.0

    def test_the_streak_resets_on_an_unstable_bar(self, tmp_path):
        env = _env(tmp_path, csv=_csv(tmp_path, stable=False,
                                      name="unstable.csv"))
        frames = _drive(env, LONG)
        streaks = [f["info"].get("session_reopen_stability_streak")
                   for f in frames
                   if f["info"]["session_state"] == "REOPEN_BLACKOUT"]
        assert streaks
        assert any(s == 0 for s in streaks), (
            "a violent post-reopen move must reset the streak")

    def test_missing_spread_evidence_fails_closed(self, tmp_path):
        env = _env(tmp_path, spread_column=None)
        frames = _drive(env, LONG)
        after = [f for f in frames
                 if f["info"]["session_state"] == "REOPEN_BLACKOUT"]
        assert after
        assert all(f["info"]["session_reopen_stability_streak"] == 0
                   for f in after), (
            "no spread evidence is not a passing check")
        assert all(f["info"]["session_state"] == "REOPEN_BLACKOUT"
                   for f in after)
        last = after[-1]["info"]["session_reopen_last_check"]
        assert "spread_unavailable" in last["reasons"]

    def test_no_future_bar_influences_a_check(self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(12):
            _o, _r, term, _t, _i = env.step([1.0])
            if term:
                break
        idx = 10
        before = env._session_stability_check(idx)
        numeric = env.dataframe.columns
        env.dataframe.loc[env.dataframe.index[idx + 1:], numeric] = \
            env.dataframe.loc[
                env.dataframe.index[idx + 1:], numeric] * 3.0
        after = env._session_stability_check(idx)
        assert before == after, (
            "a stability check may read only past bars and its own")

    def test_check_is_deterministic_across_resets(self, tmp_path):
        env = _env(tmp_path)
        keys = ("session_state", "session_reopen_closed_bars",
                "session_reopen_stability_streak")
        first = [{k: f["info"].get(k) for k in keys}
                 for f in _drive(env, LONG)]
        second = [{k: f["info"].get(k) for k in keys}
                  for f in _drive(env, LONG)]
        assert first == second

    def test_insufficient_history_never_passes(self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        check = env._session_stability_check(1)
        assert check["passed"] is False
        assert "insufficient_history" in check["reasons"]


# =================================================================== #
# G4: order inventory and fresh reconciliation                        #
# =================================================================== #

def _order(ref, parent_ref, side, size):
    return {"ref": ref, "parent_ref": parent_ref, "side": side,
            "size": size, "exectype": None,
            "reduce_only": parent_ref is not None}


class TestG4OrderInventoryAndReconciliation:

    def test_pending_entry_is_distinguished_from_protective_children(
            self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.step([1.0])
        env.bridge.open_order_inventory = (
            _order(11, None, "buy", 5.0),       # pending ENTRY
            _order(12, 9, "sell", 5.0),         # protective child
        )
        facts = env._session_exposure_facts()
        assert facts.protective_orders == 1
        assert facts.entry_orders == 1, (
            "a simultaneous pending entry must not be hidden behind "
            "the protective bracket")
        assert facts.pending_entry_side == "long"

    def test_unavailable_inventory_refuses(self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.bridge.open_order_inventory = None
        with pytest.raises(SessionEvidenceError,
                           match="inventory is unavailable"):
            env._session_exposure_facts()

    def test_no_coercive_fallbacks_remain(self):
        import inspect
        for name in ("_session_exposure_facts",
                     "_session_signed_exposure",
                     "_session_order_inventory"):
            source = inspect.getsource(getattr(GymFxEnv, name))
            assert " or 0" not in source, (
                f"{name} still coerces an unavailable value to zero")

    def test_contradictory_exposure_facts_refuse(self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.bridge.position_units = 0.0
        env.bridge.position = 1
        with pytest.raises(SessionEvidenceError,
                           match="contradictory exposure"):
            env._session_signed_exposure()

    def test_bridge_publishes_a_real_order_inventory(self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG[:4])
        inventory = env.bridge.open_order_inventory
        assert inventory is not None
        assert inventory, "the bracket legs must be visible"
        for record in inventory:
            assert set(record) >= {"ref", "parent_ref", "side", "size",
                                   "reduce_only"}
            assert isinstance(record["reduce_only"], bool)

    def test_real_bracket_legs_are_protective_not_pending_entries(
            self, tmp_path):
        """backtrader does NOT keep .parent populated on live child
        orders, so deriving reduce-only from the parent link alone
        labels the STOP and LIMIT legs pending ENTRIES -- and the
        wind-down overlay would then cancel the protection it exists
        to preserve."""
        env = _env(tmp_path)
        frames = _drive(env, LONG[:5])
        inventory = env.bridge.open_order_inventory
        assert len(inventory) == 2
        assert all(r["parent_ref"] is None for r in inventory), (
            "this test is only meaningful while backtrader drops the "
            "parent link; if it starts populating it, revisit")
        assert all(r["reduce_only"] for r in inventory)
        assert all(r["exectype"] in ("Stop", "Limit")
                   for r in inventory)
        assert frames[-1]["info"]["session_entry_orders"] == 0
        assert frames[-1]["info"]["session_protective_orders"] == 2

    def test_an_oversized_opposite_order_is_not_reduce_only(self):
        from app.bt_bridge import _describe_order

        class _O:
            ref = 9
            parent = None
            size = 250.0
            exectype = None

            def isbuy(self):
                return False

        order = _O()
        order.exectype = 3          # Stop, via ExecTypes lookup below
        order.ExecTypes = ["Market", "Close", "Limit", "Stop",
                           "StopLimit"]
        record = _describe_order(order, position_size=100.0)
        assert record["exectype"] == "Stop"
        assert record["reduce_only"] is False, (
            "an opposite-side order larger than the position would "
            "FLIP it; that is risk-increasing, not protective")

    def test_a_same_side_resting_order_is_a_pending_entry(self):
        from app.bt_bridge import _describe_order

        class _O:
            ref = 10
            parent = None
            size = 50.0
            exectype = 2
            ExecTypes = ["Market", "Close", "Limit", "Stop"]

            def isbuy(self):
                return True

        record = _describe_order(_O(), position_size=100.0)
        assert record["reduce_only"] is False

    def test_forced_close_runs_the_shared_reconciliation_gate(
            self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG)
        forced = [f for f in frames
                  if f["info"]["session_overlay"] == "forced_close"]
        assert forced
        gate = forced[0]["info"]["session_flatten_reconciliation"]
        assert gate is not None, (
            "a forced flatten must be checked by the shared typed "
            "gate, not accepted on a reported position alone")
        assert "flat_confirmed" in gate
        assert gate["flat_confirmed"] is False, (
            "at the moment the close is submitted the position is "
            "still open, so the flatten is an in-flight ATTEMPT")

    def test_the_gate_refuses_while_exposure_remains(self):
        from app.session_exposure import reconciliation_gate
        assert reconciliation_gate(
            positions_total=1, orders_total=0,
            evidence_age_seconds=0.0)["flat_confirmed"] is False
        assert reconciliation_gate(
            positions_total=0, orders_total=2,
            evidence_age_seconds=0.0)["flat_confirmed"] is False
        assert reconciliation_gate(
            positions_total=0, orders_total=0,
            evidence_age_seconds=0.0)["flat_confirmed"] is True


# =================================================================== #
# G5: non-vacuous lifecycle tests                                     #
# =================================================================== #

NO_CLOSURE = [["2030-01-05 00:00:00+00:00",
               "2030-01-06 00:00:00+00:00"]]


class TestG5LifecycleIsNonVacuous:

    def test_termination_while_exposed_preserves_the_exposure(
            self, tmp_path):
        env = _env(tmp_path, intervals=NO_CLOSURE)
        frames = _drive(env, LONG * 3)
        last = frames[-1]["info"]
        assert frames[-1]["terminated"] is True
        assert last["position"] != 0, (
            "this test is meaningless unless the episode really "
            "terminates while exposed")
        assert last["session_exposure_survived_termination"] is True
        assert last["session_carried_position_requires_migration"] \
            is True
        assert last["session_carried_exposure"] != 0.0
        assert last["termination_does_not_close_exposure"] is True

    def test_termination_while_flat_reports_no_carried_exposure(
            self, tmp_path):
        env = _env(tmp_path, intervals=NO_CLOSURE)
        frames = _drive(env, [0.0] * 60)
        last = frames[-1]["info"]
        assert frames[-1]["terminated"] is True
        assert last["position"] == 0, (
            "this test is meaningless unless the episode really "
            "terminates flat")
        assert last["session_exposure_survived_termination"] is False
        assert last["session_carried_exposure"] == 0.0

    def test_every_overlay_branch_is_actually_exercised(self,
                                                        tmp_path):
        env = _env(tmp_path)
        seen = set()
        for actions in (LONG, SHORT,
                        [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
                        [0.0] * 4 + [1.0] * 8):
            for frame in _drive(env, actions):
                seen.add(frame["info"]["session_overlay"])
        assert {"pass_through", "forced_close", "masked_risk_increase",
                "masked_entry_during_blackout"} <= seen, seen


# =================================================================== #
# observation contract and defaults                                   #
# =================================================================== #

class TestObservationContract:

    def test_session_fields_are_declared_and_emitted(self, tmp_path):
        env = _env(tmp_path)
        for name in GymFxEnv.SESSION_OBSERVATION_NAMES:
            assert name in env.observation_space.spaces
        for frame in _drive(env, LONG):
            obs = frame["obs"]
            for name in GymFxEnv.SESSION_OBSERVATION_NAMES:
                assert obs[name].shape == (1,)
                assert obs[name].dtype == np.float32
            assert env.observation_space.contains(obs)

    def test_flags_track_the_state(self, tmp_path):
        env = _env(tmp_path)
        for frame in _drive(env, LONG):
            state = frame["info"]["session_state"]
            obs = frame["obs"]
            assert obs["session_reopen_blackout"][0] == (
                1.0 if state == "REOPEN_BLACKOUT" else 0.0)
            assert obs["session_wind_down"][0] == (
                1.0 if frame["info"]["session_wind_down"] else 0.0)
            assert obs["session_market_closed"][0] == 0.0, (
                "no step exists inside a closure any more")

    def test_disabled_by_default_keeps_the_legacy_contract(self,
                                                           tmp_path):
        env = _env(tmp_path, session=False)
        assert env.session_exposure_enabled is False
        assert not any(name.startswith("session_")
                       for name in env.observation_space.spaces)
        frames = _drive(env, LONG[:6])
        assert "session_state" not in frames[-1]["info"]


class TestSessionEvidenceFailsClosed:

    def test_missing_calendar_intervals_fail_closed_to_wind_down(
            self, tmp_path):
        env = _env(tmp_path, intervals=None)
        assert env._session_calendar is None
        for frame in _drive(env, LONG[:6]):
            info = frame["info"]
            assert info["session_state"] == "WIND_DOWN"
            assert info["session_evidence_ok"] is False
            assert info["session_evidence_failed_closed"] is True
            assert info["session_final_action"] in (0, 3)
            assert frame["submitted"] in (0, 3)

    def test_calendar_identity_mismatch_refuses(self, tmp_path):
        env = _env(tmp_path)
        env._session_calendar = SessionCalendar.build(
            venue=VENUE, account_fingerprint=ACCOUNT, symbol=SYMBOL,
            calendar_digest="some-other-calendar",
            intervals=[(GymFxEnv._session_utc(CLOSE_AT),
                        GymFxEnv._session_utc(REOPEN_AT))])
        with pytest.raises(Exception, match="calendar identity"):
            _drive(env, LONG[:3])

    def test_malformed_policy_refuses_at_construction(self, tmp_path):
        broken = policy()
        del broken["forced_flatten_hours"]
        with pytest.raises(Exception, match="forced_flatten_hours"):
            _env(tmp_path, session_exposure_policy=broken)

    def test_missing_reopen_baseline_keys_refuse(self, tmp_path):
        broken = policy()
        del broken["reopen_baseline_bars"]
        with pytest.raises(Exception, match="reopen_baseline_bars"):
            _env(tmp_path, session_exposure_policy=broken)


class TestSessionEvidenceReadsTheRealTimestamps:
    """The default data feed promotes the date column to the INDEX.
    An overlay that reads only .columns loses every timestamp and
    fails closed forever, indistinguishably from a broken calendar."""

    def test_timestamps_are_found_when_the_date_is_the_index(
            self, tmp_path):
        env = _env(tmp_path)
        assert env._date_column not in env.dataframe.columns
        assert isinstance(env.dataframe.index, pd.DatetimeIndex)
        assert env._session_now(5) is not None
        frames = _drive(env, LONG[:6])
        assert any(f["info"]["session_evidence_ok"] for f in frames)
        assert not any(f["info"]["session_evidence_failed_closed"]
                       for f in frames)
