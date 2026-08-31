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
        "release_probation_factor": 2,
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
            "session_flatten_custody_root": str(
                tmp_path / "flatten_custody"),
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
        # F2: the release requirement is the declared count times
        # the predeclared probation factor
        from app.session_exposure import RELEASE_PROBATION_FACTOR
        requirement = (policy()["stability_consecutive_checks"]
                       * RELEASE_PROBATION_FACTOR)
        assert prior["session_reopen_stability_streak"] < requirement
        assert first["session_reopen_stability_streak"] >= \
            requirement
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

def _order(ref, parent_ref, side, size, role="entry"):
    return {"ref": ref, "parent_ref": parent_ref, "side": side,
            "size": size, "exectype": None, "role": role,
            "reduce_only": None if role is None else role != "entry"}


class TestG4OrderInventoryAndReconciliation:

    def test_role_comes_from_the_registry_not_from_geometry(
            self, tmp_path):
        """Musashi's adversary: an INDEPENDENT reversal entry,
        opposite the position and of exactly its size, is
        geometrically indistinguishable from a protective leg. Under
        the old heuristic it was called protection and would have
        survived the wind-down cancellation."""
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.step([1.0])
        env.bridge.register_order_role(701, "entry")
        env.bridge.open_order_inventory = (
            _order(701, None, "sell", 99.8, role="entry"),
            _order(702, None, "sell", 99.8,
                   role="protective_stop"),
        )
        facts = env._session_exposure_facts()
        assert facts.entry_orders == 1, (
            "the reversal entry must NOT be mistaken for protection")
        assert facts.protective_orders == 1
        assert facts.pending_entry_side == "short"

    def test_an_unregistered_order_is_ambiguous_and_refuses(
            self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.bridge.open_order_inventory = (
            _order(999, None, "sell", 1.0, role=None),)
        with pytest.raises(SessionEvidenceError,
                           match="no registered role"):
            env._session_exposure_facts()

    def test_the_registry_refuses_a_conflicting_reregistration(
            self, tmp_path):
        from app.bt_bridge import TradeCloseConflictError
        env = _env(tmp_path)
        env.reset(seed=7)
        env.bridge.register_order_role(55, "entry")
        env.bridge.register_order_role(55, "entry")      # idempotent
        with pytest.raises(TradeCloseConflictError,
                           match="already registered"):
            env.bridge.register_order_role(55, "protective_stop")

    def test_the_registry_refuses_unknown_roles_and_bad_refs(self,
                                                             tmp_path):
        from app.bt_bridge import TradeCloseValidationError
        env = _env(tmp_path)
        env.reset(seed=7)
        for ref, role in ((1, "hedge"), (-1, "entry"),
                          (True, "entry"), ("x", "entry")):
            with pytest.raises(TradeCloseValidationError):
                env.bridge.register_order_role(ref, role)

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
        assert {r["role"] for r in inventory} == {
            "protective_stop", "protective_take_profit"}
        assert all(r["reduce_only"] for r in inventory)
        assert all(r["exectype"] in ("Stop", "Limit")
                   for r in inventory)
        assert frames[-1]["info"]["session_entry_orders"] == 0
        assert frames[-1]["info"]["session_protective_orders"] == 2

    def test_geometry_no_longer_decides_the_role(self):
        """The describer records geometry for DIAGNOSIS only. An
        unregistered order gets role None and reduce_only None --
        ambiguity, which refuses upstream -- no matter how protective
        it looks."""
        from app.bt_bridge import _describe_order

        class _O:
            ref = 9
            parent = None
            size = 99.8
            exectype = 3
            ExecTypes = ["Market", "Close", "Limit", "Stop"]

            def isbuy(self):
                return False

        record = _describe_order(_O(), roles={})
        assert record["role"] is None
        assert record["reduce_only"] is None
        assert record["exectype"] == "Stop"
        registered = _describe_order(_O(), roles={9: "entry"})
        assert registered["reduce_only"] is False
        protective = _describe_order(_O(),
                                     roles={9: "protective_stop"})
        assert protective["reduce_only"] is True


# =================================================================== #
# C2/C3: executing cancellation and post-fill confirmation            #
# =================================================================== #

class TestC2ExecutingCancellation:

    def test_pending_entries_are_actually_cancelled(self, tmp_path,
                                                    monkeypatch):
        """The overlay must hand the broker the entry refs, not merely
        publish cancel_pending in info."""
        env = _env(tmp_path)
        original = GymFxEnv._session_order_inventory
        synthetic = _order(4242, None, "buy", 10.0, role="entry")

        def with_pending_entry(self):
            entries, protective = original(self)
            return tuple(list(entries) + [synthetic]), protective

        monkeypatch.setattr(GymFxEnv, "_session_order_inventory",
                            with_pending_entry)
        env.reset(seed=7)
        env.bridge.register_order_role(4242, "entry")
        frames = []
        for a in LONG[:6]:
            obs, r, term, tr, info = env.step([a])
            frames.append({"obs": obs, "reward": r, "info": info,
                           "terminated": term,
                           "submitted": int(env.bridge.action_slot)})
            if term:
                break
        wind = [f for f in frames
                if f["info"]["session_state"] in
                ("WIND_DOWN", "FORCED_FLATTEN")]
        assert wind, "the fixture must reach wind-down"
        assert any(f["info"]["session_cancel_pending"] for f in wind)
        assert 4242 in env._session_cancel_requested, (
            "the overlay must REQUEST the cancellation, not only "
            "publish the boolean")
        assert any(4242 in f["info"]["session_cancel_requested_refs"]
                   for f in wind)
        assert env.bridge.cancel_outcomes.get(4242) == "not_open", (
            "the strategy looked for it in the real order book")

    def test_the_strategy_reverifies_the_role_before_cancelling(
            self, tmp_path):
        """Defence in depth: even if a ref reaches the cancel channel,
        the strategy refuses to cancel anything whose role it cannot
        verify in the registry."""
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.bridge.cancel_entry_request = (98765,)
        env.step([1.0])
        assert env.bridge.cancel_outcomes[98765] == "refused_role_None"

    def test_the_broker_channel_receives_only_registered_entries(
            self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(3):
            env.step([1.0])
        assert env.bridge.open_order_inventory, "brackets must be live"
        # protective legs must never be honoured by the cancel channel
        env.bridge.cancel_entry_request = tuple(
            r["ref"] for r in env.bridge.open_order_inventory)
        env.step([1.0])
        outcomes = env.bridge.cancel_outcomes
        assert outcomes, "the strategy must have processed the request"
        assert all(v.startswith("refused_role") for v in
                   outcomes.values()), outcomes
        assert env.bridge.open_order_inventory, (
            "the protective bracket must still be alive")

    def test_protective_brackets_survive_wind_down_until_the_close(
            self, tmp_path):
        """C2: the wind-down cancels entries only. The protective
        legs must stay alive right up to the close, and it is the
        CLOSE that retires them, not the session overlay."""
        env = _env(tmp_path)
        env.reset(seed=7)
        live = []
        for _ in range(6):
            _o, _r, term, _t, info = env.step([1.0])
            live.append((info["session_state"],
                         info["session_overlay"],
                         tuple(r["role"] for r in
                               (env.bridge.open_order_inventory or ()))))
            if term:
                break
        wind = [row for row in live if row[0] == "WIND_DOWN"]
        assert wind, "the fixture must reach wind-down"
        for _state, _overlay, roles in wind:
            assert set(roles) == {"protective_stop",
                                  "protective_take_profit"}, (
                "protection must survive the wind-down")
        forced = [row for row in live if row[1] == "forced_close"]
        assert forced and forced[0][2] == (), (
            "the CLOSE retires the brackets, and only then")

    def test_pre_dispatch_and_post_fill_read_different_moments(
            self, tmp_path):
        """The overlay's view is taken BEFORE the action reaches the
        broker; the authority is taken AFTER. They are meant to
        disagree at the closing bar, and that disagreement is the
        whole point of the split."""
        env = _env(tmp_path)
        frames = _drive(env, LONG[:6])
        closing = next(f for f in frames
                       if f["info"]["session_overlay"] ==
                       "forced_close")
        pre = closing["info"]["session_flatten_pre_dispatch"]
        post = closing["info"]["session_flatten_reconciliation"]
        assert pre["diagnostic_only"] is True
        assert pre["orders"] == 2, (
            "before dispatch the protective legs are still resting")
        assert post["orders"] == 0, (
            "after the close executed they are gone")
        assert post["positions"] == 1, (
            "and the position has not settled yet, so this bar cannot "
            "confirm")
        assert closing["info"]["session_flatten_confirmed"] is False

    def test_a_rejected_cancellation_is_a_typed_incident(self,
                                                         tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env._session_cancel_requested.add(31337)
        env.bridge.order_terminal_status[31337] = "Rejected"
        outcomes = env._session_cancellation_outcomes()
        assert outcomes["session_cancellations"][31337] == "rejected"
        assert "ENTRY_CANCELLATION_REJECTED" in \
            outcomes["session_cancellation_incident"]

    def test_an_entry_that_filled_despite_cancellation_is_reported(
            self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env._session_cancel_requested.add(31338)
        env.bridge.order_terminal_status[31338] = "Completed"
        outcomes = env._session_cancellation_outcomes()
        assert outcomes["session_cancellations"][31338] == \
            "filled_before_cancel"
        assert "ENTRY_FILLED_DESPITE_CANCELLATION" in \
            outcomes["session_cancellation_incident"]

    def test_an_order_still_resting_is_pending_not_cancelled(self,
                                                             tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(3):
            env.step([1.0])
        assert env.bridge.open_order_inventory
        ref = env.bridge.open_order_inventory[0]["ref"]
        env._session_cancel_requested.add(ref)
        outcomes = env._session_cancellation_outcomes()
        assert outcomes["session_cancellations"][ref] == "still_open"
        assert outcomes["session_cancellations_pending"] == 1


class TestC3PostFillFlattenLifecycle:

    def _run(self, tmp_path, actions):
        env = _env(tmp_path)
        return env, _drive(env, actions)

    @pytest.mark.parametrize("actions,label", [(LONG, "long"),
                                               (SHORT, "short")])
    def test_the_three_phases_occur_in_order(self, tmp_path, actions,
                                             label):
        env, frames = self._run(tmp_path, actions)
        phases = [f["info"].get("session_flatten_phase")
                  for f in frames]
        assert "flatten_in_flight" in phases, label
        assert "flatten_confirmed" in phases, label
        assert phases.index("flatten_in_flight") < \
            phases.index("flatten_confirmed")

    def test_confirmation_never_precedes_the_close(self, tmp_path):
        env, frames = self._run(tmp_path, LONG)
        submit = next(i for i, f in enumerate(frames)
                      if f["submitted"] == 3)
        confirm = next(i for i, f in enumerate(frames)
                       if f["info"].get("session_flatten_confirmed"))
        assert confirm > submit, (
            "a flatten confirmed at or before the bar that submitted "
            "the CLOSE is a pre-dispatch check, not a confirmation")

    def test_the_pre_dispatch_view_is_diagnostic_only(self, tmp_path):
        env, frames = self._run(tmp_path, LONG)
        pre = [f["info"]["session_flatten_pre_dispatch"]
               for f in frames
               if f["info"].get("session_flatten_pre_dispatch")]
        assert pre
        for view in pre:
            assert view["diagnostic_only"] is True
            assert view["flat_confirmed"] is False

    def test_the_fill_is_delayed_by_one_bar(self, tmp_path):
        env, frames = self._run(tmp_path, LONG)
        in_flight = next(f for f in frames
                         if f["info"].get("session_flatten_phase") ==
                         "flatten_in_flight")
        assert in_flight["info"][
            "session_flatten_reconciliation"]["positions"] == 1, (
            "at the submitting bar the position is still open")
        confirmed = next(f for f in frames
                         if f["info"].get(
                             "session_flatten_confirmed"))
        assert confirmed["info"][
            "session_flatten_confirmed_at_bar"] == \
            in_flight["info"]["bar_index"] + 1

    def test_a_rejected_close_never_confirms(self, tmp_path,
                                             monkeypatch):
        from strategy_plugins import shared_execution_envelope as env_mod
        original = env_mod.Plugin.apply_action

        def refusing(self, s, action, config):
            if int(action) == 3:
                return          # the broker refused the close
            return original(self, s, action, config)

        monkeypatch.setattr(env_mod.Plugin, "apply_action", refusing)
        env, frames = self._run(tmp_path, LONG)
        phases = [f["info"].get("session_flatten_phase")
                  for f in frames]
        assert "flatten_in_flight" in phases
        assert "flatten_confirmed" not in phases, (
            "a close that never executed must not confirm")
        incidents = {f["info"].get("session_flatten_incident")
                     for f in frames}
        assert any(i and "FORCED_FLATTEN" in i for i in incidents)

    def test_a_confirmed_flatten_is_terminal(self, tmp_path):
        env, frames = self._run(tmp_path, LONG)
        confirmed_from = next(
            i for i, f in enumerate(frames)
            if f["info"].get("session_flatten_confirmed"))
        for frame in frames[confirmed_from:]:
            info = frame["info"]
            assert info["session_flatten_phase"] == \
                "flatten_confirmed"
            assert info["session_flatten_incident"] is None, (
                "a retired attempt must not report the agent's NEXT "
                "position as its own failure")

    def test_a_restart_recovers_the_obligation_it_must_not_forget(
            self, tmp_path):
        """The obligation survives reset and BLOCKS the new episode."""
        env = _env(tmp_path)
        frames = _drive(env, LONG[:6])
        assert frames[-1]["info"]["session_flatten_phase"] == \
            "flatten_in_flight"
        outstanding = env._flatten_store.outstanding()
        assert len(outstanding) == 1
        obligation_id = outstanding[0]["obligation_id"]

        env.reset(seed=7)
        assert env._session_recovery is not None
        assert env._session_recovery["blocks_risk_increase"] is True

        _o, _r, _t, _tr, first = env.step([1.0])
        assert first["session_recovery_active"] is True
        assert first["session_overlay"] == "blocked_by_flatten_recovery"
        assert first["session_final_action"] == 0
        record = env._flatten_store.read(obligation_id)
        assert record["state"] == "interrupted_unresolved"
        assert record["closure_claimed"] is False

    def test_a_new_empty_account_can_never_certify_the_old_close(
            self, tmp_path):
        """FROZEN COUNTEREXAMPLE. reset() builds a new bridge and
        broker that are born flat. That zero/zero is a fact about the
        NEW episode and says nothing about whether the PREVIOUS
        exposure was ever closed, so it must never discharge the
        obligation."""
        env = _env(tmp_path)
        _drive(env, LONG[:6])
        outstanding = env._flatten_store.outstanding()
        obligation_id = outstanding[0]["obligation_id"]
        assert outstanding[0]["signed_exposure_at_request"] != 0.0, (
            "the obligation must name a REAL open exposure")

        env.reset(seed=7)
        assert env.bridge.position_units == 0.0, (
            "the new episode really is born flat")
        infos = []
        for _ in range(6):
            _o, _r, term, _t, info = env.step([1.0])
            infos.append(info)
            if term:
                break
        record = env._flatten_store.read(obligation_id)
        assert record["state"] == "interrupted_unresolved", record
        assert record["closure_claimed"] is False
        assert all(i["session_recovery_active"] for i in infos), (
            "the new episode stays blocked; an empty account is not a "
            "discharge")
        assert all(i["session_final_action"] == 0 for i in infos
                   if i["session_mapped_action"]["risk_increasing"])

    def test_a_process_restart_recovers_the_same_obligation(self,
                                                            tmp_path):
        """A brand-new env object over the same custody root finds the
        obligation and is blocked by it."""
        first_env = _env(tmp_path)
        _drive(first_env, LONG[:6])
        root = first_env.config["session_flatten_custody_root"]
        ids = {o["obligation_id"]
               for o in first_env._flatten_store.outstanding()}
        assert ids

        reborn = _env(tmp_path, session_flatten_custody_root=root)
        assert reborn._flatten_store.outstanding()
        reborn.reset(seed=7)
        assert reborn._session_recovery is not None
        _o, _r, _t, _tr, info = reborn.step([1.0])
        assert info["session_recovery_active"] is True
        assert info["session_final_action"] == 0
        for obligation_id in ids:
            assert reborn._flatten_store.read(obligation_id)[
                "state"] == "interrupted_unresolved"

    def test_the_store_itself_refuses_a_foreign_episode(self,
                                                        tmp_path):
        from app.flatten_custody import (FlattenObligationError,
                                         FlattenObligationStore)
        store = FlattenObligationStore(tmp_path / "foreign")
        store.open_obligation(
            "o-1", venue="mt5_demo", account_fingerprint="fp-1",
            symbol="ETHUSD", position_identity="pos-1",
            episode_identity="ep-A", signed_exposure=99.8,
            requested_at_bar=5, code_identity="code-1")
        with pytest.raises(FlattenObligationError,
                           match="cannot be advanced by episode"):
            store.confirm("o-1",
                          reconciliation={"flat_confirmed": True},
                          bar_index=6, episode_identity="ep-B")
        assert store.read("o-1")["state"] == "flatten_requested"
        store.confirm("o-1", reconciliation={"flat_confirmed": True},
                      bar_index=6, episode_identity="ep-A")
        assert store.read("o-1")["state"] == "flatten_confirmed"

    def test_a_same_episode_obligation_still_discharges(self,
                                                        tmp_path):
        """The normal path is untouched: within ONE episode the close
        is confirmed on post-fill evidence."""
        env = _env(tmp_path)
        frames = _drive(env, LONG[:7])
        assert [f for f in frames
                if f["info"].get("session_flatten_confirmed")]
        assert env._flatten_store.outstanding() == ()
        ids = [p.stem for p in env._flatten_store.root.glob("*.json")]
        assert len(ids) == 1
        assert env._flatten_store.read(ids[0])["state"] == \
            "flatten_confirmed"

    def test_multiple_open_obligations_require_an_operator(self,
                                                           tmp_path):
        from app.flatten_custody import (FlattenDispositionRequired,
                                         FlattenObligationStore)
        root = tmp_path / "multi_custody"
        store = FlattenObligationStore(root)
        for i in (1, 2, 3):
            store.open_obligation(
                f"m-{i}", venue="mt5_demo",
                account_fingerprint="fp-1", symbol="ETHUSD",
                position_identity=f"pos-{i}",
                episode_identity=f"ep-{i}", signed_exposure=float(i),
                requested_at_bar=i, code_identity="code-1")
        with pytest.raises(FlattenDispositionRequired,
                           match="no automatic resolution"):
            store.require_single_open()

        env = _env(tmp_path, session_flatten_custody_root=str(root))
        env.reset(seed=7)
        assert env._session_recovery["reason"] == \
            "multiple_open_obligations"
        assert env._session_recovery[
            "requires_operator_disposition"] is True
        _o, _r, _t, _tr, info = env.step([1.0])
        assert info["session_recovery_active"] is True
        assert info["session_final_action"] == 0
        assert len(env._flatten_store.outstanding()) == 3

    def test_the_final_state_is_exactly_flat_with_one_close_event(
            self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG[:7])
        confirmed = frames[-1]["info"]
        assert confirmed["session_flatten_confirmed"] is True
        gate = confirmed["session_flatten_reconciliation"]
        assert gate["positions"] == 0 and gate["orders"] == 0
        policy_closes = [e for e in env.bridge.close_events
                         if e.get("reason") == "policy_close"]
        assert len(policy_closes) == 1, (
            f"exactly one forced close, got {policy_closes}")
        # exactly ONE economic closure event, with costs and a PnL
        # identity that holds
        assert len(env.bridge.closed_trade_stream) == 1, \
            env.bridge.closed_trade_stream
        event = env.bridge.closed_trade_stream[0]
        assert event["costs"] >= 0.0
        assert event["net_pnl"] == pytest.approx(
            event["gross_pnl"] - event["costs"], abs=1e-6)
        assert env.bridge.trade_count == 1


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


# =================================================================== #
# C4: observable reopen evidence and an independent vol baseline      #
# =================================================================== #

EVIDENCE_FIELDS = ("session_reopen_bar_progress",
                   "session_reopen_stability_progress",
                   "session_spread_ratio_norm",
                   "session_gap_sigma_norm",
                   "session_vol_ratio_norm",
                   "session_quote_continuous")


class TestC4ObservableReopenEvidence:

    def test_the_governing_evidence_is_declared_and_emitted(self,
                                                            tmp_path):
        env = _env(tmp_path)
        for name in EVIDENCE_FIELDS:
            assert name in GymFxEnv.SESSION_OBSERVATION_NAMES
            assert name in env.observation_space.spaces
        for frame in _drive(env, LONG):
            obs = frame["obs"]
            for name in EVIDENCE_FIELDS:
                value = float(obs[name][0])
                assert 0.0 <= value <= 1.0, (name, value)
                assert np.isfinite(value)
            assert env.observation_space.contains(obs)

    def test_the_agent_can_tell_why_it_is_still_blocked(self,
                                                        tmp_path):
        env = _env(tmp_path)
        blocked = [f for f in _drive(env, LONG)
                   if f["info"]["session_state"] == "REOPEN_BLACKOUT"]
        assert blocked
        progress = [float(f["obs"]["session_reopen_stability_progress"][0])
                    for f in blocked]
        assert progress[0] < progress[-1] or len(set(progress)) > 1, (
            "the stability progress must be visible and move")
        bars = [float(f["obs"]["session_reopen_bar_progress"][0])
                for f in blocked]
        assert bars[-1] >= bars[0]

    def test_missing_evidence_is_observed_as_the_WORST_value(self,
                                                             tmp_path):
        env = _env(tmp_path, spread_column=None)
        blocked = [f for f in _drive(env, LONG)
                   if f["info"]["session_state"] == "REOPEN_BLACKOUT"]
        assert blocked
        for frame in blocked:
            assert float(frame["obs"]["session_spread_ratio_norm"][0]) \
                == 1.0, (
                "an unavailable ratio must read as the threshold, "
                "never as a safe zero")
        assert all(float(f["obs"][
            "session_reopen_stability_progress"][0]) == 0.0
            for f in blocked)

    def test_a_gap_at_a_checkable_bar_is_a_discontinuity(self,
                                                          tmp_path):
        """The default fixture's gap sits at bar 6, earlier than the
        history both windows need, so it is reported
        insufficient_history rather than continuous -- which is also
        fail-closed. With a longer pre-block the gap lands on a
        checkable bar and IS detected."""
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(9):
            env.step([1.0])
        assert env._session_stability_check(PRE_BARS)["reasons"] == [
            "insufficient_history"]

        # a fixture whose gap lands on a CHECKABLE bar: 14 contiguous
        # bars, then a closure declared after them
        late_close = pd.Timestamp("2024-01-03 08:00:00", tz="UTC")
        late_reopen = pd.Timestamp("2024-01-04 08:00:00", tz="UTC")
        before = pd.date_range("2024-01-01 00:00:00", periods=14,
                               freq=f"{BAR_HOURS}h", tz="UTC")
        after_block = pd.date_range(late_reopen, periods=18,
                                    freq=f"{BAR_HOURS}h", tz="UTC")
        stamps = before.append(after_block).tz_localize(None)
        long_env = _env(
            tmp_path,
            csv=_csv(tmp_path, stamps=stamps, name="longpre.csv"),
            intervals=[[str(late_close), str(late_reopen)]])
        long_env.reset(seed=7)
        for _ in range(20):
            _o, _r, term, _t, _i = long_env.step([1.0])
            if term:
                break
        at_gap = long_env._session_stability_check(14)
        after = long_env._session_stability_check(16)
        assert at_gap["quote_continuous"] is False
        assert "quote_discontinuity" in at_gap["reasons"]
        assert after["quote_continuous"] is True

    def test_the_volatility_baseline_is_independent_of_the_gap_sigma(
            self, tmp_path):
        """The ratio must compare two realized-volatility windows, not
        divide a recent volatility by the GAP return sigma."""
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(14):
            env.step([1.0])
        check = env._session_stability_check(12)
        assert check["baseline_vol"] is not None
        assert check["vol_ratio"] == pytest.approx(
            check["recent_vol"] / check["baseline_vol"], rel=1e-12)

    def test_the_baseline_window_precedes_the_recent_window(self,
                                                            tmp_path):
        """Changing a bar inside the RECENT window moves the ratio;
        the baseline window is disjoint and strictly earlier."""
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(16):
            env.step([1.0])
        idx = 13
        vol_n = env._session_policy["reopen_realized_vol_bars"]
        base_n = env._session_policy["reopen_baseline_bars"]
        before = env._session_stability_check(idx)
        column = env.price_column
        # a bar strictly before the baseline window must not matter
        untouched = idx - vol_n - base_n - 1
        env.dataframe.iloc[untouched,
                           env.dataframe.columns.get_loc(column)] *= 5.0
        after = env._session_stability_check(idx)
        assert before["vol_ratio"] == after["vol_ratio"], (
            f"a bar at {untouched}, outside both windows, changed the "
            "ratio")

    def test_insufficient_history_covers_both_windows(self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        policy = env._session_policy
        need = max(policy["reopen_baseline_bars"],
                   policy["reopen_gap_sigma_bars"],
                   policy["reopen_realized_vol_bars"] +
                   policy["reopen_baseline_bars"]) + 1
        assert env._session_stability_check(need - 1)["reasons"] == [
            "insufficient_history"]


# =================================================================== #
# C5: strict typed termination boundaries                             #
# =================================================================== #

class TestC5StrictTerminationBoundaries:

    def test_no_coercion_remains_in_the_termination_record(self):
        import inspect
        source = inspect.getsource(
            GymFxEnv._session_termination_record)
        assert " or 0" not in source
        assert " or 0.0" not in source

    @pytest.mark.parametrize("bad", [None, True, float("nan"),
                                     float("inf"), "1.0"])
    def test_invalid_exposure_refuses_instead_of_reading_flat(
            self, tmp_path, bad):
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        env.bridge.position_units = bad
        with pytest.raises(SessionEvidenceError):
            env._session_termination_record()

    @pytest.mark.parametrize("field", ["episode_seq", "bar_index"])
    def test_invalid_counters_refuse(self, tmp_path, field):
        env = _env(tmp_path)
        env.reset(seed=7)
        for _ in range(3):
            env.step([1.0])
        setattr(env.bridge, field, None)
        with pytest.raises(Exception):
            env._session_termination_record()

    def test_a_genuinely_flat_account_still_reports_flat(self,
                                                         tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        record = env._session_termination_record()
        assert record["session_exposure_survived_termination"] is False
        assert record["session_carried_exposure"] == 0.0


# =================================================================== #
# R1: strict order-reference identity                                 #
# =================================================================== #

class TestR1StrictOrderReferences:

    @pytest.mark.parametrize("ref", [1.5, 2.9, 0.0, 3.0,
                                     float("nan"), float("inf"),
                                     "3", True, False, None,
                                     -1, [1]])
    def test_non_integer_references_refuse_without_coercion(
            self, tmp_path, ref):
        from app.bt_bridge import TradeCloseValidationError
        env = _env(tmp_path)
        env.reset(seed=7)
        with pytest.raises(TradeCloseValidationError):
            env.bridge.register_order_role(ref, "entry")
        assert not any(k in (1, 2, 3) and v == "entry"
                       for k, v in env.bridge.order_roles.items()
                       if k not in (1, 2, 3)), env.bridge.order_roles

    def test_a_fractional_reference_can_no_longer_collide(self,
                                                          tmp_path):
        """PRE: 1.5 was truncated to 1 and 2.9 to 2 -- real identities
        belonging to other live orders."""
        from app.bt_bridge import TradeCloseValidationError
        env = _env(tmp_path)
        env.reset(seed=7)
        env.step([1.0])
        before = dict(env.bridge.order_roles)
        for ref in (1.5, 2.9):
            with pytest.raises(TradeCloseValidationError,
                               match="no coercion"):
                env.bridge.register_order_role(ref, "close")
        assert env.bridge.order_roles == before, (
            "a refused registration must leave the registry untouched")

    def test_valid_integer_references_are_accepted(self, tmp_path):
        env = _env(tmp_path)
        env.reset(seed=7)
        assert env.bridge.register_order_role(9001, "entry") == 9001
        assert env.bridge.role_of(9001) == "entry"


# =================================================================== #
# R2: a REAL resting entry order, cancelled on the real path          #
# =================================================================== #

class _EnvelopeWithRestingEntry(Envelope):
    """Submits ONE genuine resting Limit entry through the executing
    path at a chosen bar, registered as ``entry`` like any other
    order. It is a real order in backtrader's book, so cancelling it
    exercises the real broker, not a synthetic dictionary."""

    resting_ref = None
    submit_at_bar = 2

    def apply_action(self, s, action, config):
        result = super().apply_action(s, action, config)
        bar = int(getattr(s.bridge, "bar_index", 0))
        if (type(self).resting_ref is None
                and bar >= type(self).submit_at_bar):
            import backtrader as bt
            # small enough to fit the cash the open position leaves
            # free, so the broker ACCEPTS it and it really rests
            price = float(s.data.close[0]) * 0.90
            order = s.buy(exectype=bt.Order.Limit, price=price,
                          size=0.05)
            s.bridge.register_order_role(int(order.ref), "entry")
            type(self).resting_ref = int(order.ref)
        return result


def _env_with_resting_entry(tmp_path, **kw):
    _EnvelopeWithRestingEntry.resting_ref = None
    config = {
        "input_data_file": str(_csv(tmp_path)),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "include_price_window": False,
        "window_size": 4, "initial_cash": 10000.0,
        "position_size": 1.0, "min_equity": 0.0,
        "env_mode": "training", "commission": 0.0, "leverage": 1.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        # a HALF-size position, so the account keeps real free cash
        # and the broker can actually accept a second resting entry
        "execution_envelope": {"envelope_mode": "fixed_fraction",
                               "sl_fraction": 0.50, "tp_fraction": 0.50,
                               "leverage_cap": 0.4},
        "session_exposure_enabled": True,
        "session_exposure_policy": policy(),
        "session_venue": VENUE,
        "session_account_fingerprint": ACCOUNT,
        "session_symbol": SYMBOL,
        "session_spread_column": "SPREAD",
        "session_flatten_custody_root": str(tmp_path / "r2_custody"),
        "session_calendar_intervals": [[str(CLOSE_AT),
                                        str(REOPEN_AT)]],
    }
    config.update(kw)
    return GymFxEnv(config, DataFeed(config), Broker(config),
                    _EnvelopeWithRestingEntry(config),
                    Preprocessor(config), Reward(config),
                    Metrics(config))


class TestR2RealPendingOrderCancellation:

    def _run(self, tmp_path):
        env = _env_with_resting_entry(tmp_path)
        frames = _drive(env, LONG[:8])
        return env, frames, _EnvelopeWithRestingEntry.resting_ref

    def test_a_real_resting_entry_exists_and_is_registered(self,
                                                           tmp_path):
        env, frames, ref = self._run(tmp_path)
        assert ref is not None
        assert env.bridge.role_of(ref) == "entry"
        seen = [f for f in frames
                if any(r["ref"] == ref
                       for r in (f["info"].get("_inv") or []))]
        # the order was really in the broker's book at some point
        assert any(ref in f["info"]["session_cancel_requested_refs"]
                   for f in frames), (
            "the request must name the REAL broker ref")

    def test_the_order_is_visible_when_wind_down_begins(self,
                                                        tmp_path):
        env = _env_with_resting_entry(tmp_path)
        env.reset(seed=7)
        states, saw = [], False
        for _ in range(6):
            _o, _r, term, _t, info = env.step([1.0])
            states.append(info["session_state"])
            ref = _EnvelopeWithRestingEntry.resting_ref
            if info["session_state"] == "WIND_DOWN" and ref:
                saw = info["session_entry_orders"] >= 1
                break
            if term:
                break
        assert "WIND_DOWN" in states
        assert saw, (
            "the resting entry must still be visible as an ENTRY when "
            "wind-down begins")

    def test_the_real_order_is_cancelled_and_never_fills(self,
                                                         tmp_path):
        env, frames, ref = self._run(tmp_path)
        outcomes = env.bridge.cancel_outcomes
        assert outcomes.get(ref) == "cancel_submitted", (
            f"the strategy must have called cancel() on {ref}: "
            f"{outcomes}")
        terminal = env.bridge.order_terminal_status.get(ref)
        assert terminal in ("Canceled", "Cancelled", "Expired"), (
            f"terminal broker verdict for {ref} was {terminal!r}")
        assert terminal != "Completed", "the order must never fill"
        final = frames[-1]["info"]["session_cancellations"]
        assert final.get(ref) == "cancelled"
        assert frames[-1]["info"][
            "session_cancellation_incident"] is None

    def test_the_cancellation_precedes_the_action_of_that_bar(self,
                                                              tmp_path):
        """The strategy cancels BEFORE _apply_action, so the entry
        cannot fill on the bar whose action closes the position."""
        import inspect
        from app.bt_bridge import BTBridgeStrategy
        source = inspect.getsource(BTBridgeStrategy.next)
        assert source.index("_process_cancel_requests") < \
            source.index("self._apply_action(action)")

    def test_both_protective_brackets_survive_the_cancellation(
            self, tmp_path):
        env = _env_with_resting_entry(tmp_path)
        env.reset(seed=7)
        roles_at_wind_down = None
        for _ in range(6):
            _o, _r, term, _t, info = env.step([1.0])
            if info["session_state"] == "WIND_DOWN":
                roles_at_wind_down = sorted(
                    r["role"] for r in
                    (env.bridge.open_order_inventory or ()))
            if term:
                break
        assert roles_at_wind_down is not None
        assert roles_at_wind_down.count("protective_stop") == 1
        assert roles_at_wind_down.count(
            "protective_take_profit") == 1

    def test_the_flatten_still_reaches_zero_zero(self, tmp_path):
        env, frames, ref = self._run(tmp_path)
        confirmed = [f for f in frames
                     if f["info"].get("session_flatten_confirmed")]
        assert confirmed
        gate = confirmed[0]["info"]["session_flatten_reconciliation"]
        assert gate["positions"] == 0 and gate["orders"] == 0

    def test_a_rejected_or_unfilled_cancellation_is_never_success(
            self, tmp_path):
        """R2.6: rejection, fill-before-cancel and an order left
        resting must each fail the flatten, not pass it."""
        env, frames, ref = self._run(tmp_path)
        for verdict, expected, incident in (
                ("Rejected", "rejected",
                 "ENTRY_CANCELLATION_REJECTED"),
                ("Completed", "filled_before_cancel",
                 "ENTRY_FILLED_DESPITE_CANCELLATION")):
            env.bridge.order_terminal_status[ref] = verdict
            outcomes = env._session_cancellation_outcomes()
            assert outcomes["session_cancellations"][ref] == expected
            assert incident in outcomes["session_cancellation_incident"]

    def test_an_order_left_resting_counts_as_pending(self, tmp_path):
        env, frames, ref = self._run(tmp_path)
        env.bridge.order_terminal_status.pop(ref, None)
        env.bridge.open_order_inventory = (
            _order(ref, None, "buy", 0.05, role="entry"),)
        outcomes = env._session_cancellation_outcomes()
        assert outcomes["session_cancellations"][ref] == "still_open"
        assert outcomes["session_cancellations_pending"] == 1


# =================================================================== #
# R4: reconciliation provenance                                       #
# =================================================================== #

class TestR4ReconciliationProvenance:

    def test_the_check_is_labelled_simulator_local(self, tmp_path):
        env = _env(tmp_path)
        frames = _drive(env, LONG[:7])
        gates = [f["info"]["session_flatten_reconciliation"]
                 for f in frames
                 if f["info"].get("session_flatten_reconciliation")]
        assert gates
        for gate in gates:
            assert gate["evidence_provenance"] == "simulator_bar_local"
            assert gate["venue_direct"] is False, (
                "WP3 must replace this with typed DIRECT venue "
                "evidence and may not inherit this provenance")

    def test_the_age_is_bound_to_a_bar_not_asserted(self, tmp_path):
        import inspect
        source = inspect.getsource(
            GymFxEnv._session_post_fill_reconciliation)
        assert "evidence_age_seconds=0.0" not in source, (
            "the freshness must be derived, not asserted"
        )
        env = _env(tmp_path)
        frames = _drive(env, LONG[:7])
        gate = next(f["info"]["session_flatten_reconciliation"]
                    for f in frames
                    if f["info"].get("session_flatten_confirmed"))
        assert gate["observed_at_bar"] == gate["evaluated_at_bar"]
        assert gate["observed_at"] is not None
        assert gate["age_seconds"] >= 0.0

    def test_a_backwards_bar_clock_refuses(self, tmp_path):
        env = _env(tmp_path)
        _drive(env, LONG[:6])
        env._session_last_evidence_bar = 10_000
        with pytest.raises(SessionEvidenceError,
                           match="bar clock went backwards"):
            env._session_evidence_provenance()

    def test_the_max_age_is_configurable_not_hardcoded(self,
                                                       tmp_path):
        env = _env(tmp_path, session_max_evidence_age_seconds=7.5)
        assert env.session_max_evidence_age_seconds == 7.5
        default = _env(tmp_path)
        assert default.session_max_evidence_age_seconds == 120.0
