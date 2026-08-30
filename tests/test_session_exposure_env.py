"""C5: the weekly session-exposure state machine through the REAL
GymFxEnv path — real env, real backtrader, real shared execution
envelope. Every assertion here is about production wiring, not about
the pure functions of app/session_exposure.py (covered separately in
tests/test_session_exposure.py).

Proves: closed intervals yield no actionable step and no reward while
account state carries forward; wind-down / blackout / closed reach the
observation and satisfy observation_space; a forced close travels the
SHARED cost/fill envelope and is economically settled; episode
termination cannot erase open exposure; and raw / mapped / overlay /
final stay four DISTINCT records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.env import GymFxEnv
from app.session_exposure import SessionCalendar
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from metrics_plugins.default_metrics import Plugin as Metrics
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor)
from reward_plugins.pnl_reward import Plugin as Reward
from strategy_plugins.shared_execution_envelope import Plugin as Envelope


START = "2024-01-01 00:00:00"       # Monday
VENUE = "mt5_demo"
ACCOUNT = "fp-1"
SYMBOL = "ETHUSD"
CALENDAR_DIGEST = "cal-weekly-v1"

# One closure inside the sample: bars are 4h from Monday 00:00, so
# bar 12 is Tuesday 00:00 and bar 18 is Wednesday 00:00.
CLOSE_AT = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
REOPEN_AT = pd.Timestamp("2024-01-03 00:00:00", tz="UTC")


def policy(**kw):
    base = {
        "enabled": True,
        "session_source": "venue_symbol_sessions_v1",
        "wind_down_hours": 8.0,
        "forced_flatten_hours": 4.0,
        "cancel_pending_on_wind_down": True,
        "allow_risk_increase_during_wind_down": False,
        "reopen_min_hours": 4.0,
        "reopen_min_closed_bars": 1,
        "stability_consecutive_checks": 2,
        "max_spread_relative_to_baseline": 3.0,
        "max_gap_sigma": 4.0,
        "max_realized_vol_relative_to_baseline": 3.0,
        "carried_position_recovery":
            "protected_opportunistic_then_forced",
        "holiday_policy": "same_as_weekly",
        "calendar_identity": CALENDAR_DIGEST,
    }
    base.update(kw)
    return base


def _csv(tmp_path, closes, name="bars.csv"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    frame = pd.DataFrame({
        "DATE_TIME": pd.date_range(START, periods=n, freq="4h"),
        "OPEN": closes, "HIGH": closes * 1.0005,
        "LOW": closes * 0.9995, "CLOSE": closes, "VOLUME": 1000.0,
        "feat": np.linspace(0.0, 1.0, n),
    })
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _env(tmp_path, closes, *, session=True, intervals="default", **kw):
    config = {
        "input_data_file": str(_csv(tmp_path, closes)),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "window_size": 4, "initial_cash": 10000.0,
        "include_price_window": False,
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
                       "terminated": term, "info": info})
        if term:
            break
    return frames


FLAT = [100.0] * 30


class TestC5ClosedIntervalHasNoActionableStep:

    def test_closed_bars_produce_zero_reward_and_no_action(self,
                                                           tmp_path):
        rising = [100.0 + i for i in range(30)]
        env = _env(tmp_path, rising)
        frames = _drive(env, [1.0] * 25)
        closed = [f for f in frames
                  if f["info"].get("session_state") ==
                  "EXPECTED_MARKET_CLOSED"]
        assert closed, "the fixture must cross the closure"
        for frame in closed:
            assert frame["reward"] == 0.0, (
                "a closed interval offers no actionable step, so no "
                "reward may be attributed to it")
            assert frame["info"]["session_no_actionable_step"] is True
            assert frame["info"]["session_final_action"] is None
            assert frame["info"]["session_mapped_action"] is None
            assert frame["info"]["session_action_after_overlay"] == 0

    def test_account_state_carries_forward_across_the_closure(
            self, tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        states = [f["info"]["session_state"] for f in frames]
        first = states.index("EXPECTED_MARKET_CLOSED")
        last = len(states) - 1 - states[::-1].index(
            "EXPECTED_MARKET_CLOSED")
        entering = frames[first]["info"]
        leaving = frames[last]["info"]
        # equity and position SURVIVE the closed interval: the bars
        # advanced, nothing was reset, nothing was liquidated
        assert leaving["equity"] == pytest.approx(
            entering["equity"], rel=1e-9)
        assert leaving["position"] == entering["position"]

    def test_disabled_by_default_keeps_the_legacy_contract(self,
                                                           tmp_path):
        env = _env(tmp_path, FLAT, session=False)
        assert env.session_exposure_enabled is False
        assert not any(name.startswith("session_")
                       for name in env.observation_space.spaces)
        frames = _drive(env, [1.0] * 6)
        assert "session_state" not in frames[-1]["info"]


class TestC5ObservationContract:

    def test_session_fields_are_in_the_space_and_the_observation(
            self, tmp_path):
        env = _env(tmp_path, FLAT)
        for name in GymFxEnv.SESSION_OBSERVATION_NAMES:
            assert name in env.observation_space.spaces
        frames = _drive(env, [1.0] * 25)
        for frame in frames:
            obs = frame["obs"]
            for name in GymFxEnv.SESSION_OBSERVATION_NAMES:
                assert obs[name].shape == (1,)
                assert obs[name].dtype == np.float32
            assert env.observation_space.contains(obs), (
                "every emitted observation must satisfy the declared "
                "space")

    def test_wind_down_and_closed_flags_track_the_state(self,
                                                        tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        seen = set()
        for frame in frames:
            state = frame["info"]["session_state"]
            seen.add(state)
            obs = frame["obs"]
            assert obs["session_market_closed"][0] == (
                1.0 if state == "EXPECTED_MARKET_CLOSED" else 0.0)
            assert obs["session_wind_down"][0] == (
                1.0 if frame["info"]["session_wind_down"] else 0.0)
            assert obs["session_reopen_blackout"][0] == (
                1.0 if state == "REOPEN_BLACKOUT" else 0.0)
        assert {"NORMAL_TRADING", "WIND_DOWN", "FORCED_FLATTEN",
                "EXPECTED_MARKET_CLOSED", "REOPEN_BLACKOUT"} <= seen, (
            f"the fixture must exercise all five states, saw {seen}")


class TestC5ForcedCloseUsesTheSharedEnvelope:

    def test_forced_flatten_closes_through_the_envelope(self,
                                                        tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        forced = [f for f in frames
                  if f["info"].get("session_overlay") == "forced_close"]
        assert forced, "the fixture must reach FORCED_FLATTEN holding"
        assert forced[0]["info"]["session_final_action"] == "CLOSE"
        assert forced[0]["info"]["session_action_after_overlay"] == 3

        # the close is ECONOMICALLY SETTLED by the shared envelope:
        # it appears as a policy_close event with a real price, not as
        # a silent bypass
        events = list(getattr(env.bridge, "close_events", []))
        assert any(e.get("reason") == "policy_close" for e in events), (
            "a forced close must travel the shared cost/fill envelope; "
            "flatten_step/force_flat_request returns before the plugin "
            "dispatch and would leave no close event")
        for event in events:
            if event.get("reason") == "policy_close":
                assert float(event["price"]) > 0.0

    def test_position_is_flat_after_the_forced_close(self, tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        states = [f["info"]["session_state"] for f in frames]
        first_closed = states.index("EXPECTED_MARKET_CLOSED")
        assert frames[first_closed]["info"]["position"] == 0, (
            "exposure must be flat before the closed interval begins")

    def test_protective_orders_are_never_counted_as_entry_orders(
            self, tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        holding = [f for f in frames if f["info"]["position"] != 0]
        assert holding
        for frame in holding:
            assert frame["info"]["session_entry_orders"] == 0, (
                "native SL/TP brackets are protective and must never "
                "be cancelled by the weekly overlay")


class TestC5RawMappedOverlayFinalAreDistinct:

    def test_four_records_are_kept_separately(self, tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        for frame in frames:
            info = frame["info"]
            assert "session_raw_model_action" in info
            assert "session_mapped_action" in info
            assert "session_overlay" in info
            assert "session_final_action" in info
            # the submitted action is a FIFTH, separate record
            assert "session_action_after_overlay" in info

    def test_masked_entry_does_not_rewrite_the_raw_action(self,
                                                          tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        masked = [f for f in frames
                  if f["info"].get("session_overlay") in
                  ("masked_risk_increase",
                   "masked_entry_during_blackout")]
        assert masked, "the fixture must mask at least one entry"
        for frame in masked:
            info = frame["info"]
            assert info["session_raw_model_action"] == 1.0, (
                "the model's raw request must survive verbatim")
            assert info["session_final_action"] != 1.0
            assert info["session_action_after_overlay"] == 0
            assert info["session_mapped_action"]["risk_increasing"] \
                is True

    def test_normal_trading_passes_through_unchanged(self, tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        normal = [f for f in frames
                  if f["info"]["session_state"] == "NORMAL_TRADING"]
        assert normal
        for frame in normal:
            info = frame["info"]
            assert info["session_overlay"] == "pass_through"
            assert info["session_final_action"] == \
                info["session_raw_model_action"]
            assert info["session_action_after_overlay"] == \
                info["session_action_before_overlay"]


class TestC5TerminationCannotEraseExposure:

    def test_open_exposure_survives_termination_as_a_migration(
            self, tmp_path):
        # no calendar closure at all, so nothing forces a flatten and
        # the episode ends with the position still open
        env = _env(tmp_path, FLAT, intervals=[
            ["2030-01-06 00:00:00+00:00", "2030-01-07 00:00:00+00:00"]])
        frames = _drive(env, [1.0] * 40)
        last = frames[-1]
        assert last["terminated"] is True
        if last["info"]["position"] != 0:
            assert last["info"][
                "session_exposure_survived_termination"] is True
            assert last["info"][
                "session_carried_position_requires_migration"] is True
            assert last["info"]["session_carried_exposure"] != 0.0
            assert last["info"][
                "termination_does_not_close_exposure"] is True

    def test_flat_termination_reports_no_carried_exposure(self,
                                                          tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 12 + [0.0] * 40)
        last = frames[-1]
        assert last["terminated"] is True
        if last["info"]["position"] == 0:
            assert last["info"][
                "session_exposure_survived_termination"] is False
            assert last["info"]["session_carried_exposure"] == 0.0


class TestC5EvidenceFailsClosed:

    def test_missing_calendar_intervals_fail_closed_to_wind_down(
            self, tmp_path):
        env = _env(tmp_path, FLAT, intervals=None)
        assert env._session_calendar is None
        frames = _drive(env, [1.0] * 6)
        for frame in frames:
            info = frame["info"]
            assert info["session_state"] == "WIND_DOWN"
            assert info["session_evidence_ok"] is False
            assert info["session_evidence_failed_closed"] is True
            # and it MASKS the entry rather than degrading to neutral
            assert info["session_overlay"] == "masked_risk_increase"
            assert info["session_action_after_overlay"] == 0

    def test_calendar_identity_mismatch_refuses_at_construction(
            self, tmp_path):
        # a valid but WRONG calendar can never govern the strategy
        env = _env(tmp_path, FLAT)
        wrong = SessionCalendar.build(
            venue=VENUE, account_fingerprint=ACCOUNT, symbol=SYMBOL,
            calendar_digest="some-other-calendar",
            intervals=[(GymFxEnv._session_utc(CLOSE_AT),
                        GymFxEnv._session_utc(REOPEN_AT))])
        env._session_calendar = wrong
        with pytest.raises(Exception, match="calendar identity"):
            _drive(env, [1.0] * 3)

    def test_malformed_policy_refuses_at_construction(self, tmp_path):
        broken = policy()
        del broken["forced_flatten_hours"]
        with pytest.raises(Exception, match="forced_flatten_hours"):
            _env(tmp_path, FLAT, session_exposure_policy=broken)


class TestC5SessionEvidenceReadsTheRealTimestamps:
    """The default data feed promotes the date column to the INDEX.
    A session overlay that reads only .columns silently loses every
    timestamp and fails closed forever, which is indistinguishable
    from a broken calendar."""

    def test_timestamps_are_found_when_the_date_is_the_index(
            self, tmp_path):
        env = _env(tmp_path, FLAT)
        assert env._date_column not in env.dataframe.columns
        assert isinstance(env.dataframe.index, pd.DatetimeIndex)
        assert env._session_now(5) is not None
        frames = _drive(env, [1.0] * 8)
        assert any(f["info"]["session_evidence_ok"] for f in frames)
        assert not any(
            f["info"]["session_evidence_failed_closed"]
            for f in frames), (
            "valid session evidence must never be reported as failed "
            "closed")

    def test_state_machine_reaches_all_five_states_in_order(self,
                                                            tmp_path):
        env = _env(tmp_path, FLAT)
        frames = _drive(env, [1.0] * 25)
        order = []
        for frame in frames:
            state = frame["info"]["session_state"]
            if not order or order[-1] != state:
                order.append(state)
        assert order == ["NORMAL_TRADING", "WIND_DOWN",
                         "FORCED_FLATTEN", "EXPECTED_MARKET_CLOSED",
                         "REOPEN_BLACKOUT"], order


class TestPreExistingObservationSpaceDivergence:
    """NOT a C5 defect and NOT fixed here: recorded so it cannot be
    lost. build_base_observation_space defaults include_price_window
    to ``not feature_columns``, while the preprocessor defaults it to
    its own params. With feature_columns set and the key unspecified,
    the env DECLARES a space without prices/returns and then EMITS
    them, so observation_space.contains(obs) is False on a plain
    feature-column config, with or without the session overlay."""

    def test_declared_space_diverges_from_emitted_observation(
            self, tmp_path):
        # a plain feature-column config that never mentions
        # include_price_window -- exactly what a caller would write
        config = {
            "input_data_file": str(_csv(tmp_path, FLAT)),
            "date_column": "DATE_TIME", "price_column": "CLOSE",
            "feature_columns": ["feat"], "feature_binary_columns": [],
            "window_size": 4, "initial_cash": 10000.0,
            "position_size": 1.0, "min_equity": 0.0,
            "env_mode": "training", "commission": 0.0, "leverage": 1.0,
            "action_space_mode": "continuous",
            "continuous_action_threshold": 0.0,
        }
        env = GymFxEnv(config, DataFeed(config), Broker(config),
                       Envelope(config), Preprocessor(config),
                       Reward(config), Metrics(config))
        obs, _info = env.reset(seed=7)
        assert "prices" not in env.observation_space.spaces
        assert "prices" in obs, (
            "the preprocessor emits prices the declared space omits")
        assert env.observation_space.contains(obs) is False, (
            "pre-existing: the declared observation space does not "
            "match the emitted observation on a feature-column config")
