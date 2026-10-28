"""G6: the foundational environment repairs.

F-A -- the OANDA calendar helper read the timestamp only from
``.columns`` while the default data feed promotes the date to the
INDEX, so all eleven of its fields were constant zeros under that
feed and every run observing them trained on no signal.

F-B -- the declared observation space omitted prices/returns that the
preprocessor emits, so ``observation_space.contains(obs)`` was False
on any feature-column config that did not set ``include_price_window``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.env import GymFxEnv, build_base_observation_space
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from metrics_plugins.default_metrics import Plugin as Metrics
from preprocessor_plugins.default_preprocessor import (
    Plugin as DefaultPreprocessor)
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor)
from reward_plugins.pnl_reward import Plugin as Reward
from strategy_plugins.shared_execution_envelope import Plugin as Envelope


# the eleven fields the OANDA calendar block contributes
CALENDAR_FIELDS = (
    "hours_to_fx_daily_break", "bars_to_fx_daily_break",
    "hours_to_friday_close", "bars_to_friday_close",
    "is_friday_risk_reduction_window", "is_no_new_position_window",
    "is_force_flat_window", "is_broker_daily_break_near",
    "broker_market_open", "is_no_trade_window",
)
OBSERVATION_MARGIN_FIELDS = ("margin_closeout_percent",
                             "margin_available_norm")


def _csv(tmp_path, periods=120, freq="1h", name="bars.csv"):
    stamps = pd.date_range("2024-06-03 00:00:00", periods=periods,
                           freq=freq)
    closes = 1.08 + 0.001 * np.sin(np.arange(periods, dtype=float))
    frame = pd.DataFrame({
        "DATE_TIME": stamps,
        "OPEN": closes, "HIGH": closes * 1.0005,
        "LOW": closes * 0.9995, "CLOSE": closes, "VOLUME": 1000.0,
        "feat": np.linspace(0.0, 1.0, periods),
    })
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _env(tmp_path, **kw):
    config = {
        "input_data_file": str(kw.pop("csv", None) or _csv(tmp_path)),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "window_size": 4, "initial_cash": 10000.0,
        "position_size": 1.0, "min_equity": 0.0,
        "env_mode": "training", "commission": 0.0, "leverage": 1.0,
        "timeframe": "1h",
        "feature_columns": ["feat"], "feature_binary_columns": [],
    }
    preprocessor = kw.pop("preprocessor", Preprocessor)
    config.update(kw)
    return GymFxEnv(config, DataFeed(config), Broker(config),
                    Envelope(config), preprocessor(config),
                    Reward(config), Metrics(config))


# =================================================================== #
# G6-1: F-A                                                           #
# =================================================================== #

class TestFACalendarTimestampLookup:

    def test_the_default_feed_really_promotes_the_date_to_the_index(
            self, tmp_path):
        env = _env(tmp_path, oanda_fx_calendar_obs=True)
        assert env._date_column not in env.dataframe.columns
        assert isinstance(env.dataframe.index, pd.DatetimeIndex), (
            "this whole finding only exists because of this layout")

    def test_all_eleven_fields_are_live_not_constant_zeros(self,
                                                           tmp_path):
        env = _env(tmp_path, oanda_fx_calendar_obs=True)
        rows = [env._oanda_calendar_features(i)
                for i in range(len(env.dataframe))]
        for field in CALENDAR_FIELDS:
            values = {row[field] for row in rows}
            assert values != {0.0}, (
                f"{field} is constant zero — the helper is inert")
        # the binary windows must actually toggle across a week
        for field in ("is_friday_risk_reduction_window",
                      "is_no_new_position_window",
                      "is_force_flat_window", "broker_market_open"):
            values = {row[field] for row in rows}
            assert values == {0.0, 1.0}, (
                f"{field} never changes state: {values}")

    def test_index_and_column_layouts_are_bit_identical(self,
                                                        tmp_path):
        """Equivalent inputs, two layouts, identical features."""
        indexed = _env(tmp_path, oanda_fx_calendar_obs=True)
        columnar = _env(tmp_path, oanda_fx_calendar_obs=True)
        columnar.dataframe = indexed.dataframe.reset_index()
        assert columnar._date_column in columnar.dataframe.columns
        assert not isinstance(columnar.dataframe.index,
                              pd.DatetimeIndex)
        for i in range(len(indexed.dataframe)):
            left = indexed._oanda_calendar_features(i)
            right = columnar._oanda_calendar_features(i)
            assert left == right, f"bar {i}: {left} != {right}"
            # bit-exact, not merely close
            for field in CALENDAR_FIELDS:
                assert repr(left[field]) == repr(right[field])

    def test_dst_correctness_survives_the_index_layout(self,
                                                       tmp_path):
        # 2024-06-07 19:30 UTC is Friday 15:30 EDT
        stamps = pd.date_range("2024-06-07 19:30:00", periods=8,
                               freq="1h")
        closes = np.full(8, 1.08)
        frame = pd.DataFrame({
            "DATE_TIME": stamps, "OPEN": closes,
            "HIGH": closes, "LOW": closes, "CLOSE": closes,
            "VOLUME": 1000.0, "feat": np.linspace(0, 1, 8)})
        path = tmp_path / "dst.csv"
        frame.to_csv(path, index=False)
        env = _env(tmp_path, csv=path, oanda_fx_calendar_obs=True,
                   window_size=2)
        features = env._oanda_calendar_features(0)
        assert features["is_no_new_position_window"] == 1.0
        assert features["is_friday_risk_reduction_window"] == 1.0
        assert features["is_force_flat_window"] == 0.0
        assert features["broker_market_open"] == 1.0
        assert 1.0 <= features["hours_to_friday_close"] <= 2.0

    def test_the_fields_reach_the_observation_and_the_info(self,
                                                           tmp_path):
        env = _env(tmp_path, oanda_fx_calendar_obs=True)
        obs, info = env.reset(seed=7)
        try:
            for field in CALENDAR_FIELDS[:-1]:      # no_trade is info
                assert field in obs
            for field in OBSERVATION_MARGIN_FIELDS:
                assert field in obs
            assert "is_no_trade_window" in info
            assert len(CALENDAR_FIELDS) - 1 + len(
                OBSERVATION_MARGIN_FIELDS) == 11
        finally:
            env.close()

    def test_a_frame_with_neither_layout_still_never_raises(self,
                                                            tmp_path):
        env = _env(tmp_path, oanda_fx_calendar_obs=True)
        env.dataframe = env.dataframe.reset_index(drop=True)
        features = env._oanda_calendar_features(3)
        assert set(features) == set(CALENDAR_FIELDS)
        assert set(features.values()) == {0.0}, (
            "with no timestamp anywhere the helper is neutral, and "
            "the env still does not raise mid-rollout")


# =================================================================== #
# G6-2: F-B                                                           #
# =================================================================== #

class TestFBObservationSpaceMatchesTheObservation:

    @pytest.mark.parametrize("label,extra", [
        ("features_only", {"feature_columns": ["feat"],
                           "feature_binary_columns": []}),
        ("features_no_price_window",
         {"feature_columns": ["feat"], "feature_binary_columns": [],
          "include_price_window": False}),
        ("features_with_price_window",
         {"feature_columns": ["feat"], "feature_binary_columns": [],
          "include_price_window": True}),
        ("prices_only_default_preprocessor",
         {"preprocessor": DefaultPreprocessor,
          "feature_columns": [], "feature_binary_columns": []}),
        ("live_stationary_contract",
         {"feature_columns": ["feat"], "feature_binary_columns": [],
          "agent_state_contract": "live_stationary_v2"}),
        ("oanda_calendar", {"oanda_fx_calendar_obs": True}),
        ("execution_cost",
         {"execution_cost_observation_enabled": True}),
        ("force_close_zone", {"stage_b_force_close_obs": True,
                              "timeframe": "1h"}),
    ])
    def test_declared_space_contains_every_emitted_observation(
            self, tmp_path, label, extra):
        env = _env(tmp_path, **extra)
        try:
            obs, _info = env.reset(seed=7)
            assert env.observation_space.contains(obs), (
                f"{label}: declared "
                f"{sorted(env.observation_space.spaces)} but emitted "
                f"{sorted(obs)}")
            for _ in range(5):
                obs, _r, term, _t, _i = env.step(1)
                assert env.observation_space.contains(obs), label
                if term:
                    break
        finally:
            env.close()

    def test_the_repaired_default_keeps_the_emitted_content(self,
                                                            tmp_path):
        """The space follows the EMITTER, so the repair adds the
        missing declarations rather than removing observed data."""
        env = _env(tmp_path, feature_columns=["feat"],
                   feature_binary_columns=[])
        try:
            obs, _info = env.reset(seed=7)
            assert "prices" in obs and "returns" in obs, (
                "the preprocessor still emits its price window")
            assert "prices" in env.observation_space.spaces
            assert "returns" in env.observation_space.spaces
            assert env.config["include_price_window"] is True, (
                "the flag is resolved ONCE and pinned into the config "
                "both sides read")
        finally:
            env.close()

    def test_an_explicit_false_is_still_honoured(self, tmp_path):
        env = _env(tmp_path, feature_columns=["feat"],
                   feature_binary_columns=[],
                   include_price_window=False)
        try:
            obs, _info = env.reset(seed=7)
            assert "prices" not in obs
            assert "prices" not in env.observation_space.spaces
            assert env.observation_space.contains(obs)
        finally:
            env.close()

    def test_the_unresolved_default_still_diverges(self):
        """The underlying mismatch is unchanged and is only masked by
        the env resolving the flag: build_base_observation_space on a
        bare feature config still omits prices, while the preprocessor
        default still emits them. Pinned so the repair cannot be
        quietly reverted to 'the defaults agree'."""
        declared = build_base_observation_space(
            {"feature_columns": ["feat"]}, window_size=4)
        assert "prices" not in declared.spaces
        assert Preprocessor({}).params["include_price_window"] is True
