"""
env.py — Gymnasium-style, backtrader-powered trading env.

The heavy lifting (order book, fills, analyzers) runs inside a background
thread via a bt.Cerebro driven by BTBridgeStrategy. The env is the thin
step/reset API agents interact with.

Action space (v0): Discrete(3) where 0=hold, 1=long, 2=short.
Observation space: Dict provided by the preprocessor plugin. This env
forwards a bridge_state dict so the preprocessor can include the agent's
own position/equity/unrealized-pnl/steps-remaining features.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for GymFxEnv. Install with: pip install gymnasium"
    ) from exc

from app.bt_bridge import BTBridge, build_cerebro


class GymFxEnv(gym.Env):
    """Backtrader-backed forex trading env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Dict[str, Any],
        data_feed_plugin,
        broker_plugin,
        strategy_plugin,  # overlay, not required for v0 logic
        preprocessor_plugin,
        reward_plugin,
        metrics_plugin,
    ):
        super().__init__()
        self.config = dict(config)
        self.data_feed_plugin = data_feed_plugin
        self.broker_plugin = broker_plugin
        self.strategy_plugin = strategy_plugin
        self.preprocessor_plugin = preprocessor_plugin
        self.reward_plugin = reward_plugin
        self.metrics_plugin = metrics_plugin

        # --- market / env parameters ----------------------------------------
        self.initial_cash = float(self.config.get("initial_cash", 10000.0))
        self.position_size = float(self.config.get("position_size", 1.0))
        self.window_size = int(self.config.get("window_size", 32))
        self.price_column = self.config.get("price_column", "CLOSE")
        self.min_equity = float(self.config.get("min_equity", self.initial_cash * 0.01))

        # --- load feed + sanity ---------------------------------------------
        self.dataframe = self.data_feed_plugin.load_data(self.config)
        if self.dataframe is None or len(self.dataframe) < self.window_size + 2:
            raise ValueError("input data is empty or too short for the configured window")
        if self.price_column not in self.dataframe.columns:
            raise ValueError(f"price_column '{self.price_column}' not found in data")
        self.total_bars = int(len(self.dataframe))

        # --- action / observation spaces ------------------------------------
        self.action_space_mode = str(self.config.get("action_space_mode", "discrete")).lower()
        if self.action_space_mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            # Threshold for mapping continuous actions to {-1, 0, +1}
            self.continuous_action_threshold = float(
                self.config.get("continuous_action_threshold", 0.33)
            )
        else:
            self.action_space = spaces.Discrete(3)
            self.continuous_action_threshold = None
        self.observation_space = spaces.Dict(
            {
                "prices": spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32),
                "returns": spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32),
                "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "equity_norm": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "unrealized_pnl_norm": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "steps_remaining_norm": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        # Optional Stage B force-close / session-window context. Disabled by
        # default so legacy PPO/SAC/DQN configs are unaffected; when enabled
        # (Stage B diagnostic configs), the env exposes four extra obs fields
        # and surfaces the raw values in info[] for trace evidence.
        self.stage_b_force_close_obs = bool(
            self.config.get("stage_b_force_close_obs", False)
        )
        # Default to Friday 20:00 UTC force-close zone (1 hour window) and
        # Monday entry window of the first 4 hours UTC. Both knobs are
        # config-overridable.
        self.force_close_dow = int(self.config.get("force_close_dow", 4))  # Friday
        self.force_close_hour = int(self.config.get("force_close_hour", 20))
        self.force_close_window_hours = int(self.config.get("force_close_window_hours", 4))
        self.monday_entry_window_hours = int(self.config.get("monday_entry_window_hours", 4))
        self.stage_b_force_close_reward_penalty = bool(
            self.config.get("stage_b_force_close_reward_penalty", False)
        )
        self.force_close_exposure_penalty_coef = float(
            self.config.get("force_close_exposure_penalty_coef", 0.0)
        )
        self.force_close_exposure_penalty_window_hours = float(
            self.config.get(
                "force_close_exposure_penalty_window_hours",
                self.force_close_window_hours,
            )
        )
        if self.stage_b_force_close_obs:
            self.observation_space = spaces.Dict(
                {
                    **self.observation_space.spaces,
                    "bars_to_force_close": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "hours_to_force_close": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "is_force_close_zone": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "is_monday_entry_window": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                }
            )
        # OANDA FX America/New_York calendar obs/info — opt-in via config and
        # auto-enabled for the OANDA FX broker profile. DST-aware: relies on
        # zoneinfo conversion, never on a fixed UTC Friday close.
        self.oanda_fx_calendar_obs = bool(
            self.config.get("oanda_fx_calendar_obs", False)
            or str(self.config.get("broker_profile") or "").lower() == "oanda_us_fx"
        )
        if self.oanda_fx_calendar_obs:
            self.observation_space = spaces.Dict(
                {
                    **self.observation_space.spaces,
                    "hours_to_fx_daily_break": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "bars_to_fx_daily_break": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "hours_to_friday_close": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "bars_to_friday_close": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "is_friday_risk_reduction_window": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "is_no_new_position_window": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "is_force_flat_window": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "is_broker_daily_break_near": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "broker_market_open": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "margin_closeout_percent": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "margin_available_norm": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                }
            )
        self._date_column = str(self.config.get("date_column", "DATE_TIME"))
        self._timeframe_hours = self._infer_timeframe_hours()

        # --- runtime handles -------------------------------------------------
        self.bridge: Optional[BTBridge] = None
        self._runner: Optional[threading.Thread] = None
        self._cerebro = None
        self._strategy_instance = None
        self._np_random = np.random.default_rng()
        self._reset_action_diagnostics()

    # ----------------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._teardown_runner()

        self.bridge = BTBridge(initial_cash=self.initial_cash)
        self.bridge.reset(initial_cash=self.initial_cash, total_bars=self.total_bars)
        self._reset_action_diagnostics()

        bt_feed = self.data_feed_plugin.build_bt_feed(self.dataframe, self.config)
        broker = self.broker_plugin.build_bt_broker(self.config)

        self._cerebro = build_cerebro(
            bt_feed=bt_feed,
            broker=broker,
            bridge=self.bridge,
            position_size=self.position_size,
            min_equity=self.min_equity,
            strategy_plugin=self.strategy_plugin,
            config=self.config,
        )

        self._runner = threading.Thread(target=self._run_cerebro, name="gym-fx-cerebro", daemon=True)
        self._runner.start()
        self._wait_obs()

        return self._make_observation(), self._make_info()

    def step(self, action):
        if self.bridge is None:
            raise RuntimeError("Call reset() before step().")

        raw_action = self._raw_action_value(action)
        a = self._coerce_action(action)
        self._record_action(raw_action, a)

        if self.bridge.terminated:
            return self._make_observation(), 0.0, True, False, self._make_info()

        self.bridge.action_slot = a
        self.bridge.obs_ready.clear()
        self.bridge.action_ready.set()
        self._wait_obs()

        prev_equity = self.bridge.prev_equity
        new_equity = self.bridge.equity

        base_reward = float(
            self.reward_plugin.compute_reward(
                prev_equity=prev_equity,
                new_equity=new_equity,
                step=self.bridge.bar_index,
                config=self.config,
            )
        )
        force_close_penalty = self._force_close_reward_penalty(self.bridge.bar_index)
        reward = base_reward - force_close_penalty

        terminated = bool(self.bridge.terminated or new_equity <= self.min_equity)
        truncated = False

        obs = self._make_observation()
        info = self._make_info()
        info.update(
            reward=reward,
            base_reward=base_reward,
            force_close_reward_penalty=force_close_penalty,
            pnl=new_equity - prev_equity,
            trade_cost=self.bridge.last_trade_cost,
        )

        if terminated:
            self.bridge.stop_requested = True
            self.bridge.action_ready.set()

        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        return None

    def close(self):
        self._teardown_runner()

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Action handling
    # ----------------------------------------------------------------------
    def _coerce_action(self, action) -> int:
        """Map the agent action (Discrete int or Box[-1,+1]) to {0,1,2}."""
        if self.action_space_mode == "continuous":
            try:
                val = float(np.asarray(action).reshape(-1)[0])
            except Exception:
                val = 0.0
            thr = self.continuous_action_threshold or 0.33
            if val >= thr:
                return 1  # long
            if val <= -thr:
                return 2  # short
            return 0  # hold
        try:
            a = int(action)
        except Exception:
            a = 0
        return a if a in (0, 1, 2) else 0
    def _run_cerebro(self):
        try:
            result = self._cerebro.run(maxcpus=1, stdstats=False)
            self._strategy_instance = result[0] if result else None
        except Exception:  # pragma: no cover
            self.bridge.terminated = True
            self.bridge.obs_ready.set()

    def _teardown_runner(self) -> None:
        if self.bridge is not None:
            self.bridge.stop_requested = True
            self.bridge.action_ready.set()
        if self._runner is not None and self._runner.is_alive():
            self._runner.join(timeout=2.0)
        self._runner = None
        self._cerebro = None

    def _wait_obs(self) -> None:
        if not self.bridge.obs_ready.wait(timeout=30.0):
            self.bridge.terminated = True

    def _make_observation(self) -> Dict[str, np.ndarray]:
        assert self.bridge is not None
        step_idx = max(0, min(self.bridge.bar_index, self.total_bars))
        bridge_state = {
            "position": self.bridge.position,
            "equity": self.bridge.equity,
            "initial_cash": self.initial_cash,
            "price": self.bridge.price,
            "bar_index": self.bridge.bar_index,
            "total_bars": self.total_bars,
        }
        obs = self.preprocessor_plugin.make_observation(
            data=self.dataframe,
            step=step_idx,
            bridge_state=bridge_state,
            config=self.config,
        )
        if self.stage_b_force_close_obs:
            fc = self._force_close_features(step_idx)
            obs = dict(obs)
            obs["bars_to_force_close"] = np.array([fc["bars_to_force_close"]], dtype=np.float32)
            obs["hours_to_force_close"] = np.array([fc["hours_to_force_close"]], dtype=np.float32)
            obs["is_force_close_zone"] = np.array([fc["is_force_close_zone"]], dtype=np.float32)
            obs["is_monday_entry_window"] = np.array([fc["is_monday_entry_window"]], dtype=np.float32)
        if self.oanda_fx_calendar_obs:
            obs = dict(obs)
            cal = self._oanda_calendar_features(step_idx)
            for k in (
                "hours_to_fx_daily_break",
                "bars_to_fx_daily_break",
                "hours_to_friday_close",
                "bars_to_friday_close",
                "is_friday_risk_reduction_window",
                "is_no_new_position_window",
                "is_force_flat_window",
                "is_broker_daily_break_near",
                "broker_market_open",
            ):
                obs[k] = np.array([cal[k]], dtype=np.float32)
            obs["margin_closeout_percent"] = np.array(
                [self._safe_margin_closeout_percent()], dtype=np.float32
            )
            obs["margin_available_norm"] = np.array(
                [self._safe_margin_available_norm()], dtype=np.float32
            )
        return obs

    def _infer_timeframe_hours(self) -> float:
        raw = str(
            self.config.get("timeframe")
            or self.config.get("timeframe_label")
            or self.config.get("bar_timeframe")
            or ""
        ).strip().lower()
        if "_" in raw:
            raw = raw.rsplit("_", 1)[-1]
        try:
            if raw.endswith("m"):
                return max(0.0, int(raw[:-1]) / 60.0)
            if raw.endswith("h"):
                return float(int(raw[:-1]))
            if raw.endswith("d"):
                return float(int(raw[:-1]) * 24)
        except ValueError:
            return 0.0
        return 0.0

    def _force_close_features(self, step_idx: int) -> Dict[str, float]:
        """Compute Stage B force-close / Monday-entry-window features.

        All fields are 0.0 if the underlying timestamp cannot be resolved.
        This helper never raises — it returns neutral values so a missing
        date column degrades gracefully into "no signal" rather than a step
        failure mid-rollout.
        """
        try:
            import pandas as pd

            if self._date_column not in self.dataframe.columns:
                return {
                    "bars_to_force_close": 0.0,
                    "hours_to_force_close": 0.0,
                    "is_force_close_zone": 0.0,
                    "is_monday_entry_window": 0.0,
                }
            idx = max(0, min(step_idx, len(self.dataframe) - 1))
            ts = pd.to_datetime(self.dataframe.iloc[idx][self._date_column], errors="coerce")
            if ts is None or ts is pd.NaT:
                return {
                    "bars_to_force_close": 0.0,
                    "hours_to_force_close": 0.0,
                    "is_force_close_zone": 0.0,
                    "is_monday_entry_window": 0.0,
                }
            tf_hours = self._timeframe_hours or 1.0
            dow = int(ts.weekday())
            hour = int(ts.hour)
            # Hours until next force-close moment.
            days_ahead = (self.force_close_dow - dow) % 7
            target_total_hours = days_ahead * 24 + (self.force_close_hour - hour)
            if target_total_hours < 0:
                target_total_hours += 7 * 24
            hours_to_fc = float(target_total_hours)
            bars_to_fc = hours_to_fc / max(tf_hours, 1e-9)
            in_fc_zone = 1.0 if (
                dow == self.force_close_dow
                and self.force_close_hour <= hour < self.force_close_hour + self.force_close_window_hours
            ) else 0.0
            in_monday_window = 1.0 if (dow == 0 and hour < self.monday_entry_window_hours) else 0.0
            return {
                "bars_to_force_close": bars_to_fc,
                "hours_to_force_close": hours_to_fc,
                "is_force_close_zone": in_fc_zone,
                "is_monday_entry_window": in_monday_window,
            }
        except Exception:
            return {
                "bars_to_force_close": 0.0,
                "hours_to_force_close": 0.0,
                "is_force_close_zone": 0.0,
                "is_monday_entry_window": 0.0,
            }

    def _oanda_calendar_features(self, step_idx: int) -> Dict[str, float]:
        """Resolve the OANDA NY-time calendar features for ``step_idx``.

        Returns neutral (all-zero) values if the date column is missing or
        the timestamp cannot be parsed; the env never raises mid-rollout.
        """
        try:
            from app.oanda_calendar import compute_fx_calendar_features
        except Exception:
            return {
                "hours_to_fx_daily_break": 0.0,
                "bars_to_fx_daily_break": 0.0,
                "hours_to_friday_close": 0.0,
                "bars_to_friday_close": 0.0,
                "is_friday_risk_reduction_window": 0.0,
                "is_no_new_position_window": 0.0,
                "is_force_flat_window": 0.0,
                "is_broker_daily_break_near": 0.0,
                "broker_market_open": 0.0,
                "is_no_trade_window": 0.0,
            }
        if self._date_column not in self.dataframe.columns:
            ts = None
        else:
            idx = max(0, min(step_idx, len(self.dataframe) - 1))
            ts = self.dataframe.iloc[idx][self._date_column]
        tf_h = float(self._timeframe_hours or 1.0) or 1.0
        return compute_fx_calendar_features(ts, timeframe_hours=tf_h)

    def _safe_margin_closeout_percent(self) -> float:
        """Read margin_closeout_percent from the bridge if available; else 0.0."""
        if self.bridge is None:
            return 0.0
        val = getattr(self.bridge, "margin_closeout_percent", None)
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _safe_margin_available_norm(self) -> float:
        """Margin available normalised by initial cash; deterministic placeholder."""
        if self.bridge is None:
            return 0.0
        val = getattr(self.bridge, "margin_available", None)
        if val is None:
            equity = getattr(self.bridge, "equity", None)
            val = equity if equity is not None else self.initial_cash
        try:
            base = float(self.initial_cash) if self.initial_cash else 1.0
            return float(val) / base
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    def _force_close_reward_penalty(self, step_idx: int) -> float:
        """Optional normalized reward penalty for late-Friday exposure.

        This is a Stage B diagnostic knob, disabled by default. It does not
        change the tradability metrics; it only shapes training reward when a
        config explicitly asks whether Friday-close context needs a behavioral
        incentive in addition to observation fields.
        """
        if not (
            self.stage_b_force_close_obs
            and self.stage_b_force_close_reward_penalty
            and self.force_close_exposure_penalty_coef > 0
        ):
            return 0.0
        if self.bridge is None or int(getattr(self.bridge, "position", 0) or 0) == 0:
            return 0.0
        fc = self._force_close_features(step_idx)
        hours_to_fc = float(fc.get("hours_to_force_close", 0.0) or 0.0)
        in_force_close_zone = float(fc.get("is_force_close_zone", 0.0) or 0.0) > 0
        in_penalty_window = 0.0 <= hours_to_fc <= max(
            0.0, self.force_close_exposure_penalty_window_hours
        )
        if not (in_force_close_zone or in_penalty_window):
            return 0.0
        return self.force_close_exposure_penalty_coef * abs(
            float(getattr(self.bridge, "position", 0) or 0)
        )

    def _make_info(self) -> Dict[str, Any]:
        assert self.bridge is not None
        info = {
            "equity": self.bridge.equity,
            "position": self.bridge.position,
            "price": self.bridge.price,
            "bar_index": self.bridge.bar_index,
            "total_bars": self.total_bars,
            "trades": self.bridge.trade_count,
            "commission_paid": self.bridge.commission_paid,
            "raw_action_value": self._last_raw_action_value,
            "coerced_action": self._last_coerced_action,
            "action_diagnostics": dict(self._action_diagnostics),
            "execution_diagnostics": dict(getattr(self.bridge, "execution_diagnostics", {}) or {}),
        }
        if self.stage_b_force_close_obs:
            step_idx = max(0, min(self.bridge.bar_index, self.total_bars))
            info.update(self._force_close_features(step_idx))
        if self.oanda_fx_calendar_obs:
            step_idx = max(0, min(self.bridge.bar_index, self.total_bars))
            info.update(self._oanda_calendar_features(step_idx))
            info["margin_closeout_percent"] = self._safe_margin_closeout_percent()
            info["margin_available_norm"] = self._safe_margin_available_norm()
            for k in ("broker_profile", "market_type", "trade_rate_band_id", "calendar_policy_id"):
                v = self.config.get(k)
                if v is not None:
                    info[k] = v
        return info

    def summary(self) -> Dict[str, Any]:
        analyzers: Dict[str, Any] = {}
        if self._strategy_instance is not None:
            for name in ("trades", "sharpe", "drawdown", "sqn", "time_return"):
                an = getattr(self._strategy_instance.analyzers, name, None)
                if an is not None:
                    try:
                        analyzers[name] = an.get_analysis()
                    except Exception:
                        analyzers[name] = None
        summary = self.metrics_plugin.summarize(
            initial_cash=self.initial_cash,
            final_equity=self.bridge.equity if self.bridge else self.initial_cash,
            analyzers=analyzers,
            config=self.config,
        )
        summary["action_diagnostics"] = dict(self._action_diagnostics)
        summary["execution_diagnostics"] = dict(getattr(self.bridge, "execution_diagnostics", {}) or {})
        return summary

    def _reset_action_diagnostics(self) -> None:
        self._last_raw_action_value = 0.0
        self._last_coerced_action = 0
        self._action_diagnostics = {
            "steps": 0,
            "hold_actions": 0,
            "long_actions": 0,
            "short_actions": 0,
            "non_hold_actions": 0,
            "continuous_deadband_actions": 0,
            "raw_abs_sum": 0.0,
            "raw_min": None,
            "raw_max": None,
            "continuous_action_threshold": self.continuous_action_threshold,
        }

    def _raw_action_value(self, action) -> float:
        try:
            return float(np.asarray(action).reshape(-1)[0])
        except Exception:
            try:
                return float(action)
            except Exception:
                return 0.0

    def _record_action(self, raw_action: float, coerced_action: int) -> None:
        self._last_raw_action_value = float(raw_action)
        self._last_coerced_action = int(coerced_action)
        diag = self._action_diagnostics
        diag["steps"] += 1
        diag["raw_abs_sum"] += abs(float(raw_action))
        diag["raw_min"] = raw_action if diag["raw_min"] is None else min(float(diag["raw_min"]), raw_action)
        diag["raw_max"] = raw_action if diag["raw_max"] is None else max(float(diag["raw_max"]), raw_action)
        if coerced_action == 1:
            diag["long_actions"] += 1
            diag["non_hold_actions"] += 1
        elif coerced_action == 2:
            diag["short_actions"] += 1
            diag["non_hold_actions"] += 1
        else:
            diag["hold_actions"] += 1
            if self.action_space_mode == "continuous":
                diag["continuous_deadband_actions"] += 1
