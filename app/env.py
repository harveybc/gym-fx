"""
env.py — Gymnasium-style, backtrader-powered trading env.

The heavy lifting (order book, fills, analyzers) runs inside a background
thread via a bt.Cerebro driven by BTBridgeStrategy. The env is the thin
step/reset API agents interact with.

Action space (v0): Discrete(3) where 0=hold, 1=long, 2=short.
Observation space: Dict provided by the preprocessor plugin. This env
forwards a bridge_state dict so the preprocessor can include live-observable
position, equity, unrealized-PnL and holding-duration features.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for GymFxEnv. Install with: pip install gymnasium"
    ) from exc

from app.bt_bridge import BTBridge, build_cerebro


EXECUTION_COST_OBSERVATION_NAMES = (
    "execution_cost_commission_fraction_per_side_normalized",
    "execution_cost_full_spread_rate_normalized",
    "execution_cost_slippage_bps_per_side_normalized",
    "execution_cost_financing_enabled",
    "execution_cost_phase_progress",
)


def build_base_observation_space(
    config: Dict[str, Any],
    *,
    window_size: int,
) -> spaces.Dict:
    """Describe exactly the observation blocks emitted by the preprocessor.

    Legacy/default preprocessing has no feature list and therefore keeps the
    historical prices + returns + agent-state contract. Feature-aware runs can
    remove raw prices without leaving stale keys in Gymnasium's Dict space.
    """
    feature_columns = list(config.get("feature_columns") or [])
    include_prices = bool(
        config.get("include_price_window", not feature_columns)
    )
    include_agent_state = bool(config.get("include_agent_state", True))
    agent_state_contract = str(
        config.get("agent_state_contract", "legacy_episode_v1")
    ).strip().lower()
    observation_spaces: Dict[str, spaces.Space] = {}

    if feature_columns:
        observation_spaces["features"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(feature_columns)),
            dtype=np.float32,
        )
    if include_prices:
        observation_spaces.update({
            "prices": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size,),
                dtype=np.float32,
            ),
            "returns": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size,),
                dtype=np.float32,
            ),
        })
    if include_agent_state:
        if agent_state_contract not in {
            "legacy_episode_v1", "live_stationary_v2"
        }:
            raise ValueError(
                "agent_state_contract must be legacy_episode_v1 or "
                f"live_stationary_v2; got {agent_state_contract!r}"
            )
        observation_spaces.update({
            "position": spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "equity_norm": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "unrealized_pnl_norm": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            (
                "holding_duration_norm"
                if agent_state_contract == "live_stationary_v2"
                else "steps_remaining_norm"
            ): spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })
    if not observation_spaces:
        raise ValueError(
            "preprocessor observation contract emits no observation blocks"
        )
    return spaces.Dict(observation_spaces)


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

        # --- solvency mode (owner curriculum order, WP-C) -------------------
        # normal_realistic: breach terminates exactly as before (mandatory
        # for train-tail, validation, protected test and any Paper check).
        # easy_chronological_continuation: TRAIN-ONLY — a would-be margin
        # call liquidates (keeping the loss), recapitalizes operational
        # capital as journaled debt and continues chronologically.
        self.solvency_mode = str(
            self.config.get("solvency_mode", "normal_realistic"))
        if self.solvency_mode not in (
                "normal_realistic", "easy_chronological_continuation"):
            raise ValueError(
                f"unknown solvency_mode {self.solvency_mode!r}")
        if self.solvency_mode == "easy_chronological_continuation":
            env_mode = str(self.config.get("env_mode", ""))
            if env_mode != "training":
                raise ValueError(
                    "easy_chronological_continuation is train-only:"
                    f" env_mode={env_mode!r} may never enable relaxed"
                    " solvency dynamics (validation, test and Paper/Demo"
                    " always run normal_realistic)")
        self.config["recap_target_equity"] = float(
            self.config.get("recap_target_equity", self.initial_cash))
        self._last_recap_debt = 0.0

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
            self.continuous_action_contract = str(
                self.config.get(
                    "continuous_action_contract", "legacy_directional_v1"
                )
            )
            if self.continuous_action_contract not in (
                "legacy_directional_v1", "target_exposure_hysteresis_v2"
            ):
                raise ValueError(
                    "continuous_action_contract must be "
                    "legacy_directional_v1 or target_exposure_hysteresis_v2"
                )
            # Entry threshold for mapping continuous actions to {-1, 0, +1}.
            self.continuous_action_threshold = float(
                self.config.get("continuous_action_threshold", 0.33)
            )
            if (
                not np.isfinite(self.continuous_action_threshold)
                or not 0.0 <= self.continuous_action_threshold < 1.0
            ):
                raise ValueError(
                    "continuous_action_threshold must be finite in [0, 1)"
                )
            exit_default = min(0.05, self.continuous_action_threshold / 2.0)
            self.continuous_exit_threshold = float(
                self.config.get("continuous_exit_threshold", exit_default)
            )
            if (
                not np.isfinite(self.continuous_exit_threshold)
                or self.continuous_exit_threshold < 0.0
                or (
                    self.continuous_action_contract
                    == "target_exposure_hysteresis_v2"
                    and (
                        self.continuous_exit_threshold
                        >= self.continuous_action_threshold
                        and not (
                            self.continuous_action_threshold == 0.0
                            and self.continuous_exit_threshold == 0.0
                        )
                    )
                )
            ):
                raise ValueError(
                    "continuous_exit_threshold must be finite, non-negative, "
                    "and below continuous_action_threshold for the v2 contract"
                )
        else:
            self.action_space = spaces.Discrete(3)
            self.continuous_action_contract = None
            self.continuous_action_threshold = None
            self.continuous_exit_threshold = None
        # F-B: build_base_observation_space defaults
        # include_price_window to `not feature_columns` while the
        # preprocessor defaults it to its own params, so a
        # feature-column config that never mentions the key DECLARED a
        # space without prices/returns and then EMITTED them --
        # observation_space.contains(obs) was False. Resolve the flag
        # ONCE from the actual emitter and pin it into the config both
        # sides read, so the declared space and the emitted
        # observation agree by construction. The emitter is the
        # authority on what it emits, so no observation CONTENT
        # changes; only the declaration is repaired.
        if "include_price_window" not in self.config:
            emitter_params = getattr(
                self.preprocessor_plugin, "params", None) or {}
            if "include_price_window" in emitter_params:
                self.config["include_price_window"] = bool(
                    emitter_params["include_price_window"])
        self.observation_space = build_base_observation_space(
            self.config,
            window_size=self.window_size,
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
        self.execution_cost_observation_enabled = bool(
            self.config.get("execution_cost_observation_enabled", False)
        )
        self._execution_cost_observation = np.zeros(
            len(EXECUTION_COST_OBSERVATION_NAMES), dtype=np.float32
        )
        self._execution_cost_context: Dict[str, Any] = {}
        if self.execution_cost_observation_enabled:
            self.observation_space = spaces.Dict(
                {
                    **self.observation_space.spaces,
                    **{
                        name: spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float32,
                        )
                        for name in EXECUTION_COST_OBSERVATION_NAMES
                    },
                }
            )
        self._date_column = str(self.config.get("date_column", "DATE_TIME"))
        self._timeframe_hours = self._infer_timeframe_hours()
        self.event_context_execution_overlay = bool(
            self.config.get("event_context_execution_overlay", False)
        )
        self.event_context_no_trade_column = str(
            self.config.get("event_context_no_trade_column", "event_no_trade_window_active")
        )
        self.event_context_no_trade_threshold = float(
            self.config.get("event_context_no_trade_threshold", 0.5)
        )
        self.event_context_block_new_entries = bool(
            self.config.get("event_context_block_new_entries", True)
        )
        self.event_context_force_flat = bool(
            self.config.get("event_context_force_flat", False)
        )
        self.event_context_spread_stress_column = str(
            self.config.get(
                "event_context_spread_stress_column",
                "event_spread_stress_multiplier",
            )
        )
        self.event_context_slippage_stress_column = str(
            self.config.get(
                "event_context_slippage_stress_column",
                "event_slippage_stress_multiplier",
            )
        )

        # --- C5: weekly session-exposure overlay -----------------------------
        # Opt-in. The five-state machine of app/session_exposure.py governs
        # the REAL env path here: it is the only place where a session state
        # can change what reaches the broker. Disabled by default so every
        # pre-existing config keeps its exact observation space and behaviour.
        self._session_exposure_init()

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
        self._last_recap_debt = 0.0
        self._last_session_info = {}
        self._session_carried_migration = None

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
        a, event_context_info = self._apply_event_context_overlay(a)
        self._last_event_context_info = event_context_info
        a, session_info = self._apply_session_exposure_overlay(
            a, raw_action)
        self._record_action(raw_action, a)

        if self.bridge.terminated:
            return self._make_observation(), 0.0, True, False, self._make_info()

        self.bridge.action_slot = a
        self.bridge.raw_action_slot = raw_action
        self.bridge.obs_ready.clear()
        self.bridge.action_ready.set()
        self._wait_obs()

        prev_equity = self.bridge.prev_equity
        new_equity = self.bridge.equity

        # WP-C: reward always flows from ECONOMIC equity (operational
        # equity minus recapitalization debt), so a recapitalization can
        # never manufacture reward — in normal mode debt is always zero
        # and this is identical to the previous behavior.
        new_debt = float(self.bridge.recapitalization_debt)
        economic_prev = prev_equity - self._last_recap_debt
        economic_new = new_equity - new_debt
        self._last_recap_debt = new_debt

        base_reward = float(
            self.reward_plugin.compute_reward(
                prev_equity=economic_prev,
                new_equity=economic_new,
                step=self.bridge.bar_index,
                config=self.config,
            )
        )
        force_close_penalty = self._force_close_reward_penalty(self.bridge.bar_index)
        reward = base_reward - force_close_penalty

        terminated = bool(
            self.bridge.terminated
            or (self.solvency_mode == "normal_realistic"
                and new_equity <= self.min_equity)
        )
        truncated = False

        obs = self._make_observation()
        info = self._make_info()
        info.update(
            reward=reward,
            base_reward=base_reward,
            force_close_reward_penalty=force_close_penalty,
            pnl=economic_new - economic_prev,
            operational_pnl=new_equity - prev_equity,
            trade_cost=self.bridge.last_trade_cost,
            solvency_mode=self.solvency_mode,
            economic_equity=economic_new,
            recapitalization_debt=new_debt,
            recapitalization_count=int(self.bridge.recapitalization_count),
            would_margin_call_count=len(
                self.bridge.would_margin_call_events),
            termination_cause=self.bridge.termination_cause,
        )
        if self.session_exposure_enabled and terminated:
            info.update(self._session_termination_record())

        if terminated:
            self.bridge.stop_requested = True
            self.bridge.action_ready.set()

        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        return None

    def close(self):
        self._teardown_runner()

    def set_execution_cost_context(
        self,
        *,
        observable_names,
        observable_vector,
        cost_patch: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Apply one visible execution-cost scenario before the next reset."""
        names = tuple(str(name) for name in observable_names)
        if names != EXECUTION_COST_OBSERVATION_NAMES:
            raise ValueError(
                "execution cost observation contract mismatch: "
                f"expected {EXECUTION_COST_OBSERVATION_NAMES}, got {names}"
            )
        vector = np.asarray(observable_vector, dtype=np.float32).reshape(-1)
        if vector.shape != (len(EXECUTION_COST_OBSERVATION_NAMES),):
            raise ValueError("execution cost observation vector has invalid shape")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError("execution cost observation values must be finite in [0, 1]")

        commission = float(cost_patch["commission_fraction_per_side"])
        full_spread = float(cost_patch["full_spread_rate"])
        slippage_bps = float(cost_patch["slippage_bps_per_side"])
        if (
            not np.all(np.isfinite([commission, full_spread, slippage_bps]))
            or min(commission, full_spread, slippage_bps) < 0.0
        ):
            raise ValueError("execution cost patch values must be finite and nonnegative")
        financing = cost_patch.get("financing_enabled", False)
        if not isinstance(financing, bool):
            raise ValueError("financing_enabled must be boolean")

        self._execution_cost_observation = vector
        self._execution_cost_context = dict(metadata or {})
        self._execution_cost_context.update(
            {
                "commission_fraction_per_side": commission,
                "full_spread_rate": full_spread,
                "slippage_bps_per_side": slippage_bps,
                "financing_enabled": financing,
            }
        )
        self.config.update(self._execution_cost_context)
        # Backtrader compatibility: one adverse quote displacement per fill.
        adverse_fill_rate = full_spread / 2.0 + slippage_bps / 10_000.0
        self.config["commission"] = commission
        self.config["slippage"] = adverse_fill_rate
        self.config["slippage_perc"] = adverse_fill_rate

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Action handling
    # ----------------------------------------------------------------------
    def _coerce_action(self, action) -> int:
        """Map an agent action to hold/long/short/explicit-close.

        ``legacy_directional_v1`` preserves the historical three-action
        mapping. ``target_exposure_hysteresis_v2`` interprets the scalar as a
        desired exposure: strong values target long/short, weak values close
        existing exposure, and the band between the two thresholds holds.
        """
        if self.action_space_mode == "continuous":
            try:
                val = float(np.asarray(action).reshape(-1)[0])
            except Exception:
                val = 0.0
            # Zero is an intentional curriculum setting: every non-zero
            # policy output becomes directional while an exact zero remains
            # HOLD. Do not replace it with the legacy default through truthy
            # fallback semantics.
            thr = (
                0.33
                if self.continuous_action_threshold is None
                else float(self.continuous_action_threshold)
            )
            if thr == 0.0:
                if val > 0.0:
                    return 1
                if val < 0.0:
                    return 2
                if (
                    getattr(self, "continuous_action_contract", None)
                    == "target_exposure_hysteresis_v2"
                    and self._has_actionable_exposure()
                ):
                    return 3
                return 0
            if val >= thr:
                return 1  # long
            if val <= -thr:
                return 2  # short
            if (
                getattr(self, "continuous_action_contract", None)
                == "target_exposure_hysteresis_v2"
                and abs(val) <= float(self.continuous_exit_threshold)
                and self._has_actionable_exposure()
            ):
                return 3  # model-requested close to flat
            return 0  # hold
        try:
            a = int(action)
        except Exception:
            a = 0
        return a if a in (0, 1, 2) else 0

    def _has_actionable_exposure(self) -> bool:
        bridge = getattr(self, "bridge", None)
        if bridge is None:
            return False
        position = float(
            getattr(bridge, "position_units", None)
            or getattr(bridge, "position", 0.0)
            or 0.0
        )
        open_orders = int(getattr(bridge, "open_order_count", 0) or 0)
        return abs(position) > 1e-12 or open_orders > 0

    def _event_context_features(self, step_idx: int) -> Dict[str, float]:
        """Read engineered event-context controls for the current bar.

        The event overlay intentionally treats missing fields as neutral. This
        keeps legacy configs unchanged while allowing explicit event-overlay
        configs to alter execution when the engineered event columns exist.
        """
        idx = max(0, min(int(step_idx), len(self.dataframe) - 1))
        row = self.dataframe.iloc[idx]

        def read_float(column: str, default: float) -> float:
            if not column or column not in self.dataframe.columns:
                return float(default)
            try:
                val = row[column]
                if val is None:
                    return float(default)
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        no_trade_value = read_float(self.event_context_no_trade_column, 0.0)
        spread_mult = read_float(self.event_context_spread_stress_column, 1.0)
        slippage_mult = read_float(self.event_context_slippage_stress_column, 1.0)
        active = 1.0 if no_trade_value >= self.event_context_no_trade_threshold else 0.0
        return {
            "event_context_no_trade_value": no_trade_value,
            "event_context_no_trade_active": active,
            "event_context_spread_stress_multiplier": spread_mult,
            "event_context_slippage_stress_multiplier": slippage_mult,
        }

    def _apply_event_context_overlay(self, action: int) -> tuple[int, Dict[str, Any]]:
        if self.bridge is None:
            return int(action), {}
        step_idx = max(0, min(int(getattr(self.bridge, "bar_index", 0) or 0), self.total_bars))
        features = self._event_context_features(step_idx)
        active = bool(features["event_context_no_trade_active"] > 0.0)
        before = int(action)
        after = before
        position = int(getattr(self.bridge, "position", 0) or 0)
        blocked_entry = False
        forced_flat = False
        if self.event_context_execution_overlay and active:
            diag = getattr(self.bridge, "execution_diagnostics", {}) or {}
            diag["event_context_no_trade_active_steps"] = (
                diag.get("event_context_no_trade_active_steps", 0) + 1
            )
            self.bridge.execution_diagnostics = diag
            if self.event_context_force_flat and position != 0:
                after = 3
                forced_flat = True
            elif self.event_context_block_new_entries and position == 0 and before in (1, 2):
                after = 0
                blocked_entry = True
            if after != before:
                diag["event_context_action_overrides"] = (
                    diag.get("event_context_action_overrides", 0) + 1
                )
                if blocked_entry:
                    diag["event_context_blocked_entries"] = (
                        diag.get("event_context_blocked_entries", 0) + 1
                    )
                if forced_flat:
                    diag["event_context_forced_flat_actions"] = (
                        diag.get("event_context_forced_flat_actions", 0) + 1
                    )
                self.bridge.execution_diagnostics = diag

        return after, {
            **features,
            "event_context_execution_overlay": bool(self.event_context_execution_overlay),
            "event_context_action_before_overlay": before,
            "event_context_action_after_overlay": after,
            "event_context_action_overridden": bool(after != before),
            "event_context_blocked_entry": bool(blocked_entry),
            "event_context_forced_flat": bool(forced_flat),
            "event_context_position_before_overlay": position,
        }

    # ------------------------------------------------------------------
    # C5: weekly session exposure through the REAL env path
    # ------------------------------------------------------------------
    SESSION_OBSERVATION_NAMES = (
        "session_wind_down",
        "session_forced_flatten",
        "session_market_closed",
        "session_reopen_blackout",
        "session_hours_to_next_close",
        "session_hours_since_reopen",
    )

    def _session_exposure_init(self) -> None:
        from app.session_exposure import (
            SessionCalendar, validate_policy)

        self.session_exposure_enabled = bool(
            self.config.get("session_exposure_enabled", False))
        self._session_policy: Optional[Dict[str, Any]] = None
        self._session_calendar = None
        self._session_calendar_error: Optional[str] = None
        self.session_spread_column = None
        self._last_session_info: Dict[str, Any] = {}
        self._session_carried_migration: Optional[Dict[str, Any]] = None
        if not self.session_exposure_enabled:
            return

        # The policy is VALIDATED, never defaulted. A malformed policy
        # is a construction-time refusal: an env that cannot state its
        # own session contract must not run at all.
        self._session_policy = validate_policy(
            dict(self.config.get("session_exposure_policy") or {}))
        self.session_venue = str(self.config.get("session_venue", ""))
        self.session_account_fingerprint = str(
            self.config.get("session_account_fingerprint", ""))
        self.session_symbol = str(self.config.get("session_symbol", ""))
        self.session_spread_column = self.config.get(
            "session_spread_column")

        intervals_raw = self.config.get("session_calendar_intervals")
        if intervals_raw is None:
            # Fail CLOSED, and say so. Unlike the OANDA-calendar and
            # event-context helpers, missing session evidence is never
            # degraded to neutral zeros: session_state(calendar=None)
            # returns WIND_DOWN with evidence_failed_closed=True.
            self._session_calendar_error = (
                "no session_calendar_intervals configured")
        else:
            try:
                self._session_calendar = SessionCalendar.build(
                    venue=self.session_venue,
                    account_fingerprint=self.session_account_fingerprint,
                    symbol=self.session_symbol,
                    calendar_digest=str(
                        self._session_policy["calendar_identity"]),
                    intervals=[
                        (self._session_utc(item[0]),
                         self._session_utc(item[1]))
                        for item in intervals_raw])
            except Exception as exc:
                self._session_calendar = None
                self._session_calendar_error = (
                    f"{type(exc).__name__}: {exc}")

        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "session_wind_down": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "session_forced_flatten": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "session_market_closed": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "session_reopen_blackout": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "session_hours_to_next_close": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "session_hours_since_reopen": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

    @staticmethod
    def _session_utc(value):
        """Canonical UTC as a STDLIB datetime. session_exposure's
        require_utc returns value.astimezone(timezone.utc), and
        SessionCalendar.__post_init__ asserts object identity against
        that result — a pandas Timestamp always allocates a new object
        under astimezone and would be refused as non-canonical."""
        from datetime import timezone
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.to_pydatetime().astimezone(timezone.utc)

    def _session_now(self, step_idx: int):
        """UTC timestamp of the current bar. Naive stamps are read as
        UTC, matching oanda_calendar._to_ny."""
        try:
            raw = self._bar_timestamp(step_idx)
            if raw is None:
                return None
            ts = pd.Timestamp(raw)
        except Exception:
            return None
        if ts is None or pd.isna(ts):
            return None
        return self._session_utc(ts)

    def _session_order_inventory(self):
        """G4: typed pending-entry / protective split derived from
        ACTUAL order identities. No `or 0` coercion: an unavailable
        inventory is a typed refusal, because an unknown order book
        must never read as 'no pending entries'."""
        from app.session_exposure import SessionEvidenceError
        inventory = getattr(self.bridge, "open_order_inventory", None)
        if inventory is None:
            raise SessionEvidenceError(
                "open order inventory is unavailable — the pending "
                "entry / protective split cannot be derived and the "
                "overlay refuses rather than assuming zero entries")
        entries, protective = [], []
        for record in inventory:
            if not isinstance(record, dict) or "reduce_only" not in \
                    record:
                raise SessionEvidenceError(
                    f"malformed order record {record!r}")
            (protective if record["reduce_only"] else entries).append(
                record)
        return tuple(entries), tuple(protective)

    def _session_signed_exposure(self) -> float:
        """G4: signed exposure with NO coercive fallback. A bridge
        that cannot state its own position refuses."""
        from app.session_exposure import SessionEvidenceError
        units = getattr(self.bridge, "position_units", None)
        if isinstance(units, bool) or not isinstance(
                units, (int, float)):
            raise SessionEvidenceError(
                f"position_units is unavailable ({units!r}) — signed "
                "exposure refuses")
        units = float(units)
        if units != 0.0:
            return units
        # a flat units reading is only trustworthy if the discrete
        # position agrees; a disagreement is a typed contradiction
        position = getattr(self.bridge, "position", None)
        if isinstance(position, bool) or not isinstance(
                position, (int, float)):
            raise SessionEvidenceError(
                f"position is unavailable ({position!r})")
        if int(position) != 0:
            raise SessionEvidenceError(
                f"position_units is 0.0 but position is {position!r} "
                "— contradictory exposure facts refuse")
        return 0.0

    def _session_exposure_facts(self):
        from app.session_exposure import (
            ExposureFacts, SessionEvidenceError)
        entries, protective = self._session_order_inventory()
        pending_side = None
        pending_size = 0.0
        if entries:
            sides = {record["side"] for record in entries}
            if len(sides) == 1 and sides <= {"buy", "sell"}:
                pending_side = ("long" if entries[0]["side"] == "buy"
                                else "short")
                sizes = [record["size"] for record in entries]
                if any(size is None for size in sizes):
                    raise SessionEvidenceError(
                        "a pending entry order has no reported size; "
                        "an unknown size is refused, never read as 0")
                pending_size = float(sum(sizes))
                if pending_size <= 0.0:
                    raise SessionEvidenceError(
                        f"pending entry size {pending_size} is not "
                        "positive")
        return ExposureFacts.build(
            signed_exposure=self._session_signed_exposure(),
            pending_orders=len(entries) + len(protective),
            protective_orders=len(protective),
            pending_entry_side=pending_side,
            pending_entry_size=pending_size,
            action_mapping="discrete_command_v1")

    # -- G2: causal reopen evidence ---------------------------------
    def _session_stability_check(self, idx: int) -> dict:
        """One PAST-ONLY stability observation for bar ``idx``.

        Every input is computed from bars strictly BEFORE idx plus the
        bar itself; no future bar can influence it. A missing input is
        reported unavailable and FAILS the check, never passes it."""
        policy = self._session_policy
        frame = self.dataframe
        reasons = []
        baseline_n = policy["reopen_baseline_bars"]
        gap_n = policy["reopen_gap_sigma_bars"]
        vol_n = policy["reopen_realized_vol_bars"]
        need = max(baseline_n, gap_n, vol_n) + 1
        if idx < need:
            return {"passed": False, "reasons": ["insufficient_history"],
                    "spread_ratio": None, "gap_sigma": None,
                    "vol_ratio": None, "quote_continuous": None}

        closes = frame[self.price_column].to_numpy(dtype=float)

        # spread relative to a PAST-ONLY baseline
        spread_ratio = None
        column = self.session_spread_column
        if column and column in frame.columns:
            spreads = frame[column].to_numpy(dtype=float)
            baseline = spreads[idx - baseline_n:idx]
            current = float(spreads[idx])
            mean = float(np.mean(baseline)) if len(baseline) else 0.0
            if not np.isfinite(current) or not np.isfinite(mean) \
                    or mean <= 0.0:
                reasons.append("spread_unavailable")
            else:
                spread_ratio = current / mean
                if spread_ratio > policy[
                        "max_spread_relative_to_baseline"]:
                    reasons.append("spread_above_baseline")
        else:
            # no spread evidence is NOT a pass
            reasons.append("spread_unavailable")

        # opening gap in sigmas of past-only returns
        past = closes[idx - gap_n:idx]
        returns = np.diff(past) / past[:-1]
        sigma = float(np.std(returns)) if len(returns) > 1 else 0.0
        gap_sigma = None
        if sigma <= 0.0 or not np.isfinite(sigma):
            reasons.append("gap_sigma_unavailable")
        else:
            gap = (closes[idx] - closes[idx - 1]) / closes[idx - 1]
            gap_sigma = abs(float(gap)) / sigma
            if gap_sigma > policy["max_gap_sigma"]:
                reasons.append("gap_above_sigma")

        # realized volatility relative to the same past-only baseline
        recent = closes[idx - vol_n:idx + 1]
        recent_ret = np.diff(recent) / recent[:-1]
        recent_vol = float(np.std(recent_ret)) if len(recent_ret) > 1 \
            else 0.0
        vol_ratio = None
        if sigma <= 0.0 or not np.isfinite(recent_vol):
            reasons.append("realized_vol_unavailable")
        else:
            vol_ratio = recent_vol / sigma
            if vol_ratio > policy[
                    "max_realized_vol_relative_to_baseline"]:
                reasons.append("realized_vol_above_baseline")

        # Quote continuity. The expected spacing is derived from the
        # PAST-ONLY baseline bars themselves, never from an optional
        # `timeframe` config label: a helper that silently reports
        # "unavailable" when a label is absent is exactly the F-A
        # failure mode. A DECLARED timeframe that contradicts the data
        # is a typed contradiction, not a tie broken in its favour.
        continuous = None
        stamps = [self._session_now(j)
                  for j in range(idx - baseline_n, idx + 1)]
        if any(stamp is None for stamp in stamps):
            reasons.append("quote_continuity_unavailable")
        else:
            deltas = [(b - a).total_seconds() / 3600.0
                      for a, b in zip(stamps, stamps[1:])]
            expected = float(np.median(deltas[:-1]))
            declared = float(self._timeframe_hours or 0.0)
            if expected <= 0.0:
                reasons.append("quote_continuity_unavailable")
            elif declared > 0.0 and abs(declared - expected) > 1e-6:
                reasons.append("timeframe_contradicts_data")
            else:
                continuous = abs(deltas[-1] - expected) <= 1e-6
                if not continuous:
                    reasons.append("quote_discontinuity")

        return {"passed": not reasons, "reasons": reasons,
                "spread_ratio": spread_ratio, "gap_sigma": gap_sigma,
                "vol_ratio": vol_ratio, "quote_continuous": continuous}

    def _session_reopen_evidence(self, step_idx: int, now):
        """G2: materialize ReopenEvidence from causal observations.

        Counts FULLY CLOSED bars since the bound reopen instant and
        CONSECUTIVE passing stability checks — a single failing check
        resets the streak to zero, so a blackout can only be exited by
        an uninterrupted run of stable bars."""
        from app.session_exposure import ReopenEvidence
        calendar = self._session_calendar
        if calendar is None or now is None:
            return None, {}
        reopen_at = calendar.most_recent_reopen(now)
        if reopen_at is None:
            return None, {}
        closed_bars = 0
        streak = 0
        detail = []
        for idx in range(step_idx + 1):
            stamp = self._session_now(idx)
            if stamp is None or stamp <= reopen_at:
                continue
            # a bar is FULLY CLOSED only if the next bar's stamp has
            # been reached; the bar under decision is still forming
            if idx >= step_idx:
                continue
            closed_bars += 1
            check = self._session_stability_check(idx)
            streak = streak + 1 if check["passed"] else 0
            detail.append({"bar": idx, **check})
        evidence = ReopenEvidence.build(
            closed_bars_since_reopen=closed_bars,
            stability_checks_passed=streak,
            hint_time_since_reopen_hours=None)
        return evidence, {
            "session_reopen_closed_bars": closed_bars,
            "session_reopen_stability_streak": streak,
            "session_reopen_last_check": detail[-1] if detail else None,
        }

    def _apply_session_exposure_overlay(
            self, action: int, raw_model_output) -> tuple[
                int, Dict[str, Any]]:
        """G1: the ORIGINAL model output, the MAPPED discrete command
        and the CURRENT signed exposure are three separate values.
        Risk is classified from the mapped command by a discrete
        adapter -- command ids are never fed to a target-value
        classifier -- and a masked entry, enlargement or reversal
        submits HOLD, never the command it claims to have masked."""
        if not self.session_exposure_enabled or self.bridge is None:
            return int(action), {}
        from app.session_exposure import (
            HOLD_COMMAND, CLOSE_COMMAND, SessionDataContradictionError,
            classify_discrete_command, overlay_action,
            reconciliation_gate, session_state)

        step_idx = max(0, min(
            int(getattr(self.bridge, "bar_index", 0) or 0),
            self.total_bars))
        now = self._session_now(step_idx)
        reopen_evidence, reopen_info = self._session_reopen_evidence(
            step_idx, now)
        if now is None:
            state_block = {"state": "WIND_DOWN", "policy_enabled": True,
                           "evidence_ok": False,
                           "evidence_failed_closed": True,
                           "time_to_next_close_hours": None,
                           "time_since_reopen_hours": None,
                           "wind_down": True, "forced_flatten": False}
        else:
            state_block = session_state(
                self._session_policy, now=now,
                calendar=self._session_calendar,
                reopen_evidence=reopen_evidence,
                expected_venue=self.session_venue or None,
                expected_account_fingerprint=(
                    self.session_account_fingerprint or None),
                expected_symbol=self.session_symbol or None)

        state = state_block["state"]
        if state == "EXPECTED_MARKET_CLOSED":
            # G3: a tradable bar inside a declared closure is a
            # contradiction between the data and the bound calendar.
            # Real historical data has a TIMESTAMP GAP there and the
            # simulator performs no step at all. Zeroing the reward on
            # a fabricated bar would conceal whatever economic change
            # that bar carried, so this refuses instead.
            raise SessionDataContradictionError(
                f"bar {step_idx} at {now} falls inside a declared "
                "session closure: the simulator must perform no step "
                "during a closure, and a fabricated tradable bar "
                "inside one is prohibited")

        exposure = self._session_exposure_facts()
        command = int(action)
        classification = classify_discrete_command(command, exposure)
        # the state-driven policy stays the single authority; only
        # the CLASSIFICATION crosses the domain boundary
        decision = overlay_action(
            self._session_policy, state_block, exposure,
            float(command), classification=classification)

        overlay = decision["overlay"]
        if overlay == "forced_close":
            after = CLOSE_COMMAND
        elif overlay in ("masked_risk_increase",
                         "masked_entry_during_blackout"):
            # HOLD is the safe command in this domain: it preserves
            # the current position and adds no risk. Submitting the
            # original command would execute the masked entry or
            # reversal.
            after = HOLD_COMMAND
        else:
            after = command
        decision["final_action"] = after

        gate = None
        if overlay == "forced_close":
            # G4: the forced flatten is only ACCEPTED once the shared
            # typed gate proves fresh zero positions and zero orders.
            # Until then it is an in-flight attempt, not a success.
            entries, protective = self._session_order_inventory()
            try:
                gate = reconciliation_gate(
                    positions_total=(
                        0 if not exposure.has_position else 1),
                    orders_total=len(entries) + len(protective),
                    evidence_age_seconds=0.0,
                    max_age_seconds=120.0)
            except Exception as exc:
                # a gate that cannot be evaluated is a typed INCIDENT,
                # never a silent success
                gate = {"flat_confirmed": False,
                        "incident": f"{type(exc).__name__}: {exc}"}

        if after != command:
            diag = getattr(self.bridge, "execution_diagnostics", {}) or {}
            diag["session_overlay_overrides"] = (
                diag.get("session_overlay_overrides", 0) + 1)
            key = f"session_{overlay}_actions"
            diag[key] = diag.get(key, 0) + 1
            self.bridge.execution_diagnostics = diag

        info = {
            "session_decision_bar_index": step_idx,
            "session_state": state,
            "session_wind_down": bool(state_block.get("wind_down")),
            "session_forced_flatten": bool(
                state_block.get("forced_flatten")),
            "session_evidence_ok": bool(state_block.get("evidence_ok")),
            "session_evidence_failed_closed": bool(
                state_block.get("evidence_failed_closed", False)),
            "session_calendar_error": self._session_calendar_error,
            "session_time_to_next_close_hours": state_block.get(
                "time_to_next_close_hours"),
            "session_time_since_reopen_hours": state_block.get(
                "time_since_reopen_hours"),
            # G1: three separate values, none derived from another
            "session_raw_model_output": raw_model_output,
            "session_mapped_command": command,
            "session_mapped_action": dict(classification),
            "session_signed_exposure": exposure.signed_exposure,
            "session_overlay": overlay,
            "session_final_action": after,
            "session_action_before_overlay": command,
            "session_action_after_overlay": after,
            "session_cancel_pending": bool(decision["cancel_pending"]),
            "session_cancel_scope": decision.get("cancel_scope"),
            "session_no_actionable_step": False,
            "session_entry_orders": exposure.entry_orders,
            "session_protective_orders": exposure.protective_orders,
            "session_pending_entry_side": exposure.pending_entry_side,
            "session_flatten_reconciliation": gate,
            **reopen_info,
        }
        self._last_session_info = info
        return after, info

    def _session_termination_record(self) -> Dict[str, Any]:
        """Episode termination is an EPISODE boundary, not a venue
        event. If exposure is open when the episode ends, it survives
        and is reported as a carried position requiring migration --
        never silently zeroed by reset()."""
        signed = float(getattr(self.bridge, "position_units", 0.0) or 0.0)
        if signed == 0.0:
            signed = float(getattr(self.bridge, "position", 0) or 0)
        if signed == 0.0:
            return {"session_exposure_survived_termination": False,
                    "session_carried_exposure": 0.0}
        record = {
            "session_exposure_survived_termination": True,
            "session_carried_exposure": signed,
            "session_carried_position_requires_migration": True,
            "session_carried_episode_seq": int(
                getattr(self.bridge, "episode_seq", 0) or 0),
            "session_carried_bar_index": int(
                getattr(self.bridge, "bar_index", 0) or 0),
            "termination_does_not_close_exposure": True,
        }
        self._session_carried_migration = record
        return record

    def _session_observation(self) -> Dict[str, np.ndarray]:
        info = self._last_session_info or {}

        def _hours(key):
            value = info.get(key)
            return 0.0 if value is None else max(0.0, float(value))

        state = info.get("session_state")
        return {
            "session_wind_down": np.array(
                [1.0 if info.get("session_wind_down") else 0.0],
                dtype=np.float32),
            "session_forced_flatten": np.array(
                [1.0 if info.get("session_forced_flatten") else 0.0],
                dtype=np.float32),
            "session_market_closed": np.array(
                [1.0 if state == "EXPECTED_MARKET_CLOSED" else 0.0],
                dtype=np.float32),
            "session_reopen_blackout": np.array(
                [1.0 if state == "REOPEN_BLACKOUT" else 0.0],
                dtype=np.float32),
            "session_hours_to_next_close": np.array(
                [_hours("session_time_to_next_close_hours")],
                dtype=np.float32),
            "session_hours_since_reopen": np.array(
                [_hours("session_time_since_reopen_hours")],
                dtype=np.float32),
        }

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
            "position_units": self.bridge.position_units,
            "equity": self.bridge.equity,
            "initial_cash": self.initial_cash,
            "price": self.bridge.price,
            "entry_price": self.bridge.entry_price,
            "holding_bars": self.bridge.holding_bars,
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
        if getattr(self, "session_exposure_enabled", False):
            obs = dict(obs)
            obs.update(self._session_observation())
        if getattr(self, "execution_cost_observation_enabled", False):
            obs = dict(obs)
            for name, value in zip(
                EXECUTION_COST_OBSERVATION_NAMES,
                self._execution_cost_observation,
            ):
                obs[name] = np.array([value], dtype=np.float32)
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
        # F-A: the default data feed promotes the date column to the
        # INDEX. Reading only .columns made this helper take its
        # ts=None branch under that feed, so ALL ELEVEN fields were
        # constant zeros and every run that observed them trained on
        # no signal at all. Both layouts now resolve to the SAME
        # timestamp, so equivalent inputs give bit-identical features.
        ts = self._bar_timestamp(step_idx)
        tf_h = float(self._timeframe_hours or 1.0) or 1.0
        return compute_fx_calendar_features(ts, timeframe_hours=tf_h)

    def _bar_timestamp(self, step_idx: int):
        """The raw timestamp of a bar, from the date COLUMN or, when
        the feed has promoted it, from the DatetimeIndex. Returns None
        only when neither carries one."""
        if len(self.dataframe) == 0:
            return None
        idx = max(0, min(int(step_idx), len(self.dataframe) - 1))
        if self._date_column in self.dataframe.columns:
            return self.dataframe.iloc[idx][self._date_column]
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            return self.dataframe.index[idx]
        return None

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

    def flatten_step(self, max_bars: int = 5):
        """Close any open exposure through the SAME execution path the
        policy uses (AUD-F1-20260806-152).

        Action 3 is the simulator's explicit risk-reducing close: it
        cancels resting orders and closes the position with the real
        configured commission/slippage. Returns the post-close info so
        the caller can PROVE flatness instead of asserting it.
        """
        if self.bridge is None:
            raise RuntimeError("Call reset() before flatten_step().")
        if self.bridge.terminated:
            return self._make_info()
        # Bounded liquidation: submit the cancel+close, then advance
        # until the simulator reports flat (an order submitted on bar N
        # fills on bar N+1) or the bounded attempts are exhausted. The
        # caller PROVES flatness from the returned facts.
        self.bridge.force_flat_request = True
        try:
            for _attempt in range(int(max_bars)):
                self.bridge.action_slot = 3
                self.bridge.raw_action_slot = 0.0
                self.bridge.obs_ready.clear()
                self.bridge.action_ready.set()
                self._wait_obs()
                if self.bridge.terminated:
                    break
                flat = (
                    abs(float(getattr(self.bridge, "position_units", 0)
                              or 0.0)) <= 1e-12
                    and int(getattr(self.bridge, "open_order_count", 0)
                            or 0) == 0)
                if flat:
                    break
        finally:
            self.bridge.force_flat_request = False
        return self._make_info()

    def _make_info(self) -> Dict[str, Any]:
        assert self.bridge is not None
        info = {
            "equity": self.bridge.equity,
            "position": self.bridge.position,
            "position_units": getattr(self.bridge, "position_units", None),
            "open_order_count": getattr(
                self.bridge, "open_order_count", None),
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
        info.update(dict(getattr(self, "_last_event_context_info", {}) or {}))
        if getattr(self, "session_exposure_enabled", False):
            info.update(dict(getattr(self, "_last_session_info", {}) or {}))
            info["session_policy_enabled"] = True
        if getattr(self, "execution_cost_observation_enabled", False):
            info["execution_cost_context"] = dict(self._execution_cost_context)
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
        # Steps-1-2 correction order 2026-08-28 (finding 1): EVERY
        # authoritative trade statistic derives from the bridge's ONE
        # economically complete closed-trade stream. Backtrader
        # analyzer results — the population proven blind to direct
        # settlements — move wholesale under the explicit
        # analyzer_*_diagnostic namespace and are never mixed with
        # authoritative totals again.
        stream = list(getattr(self.bridge, "closed_trade_stream", []))
        for legacy_key in ("trades_total", "trades_won",
                           "trades_lost", "avg_trade_pnl"):
            summary[f"analyzer_{legacy_key}_diagnostic"] =                 summary.pop(legacy_key, None)
        # final hardening 2026-08-28: events are STRICTLY validated
        # at append time, so fields are accessed directly — a missing
        # field is a loud bug, never a silent breakeven/zero
        won = sum(1 for e in stream if e["net_pnl"] > 0.0)
        lost = sum(1 for e in stream if e["net_pnl"] < 0.0)
        breakeven = sum(1 for e in stream if e["net_pnl"] == 0.0)
        by_source: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for event in stream:
            src = str(event.get("source"))
            by_source[src] = by_source.get(src, 0) + 1
            reason = str(event.get("reason"))
            by_reason[reason] = by_reason.get(reason, 0) + 1
        # conservation identities are ASSERTED, not hoped (finding 1)
        if won + lost + breakeven != len(stream):
            raise RuntimeError(
                "trade-stat conservation violated: "
                f"{won}+{lost}+{breakeven} != {len(stream)}")
        if sum(by_source.values()) != len(stream) or                 sum(by_reason.values()) != len(stream):
            raise RuntimeError(
                "trade source/reason counts do not sum to the total")
        net_values = [e["net_pnl"] for e in stream]
        summary["trade_stats_authority"] = "closed_trade_stream_v2"
        summary["trades_total"] = len(stream)
        summary["trades_won"] = won
        summary["trades_lost"] = lost
        summary["trades_breakeven"] = breakeven
        summary["avg_trade_pnl"] = (
            sum(net_values) / len(net_values) if net_values else None)
        summary["trade_costs_total"] = sum(
            e["costs"] for e in stream)
        summary["closed_trades_by_source"] = by_source
        summary["close_reason_counts"] = by_reason
        summary["open_position_at_end"] = bool(
            getattr(self.bridge, "position", 0))
        summary["duplicate_close_events_ignored"] = (
            self.bridge.execution_diagnostics.get(
                "duplicate_close_events_ignored", 0)
            if self.bridge else 0)
        summary["action_diagnostics"] = dict(self._action_diagnostics)
        summary["execution_diagnostics"] = dict(getattr(self.bridge, "execution_diagnostics", {}) or {})
        summary["event_context_diagnostics"] = dict(getattr(self, "_last_event_context_info", {}) or {})
        if getattr(self, "execution_cost_observation_enabled", False):
            summary["execution_cost_context"] = dict(self._execution_cost_context)
        return summary

    def _reset_action_diagnostics(self) -> None:
        self._last_raw_action_value = 0.0
        self._last_coerced_action = 0
        self._last_event_context_info = {}
        self._action_diagnostics = {
            "steps": 0,
            "hold_actions": 0,
            "long_actions": 0,
            "short_actions": 0,
            "explicit_close_actions": 0,
            "non_hold_actions": 0,
            "continuous_deadband_actions": 0,
            "raw_abs_sum": 0.0,
            "raw_min": None,
            "raw_max": None,
            "continuous_action_threshold": self.continuous_action_threshold,
            "continuous_exit_threshold": self.continuous_exit_threshold,
            "continuous_action_contract": self.continuous_action_contract,
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
        elif coerced_action == 3:
            diag["explicit_close_actions"] += 1
            diag["non_hold_actions"] += 1
        else:
            diag["hold_actions"] += 1
            if self.action_space_mode == "continuous":
                diag["continuous_deadband_actions"] += 1
