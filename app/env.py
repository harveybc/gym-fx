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
        self.action_space = spaces.Discrete(3)
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

        # --- runtime handles -------------------------------------------------
        self.bridge: Optional[BTBridge] = None
        self._runner: Optional[threading.Thread] = None
        self._cerebro = None
        self._strategy_instance = None
        self._np_random = np.random.default_rng()

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

        bt_feed = self.data_feed_plugin.build_bt_feed(self.dataframe, self.config)
        broker = self.broker_plugin.build_bt_broker(self.config)

        self._cerebro = build_cerebro(
            bt_feed=bt_feed,
            broker=broker,
            bridge=self.bridge,
            position_size=self.position_size,
            min_equity=self.min_equity,
        )

        self._runner = threading.Thread(target=self._run_cerebro, name="gym-fx-cerebro", daemon=True)
        self._runner.start()
        self._wait_obs()

        return self._make_observation(), self._make_info()

    def step(self, action):
        if self.bridge is None:
            raise RuntimeError("Call reset() before step().")

        try:
            a = int(action)
        except Exception:
            a = 0
        if a not in (0, 1, 2):
            a = 0

        if self.bridge.terminated:
            return self._make_observation(), 0.0, True, False, self._make_info()

        self.bridge.action_slot = a
        self.bridge.obs_ready.clear()
        self.bridge.action_ready.set()
        self._wait_obs()

        prev_equity = self.bridge.prev_equity
        new_equity = self.bridge.equity

        reward = float(
            self.reward_plugin.compute_reward(
                prev_equity=prev_equity,
                new_equity=new_equity,
                step=self.bridge.bar_index,
                config=self.config,
            )
        )

        terminated = bool(self.bridge.terminated or new_equity <= self.min_equity)
        truncated = False

        obs = self._make_observation()
        info = self._make_info()
        info.update(
            reward=reward,
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
    def _run_cerebro(self) -> None:
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
        return self.preprocessor_plugin.make_observation(
            data=self.dataframe,
            step=step_idx,
            bridge_state=bridge_state,
            config=self.config,
        )

    def _make_info(self) -> Dict[str, Any]:
        assert self.bridge is not None
        return {
            "equity": self.bridge.equity,
            "position": self.bridge.position,
            "price": self.bridge.price,
            "bar_index": self.bridge.bar_index,
            "total_bars": self.total_bars,
            "trades": self.bridge.trade_count,
            "commission_paid": self.bridge.commission_paid,
        }

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
        return self.metrics_plugin.summarize(
            initial_cash=self.initial_cash,
            final_equity=self.bridge.equity if self.bridge else self.initial_cash,
            analyzers=analyzers,
            config=self.config,
        )
