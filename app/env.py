from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class EnvState:
    step: int
    position: int
    equity: float
    price: float


class GymFxEnv:
    """Agent-agnostic trading env used by agent-multi.

    Action space convention:
      0 = hold, 1 = long, 2 = short
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_feed_plugin,
        broker_plugin,
        strategy_plugin,
        preprocessor_plugin,
        reward_plugin,
        metrics_plugin,
    ):
        self.config = config
        self.data_feed_plugin = data_feed_plugin
        self.broker_plugin = broker_plugin
        self.strategy_plugin = strategy_plugin
        self.preprocessor_plugin = preprocessor_plugin
        self.reward_plugin = reward_plugin
        self.metrics_plugin = metrics_plugin

        self.data = self.data_feed_plugin.load_data(config)
        if self.data is None or len(self.data) < 2:
            raise ValueError("input data is empty or too short")

        self.price_column = config.get("price_column", "CLOSE")
        if self.price_column not in self.data.columns:
            raise ValueError(f"price_column '{self.price_column}' not found in data")

        self.window_size = int(config.get("window_size", 32))
        self.initial_cash = float(config.get("initial_cash", 10000.0))
        self.position_size = float(config.get("position_size", 1.0))

        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._equity_curve: List[float] = []
        self._prices = self.data[self.price_column].astype(float).to_numpy()

        self.state = EnvState(step=self.window_size, position=0, equity=self.initial_cash, price=self._prices[self.window_size])

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.state = EnvState(step=self.window_size, position=0, equity=self.initial_cash, price=self._prices[self.window_size])
        self._actions = []
        self._rewards = []
        self._equity_curve = [self.initial_cash]
        obs = self.preprocessor_plugin.make_observation(self.data, self.state.step, self.config)
        info = {"equity": self.state.equity, "position": self.state.position}
        return obs, info

    def _normalize_action(self, action: Any) -> int:
        try:
            a = int(action)
        except Exception:
            a = 0
        if a not in (0, 1, 2):
            return 0
        return a

    def step(self, action: Any):
        action = self._normalize_action(action)
        if self.state.step >= len(self._prices) - 1:
            return self._terminal_response()

        prev_price = float(self._prices[self.state.step - 1])
        curr_price = float(self._prices[self.state.step])
        self.state.price = curr_price

        old_position = self.state.position
        new_position = {0: old_position, 1: 1, 2: -1}[action]
        trade_cost = self.broker_plugin.trade_cost(old_position, new_position, curr_price, self.config)

        pnl = old_position * (curr_price - prev_price) * self.position_size
        new_equity = self.state.equity + pnl - trade_cost

        reward = self.reward_plugin.compute_reward(
            prev_equity=self.state.equity,
            new_equity=new_equity,
            step=self.state.step,
            config=self.config,
        )

        self.state = EnvState(
            step=self.state.step + 1,
            position=new_position,
            equity=new_equity,
            price=curr_price,
        )
        self._actions.append(action)
        self._rewards.append(float(reward))
        self._equity_curve.append(float(new_equity))

        terminated = self.state.step >= len(self._prices) - 1
        truncated = False
        obs = self.preprocessor_plugin.make_observation(self.data, self.state.step, self.config)
        info = {
            "equity": self.state.equity,
            "position": self.state.position,
            "price": self.state.price,
            "pnl": pnl,
            "trade_cost": trade_cost,
        }
        return obs, float(reward), terminated, truncated, info

    def _terminal_response(self):
        obs = self.preprocessor_plugin.make_observation(self.data, self.state.step, self.config)
        info = {"equity": self.state.equity, "position": self.state.position}
        return obs, 0.0, True, False, info

    def summary(self) -> Dict[str, Any]:
        return self.metrics_plugin.summarize(
            equity_curve=self._equity_curve,
            rewards=self._rewards,
            actions=self._actions,
            config=self.config,
        )
