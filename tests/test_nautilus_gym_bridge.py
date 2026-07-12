from pathlib import Path

import pytest

pytest.importorskip("nautilus_trader")

from broker_plugins.default_broker import Plugin as BrokerPlugin
from data_feed_plugins.default_data_feed import Plugin as DataPlugin
from gym_fx import build_environment
from metrics_plugins.default_metrics import Plugin as MetricsPlugin
from preprocessor_plugins.default_preprocessor import Plugin as PreprocessorPlugin
from reward_plugins.pnl_reward import Plugin as RewardPlugin
from strategy_plugins.default_strategy import Plugin as StrategyPlugin


def test_nautilus_bridge_preserves_gym_step_contract():
    root = Path(__file__).resolve().parents[1]
    config = {
        "simulation_engine": "nautilus",
        "execution_cost_profile": str(
            root
            / "examples/config/execution_cost_profiles/project3_pessimistic_v1.json"
        ),
        "financing_rate_data_file": str(
            root / "examples/data/fx_rollover_rates_smoke.csv"
        ),
        "input_data_file": str(root / "examples/data/eurusd_sample.csv"),
        "date_column": "DATE_TIME",
        "price_column": "CLOSE",
        "instrument": "EUR_USD",
        "timeframe": "M1",
        "window_size": 4,
        "initial_cash": 10000.0,
        "position_size": 1000.0,
        "min_quantity": 1,
        "lot_size": 1,
    }
    plugins = {
        "data_feed_plugin": DataPlugin(config),
        "broker_plugin": BrokerPlugin(config),
        "strategy_plugin": StrategyPlugin(config),
        "preprocessor_plugin": PreprocessorPlugin(config),
        "reward_plugin": RewardPlugin(config),
        "metrics_plugin": MetricsPlugin(config),
    }
    env = build_environment(config=config, **plugins)
    try:
        observation, info = env.reset(seed=7)
        assert "prices" in observation
        assert info["position"] == 0
        observation, reward, terminated, truncated, info = env.step(1)
        assert isinstance(reward, float)
        assert truncated is False
        assert info["position"] == 1
        assert not terminated
    finally:
        env.close()
