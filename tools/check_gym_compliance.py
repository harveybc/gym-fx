#!/usr/bin/env python3
"""
Gymnasium API compliance check for gym-fx env.

Runs gymnasium.utils.env_checker.check_env against a freshly constructed
env backed by the default plugins and the sample data file.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gymnasium.utils.env_checker import check_env

from app.config import DEFAULT_VALUES
from app.env import GymFxEnv
from data_feed_plugins.default_data_feed import Plugin as DataFeedPlugin
from broker_plugins.default_broker import Plugin as BrokerPlugin
from strategy_plugins.default_strategy import Plugin as StrategyPlugin
from preprocessor_plugins.default_preprocessor import Plugin as PreprocessorPlugin
from reward_plugins.pnl_reward import Plugin as RewardPlugin
from metrics_plugins.default_metrics import Plugin as MetricsPlugin


def build_env():
    config = {
        **DEFAULT_VALUES,
        "mode": "inference",
        "driver_mode": "flat",
        "steps": 100,
        "input_data_file": str(REPO_ROOT / "examples" / "data" / "eurusd_sample.csv"),
        "window_size": 32,
        "initial_cash": 10000.0,
    }
    return GymFxEnv(
        config=config,
        data_feed_plugin=DataFeedPlugin(config),
        broker_plugin=BrokerPlugin(config),
        strategy_plugin=StrategyPlugin(config),
        preprocessor_plugin=PreprocessorPlugin(config),
        reward_plugin=RewardPlugin(config),
        metrics_plugin=MetricsPlugin(config),
    )


def main() -> int:
    env = build_env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    print("[check_gym_compliance] env passes gymnasium.env_checker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
