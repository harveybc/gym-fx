#!/usr/bin/env python3
"""
Smoke-test driver for gym-fx env.
Runs the environment with the buy_hold and random drivers using the sample data.
No install required — runs via Python path manipulation.

Usage:
  python tools/smoke_test.py          # buy_hold + random (default)
  python tools/smoke_test.py random   # random only
  python tools/smoke_test.py buy_hold # buy_hold only
"""
import sys
import json
import os
from pathlib import Path

# Make sure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import DEFAULT_VALUES
from app.config_merger import merge_config
from app.env import GymFxEnv
from data_feed_plugins.default_data_feed import Plugin as DataFeedPlugin
from broker_plugins.default_broker import Plugin as BrokerPlugin
from strategy_plugins.default_strategy import Plugin as StrategyPlugin
from preprocessor_plugins.default_preprocessor import Plugin as PreprocessorPlugin
from reward_plugins.pnl_reward import Plugin as RewardPlugin
from metrics_plugins.default_metrics import Plugin as MetricsPlugin


def run_driver(driver_mode: str) -> dict:
    base_config = {
        **DEFAULT_VALUES,
        "mode": "inference",
        "driver_mode": driver_mode,
        "steps": 490,
        "input_data_file": str(REPO_ROOT / "examples" / "data" / "eurusd_sample.csv"),
        "date_column": "DATE_TIME",
        "price_column": "CLOSE",
        "headers": True,
        "window_size": 32,
        "initial_cash": 10000.0,
        "position_size": 1.0,
        "commission": 0.0,
        "slippage": 0.0,
    }

    data_feed = DataFeedPlugin(base_config)
    broker = BrokerPlugin(base_config)
    strategy = StrategyPlugin(base_config)
    preprocessor = PreprocessorPlugin(base_config)
    reward = RewardPlugin(base_config)
    metrics = MetricsPlugin(base_config)

    env = GymFxEnv(
        config=base_config,
        data_feed_plugin=data_feed,
        broker_plugin=broker,
        strategy_plugin=strategy,
        preprocessor_plugin=preprocessor,
        reward_plugin=reward,
        metrics_plugin=metrics,
    )

    obs, info = env.reset()
    done = False
    step_count = 0
    steps = base_config["steps"]
    while not done and step_count < steps:
        action = strategy.decide_action(obs=obs, info=info, step=step_count)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_count += 1

    return env.summary()


def main():
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["buy_hold", "random"]
    results_dir = REPO_ROOT / "examples" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  Running driver: {mode}")
        print(f"{'='*50}")
        summary = run_driver(mode)
        out_path = results_dir / f"{mode}_summary.json"
        with out_path.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"  Results saved → {out_path}")

    print("\n[smoke_test] All drivers completed successfully.")


if __name__ == "__main__":
    main()
