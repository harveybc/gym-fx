#!/usr/bin/env python3
"""
Smoke test for the backtrader-backed gym-fx env.

Exits non-zero on assertion failure. Checks performed:

  1. `flat` driver leaves equity unchanged (no trades executed).
  2. `buy_hold` driver produces a positive total_return on an uptrending feed.
  3. Seeded resets yield reproducible first observations.
  4. `total_return` reported by metrics matches (final_equity - initial)/initial.

Each driver summary is written to examples/results/<mode>_summary.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config import DEFAULT_VALUES
from app.env import GymFxEnv
from data_feed_plugins.default_data_feed import Plugin as DataFeedPlugin
from broker_plugins.default_broker import Plugin as BrokerPlugin
from strategy_plugins.default_strategy import Plugin as StrategyPlugin
from preprocessor_plugins.default_preprocessor import Plugin as PreprocessorPlugin
from reward_plugins.pnl_reward import Plugin as RewardPlugin
from metrics_plugins.default_metrics import Plugin as MetricsPlugin


DATA_FILE = REPO_ROOT / "examples" / "data" / "eurusd_sample.csv"


def _base_config(driver_mode: str, steps: int = 480) -> dict:
    return {
        **DEFAULT_VALUES,
        "mode": "inference",
        "driver_mode": driver_mode,
        "steps": steps,
        "input_data_file": str(DATA_FILE),
        "date_column": "DATE_TIME",
        "price_column": "CLOSE",
        "headers": True,
        "window_size": 32,
        "initial_cash": 10000.0,
        "position_size": 1.0,
        "commission": 0.0,
        "slippage": 0.0,
    }


def _build_env(config: dict):
    return GymFxEnv(
        config=config,
        data_feed_plugin=DataFeedPlugin(config),
        broker_plugin=BrokerPlugin(config),
        strategy_plugin=StrategyPlugin(config),
        preprocessor_plugin=PreprocessorPlugin(config),
        reward_plugin=RewardPlugin(config),
        metrics_plugin=MetricsPlugin(config),
    )


def run_driver(driver_mode: str, seed: int | None = None) -> tuple[dict, dict]:
    config = _base_config(driver_mode)
    env = _build_env(config)
    strategy = env.strategy_plugin
    obs, info = env.reset(seed=seed)
    first_obs = {k: np.array(v, copy=True) for k, v in obs.items()}
    steps = config["steps"]
    step = 0
    done = False
    while not done and step < steps:
        action = strategy.decide_action(obs=obs, info=info, step=step)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    summary = env.summary()
    env.close()
    return summary, first_obs


def _write_uptrend_feed() -> Path:
    path = REPO_ROOT / "examples" / "data" / "eurusd_uptrend.csv"
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    closes = np.linspace(1.10, 1.20, n)
    df = pd.DataFrame(
        {
            "DATE_TIME": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "OPEN": closes,
            "HIGH": closes + 1e-5,
            "LOW": closes - 1e-5,
            "CLOSE": closes,
            "VOLUME": 0,
        }
    )
    df.to_csv(path, index=False)
    return path


def main() -> int:
    results_dir = REPO_ROOT / "examples" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- flat driver: equity unchanged --------------------------------------
    flat_summary, _ = run_driver("flat")
    (results_dir / "flat_summary.json").write_text(json.dumps(flat_summary, indent=2))
    assert math.isclose(
        flat_summary["final_equity"], flat_summary["initial_cash"], rel_tol=1e-9, abs_tol=1e-3
    ), f"flat driver changed equity: {flat_summary}"
    assert math.isclose(flat_summary["total_return"], 0.0, abs_tol=1e-6)

    # --- buy_hold on uptrend: total_return > 0 ------------------------------
    uptrend_path = _write_uptrend_feed()
    up_config = _base_config("buy_hold")
    up_config["input_data_file"] = str(uptrend_path)
    env = _build_env(up_config)
    strategy = env.strategy_plugin
    obs, info = env.reset(seed=42)
    step = 0
    done = False
    while not done and step < up_config["steps"]:
        action = strategy.decide_action(obs=obs, info=info, step=step)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    up_summary = env.summary()
    env.close()
    (results_dir / "buy_hold_summary.json").write_text(json.dumps(up_summary, indent=2))
    assert up_summary["total_return"] > 0.0, (
        f"buy_hold on an uptrending feed did not produce positive return: {up_summary}"
    )

    # --- seeded reset reproducibility ---------------------------------------
    _, obs_a = run_driver("flat", seed=123)
    _, obs_b = run_driver("flat", seed=123)
    for key in obs_a:
        assert np.allclose(obs_a[key], obs_b[key]), f"seed 123 not reproducible for {key}"

    # --- total_return math check --------------------------------------------
    expected = (up_summary["final_equity"] - up_summary["initial_cash"]) / up_summary[
        "initial_cash"
    ]
    assert math.isclose(up_summary["total_return"], expected, rel_tol=1e-9, abs_tol=1e-9)

    print("[smoke_test] all assertions passed")
    print(json.dumps({"flat": flat_summary, "buy_hold": up_summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
