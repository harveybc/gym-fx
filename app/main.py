#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict

from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import load_config, save_config
from app.config_merger import merge_config, process_unknown_args
from app.env import GymFxEnv
from app.plugin_loader import load_plugin


def _load_optional_config(args) -> Dict[str, Any]:
    if args.load_config:
        return load_config(args.load_config)
    return {}


def _load_plugin_instance(group: str, name: str, config: Dict[str, Any]):
    klass, _ = load_plugin(group, name)
    instance = klass(config)
    instance.set_params(**config)
    return instance


def _collect_plugin_defaults(instances: list[Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for instance in instances:
        merged.update(getattr(instance, "plugin_params", {}))
    return merged


def _run_env(config: Dict[str, Any]) -> Dict[str, Any]:
    data_feed = _load_plugin_instance("data_feed.plugins", config["data_feed_plugin"], config)
    broker = _load_plugin_instance("broker.plugins", config["broker_plugin"], config)
    strategy = _load_plugin_instance("strategy.plugins", config["strategy_plugin"], config)
    preprocessor = _load_plugin_instance("preprocessor.plugins", config["preprocessor_plugin"], config)
    reward = _load_plugin_instance("reward.plugins", config["reward_plugin"], config)
    metrics = _load_plugin_instance("metrics.plugins", config["metrics_plugin"], config)

    plugin_defaults = _collect_plugin_defaults(
        [data_feed, broker, strategy, preprocessor, reward, metrics]
    )
    config = merge_config(config, plugin_defaults, {}, {}, {}, {})

    env = GymFxEnv(
        config=config,
        data_feed_plugin=data_feed,
        broker_plugin=broker,
        strategy_plugin=strategy,
        preprocessor_plugin=preprocessor,
        reward_plugin=reward,
        metrics_plugin=metrics,
    )

    obs, info = env.reset()
    done = False
    steps = int(config.get("steps", 500))
    step_count = 0
    while not done and step_count < steps:
        action = strategy.decide_action(obs=obs, info=info, step=step_count)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_count += 1

    return env.summary()


def main():
    args, unknown_args = parse_args()
    cli_args = vars(args)

    config = DEFAULT_VALUES.copy()
    file_config = _load_optional_config(args)
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    if config.get("mode") not in {"training", "optimization", "inference"}:
        raise ValueError("mode must be one of training|optimization|inference")

    summary = _run_env(config)

    results_file = Path(config.get("results_file", "results.json"))
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if config.get("save_config"):
        save_config(config, config["save_config"])

    if not config.get("quiet_mode", False):
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
