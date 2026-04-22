import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="gym-fx env runtime (env-only, agent-agnostic)."
    )
    parser.add_argument("--mode", choices=["training", "optimization", "inference"])
    parser.add_argument("--driver_mode", choices=["random", "buy_hold", "flat", "replay"])
    parser.add_argument("--steps", type=int)

    parser.add_argument("--input_data_file", type=str)
    parser.add_argument("--date_column", type=str)
    parser.add_argument("--price_column", type=str)
    parser.add_argument("--headers", action="store_true")
    parser.add_argument("--max_rows", type=int)

    parser.add_argument("--window_size", type=int)
    parser.add_argument("--initial_cash", type=float)
    parser.add_argument("--position_size", type=float)
    parser.add_argument("--commission", type=float)
    parser.add_argument("--slippage", type=float)

    parser.add_argument("--data_feed_plugin", type=str)
    parser.add_argument("--broker_plugin", type=str)
    parser.add_argument("--strategy_plugin", type=str)
    parser.add_argument("--preprocessor_plugin", type=str)
    parser.add_argument("--reward_plugin", type=str)
    parser.add_argument("--metrics_plugin", type=str)

    parser.add_argument("--replay_actions_file", type=str)
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--load_config", type=str)
    parser.add_argument("--save_config", type=str)
    parser.add_argument("--quiet_mode", action="store_true")

    return parser.parse_known_args()
