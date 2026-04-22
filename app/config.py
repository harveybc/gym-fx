DEFAULT_VALUES = {
    # execution
    "mode": "inference",  # training|optimization|inference
    "driver_mode": "buy_hold",  # random|buy_hold|flat|replay
    "steps": 500,

    # plugin selection
    "data_feed_plugin": "default_data_feed",
    "broker_plugin": "default_broker",
    "strategy_plugin": "default_strategy",
    "preprocessor_plugin": "default_preprocessor",
    "reward_plugin": "pnl_reward",
    "metrics_plugin": "default_metrics",

    # data + symbol
    "input_data_file": "examples/data/eurusd.csv",
    "date_column": "DATE_TIME",
    "price_column": "CLOSE",
    "instrument": "EUR_USD",
    "timeframe": "M1",
    "headers": True,
    "max_rows": None,

    # env and execution settings
    "window_size": 32,
    "initial_cash": 10000.0,
    "position_size": 1.0,
    "commission": 0.0,
    "slippage": 0.0,

    # optional replay actions
    "replay_actions_file": None,

    # config I/O
    "remote_log": None,
    "remote_load_config": None,
    "remote_save_config": None,
    "username": None,
    "password": None,
    "load_config": None,
    "save_config": "./config_out.json",
    "save_log": "./debug_out.json",
    "results_file": "./results.json",
    "quiet_mode": False,
}
