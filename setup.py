from setuptools import find_packages, setup

setup(
    name="gym-fx",
    version="0.2.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "gym-fx-env=app.main:main",
        ],
        "data_feed.plugins": [
            "default_data_feed=data_feed_plugins.default_data_feed:Plugin",
        ],
        "broker.plugins": [
            "default_broker=broker_plugins.default_broker:Plugin",
            "oanda_broker=broker_plugins.oanda_broker:Plugin",
        ],
        "strategy.plugins": [
            "default_strategy=strategy_plugins.default_strategy:Plugin",
            "direct_fixed_sltp=strategy_plugins.direct_fixed_sltp:Plugin",
            "direct_atr_sltp=strategy_plugins.direct_atr_sltp:Plugin",
        ],
        "preprocessor.plugins": [
            "default_preprocessor=preprocessor_plugins.default_preprocessor:Plugin",
            "feature_window_preprocessor=preprocessor_plugins.feature_window_preprocessor:Plugin",
        ],
        "reward.plugins": [
            "pnl_reward=reward_plugins.pnl_reward:Plugin",
            "sharpe_reward=reward_plugins.sharpe_reward:Plugin",
            "dd_penalized_reward=reward_plugins.dd_penalized_reward:Plugin",
        ],
        "metrics.plugins": [
            "default_metrics=metrics_plugins.default_metrics:Plugin",
        ],
    },
    install_requires=[
        "pandas",
        "backtrader",
        "gymnasium",
        "numpy",
        "requests",
    ],
    author="Harvey Bastidas",
    author_email="your.email@example.com",
    description="Environment-only FX trading package for agent-multi integration.",
)
