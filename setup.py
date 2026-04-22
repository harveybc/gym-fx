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
        ],
        "strategy.plugins": [
            "default_strategy=strategy_plugins.default_strategy:Plugin",
        ],
        "preprocessor.plugins": [
            "default_preprocessor=preprocessor_plugins.default_preprocessor:Plugin",
        ],
        "reward.plugins": [
            "pnl_reward=reward_plugins.pnl_reward:Plugin",
        ],
        "metrics.plugins": [
            "default_metrics=metrics_plugins.default_metrics:Plugin",
        ],
    },
    install_requires=[
        "pandas",
        "numpy",
        "requests",
    ],
    author="Harvey Bastidas",
    author_email="your.email@example.com",
    description="Environment-only FX trading package for agent-multi integration.",
)
