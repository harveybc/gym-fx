"""Tests for the gym-fx feature_window_preprocessor plugin."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessor_plugins.feature_window_preprocessor import Plugin


def _make_df(rows: int = 100, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "DATE_TIME": pd.date_range("2020-01-01", periods=rows, freq="4h"),
        "CLOSE": np.cumsum(rng.normal(0, 1, rows)) + 100.0,
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(0, 1, rows)
    data["bin_flag"] = rng.integers(0, 2, rows).astype(float)
    return pd.DataFrame(data)


def _bridge_state(initial_cash: float = 10000.0, step: int = 32, total: int = 100):
    return {
        "position": 0,
        "equity": initial_cash,
        "initial_cash": initial_cash,
        "price": 100.0,
        "bar_index": step,
        "total_bars": total,
    }


def test_feature_window_shape_and_columns():
    df = _make_df()
    feature_cols = [f"feat_{i}" for i in range(5)] + ["bin_flag"]
    cfg = {
        "window_size": 32,
        "price_column": "CLOSE",
        "feature_columns": feature_cols,
        "feature_binary_columns": ["bin_flag"],
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 64,
        "include_price_window": True,
        "include_agent_state": True,
    }
    plugin = Plugin(cfg)
    obs = plugin.make_observation(
        data=df, step=40, bridge_state=_bridge_state(step=40), config=cfg
    )

    assert obs["features"].shape == (32, len(feature_cols))
    assert obs["features"].dtype == np.float32
    assert obs["prices"].shape == (32,)
    assert obs["returns"].shape == (32,)
    for k in ("position", "equity_norm", "unrealized_pnl_norm", "steps_remaining_norm"):
        assert obs[k].shape == (1,)
    assert np.all(np.isfinite(obs["features"]))


def test_binary_columns_passthrough():
    df = _make_df()
    feature_cols = ["feat_0", "bin_flag"]
    cfg = {
        "window_size": 16,
        "feature_columns": feature_cols,
        "feature_binary_columns": ["bin_flag"],
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 32,
    }
    plugin = Plugin(cfg)
    obs = plugin.make_observation(
        data=df, step=40, bridge_state=_bridge_state(step=40), config=cfg
    )
    bin_idx = feature_cols.index("bin_flag")
    expected = df["bin_flag"].to_numpy()[40 - 16 : 40].astype(np.float32)
    np.testing.assert_array_equal(obs["features"][:, bin_idx], expected)


def test_no_future_leakage_in_scaling():
    """Modifying rows AFTER the current step must not change the observation."""
    df = _make_df(rows=200)
    feature_cols = [f"feat_{i}" for i in range(5)]
    cfg = {
        "window_size": 32,
        "feature_columns": feature_cols,
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 64,
        "include_price_window": False,
        "include_agent_state": False,
    }
    plugin = Plugin(cfg)
    step = 80
    obs1 = plugin.make_observation(
        data=df, step=step, bridge_state=_bridge_state(step=step), config=cfg
    )

    df_perturbed = df.copy()
    df_perturbed.loc[step:, feature_cols] = 1e6  # poison the future
    obs2 = plugin.make_observation(
        data=df_perturbed,
        step=step,
        bridge_state=_bridge_state(step=step),
        config=cfg,
    )
    np.testing.assert_array_equal(obs1["features"], obs2["features"])


def test_missing_columns_raises():
    df = _make_df()
    cfg = {
        "feature_columns": ["feat_0", "does_not_exist"],
        "feature_scaling": "none",
    }
    plugin = Plugin(cfg)
    with pytest.raises(ValueError, match="missing from dataframe"):
        plugin.make_observation(
            data=df, step=40, bridge_state=_bridge_state(step=40), config=cfg
        )


def test_empty_feature_list_raises():
    df = _make_df()
    plugin = Plugin({"feature_columns": []})
    with pytest.raises(ValueError, match="non-empty"):
        plugin.make_observation(
            data=df, step=40, bridge_state=_bridge_state(step=40), config={"feature_columns": []}
        )


def test_warmup_padding_at_start():
    df = _make_df()
    feature_cols = [f"feat_{i}" for i in range(3)]
    cfg = {
        "window_size": 32,
        "feature_columns": feature_cols,
        "feature_scaling": "none",
        "include_price_window": False,
        "include_agent_state": False,
    }
    plugin = Plugin(cfg)
    obs = plugin.make_observation(
        data=df, step=0, bridge_state=_bridge_state(step=0), config=cfg
    )
    assert obs["features"].shape == (32, 3)
    assert np.all(np.isfinite(obs["features"]))
