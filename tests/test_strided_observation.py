import numpy as np
import pytest

from preprocessor_plugins.feature_window_preprocessor import Plugin
from test_feature_window_preprocessor import _make_df, _bridge_state
from test_env_foundations import _env


def config():
    return {"window_size": 13, "observation_stride_bars": 2,
            "require_full_observation_window": True, "feature_columns": ["feat_0"],
            "feature_scaling": "none", "feature_clip": 0,
            "include_price_window": True}


def test_exact_completed_origin_stride_and_future_perturbation():
    df = _make_df()
    df["feat_0"] = np.arange(len(df), dtype=float)
    cfg = config()
    def observe(frame):
        return Plugin(cfg).make_observation(data=frame, step=30,
                                            bridge_state=_bridge_state(), config=cfg)
    obs = observe(df)
    np.testing.assert_array_equal(obs["features"][:, 0], np.arange(5, 30, 2))
    np.testing.assert_array_equal(obs["prices"], df.CLOSE.iloc[5:30:2].to_numpy(dtype=np.float32))
    changed = df.copy()
    changed.loc[30:, ["feat_0", "CLOSE"]] = 10000
    np.testing.assert_array_equal(obs["features"], observe(changed)["features"])
    changed.loc[29, "feat_0"] += 100
    assert observe(changed)["features"][-1, 0] != obs["features"][-1, 0]


@pytest.mark.parametrize("step", [0, 1, 24])
def test_no_synthetic_padding(step):
    cfg = config()
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        Plugin(cfg).make_observation(data=_make_df(), step=step,
                                     bridge_state=_bridge_state(), config=cfg)


def test_real_env_warmup_uses_25_observed_bars(tmp_path):
    env = _env(tmp_path, window_size=13, observation_stride_bars=2,
               require_full_observation_window=True, feature_scaling="none",
               feature_clip=0, include_price_window=False)
    try:
        obs, info = env.reset(seed=7)
        assert env.bridge.bar_index == 25
        assert env.bridge.position == 0
        np.testing.assert_array_equal(obs["features"][:, 0],
                                      env.dataframe.feat.iloc[:25:2].to_numpy(dtype=np.float32))
        assert env.observation_space.contains(obs)
    finally:
        env.close()
