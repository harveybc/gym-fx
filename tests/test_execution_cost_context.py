from __future__ import annotations

import numpy as np
import pytest

from app.env import EXECUTION_COST_OBSERVATION_NAMES
from app.env import GymFxEnv


def _env() -> GymFxEnv:
    env = object.__new__(GymFxEnv)
    env.config = {}
    env._execution_cost_observation = np.zeros(5, dtype=np.float32)
    env._execution_cost_context = {}
    return env


def test_cost_context_updates_visible_vector_and_backtrader_compatibility() -> None:
    env = _env()
    env.set_execution_cost_context(
        observable_names=EXECUTION_COST_OBSERVATION_NAMES,
        observable_vector=(0.1, 0.2, 0.3, 0.0, 0.5),
        cost_patch={
            "commission_fraction_per_side": 0.0001,
            "full_spread_rate": 0.0004,
            "slippage_bps_per_side": 2.0,
            "financing_enabled": False,
        },
        metadata={"scenario_id": "nominal"},
    )

    np.testing.assert_allclose(
        env._execution_cost_observation,
        [0.1, 0.2, 0.3, 0.0, 0.5],
    )
    assert env.config["commission"] == pytest.approx(0.0001)
    assert env.config["slippage_perc"] == pytest.approx(0.0004)
    assert env._execution_cost_context["scenario_id"] == "nominal"


def test_cost_context_rejects_hidden_or_invalid_regimes() -> None:
    env = _env()
    with pytest.raises(ValueError, match="contract mismatch"):
        env.set_execution_cost_context(
            observable_names=("unknown",),
            observable_vector=(0.0,),
            cost_patch={},
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        env.set_execution_cost_context(
            observable_names=EXECUTION_COST_OBSERVATION_NAMES,
            observable_vector=(0.0, 0.0, 0.0, 0.0, 2.0),
            cost_patch={
                "commission_fraction_per_side": 0.0,
                "full_spread_rate": 0.0,
                "slippage_bps_per_side": 0.0,
                "financing_enabled": False,
            },
        )
