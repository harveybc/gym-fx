"""Continuous-action deadband semantics used by the easy curriculum."""
from __future__ import annotations

import numpy as np

from app.env import GymFxEnv


def _env(threshold: float) -> GymFxEnv:
    env = GymFxEnv.__new__(GymFxEnv)
    env.action_space_mode = "continuous"
    env.continuous_action_threshold = threshold
    return env


def test_explicit_zero_threshold_maps_nonzero_outputs_directionally() -> None:
    env = _env(0.0)

    assert env._coerce_action(np.array([0.000001], dtype=np.float32)) == 1
    assert env._coerce_action(np.array([-0.000001], dtype=np.float32)) == 2
    assert env._coerce_action(np.array([0.0], dtype=np.float32)) == 0


def test_positive_threshold_retains_deadband_and_boundary_behavior() -> None:
    env = _env(0.1)

    assert env._coerce_action(np.array([0.099], dtype=np.float32)) == 0
    assert env._coerce_action(np.array([-0.099], dtype=np.float32)) == 0
    assert env._coerce_action(np.array([0.1], dtype=np.float32)) == 1
    assert env._coerce_action(np.array([-0.1], dtype=np.float32)) == 2
