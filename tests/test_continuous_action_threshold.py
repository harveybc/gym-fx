"""Continuous-action deadband semantics used by the easy curriculum."""
from __future__ import annotations

import numpy as np

from app.env import GymFxEnv


def _env(threshold: float) -> GymFxEnv:
    env = GymFxEnv.__new__(GymFxEnv)
    env.action_space_mode = "continuous"
    env.continuous_action_contract = "legacy_directional_v1"
    env.continuous_action_threshold = threshold
    env.continuous_exit_threshold = 0.0
    env.bridge = None
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


class _Bridge:
    def __init__(self, position: float = 0.0, open_orders: int = 0) -> None:
        self.position = position
        self.position_units = position
        self.open_order_count = open_orders


def test_target_exposure_v2_neutral_closes_only_existing_exposure() -> None:
    env = _env(0.2)
    env.continuous_action_contract = "target_exposure_hysteresis_v2"
    env.continuous_exit_threshold = 0.05

    env.bridge = _Bridge(position=1.0)
    assert env._coerce_action(np.array([0.01], dtype=np.float32)) == 3
    assert env._coerce_action(np.array([0.1], dtype=np.float32)) == 0

    env.bridge = _Bridge()
    assert env._coerce_action(np.array([0.01], dtype=np.float32)) == 0


def test_target_exposure_v2_neutral_cancels_pending_entry() -> None:
    env = _env(0.2)
    env.continuous_action_contract = "target_exposure_hysteresis_v2"
    env.continuous_exit_threshold = 0.05
    env.bridge = _Bridge(open_orders=3)

    assert env._coerce_action(np.array([0.0], dtype=np.float32)) == 3


def test_target_exposure_v2_zero_easy_threshold_closes_on_exact_zero() -> None:
    env = _env(0.0)
    env.continuous_action_contract = "target_exposure_hysteresis_v2"
    env.continuous_exit_threshold = 0.0
    env.bridge = _Bridge(position=1.0)

    assert env._coerce_action(np.array([0.000001], dtype=np.float32)) == 1
    assert env._coerce_action(np.array([-0.000001], dtype=np.float32)) == 2
    assert env._coerce_action(np.array([0.0], dtype=np.float32)) == 3
