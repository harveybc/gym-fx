"""Public re-export of the observation builder.

Consumers outside this checkout import `gym_fx.observation_builder`; the implementation lives in
`app.observation_builder` next to the env whose construction it reproduces, so the two cannot
drift apart unnoticed.
"""
from app.observation_builder import (
    AGENT_STATE_NOTE,
    CONTRACT_SCHEMA,
    FLATTEN_NOTE,
    AgentState,
    BuiltObservation,
    ObservationContract,
    ObservationRefusal,
    build_observation,
    build_observation_from_csv,
    rows_from_csv,
)

__all__ = [
    "AGENT_STATE_NOTE",
    "CONTRACT_SCHEMA",
    "FLATTEN_NOTE",
    "AgentState",
    "BuiltObservation",
    "ObservationContract",
    "ObservationRefusal",
    "build_observation",
    "build_observation_from_csv",
    "rows_from_csv",
]
