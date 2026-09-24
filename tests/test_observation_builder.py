"""The observation builder: that it reproduces the env exactly, and what it refuses.

Two families of test here. The first proves EQUIVALENCE: the vector built from a tail of market
data is element-for-element the vector the env's own preprocessor emits at the same bar, flattened
the way gymnasium's FlattenObservation flattens it. If that ever stops holding, every action this
builder feeds a policy is about data the policy never saw, and nothing else in the stack would
notice. The second family proves REFUSAL: every way of arriving at a plausible-looking vector that
is not the right one -- a short window padded to length, a missing feature column, a NaN quietly
zeroed, a file whose columns come out in a different order -- has to raise, by name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.env import build_base_observation_space
from app.observation_builder import (
    AgentState,
    ObservationContract,
    ObservationRefusal,
    build_observation,
    build_observation_from_csv,
    rows_from_csv,
)
from preprocessor_plugins.feature_window_preprocessor import Plugin

WINDOW = 8
SCALING_WINDOW = 32
FEATURES = [f"feat_{i}" for i in range(4)] + ["bin_flag"]


def _frame(rows: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "DATE_TIME": pd.date_range("2020-01-01", periods=rows, freq="4h").astype(str),
        "CLOSE": np.cumsum(rng.normal(0, 1, rows)) + 1000.0,
    }
    for name in FEATURES[:-1]:
        data[name] = rng.normal(0, 3, rows)
    data["bin_flag"] = rng.integers(0, 2, rows).astype(float)
    return pd.DataFrame(data)


def _contract(**overrides) -> ObservationContract:
    config = {
        "window_size": WINDOW,
        "price_column": "CLOSE",
        "feature_columns": list(FEATURES),
        "feature_binary_columns": ["bin_flag"],
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": SCALING_WINDOW,
        "feature_clip": 10.0,
        "include_price_window": True,
        "include_agent_state": True,
        "position_size": 0.01,
    }
    config.update(overrides)
    return ObservationContract.from_config(config)


def _state(price: float, bar_index: int, total_bars: int) -> AgentState:
    return AgentState.flat_at(initial_cash=10_000.0, price=price, bar_index=bar_index,
                              total_bars=total_bars)


# --- it is the env's observation, not a lookalike ---------------------------------------------

def test_it_reproduces_the_env_preprocessor_element_for_element():
    """The whole point. A tail of rows must give the vector the env emits at that bar."""
    frame = _frame()
    contract = _contract()
    step = len(frame)
    state = _state(float(frame["CLOSE"].iloc[-1]), step, step)

    reference = Plugin().make_observation(
        data=frame, step=step, bridge_state=state.as_bridge_state(),
        config=contract.to_config())
    expected = np.concatenate(
        [np.asarray(reference[key], dtype=np.float32).ravel() for key in sorted(reference)])

    built = build_observation(frame.iloc[-contract.required_rows:], contract, state)
    assert np.array_equal(np.asarray(built.vector, dtype=np.float32), expected)


def test_the_flatten_order_is_gymnasiums_and_not_an_assumption():
    """Sorted key order is asserted against the real space, so a gymnasium change cannot pass here."""
    gymnasium_spaces = pytest.importorskip("gymnasium.spaces")
    contract = _contract()
    space = build_base_observation_space(contract.to_config(), window_size=contract.window_size)
    assert list(space.spaces) == [name for name, _size in contract.block_layout()]
    assert gymnasium_spaces.flatdim(space) == contract.expected_length


def test_extra_rows_do_not_change_the_answer():
    """A rolling scaler fitted on 'whatever was supplied' would move every value silently."""
    frame = _frame()
    contract = _contract()
    state = _state(float(frame["CLOSE"].iloc[-1]), len(frame), len(frame))
    exact = build_observation(frame.iloc[-contract.required_rows:], contract, state)
    generous = build_observation(frame, contract, state)
    assert exact.observation_sha256 == generous.observation_sha256
    assert generous.rows_consumed == contract.required_rows
    assert generous.rows_supplied == len(frame)


def test_the_agent_state_actually_moves_the_observation():
    """If it did not, requiring the caller to state it would be ceremony rather than a contract."""
    frame = _frame()
    contract = _contract()
    flat = build_observation(frame, contract, _state(1000.0, 100, 400))
    later = build_observation(frame, contract, _state(1000.0, 390, 400))
    assert flat.observation_sha256 != later.observation_sha256


# --- what it declares -------------------------------------------------------------------------

def test_it_declares_how_it_was_built():
    frame = _frame()
    contract = _contract()
    built = build_observation(frame, contract, _state(1000.0, 400, 400))
    declared = built.declaration()
    assert declared["observation_length"] == contract.expected_length == built.length
    assert declared["rows_consumed"] == contract.required_rows
    assert declared["contract"]["feature_order"] == list(FEATURES)
    assert declared["contract"]["contract_sha256"] == contract.digest
    assert len(declared["observation_sha256"]) == 64
    assert sum(block["elements"] for block in declared["contract"]["block_layout"]) == built.length


def test_a_different_feature_order_is_a_different_contract():
    """The digest has to separate them; a receipt that could not is not evidence of anything."""
    reordered = list(FEATURES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert _contract().digest != _contract(feature_columns=reordered).digest


# --- refusals ---------------------------------------------------------------------------------

def test_too_few_rows_is_refused_and_never_padded():
    contract = _contract()
    short = _frame(rows=contract.required_rows - 1)
    with pytest.raises(ObservationRefusal, match="TOO_FEW_ROWS") as refusal:
        build_observation(short, contract, _state(1000.0, 10, 400))
    assert str(contract.required_rows) in str(refusal.value)
    assert "Padding" in str(refusal.value)


def test_enough_rows_for_the_window_but_not_for_the_scaler_is_still_refused():
    """The window is full, the shape is right, and the z-scores would be fitted on the wrong sample."""
    contract = _contract()
    assert contract.required_rows > contract.window_size
    with pytest.raises(ObservationRefusal, match="TOO_FEW_ROWS"):
        build_observation(_frame(rows=contract.window_size), contract, _state(1000.0, 8, 400))


def test_a_missing_feature_column_is_refused_by_name():
    frame = _frame().drop(columns=["feat_2"])
    with pytest.raises(ObservationRefusal, match="MISSING_COLUMNS") as refusal:
        build_observation(frame, _contract(), _state(1000.0, 400, 400))
    assert "feat_2" in str(refusal.value)


def test_a_missing_price_column_is_refused_by_name():
    frame = _frame().rename(columns={"CLOSE": "close"})
    with pytest.raises(ObservationRefusal, match="MISSING_COLUMNS") as refusal:
        build_observation(frame, _contract(), _state(1000.0, 400, 400))
    assert "CLOSE" in str(refusal.value)


def test_a_reordered_file_is_refused_even_though_every_name_is_present():
    """Selecting by name would succeed here, which is exactly why the order has to be checked."""
    frame = _frame()
    swapped = frame[["DATE_TIME", "CLOSE", "feat_1", "feat_0", "feat_2", "feat_3", "bin_flag"]]
    with pytest.raises(ObservationRefusal, match="COLUMN_ORDER_MISMATCH") as refusal:
        build_observation(swapped, _contract(), _state(1000.0, 400, 400))
    assert "feat_1" in str(refusal.value) and "feat_0" in str(refusal.value)


def test_a_non_finite_cell_is_refused_and_not_zero_filled():
    frame = _frame()
    frame.loc[frame.index[-3], "feat_1"] = np.nan
    with pytest.raises(ObservationRefusal, match="NON_FINITE_VALUES") as refusal:
        build_observation(frame, _contract(), _state(1000.0, 400, 400))
    assert "feat_1" in str(refusal.value)


def test_a_non_numeric_cell_is_refused_rather_than_coerced():
    rows = rows_from_csv(_frame(rows=64).to_csv(index=False))
    rows[-1]["feat_0"] = "n/a"
    with pytest.raises(ObservationRefusal, match="NON_FINITE_VALUES"):
        build_observation(rows, _contract(), _state(1000.0, 64, 64))


def test_a_stale_non_finite_cell_outside_the_consumed_tail_is_not_an_obstacle():
    """Refusals must be about the rows that reach the policy, or they become noise the caller routes around."""
    frame = _frame()
    frame.loc[frame.index[0], "feat_1"] = np.nan
    built = build_observation(frame, _contract(), _state(1000.0, 400, 400))
    assert built.length == _contract().expected_length


def test_a_bare_list_of_numbers_is_refused_because_it_is_positional():
    with pytest.raises(ObservationRefusal, match="ROWS_MUST_BE_NAMED"):
        build_observation([[1.0, 2.0, 3.0]] * 64, _contract(), _state(1000.0, 64, 64))


def test_a_ragged_table_is_refused():
    rows = rows_from_csv(_frame(rows=64).to_csv(index=False))
    rows[5] = {key: value for key, value in rows[5].items() if key != "feat_3"}
    with pytest.raises(ObservationRefusal, match="ROW_SHAPE_MISMATCH"):
        build_observation(rows, _contract(), _state(1000.0, 64, 64))


def test_a_headerless_csv_is_refused_as_positional():
    """Read as a header, the first data row would vanish AND every column would land by position."""
    body = "\n".join(",".join(["1.0"] * 7) for _ in range(64))
    with pytest.raises(ObservationRefusal, match="HEADERS_REQUIRED"):
        build_observation_from_csv("1.0,2.0,3.0,4.0,5.0,6.0,7.0\n" + body, _contract(),
                                   _state(1000.0, 64, 64))


def test_an_empty_csv_is_refused():
    with pytest.raises(ObservationRefusal, match="CSV_EMPTY"):
        rows_from_csv("")


def test_a_csv_with_a_repeated_column_is_refused():
    with pytest.raises(ObservationRefusal, match="DUPLICATE_COLUMNS"):
        rows_from_csv("a,b,a\n1,2,3\n")


def test_the_agent_state_is_required_when_the_policy_observes_it():
    with pytest.raises(ObservationRefusal, match="AGENT_STATE_REQUIRED"):
        build_observation(_frame(), _contract(), None)


def test_an_agent_state_is_refused_when_the_policy_does_not_observe_one():
    contract = _contract(include_agent_state=False)
    with pytest.raises(ObservationRefusal, match="AGENT_STATE_NOT_OBSERVED"):
        build_observation(_frame(), contract, _state(1000.0, 400, 400))


def test_expanding_scaling_will_not_pretend_a_tail_is_the_whole_series():
    contract = _contract(feature_scaling="expanding_zscore")
    with pytest.raises(ObservationRefusal, match="EXPANDING_SCALING_NEEDS_FULL_HISTORY"):
        build_observation(_frame(), contract, _state(1000.0, 400, 400))
    built = build_observation(_frame(), contract, _state(1000.0, 400, 400),
                              rows_start_the_series=True)
    assert built.rows_consumed == 400, "an expanding scaler reads every row it was given"


def test_a_contract_without_features_is_refused():
    with pytest.raises(ObservationRefusal, match="INVALID_CONTRACT"):
        ObservationContract.from_config({"window_size": 8, "feature_columns": []})


def test_a_contract_with_an_unknown_scaling_is_refused():
    with pytest.raises(ObservationRefusal, match="INVALID_CONTRACT"):
        _contract(feature_scaling="minmax")


def test_a_zero_initial_cash_is_refused_because_two_elements_are_expressed_against_it():
    with pytest.raises(ObservationRefusal, match="INVALID_AGENT_STATE"):
        AgentState.flat_at(initial_cash=0.0, price=1000.0, bar_index=1, total_bars=10)


# --- the shape the interface actually receives ------------------------------------------------

def test_rows_of_strings_from_a_parsed_csv_build_the_same_vector_as_the_frame():
    """An attached CSV reaches a provider as row dicts of strings; it must not be a second path."""
    frame = _frame()
    contract = _contract()
    state = _state(float(frame["CLOSE"].iloc[-1]), len(frame), len(frame))
    from_frame = build_observation(frame, contract, state)
    from_rows = build_observation(rows_from_csv(frame.to_csv(index=False)), contract, state)
    assert from_rows.observation_sha256 == from_frame.observation_sha256
