"""
observation_builder.py — build the exact observation the env would emit, without an env.

Why this exists. A fitted policy answers ANY vector of the right length. It has no
way to complain that the numbers mean something else than they did during fitting,
so a caller who holds market data and not a pre-baked float vector has, until now,
had two options: paste a vector nobody can type, or reconstruct the observation by
eye. A reconstruction that is subtly wrong — a feature in the wrong slot, a z-score
fitted over 32 rows where the env used 256, a NaN quietly turned into 0.0 — still
produces an action, and the action looks exactly as authoritative as a correct one.
That is the failure this module exists to make impossible.

The construction is NOT reimplemented here. `preprocessor_plugins.feature_window_preprocessor`
is the authority on what the env emits, and this module calls it with exactly the
arguments `GymFxEnv._make_observation` passes, then flattens the resulting Dict the
way `gymnasium.wrappers.FlattenObservation` does (sorted key order — see
`FLATTEN_NOTE`). What this module adds is refusal: everything the plugin would
paper over on its own (padding a short window, zero-filling a NaN, fitting a scaler
on whatever history happens to be present) is checked and refused BY NAME before
the plugin is reached.

Deliberately importable without gymnasium or backtrader. The consumer that needs it
most — the M5PHET policy interface — runs in an environment that has neither, on
purpose, and `import gymnasium` there would turn "build me an observation" into an
install problem. Only pandas and numpy are used, the same two the preprocessor plugin
itself imports.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from preprocessor_plugins.feature_window_preprocessor import Plugin as _FeatureWindowPlugin

__all__ = [
    "ObservationRefusal",
    "ObservationContract",
    "AgentState",
    "BuiltObservation",
    "build_observation",
    "build_observation_from_csv",
    "rows_from_csv",
    "FLATTEN_NOTE",
    "AGENT_STATE_NOTE",
    "CONTRACT_SCHEMA",
]

CONTRACT_SCHEMA = "gym_fx.observation_contract.v1"

#: gymnasium's FlattenObservation concatenates a Dict space in SORTED KEY ORDER, because
#: gymnasium.spaces.Dict sorts a plain dict of subspaces on construction. The policy's first
#: layer is therefore indexed by that order and nothing else. Reproducing it here rather than
#: importing gymnasium keeps this module usable in an interface environment; the rule is one
#: line and `tests/test_observation_builder.py` pins it against the real wrapper.
FLATTEN_NOTE = "blocks concatenated in sorted key order, each block raveled in C (row-major) order"

#: The part of the observation that market data cannot supply, and why a default here would lie.
AGENT_STATE_NOTE = (
    "the agent-state block is the env's own state, not the market's: position, equity against "
    "initial cash, unrealized PnL and the episode coordinate. A row of prices does not contain it, "
    "so the caller states it. A builder that assumed it would be answering a different question "
    "from the one asked, and the action would not look any different"
)

_VALID_SCALINGS = ("none", "rolling_zscore", "expanding_zscore")
_VALID_AGENT_STATE_CONTRACTS = ("legacy_episode_v1", "live_stationary_v2")


class ObservationRefusal(ValueError):
    """A refusal that names itself.

    Every message here starts with an upper-case reason code, so the caller — and a person
    reading a receipt — can tell WHICH rule was broken without parsing prose. Nothing in this
    module substitutes a value for a refusal.
    """


def _refuse(code: str, detail: str) -> "ObservationRefusal":
    return ObservationRefusal(f"{code}: {detail}")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationContract:
    """The feature configuration a particular policy was fitted with.

    This is not a preference the caller may adjust per request: it is a property of the fitted
    artifact. A policy fitted on 83 columns in a stated order cannot be asked about 83 columns in
    a different order — it would answer, and the answer would be about data it never saw. So the
    contract is constructed once from the campaign configuration that produced the checkpoint, and
    every field that changes the observation is carried in `digest` so a receipt can prove which
    configuration was used.

    Field names are the env's own config keys on purpose: `ObservationContract.from_config` accepts
    the `environment` block of a campaign config verbatim, so there is no second spelling to drift.
    """

    window_size: int
    feature_columns: Tuple[str, ...]
    price_column: str = "CLOSE"
    feature_binary_columns: Tuple[str, ...] = ()
    feature_scaling: str = "rolling_zscore"
    feature_scaling_window: int = 256
    feature_clip: float = 10.0
    include_price_window: bool = True
    include_agent_state: bool = True
    agent_state_contract: str = "legacy_episode_v1"
    position_size: float = 1.0

    def __post_init__(self) -> None:
        if int(self.window_size) < 1:
            raise _refuse("INVALID_CONTRACT", "window_size must be >= 1")
        if not self.feature_columns:
            raise _refuse(
                "INVALID_CONTRACT",
                "feature_columns is empty; this builder serves feature-aware policies and cannot "
                "invent the columns a policy was fitted on")
        duplicates = sorted({c for c in self.feature_columns if list(self.feature_columns).count(c) > 1})
        if duplicates:
            raise _refuse("INVALID_CONTRACT", f"feature_columns repeats {duplicates}")
        if self.feature_scaling not in _VALID_SCALINGS:
            raise _refuse(
                "INVALID_CONTRACT",
                f"feature_scaling must be one of {_VALID_SCALINGS}; got {self.feature_scaling!r}")
        if self.agent_state_contract not in _VALID_AGENT_STATE_CONTRACTS:
            raise _refuse(
                "INVALID_CONTRACT",
                f"agent_state_contract must be one of {_VALID_AGENT_STATE_CONTRACTS}; "
                f"got {self.agent_state_contract!r}")
        unknown_binary = sorted(set(self.feature_binary_columns) - set(self.feature_columns))
        if unknown_binary:
            raise _refuse(
                "INVALID_CONTRACT",
                f"feature_binary_columns names columns that are not features: {unknown_binary}")
        if not (self.include_price_window or self.include_agent_state) and not self.feature_columns:
            raise _refuse("INVALID_CONTRACT", "the contract emits no observation blocks")

    # --- construction ------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ObservationContract":
        """Build from an env config mapping — the `environment` block of a campaign config.

        Only keys that change the observation are read. Anything else in the block (broker,
        commission, reward plugin) cannot move a single element of the vector and is ignored
        rather than validated, so a caller may hand over the whole block unedited.
        """
        if not isinstance(config, Mapping):
            raise _refuse("INVALID_CONTRACT", "an observation contract is built from a mapping")
        missing = [key for key in ("window_size", "feature_columns") if config.get(key) in (None, "", [])]
        if missing:
            raise _refuse(
                "INVALID_CONTRACT",
                f"the configuration does not declare {missing}; without them there is no observation to build")
        return cls(
            window_size=int(config["window_size"]),
            feature_columns=tuple(str(c) for c in config["feature_columns"]),
            price_column=str(config.get("price_column", "CLOSE")),
            feature_binary_columns=tuple(str(c) for c in (config.get("feature_binary_columns") or ())),
            feature_scaling=str(config.get("feature_scaling", "rolling_zscore")).strip().lower(),
            feature_scaling_window=int(config.get("feature_scaling_window", 256)),
            feature_clip=float(config.get("feature_clip", 10.0)),
            include_price_window=bool(config.get("include_price_window", True)),
            include_agent_state=bool(config.get("include_agent_state", True)),
            agent_state_contract=str(
                config.get("agent_state_contract", "legacy_episode_v1")).strip().lower(),
            position_size=float(config.get("position_size", 1.0) or 1.0),
        )

    # --- what it implies ---------------------------------------------------------------------
    def to_config(self) -> Dict[str, Any]:
        """Exactly the config mapping the preprocessor plugin is called with."""
        return {
            "window_size": int(self.window_size),
            "price_column": self.price_column,
            "feature_columns": list(self.feature_columns),
            "feature_binary_columns": list(self.feature_binary_columns),
            "feature_scaling": self.feature_scaling,
            "feature_scaling_window": int(self.feature_scaling_window),
            "feature_clip": float(self.feature_clip),
            "include_price_window": bool(self.include_price_window),
            "include_agent_state": bool(self.include_agent_state),
            "agent_state_contract": self.agent_state_contract,
            "position_size": float(self.position_size),
        }

    @property
    def agent_state_key(self) -> str:
        return ("holding_duration_norm" if self.agent_state_contract == "live_stationary_v2"
                else "steps_remaining_norm")

    def block_layout(self) -> Tuple[Tuple[str, int], ...]:
        """(block name, element count) in the order the flattened vector concatenates them.

        Sorted by block name, because that is what FlattenObservation does. Keeping this as data
        rather than a comment lets a receipt say where each element came from.
        """
        blocks: Dict[str, int] = {
            "features": int(self.window_size) * len(self.feature_columns)}
        if self.include_price_window:
            blocks["prices"] = int(self.window_size)
            blocks["returns"] = int(self.window_size)
        if self.include_agent_state:
            blocks["position"] = 1
            blocks["equity_norm"] = 1
            blocks["unrealized_pnl_norm"] = 1
            blocks[self.agent_state_key] = 1
        return tuple((name, blocks[name]) for name in sorted(blocks))

    @property
    def expected_length(self) -> int:
        return sum(size for _name, size in self.block_layout())

    @property
    def required_rows(self) -> int:
        """How many rows of history the caller must supply for an EXACT reproduction.

        The window itself needs `window_size` rows. A rolling z-score additionally fits its mean
        and standard deviation on the `feature_scaling_window` rows before the step — in the env,
        at any step past warm-up, that is exactly `feature_scaling_window` rows. Hand the builder
        fewer and the scaler is fitted on a different sample: same shape, same length, different
        numbers, and no error anywhere. So the larger of the two is required, not the window alone.
        """
        if self.feature_scaling == "rolling_zscore":
            return max(int(self.window_size), int(self.feature_scaling_window))
        return int(self.window_size)

    @property
    def feature_order(self) -> Tuple[str, ...]:
        return tuple(self.feature_columns)

    @property
    def digest(self) -> str:
        return _canonical_digest({"schema": CONTRACT_SCHEMA, **self.to_config()})

    def differences_from(self, other: "ObservationContract") -> List[str]:
        """What separates this contract from another, one sentence per difference.

        A caller that DERIVED a contract -- from a representation spec, from a campaign config it
        re-read, from a second export of the same bundle -- needs to know whether the derivation
        describes the policy it is about to question. Comparing digests answers "no" without
        saying what to fix, and comparing lengths answers "yes" when 83 columns came out in a
        different order. So the comparison lives here, beside the fields, and it names the length
        and the columns first because those are the two a person can act on.

        Empty means the two contracts build the same observation, element for element.
        """
        if not isinstance(other, ObservationContract):
            raise _refuse("INVALID_CONTRACT", "a contract is compared against another contract")
        found: List[str] = []
        if self.expected_length != other.expected_length:
            found.append(
                f"length: this contract builds {self.expected_length} elements and the other "
                f"builds {other.expected_length}")
        if list(self.feature_columns) != list(other.feature_columns):
            mine, theirs = list(self.feature_columns), list(other.feature_columns)
            if len(mine) != len(theirs):
                found.append(
                    f"feature columns: {len(mine)} here against {len(theirs)} there")
            missing = [name for name in theirs if name not in set(mine)]
            extra = [name for name in mine if name not in set(theirs)]
            if missing:
                found.append(f"feature columns absent here: {missing[:8]}"
                             + (" ..." if len(missing) > 8 else ""))
            if extra:
                found.append(f"feature columns absent there: {extra[:8]}"
                             + (" ..." if len(extra) > 8 else ""))
            if not missing and not extra:
                position = next(i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b)
                found.append(
                    f"feature column order: position {position} is {mine[position]!r} here and "
                    f"{theirs[position]!r} there")
        for field_name in ("window_size", "price_column", "feature_binary_columns",
                           "feature_scaling", "feature_scaling_window", "feature_clip",
                           "include_price_window", "include_agent_state",
                           "agent_state_contract", "position_size"):
            mine, theirs = getattr(self, field_name), getattr(other, field_name)
            if mine != theirs:
                found.append(f"{field_name}: {mine!r} here and {theirs!r} there")
        return found

    def declaration(self) -> Dict[str, Any]:
        """Everything about this contract that a receipt should carry."""
        return {
            "schema": CONTRACT_SCHEMA,
            "contract_sha256": self.digest,
            "observation_length": self.expected_length,
            "required_rows": self.required_rows,
            "window_size": int(self.window_size),
            "feature_count": len(self.feature_columns),
            "feature_order": list(self.feature_columns),
            "binary_columns": list(self.feature_binary_columns),
            "price_column": self.price_column,
            "feature_scaling": self.feature_scaling,
            "feature_scaling_window": int(self.feature_scaling_window),
            "feature_clip": float(self.feature_clip),
            "include_price_window": bool(self.include_price_window),
            "include_agent_state": bool(self.include_agent_state),
            "agent_state_contract": self.agent_state_contract,
            "block_layout": [{"block": name, "elements": size} for name, size in self.block_layout()],
            "flatten_order": FLATTEN_NOTE,
            "features_layout": "window rows outer, feature columns inner (row-major)",
        }


# ---------------------------------------------------------------------------------------------
# the state a row of market data cannot supply
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentState:
    """The env-side state the observation embeds alongside the market window.

    Four of the 2724 elements of the ETH contract come from here, and every one of them is
    unknowable from a price series: whether the agent is in a position, how its equity compares to
    the cash it started with, what that position is worth right now, and where the episode is in
    its own clock. This dataclass is the smallest honest interface that supplies them — it is
    required whenever `include_agent_state` is set, and it has no defaults for the fields that
    change the answer, because a default is a silent assumption about the caller's situation.

    `flat_at` is offered for the common question ("what would this policy do here, holding
    nothing?"), and it still makes the caller state the episode coordinate, since the legacy
    contract's `steps_remaining_norm` genuinely depends on it.
    """

    position: int
    equity: float
    initial_cash: float
    price: float
    bar_index: int
    total_bars: int
    entry_price: float = 0.0
    holding_bars: int = 0
    position_units: Optional[float] = None

    def __post_init__(self) -> None:
        if float(self.initial_cash) == 0.0:
            raise _refuse(
                "INVALID_AGENT_STATE",
                "initial_cash is zero; equity_norm and unrealized_pnl_norm are expressed against it")
        if int(self.total_bars) < 1:
            raise _refuse("INVALID_AGENT_STATE", "total_bars must be >= 1")
        if int(self.bar_index) < 0:
            raise _refuse("INVALID_AGENT_STATE", "bar_index must be >= 0")
        for name in ("equity", "initial_cash", "price", "entry_price"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise _refuse("INVALID_AGENT_STATE", f"{name} is not finite")

    @classmethod
    def flat_at(cls, *, initial_cash: float, price: float, bar_index: int,
                total_bars: int) -> "AgentState":
        """Holding nothing, equity equal to the cash the episode started with."""
        return cls(position=0, equity=float(initial_cash), initial_cash=float(initial_cash),
                   price=float(price), bar_index=int(bar_index), total_bars=int(total_bars))

    def as_bridge_state(self) -> Dict[str, Any]:
        """Exactly the mapping `GymFxEnv._make_observation` hands the preprocessor."""
        return {
            "position": int(self.position),
            "position_units": (float(self.position_units) if self.position_units is not None
                               else None),
            "equity": float(self.equity),
            "initial_cash": float(self.initial_cash),
            "price": float(self.price),
            "entry_price": float(self.entry_price),
            "holding_bars": int(self.holding_bars),
            "bar_index": int(self.bar_index),
            "total_bars": int(self.total_bars),
        }

    def declaration(self) -> Dict[str, Any]:
        return {"position": int(self.position), "equity": float(self.equity),
                "initial_cash": float(self.initial_cash), "price": float(self.price),
                "entry_price": float(self.entry_price), "holding_bars": int(self.holding_bars),
                "bar_index": int(self.bar_index), "total_bars": int(self.total_bars)}


# ---------------------------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BuiltObservation:
    """One observation, together with everything needed to say how it was built.

    The point of carrying the provenance next to the numbers is that the numbers alone are
    unfalsifiable: 2724 floats look identical whether they came from the right 256 rows or from a
    zero-fill. `declaration()` is what a receipt should copy verbatim.
    """

    vector: Tuple[float, ...]
    contract: ObservationContract
    agent_state: Optional[AgentState]
    rows_supplied: int
    rows_consumed: int
    scaling_rows_used: int
    blocks: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.vector)

    @property
    def observation_sha256(self) -> str:
        """Digest of the float32 bytes the policy actually receives.

        Over the bytes, not over a text rendering: two vectors that print the same at six decimals
        are not the same observation, and a receipt that could not tell them apart would be
        useless for the one job it has.
        """
        return hashlib.sha256(
            np.asarray(self.vector, dtype=np.float32).tobytes()).hexdigest()

    def as_list(self) -> List[float]:
        return [float(v) for v in self.vector]

    def declaration(self) -> Dict[str, Any]:
        return {
            "observation_length": self.length,
            "observation_sha256": self.observation_sha256,
            "rows_supplied": int(self.rows_supplied),
            "rows_consumed": int(self.rows_consumed),
            "scaling_rows_used": int(self.scaling_rows_used),
            "contract": self.contract.declaration(),
            "agent_state": (self.agent_state.declaration() if self.agent_state is not None
                            else None),
            "agent_state_note": AGENT_STATE_NOTE,
        }


# ---------------------------------------------------------------------------------------------
# reading the caller's rows
# ---------------------------------------------------------------------------------------------

def _looks_numeric(value: Any) -> bool:
    try:
        float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def rows_from_csv(text: str) -> List[Dict[str, str]]:
    """Parse CSV text into ordered row mappings, refusing anything headerless or ragged.

    Headers are not optional. Without them the columns are positional, and a file whose columns
    were exported in a different order would be read straight into the wrong feature slots — the
    exact substitution this module exists to prevent, and one that produces no error at all.
    """
    if not isinstance(text, str) or not text.strip():
        raise _refuse("CSV_EMPTY", "no CSV text was supplied")
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames
    if not header or any(not str(name).strip() for name in header):
        raise _refuse(
            "HEADERS_REQUIRED",
            "the first line must name the columns; a headerless file is positional and its columns "
            "would be read into whatever feature slot they happen to land in")
    if len(header) != len(set(header)):
        repeated = sorted({name for name in header if header.count(name) > 1})
        raise _refuse("DUPLICATE_COLUMNS", f"the header repeats {repeated}")
    if all(_looks_numeric(name) for name in header):
        # Every "name" is a number, so the first line is a data row and this file is positional.
        # Read as a header it would also silently drop that row from the window.
        raise _refuse(
            "HEADERS_REQUIRED",
            "the first line is all numbers, so it is data and not a header; a positional file "
            "cannot say which column is which feature")
    rows: List[Dict[str, str]] = []
    for number, row in enumerate(reader, start=2):
        if None in row or None in row.values():
            raise _refuse("ROW_WIDTH_MISMATCH", f"line {number} does not have {len(header)} fields")
        rows.append({name: row[name] for name in header})
    if not rows:
        raise _refuse("CSV_EMPTY", "the CSV has a header and no data rows")
    return rows


def _frame_from_rows(rows: Any) -> Tuple[pd.DataFrame, List[str]]:
    """Normalize the accepted input shapes into (frame, declared column order).

    The declared order is kept separately because pandas would happily reorder or de-duplicate it,
    and the order is evidence: it is how we detect a file produced by a different feature pipeline.
    """
    if isinstance(rows, pd.DataFrame):
        declared = [str(c) for c in rows.columns]
        if len(declared) != len(set(declared)):
            repeated = sorted({c for c in declared if declared.count(c) > 1})
            raise _refuse("DUPLICATE_COLUMNS", f"the frame repeats {repeated}")
        return rows, declared
    if isinstance(rows, str):
        parsed = rows_from_csv(rows)
        return _frame_from_rows(parsed)
    if not isinstance(rows, Sequence) or isinstance(rows, (bytes, bytearray)):
        raise _refuse(
            "ROWS_REQUIRED",
            "supply market data as a DataFrame, CSV text, or a list of row objects")
    listed = list(rows)
    if not listed:
        raise _refuse("ROWS_REQUIRED", "no rows were supplied")
    if not all(isinstance(row, Mapping) for row in listed):
        raise _refuse(
            "ROWS_MUST_BE_NAMED",
            "each row must be an object keyed by column name; a bare list of numbers is positional "
            "and this builder will not guess which column each position is")
    declared = [str(key) for key in listed[0].keys()]
    if len(declared) != len(set(declared)):
        repeated = sorted({c for c in declared if declared.count(c) > 1})
        raise _refuse("DUPLICATE_COLUMNS", f"the first row repeats {repeated}")
    expected = set(declared)
    for number, row in enumerate(listed, start=1):
        if set(str(key) for key in row.keys()) != expected:
            raise _refuse(
                "ROW_SHAPE_MISMATCH",
                f"row {number} has different columns from the first row; a ragged table cannot be "
                "one contiguous window of history")
    frame = pd.DataFrame([{str(k): v for k, v in row.items()} for row in listed],
                         columns=declared)
    return frame, declared


def _check_columns(declared_order: Sequence[str], contract: ObservationContract) -> None:
    needed = list(contract.feature_columns)
    if contract.include_price_window:
        needed = needed + [contract.price_column]
    missing = [name for name in dict.fromkeys(needed) if name not in set(declared_order)]
    if missing:
        shown = missing[:8]
        raise _refuse(
            "MISSING_COLUMNS",
            f"{len(missing)} column(s) the policy was fitted on are not in the data: {shown}"
            + (" ..." if len(missing) > len(shown) else "")
            + "; these are engineered features, and they cannot be derived from OHLC here")
    # Order check. Selecting by name is what the preprocessor does, so a reordered file would
    # still produce a well-formed vector -- which is precisely why the mismatch has to be raised
    # here. A file whose feature columns come out in a different order than the campaign's
    # model-ready export was produced by a different pipeline configuration, and the same NAME may
    # not carry the same DEFINITION across those two. Refusing costs a caller one reorder;
    # accepting costs them an action computed from columns they did not intend.
    present = [name for name in declared_order if name in set(contract.feature_columns)]
    if present != list(contract.feature_columns):
        for position, (found, wanted) in enumerate(zip(present, contract.feature_columns)):
            if found != wanted:
                raise _refuse(
                    "COLUMN_ORDER_MISMATCH",
                    f"feature column {position} is {found!r} and the policy was fitted with "
                    f"{wanted!r}; the data is ordered differently from the configuration that "
                    "produced this policy")
        raise _refuse(
            "COLUMN_ORDER_MISMATCH",
            f"the data carries {len(present)} of the fitted feature columns in a different "
            f"arrangement from the fitted order of {len(contract.feature_columns)}")


def _numeric_tail(frame: pd.DataFrame, columns: Sequence[str], rows_consumed: int) -> pd.DataFrame:
    """Coerce the consumed tail to float and refuse anything that is not a finite number.

    Two different silences are closed here. A CSV arrives as strings, so an empty cell or a stray
    'NaN' would become a float NaN; and the preprocessor's last-resort `np.nan_to_num` would then
    turn that NaN into 0.0 deep inside a z-scored window, where it reads as "exactly average"
    rather than "missing". The policy cannot tell those apart. Neither can the person.
    """
    tail = frame.iloc[-rows_consumed:]
    numeric: Dict[str, np.ndarray] = {}
    for name in columns:
        series = pd.to_numeric(tail[name], errors="coerce")
        values = series.to_numpy(dtype=np.float64)
        original = tail[name].to_numpy()
        bad = ~np.isfinite(values)
        if bool(bad.any()):
            first = int(np.argmax(bad))
            raise _refuse(
                "NON_FINITE_VALUES",
                f"column {name!r} holds {original[first]!r} at row {first + 1} of the "
                f"{rows_consumed} consumed; a non-numeric or non-finite cell would be silently "
                "turned into 0.0 inside a normalized window and read as an average value")
        numeric[name] = values
    return pd.DataFrame(numeric, columns=list(columns))


# ---------------------------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------------------------

def build_observation(
    rows: Any,
    contract: ObservationContract,
    agent_state: Optional[AgentState] = None,
    *,
    rows_start_the_series: bool = False,
) -> BuiltObservation:
    """Build the observation the env would emit for the bar AFTER the last supplied row.

    `rows` may be a pandas DataFrame, CSV text, or a list of row objects (the shape an attached
    CSV takes once the workbench has parsed it). The last row is the most recent OBSERVED bar,
    matching the env, where the window at `bar_index` is the rows strictly before it.

    Exactly `contract.required_rows` rows are consumed, taken from the end. Consuming a fixed
    count rather than "whatever was supplied" is what makes the result reproducible: with a
    rolling z-score, handing over 300 rows instead of 256 would otherwise move every scaled value
    while nothing in the output changed shape.

    `rows_start_the_series` is only meaningful for `expanding_zscore`, whose scaler is fitted on
    everything from row 0 of the episode. There is no way to see, from a tail of rows, whether it
    is a tail or the whole series, so the caller has to say, and the default is the safe answer.
    """
    if not isinstance(contract, ObservationContract):
        raise _refuse("INVALID_CONTRACT", "an ObservationContract is required")

    frame, declared_order = _frame_from_rows(rows)
    rows_supplied = int(len(frame))
    _check_columns(declared_order, contract)

    if contract.feature_scaling == "expanding_zscore" and not rows_start_the_series:
        raise _refuse(
            "EXPANDING_SCALING_NEEDS_FULL_HISTORY",
            "this policy's scaler is fitted on every row from the start of the series, so a tail of "
            "rows cannot reproduce it; supply the series from its first row and set "
            "rows_start_the_series=True")

    required = contract.required_rows
    if rows_supplied < required:
        detail = (f"{rows_supplied} row(s) supplied and this policy needs {required}: "
                  f"{contract.window_size} for the window")
        if contract.feature_scaling == "rolling_zscore":
            detail += (f" and {contract.feature_scaling_window} for the rolling scaler that "
                       "normalizes it")
        raise _refuse(
            "TOO_FEW_ROWS",
            detail + ". Padding the window to length would build a different observation, and the "
                     "policy would answer it without anyone noticing")

    rows_consumed = rows_supplied if contract.feature_scaling == "expanding_zscore" else required

    needed = list(dict.fromkeys(
        list(contract.feature_columns)
        + ([contract.price_column] if contract.include_price_window else [])))
    numeric = _numeric_tail(frame, needed, rows_consumed)

    if contract.include_agent_state and agent_state is None:
        raise _refuse(
            "AGENT_STATE_REQUIRED",
            "this policy observes its own position, equity, unrealized PnL and episode position; "
            + AGENT_STATE_NOTE)
    if not contract.include_agent_state and agent_state is not None:
        raise _refuse(
            "AGENT_STATE_NOT_OBSERVED",
            "this policy's observation carries no agent-state block, so the supplied state would "
            "change nothing and claiming otherwise in a receipt would be false")

    config = contract.to_config()
    bridge_state = agent_state.as_bridge_state() if agent_state is not None else {}
    # position_units is only consulted by the live_stationary_v2 contract; leaving a None in the
    # mapping would make the plugin's `or 0.0` swallow it, so it is dropped when unstated.
    bridge_state = {k: v for k, v in bridge_state.items() if v is not None}

    plugin = _FeatureWindowPlugin()
    step = int(len(numeric))
    obs = plugin.make_observation(data=numeric, step=step, bridge_state=bridge_state,
                                  config=config)

    expected_blocks = dict(contract.block_layout())
    produced = {name: np.asarray(value, dtype=np.float32).ravel() for name, value in obs.items()}
    if set(produced) != set(expected_blocks):
        raise _refuse(
            "BLOCK_SET_MISMATCH",
            f"the preprocessor emitted {sorted(produced)} and this contract declares "
            f"{sorted(expected_blocks)}; the contract no longer describes what the env builds")
    for name, size in expected_blocks.items():
        if produced[name].size != size:
            raise _refuse(
                "BLOCK_SIZE_MISMATCH",
                f"block {name!r} has {produced[name].size} elements and the contract declares "
                f"{size}")

    vector = np.concatenate([produced[name] for name in sorted(produced)])
    if vector.size != contract.expected_length:
        # Unreachable while the block checks above hold; kept because the one thing this module
        # must never do is hand back a vector of a length it did not verify.
        raise _refuse(
            "OBSERVATION_LENGTH_MISMATCH",
            f"built {vector.size} elements and the contract declares {contract.expected_length}")
    if not bool(np.isfinite(vector).all()):
        raise _refuse(
            "NON_FINITE_OBSERVATION",
            "the built observation contains a non-finite value even though every input cell was "
            "finite; the contract's scaling or clipping is producing it and the vector will not "
            "be served")

    scaling_rows = (rows_consumed if contract.feature_scaling != "none"
                    else 0)
    blocks = {name: int(size) for name, size in contract.block_layout()}
    return BuiltObservation(
        vector=tuple(float(v) for v in vector),
        contract=contract,
        agent_state=agent_state,
        rows_supplied=rows_supplied,
        rows_consumed=int(rows_consumed),
        scaling_rows_used=int(scaling_rows),
        blocks=blocks,
    )


def build_observation_from_csv(
    text: str,
    contract: ObservationContract,
    agent_state: Optional[AgentState] = None,
    *,
    rows_start_the_series: bool = False,
) -> BuiltObservation:
    """Convenience over `build_observation` for callers that hold raw CSV text."""
    return build_observation(rows_from_csv(text), contract, agent_state,
                             rows_start_the_series=rows_start_the_series)
