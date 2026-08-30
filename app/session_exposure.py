"""Weekly session-exposure state machine (work plan 42; corrected
under order @e303e386 C1-C4).

PURE, venue-agnostic, driver-free: the five states and the action
overlay are computed from IMMUTABLE VALIDATED values so gym-fx, the
lts runners and the calibration materializer share ONE authority.

Corrections in this revision, each with a reproduced counterexample:

* C1 exposure is TYPED and SIGNED. The previous ``open_position``
  boolean made every action legal while a position was open, so a
  reversal (-1.0 against a long) passed through WIND_DOWN. Risk
  increase now means greater absolute exposure OR sign reversal OR
  entry from flat, decided from signed facts under a declared action
  mapping. Ambiguous or non-finite actions REFUSE.
* C2 closure/reopen state derives from the BOUND interval set:
  canonical UTC, ordered, non-overlapping, ``close_at < reopen_at``,
  tied to venue/account/symbol and calendar digest. Nullable adapter
  hints may be cross-checked but can never authorize normal trading;
  missing reopen evidence after a known closure FAILS CLOSED.
* C3 expected closure suppresses ONLY bar staleness. Terminal,
  account, bracket, pending-order and exposure incidents take
  precedence, and the already-carried position gets its own
  ``CARRIED_POSITION_RECOVERY_ACTIVE`` bound to a one-use migration
  record — it never normalizes future weekend exposure.
* C4 strict typed boundaries: bool, str, NaN, inf, fractional counts,
  negatives and unavailable values are refused. No ``float(...)``,
  ``int(...)``, ``or 0`` or raw ``TypeError`` as policy behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

STATES = ("NORMAL_TRADING", "WIND_DOWN", "FORCED_FLATTEN",
          "EXPECTED_MARKET_CLOSED", "REOPEN_BLACKOUT")
SIDES = ("long", "short", "flat")
ACTION_MAPPINGS = ("target_exposure_v2", "discrete_command_v1")
# G1: the env's discrete command domain. It is NOT the signed-target
# domain classify_action consumes: feeding command id 2 ("go short")
# to a target-value classifier silently reads it as "target exposure
# +2", which reports an enlargement of a short as a reversal.
DISCRETE_COMMANDS = {0: "hold", 1: "long", 2: "short", 3: "close"}
HOLD_COMMAND = 0
CLOSE_COMMAND = 3
RECOVERY_POLICIES = ("protected_opportunistic_then_forced",)
HOLIDAY_POLICIES = ("same_as_weekly",)
SESSION_SOURCES = ("venue_symbol_sessions_v1",)
EPSILON = 1e-12


class SessionPolicyError(ValueError):
    """Invalid configuration/evidence — refused, never defaulted."""


class SessionEvidenceError(ValueError):
    """Session/exposure evidence is unusable — typed refusal."""


class SessionDataContradictionError(ValueError):
    """The market data contradicts the bound session calendar: a
    tradable bar exists inside a declared closure. Work plan 42
    prohibits synthesized tradable weekend bars, so this is a typed
    refusal and never a zeroed reward on a fabricated step."""


# ---------------------------------------------------------------- #
# C4: strict typed primitives (no coercion anywhere)               #
# ---------------------------------------------------------------- #

def require_real(name: str, value: Any, *, positive: bool = False,
                 nonnegative: bool = False) -> float:
    """A real, finite, NON-BOOL number. Strings, bools, NaN and
    infinities refuse (PRE: wind_down_hours='36' and
    max_gap_sigma=NaN were both accepted)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionPolicyError(
            f"{name}: a finite real number is required, got "
            f"{type(value).__name__} {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SessionPolicyError(f"{name}: non-finite value {value!r}")
    if positive and number <= 0.0:
        raise SessionPolicyError(f"{name}: must be > 0, got {number}")
    if nonnegative and number < 0.0:
        raise SessionPolicyError(f"{name}: must be >= 0, got {number}")
    return number


def require_count(name: str, value: Any, *, minimum: int = 0) -> int:
    """A non-bool integer count; floats (even integral) refuse."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionPolicyError(
            f"{name}: an integer count is required, got "
            f"{type(value).__name__} {value!r}")
    if value < minimum:
        raise SessionPolicyError(
            f"{name}: must be >= {minimum}, got {value}")
    return value


def require_enum(name: str, value: Any,
                 allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SessionPolicyError(
            f"{name}: {value!r} is not one of {list(allowed)}")
    return value


def require_identity(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionPolicyError(
            f"{name}: a non-empty identity string is required")
    return value


def require_utc(name: str, value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise SessionEvidenceError(
            f"{name}: a datetime is required, got "
            f"{type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise SessionEvidenceError(
            f"{name}: timezone-aware UTC datetime required "
            "(naive datetimes refuse)")
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------- #
# C4/§4: typed policy contract                                      #
# ---------------------------------------------------------------- #

REQUIRED_KEYS = (
    "enabled", "session_source", "wind_down_hours",
    "forced_flatten_hours", "cancel_pending_on_wind_down",
    "allow_risk_increase_during_wind_down", "reopen_min_hours",
    "reopen_min_closed_bars", "stability_consecutive_checks",
    "max_spread_relative_to_baseline", "max_gap_sigma",
    "max_realized_vol_relative_to_baseline",
    "carried_position_recovery", "holiday_policy",
    "calendar_identity",
    # G2: baseline windows and units are BOUND HERE, never inferred
    # at the call site. All three are counts of fully closed bars.
    "reopen_baseline_bars", "reopen_gap_sigma_bars",
    "reopen_realized_vol_bars")


def validate_policy(config: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise SessionPolicyError(
            f"session_exposure_policy missing fields {missing}")
    unknown = sorted(set(config) - set(REQUIRED_KEYS))
    if unknown:
        raise SessionPolicyError(
            f"session_exposure_policy unknown fields {unknown}")
    for key in ("enabled", "cancel_pending_on_wind_down",
                "allow_risk_increase_during_wind_down"):
        if not isinstance(config[key], bool):
            raise SessionPolicyError(f"{key} must be a bool")
    validated = {
        "enabled": config["enabled"],
        "session_source": require_enum(
            "session_source", config["session_source"],
            SESSION_SOURCES),
        "wind_down_hours": require_real(
            "wind_down_hours", config["wind_down_hours"],
            positive=True),
        "forced_flatten_hours": require_real(
            "forced_flatten_hours", config["forced_flatten_hours"],
            positive=True),
        "cancel_pending_on_wind_down":
            config["cancel_pending_on_wind_down"],
        "allow_risk_increase_during_wind_down":
            config["allow_risk_increase_during_wind_down"],
        "reopen_min_hours": require_real(
            "reopen_min_hours", config["reopen_min_hours"],
            nonnegative=True),
        "reopen_min_closed_bars": require_count(
            "reopen_min_closed_bars", config["reopen_min_closed_bars"],
            minimum=1),
        "stability_consecutive_checks": require_count(
            "stability_consecutive_checks",
            config["stability_consecutive_checks"], minimum=1),
        "max_spread_relative_to_baseline": require_real(
            "max_spread_relative_to_baseline",
            config["max_spread_relative_to_baseline"], positive=True),
        "max_gap_sigma": require_real(
            "max_gap_sigma", config["max_gap_sigma"], positive=True),
        "max_realized_vol_relative_to_baseline": require_real(
            "max_realized_vol_relative_to_baseline",
            config["max_realized_vol_relative_to_baseline"],
            positive=True),
        "carried_position_recovery": require_enum(
            "carried_position_recovery",
            config["carried_position_recovery"], RECOVERY_POLICIES),
        "holiday_policy": require_enum(
            "holiday_policy", config["holiday_policy"],
            HOLIDAY_POLICIES),
        "calendar_identity": require_identity(
            "calendar_identity", config["calendar_identity"]),
        "reopen_baseline_bars": require_count(
            "reopen_baseline_bars", config["reopen_baseline_bars"],
            minimum=2),
        "reopen_gap_sigma_bars": require_count(
            "reopen_gap_sigma_bars", config["reopen_gap_sigma_bars"],
            minimum=2),
        "reopen_realized_vol_bars": require_count(
            "reopen_realized_vol_bars",
            config["reopen_realized_vol_bars"], minimum=2),
    }
    if validated["forced_flatten_hours"] >= \
            validated["wind_down_hours"]:
        raise SessionPolicyError(
            "forced flatten must occur AFTER wind-down begins and "
            "BEFORE closure: forced_flatten_hours < wind_down_hours")
    if validated["enabled"] and validated[
            "allow_risk_increase_during_wind_down"]:
        raise SessionPolicyError(
            "allow_risk_increase_during_wind_down=true while "
            "demanding flat exposure is invalid (work plan 42 §4)")
    # F3: an ENABLED weekly-flat policy MUST cancel risk-increasing
    # pending entries at wind-down. The false cell is rejected at
    # materialization, never launched.
    if validated["enabled"] and not validated[
            "cancel_pending_on_wind_down"]:
        raise SessionPolicyError(
            "cancel_pending_on_wind_down=false is invalid while the "
            "policy is enabled: pending ENTRY orders must be "
            "cancelled before closure (protective reduce-only "
            "brackets are never cancelled by this rule)")
    return validated


# ---------------------------------------------------------------- #
# C2: bound session calendar                                        #
# ---------------------------------------------------------------- #

@dataclass(frozen=True)
class SessionCalendar:
    """Immutable VALIDATED closure intervals bound to an identity.

    F1: every invariant is enforced in ``__post_init__`` so a direct
    constructor call cannot produce an invalid object. Frozen invalid
    data is still invalid."""

    venue: str
    account_fingerprint: str
    symbol: str
    calendar_digest: str
    intervals: tuple  # ((close_at, reopen_at), ...) canonical UTC

    def __post_init__(self):
        require_identity("venue", self.venue)
        require_identity("account_fingerprint",
                         self.account_fingerprint)
        require_identity("symbol", self.symbol)
        require_identity("calendar_digest", self.calendar_digest)
        if not isinstance(self.intervals, tuple):
            raise SessionEvidenceError("intervals must be a tuple")
        previous = None
        for index, item in enumerate(self.intervals):
            if not isinstance(item, tuple) or len(item) != 2:
                raise SessionEvidenceError(
                    f"interval[{index}]: (close_at, reopen_at) tuple "
                    "required")
            close_at = require_utc(f"interval[{index}].close_at",
                                   item[0])
            reopen_at = require_utc(f"interval[{index}].reopen_at",
                                    item[1])
            if close_at is not item[0] or reopen_at is not item[1]:
                raise SessionEvidenceError(
                    f"interval[{index}]: intervals must already be "
                    "canonical UTC (use SessionCalendar.build)")
            if not close_at < reopen_at:
                raise SessionEvidenceError(
                    f"interval[{index}]: close_at must precede "
                    "reopen_at")
            if previous is not None:
                if close_at < previous[0]:
                    raise SessionEvidenceError(
                        "intervals must be sorted by close_at")
                if close_at < previous[1]:
                    raise SessionEvidenceError(
                        "overlapping closure intervals are "
                        "contradictory session evidence")
            previous = (close_at, reopen_at)

    @staticmethod
    def build(*, venue: str, account_fingerprint: str, symbol: str,
              calendar_digest: str,
              intervals: Sequence) -> "SessionCalendar":
        venue = require_identity("venue", venue)
        account = require_identity("account_fingerprint",
                                   account_fingerprint)
        symbol = require_identity("symbol", symbol)
        digest = require_identity("calendar_digest", calendar_digest)
        canonical = []
        for index, item in enumerate(intervals):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise SessionEvidenceError(
                    f"interval[{index}]: (close_at, reopen_at) pair "
                    "required")
            close_at = require_utc(f"interval[{index}].close_at",
                                   item[0])
            reopen_at = require_utc(f"interval[{index}].reopen_at",
                                    item[1])
            if not close_at < reopen_at:
                raise SessionEvidenceError(
                    f"interval[{index}]: close_at must precede "
                    "reopen_at (contradictory session evidence)")
            canonical.append((close_at, reopen_at))
        canonical.sort(key=lambda pair: pair[0])
        return SessionCalendar(venue=venue,
                               account_fingerprint=account,
                               symbol=symbol, calendar_digest=digest,
                               intervals=tuple(canonical))

    def current_closure(self, now: datetime):
        for close_at, reopen_at in self.intervals:
            if close_at <= now < reopen_at:
                return (close_at, reopen_at)
        return None

    def most_recent_reopen(self, now: datetime):
        latest = None
        for _close_at, reopen_at in self.intervals:
            if reopen_at <= now and (latest is None
                                     or reopen_at > latest):
                latest = reopen_at
        return latest

    def next_closure(self, now: datetime):
        for close_at, reopen_at in self.intervals:
            if now < close_at:
                return (close_at, reopen_at)
        return None


@dataclass(frozen=True)
class ReopenEvidence:
    """Fresh direct evidence after a reopen. Absent evidence after a
    known closure FAILS CLOSED (C2)."""

    closed_bars_since_reopen: int
    stability_checks_passed: int
    hint_time_since_reopen_hours: Optional[float] = None

    def __post_init__(self):
        require_count("closed_bars_since_reopen",
                      self.closed_bars_since_reopen, minimum=0)
        require_count("stability_checks_passed",
                      self.stability_checks_passed, minimum=0)
        if self.hint_time_since_reopen_hours is not None:
            require_real("hint_time_since_reopen_hours",
                         self.hint_time_since_reopen_hours,
                         nonnegative=True)

    @staticmethod
    def build(*, closed_bars_since_reopen: Any,
              stability_checks_passed: Any,
              hint_time_since_reopen_hours: Any = None
              ) -> "ReopenEvidence":
        bars = require_count("closed_bars_since_reopen",
                             closed_bars_since_reopen, minimum=0)
        checks = require_count("stability_checks_passed",
                               stability_checks_passed, minimum=0)
        hint = (None if hint_time_since_reopen_hours is None
                else require_real("hint_time_since_reopen_hours",
                                  hint_time_since_reopen_hours,
                                  nonnegative=True))
        return ReopenEvidence(closed_bars_since_reopen=bars,
                              stability_checks_passed=checks,
                              hint_time_since_reopen_hours=hint)


# ---------------------------------------------------------------- #
# C1: typed signed exposure                                         #
# ---------------------------------------------------------------- #

@dataclass(frozen=True)
class ExposureFacts:
    """Signed exposure facts sufficient to classify a target action.

    ``signed_exposure``: current signed target/quantity (>0 long,
    <0 short, 0 flat) under ``action_mapping``."""

    signed_exposure: float
    side: str
    pending_entry_side: Optional[str]
    pending_entry_size: float
    pending_orders: int
    action_mapping: str
    protective_orders: int = 0

    def __post_init__(self):
        exposure = require_real("signed_exposure",
                                self.signed_exposure)
        derived = ("flat" if abs(exposure) <= EPSILON
                   else ("long" if exposure > 0 else "short"))
        require_enum("side", self.side, SIDES)
        if self.side != derived:
            raise SessionEvidenceError(
                f"side {self.side!r} contradicts signed_exposure "
                f"{exposure} (derived {derived!r})")
        require_enum("action_mapping", self.action_mapping,
                     ACTION_MAPPINGS)
        require_real("pending_entry_size", self.pending_entry_size,
                     nonnegative=True)
        require_count("pending_orders", self.pending_orders,
                      minimum=0)
        require_count("protective_orders", self.protective_orders,
                      minimum=0)
        if self.protective_orders > self.pending_orders:
            raise SessionEvidenceError(
                "protective_orders cannot exceed pending_orders")
        if self.pending_entry_side is not None:
            require_enum("pending_entry_side",
                         self.pending_entry_side, ("long", "short"))
            if self.pending_entry_size <= 0.0:
                raise SessionEvidenceError(
                    "pending_entry_side declared with zero size")
            if self.pending_orders < 1:
                raise SessionEvidenceError(
                    "pending_entry_side declared with zero pending "
                    "orders")

    @staticmethod
    def build(*, signed_exposure: Any = 0.0, side: Any = None,
              pending_entry_side: Any = None,
              pending_entry_size: Any = 0.0,
              pending_orders: Any = 0,
              action_mapping: Any = "target_exposure_v2",
              protective_orders: Any = 0) -> "ExposureFacts":
        exposure = require_real("signed_exposure", signed_exposure)
        derived = ("flat" if abs(exposure) <= EPSILON
                   else ("long" if exposure > 0 else "short"))
        if side is None:
            side = derived
        side = require_enum("side", side, SIDES)
        if side != derived:
            raise SessionEvidenceError(
                f"side {side!r} contradicts signed_exposure "
                f"{exposure} (derived {derived!r})")
        mapping = require_enum("action_mapping", action_mapping,
                               ACTION_MAPPINGS)
        pending_size = require_real("pending_entry_size",
                                    pending_entry_size,
                                    nonnegative=True)
        orders = require_count("pending_orders", pending_orders,
                               minimum=0)
        protective = require_count("protective_orders",
                                   protective_orders, minimum=0)
        if pending_entry_side is not None:
            pending_entry_side = require_enum(
                "pending_entry_side", pending_entry_side,
                ("long", "short"))
            if pending_size <= 0.0:
                raise SessionEvidenceError(
                    "pending_entry_side declared with zero size")
            if orders < 1:
                raise SessionEvidenceError(
                    "pending_entry_side declared with zero pending "
                    "orders")
        return ExposureFacts(signed_exposure=exposure, side=side,
                             pending_entry_side=pending_entry_side,
                             pending_entry_size=pending_size,
                             pending_orders=orders,
                             action_mapping=mapping,
                             protective_orders=protective)

    @property
    def has_position(self) -> bool:
        return abs(self.signed_exposure) > EPSILON

    @property
    def entry_orders(self) -> int:
        """Pending orders that would INCREASE risk. Protective
        reduce-only brackets are excluded: native SL/TP protection is
        never cancelled by the weekly overlay (F3)."""
        return max(0, self.pending_orders - self.protective_orders)


def classify_discrete_command(command: Any,
                              exposure: ExposureFacts
                              ) -> dict[str, Any]:
    """G1: classify a DISCRETE COMMAND against signed exposure.

    The discrete domain carries a fixed position size, so enlargement
    and reduction are not expressible in it and never appear; a
    directional command that agrees with the open side is a HOLD of
    that side, not an enlargement. Anything outside
    ``DISCRETE_COMMANDS`` REFUSES rather than defaulting to hold."""
    if exposure.action_mapping != "discrete_command_v1":
        raise SessionEvidenceError(
            f"classify_discrete_command requires the "
            f"discrete_command_v1 mapping, got "
            f"{exposure.action_mapping!r}")
    if isinstance(command, bool) or not isinstance(command, int):
        raise SessionEvidenceError(
            f"discrete command must be an int, got "
            f"{type(command).__name__} {command!r}")
    if command not in DISCRETE_COMMANDS:
        raise SessionEvidenceError(
            f"unknown discrete command {command!r}; allowed "
            f"{sorted(DISCRETE_COMMANDS)}")
    current = exposure.signed_exposure
    name = DISCRETE_COMMANDS[command]
    if name == "close":
        kind = "close" if exposure.has_position else "hold_flat"
        return {"kind": kind, "risk_increasing": False,
                "command": command, "command_name": name,
                "current": current}
    if name == "hold":
        kind = "hold" if exposure.has_position else "hold_flat"
        return {"kind": kind, "risk_increasing": False,
                "command": command, "command_name": name,
                "current": current}
    wanted = 1.0 if name == "long" else -1.0
    if not exposure.has_position:
        return {"kind": "entry_from_flat", "risk_increasing": True,
                "command": command, "command_name": name,
                "current": current}
    if (wanted > 0) != (current > 0):
        return {"kind": "reversal", "risk_increasing": True,
                "command": command, "command_name": name,
                "current": current}
    return {"kind": "hold", "risk_increasing": False,
            "command": command, "command_name": name,
            "current": current}


def classify_action(raw_action: Any,
                    exposure: ExposureFacts) -> dict[str, Any]:
    """C1: classify a TARGET-exposure action against signed facts.

    Risk increases when absolute exposure grows OR the sign reverses
    OR an entry is opened from flat. Reduction and explicit close are
    always legal. Ambiguous or non-finite actions REFUSE."""
    if exposure.action_mapping != "target_exposure_v2":
        raise SessionEvidenceError(
            f"unsupported action mapping "
            f"{exposure.action_mapping!r}")
    if raw_action is None:
        raise SessionEvidenceError(
            "action is unavailable — ambiguous actions refuse")
    if isinstance(raw_action, bool) or not isinstance(
            raw_action, (int, float)):
        raise SessionEvidenceError(
            f"action must be a finite real target, got "
            f"{type(raw_action).__name__} {raw_action!r}")
    target = float(raw_action)
    if not math.isfinite(target):
        raise SessionEvidenceError(
            f"non-finite action {raw_action!r} refuses")
    current = exposure.signed_exposure
    if abs(target) <= EPSILON:
        kind = "close" if exposure.has_position else "hold_flat"
        return {"kind": kind, "risk_increasing": False,
                "target": target, "current": current}
    if not exposure.has_position:
        return {"kind": "entry_from_flat", "risk_increasing": True,
                "target": target, "current": current}
    if (target > 0) != (current > 0):
        return {"kind": "reversal", "risk_increasing": True,
                "target": target, "current": current}
    if abs(target) > abs(current) + EPSILON:
        return {"kind": "enlargement", "risk_increasing": True,
                "target": target, "current": current}
    if abs(target) < abs(current) - EPSILON:
        return {"kind": "reduction", "risk_increasing": False,
                "target": target, "current": current}
    return {"kind": "hold", "risk_increasing": False,
            "target": target, "current": current}


# ---------------------------------------------------------------- #
# state machine                                                     #
# ---------------------------------------------------------------- #

def session_state(policy: dict, *, now: Any,
                  calendar: Optional[SessionCalendar],
                  reopen_evidence: Optional[ReopenEvidence] = None,
                  expected_venue: Optional[str] = None,
                  expected_account_fingerprint: Optional[str] = None,
                  expected_symbol: Optional[str] = None
                  ) -> dict[str, Any]:
    """The five-state machine as a pure function over VALIDATED
    values. ``calendar=None`` means session evidence is unavailable:
    it fails closed for new entries."""
    now = require_utc("now", now)
    if not policy["enabled"]:
        return {"state": "NORMAL_TRADING", "policy_enabled": False,
                "evidence_ok": calendar is not None,
                "time_to_next_close_hours": None,
                "time_since_reopen_hours": None,
                "wind_down": False, "forced_flatten": False}
    if calendar is None:
        return {"state": "WIND_DOWN", "policy_enabled": True,
                "evidence_ok": False,
                "evidence_failed_closed": True,
                "time_to_next_close_hours": None,
                "time_since_reopen_hours": None,
                "wind_down": True, "forced_flatten": False}
    # F2: the policy's calendar identity MUST equal the calendar's
    # digest before any state derives from it. A valid but WRONG
    # calendar can never govern the strategy.
    if not isinstance(calendar, SessionCalendar):
        raise SessionEvidenceError(
            "calendar must be a validated SessionCalendar")
    if policy["calendar_identity"] != calendar.calendar_digest:
        raise SessionEvidenceError(
            f"calendar identity mismatch: policy declares "
            f"{policy['calendar_identity']!r} but the calendar "
            f"carries {calendar.calendar_digest!r} — cross-calendar "
            "substitution refuses")
    for label, expected, actual in (
            ("venue", expected_venue, calendar.venue),
            ("account_fingerprint", expected_account_fingerprint,
             calendar.account_fingerprint),
            ("symbol", expected_symbol, calendar.symbol)):
        if expected is not None and expected != actual:
            raise SessionEvidenceError(
                f"{label} mismatch: adapter expects {expected!r}, "
                f"calendar carries {actual!r} — cross-{label} "
                "substitution refuses")
    if reopen_evidence is not None and not isinstance(
            reopen_evidence, ReopenEvidence):
        raise SessionEvidenceError(
            "reopen_evidence must be a validated ReopenEvidence")
    base = {"policy_enabled": True, "evidence_ok": True,
            "calendar_identity": calendar.calendar_digest,
            "venue": calendar.venue, "symbol": calendar.symbol,
            "account_fingerprint": calendar.account_fingerprint}
    current = calendar.current_closure(now)
    if current is not None:
        return {**base, "state": "EXPECTED_MARKET_CLOSED",
                "time_to_next_close_hours": 0.0,
                "time_since_reopen_hours": None,
                "closure_started_at": current[0].isoformat(),
                "closure_reopens_at": current[1].isoformat(),
                "wind_down": False, "forced_flatten": False}
    upcoming = calendar.next_closure(now)
    hours_to_close = (None if upcoming is None else
                      (upcoming[0] - now).total_seconds() / 3600.0)
    # C2: blackout identity derives from the BOUND interval set
    last_reopen = calendar.most_recent_reopen(now)
    since_reopen = (None if last_reopen is None else
                    (now - last_reopen).total_seconds() / 3600.0)
    if last_reopen is not None:
        if reopen_evidence is None:
            # missing reopen evidence after a KNOWN closure fails
            # closed: no entries until direct evidence exists
            return {**base, "state": "REOPEN_BLACKOUT",
                    "evidence_failed_closed": True,
                    "time_to_next_close_hours": (
                        None if hours_to_close is None
                        else round(hours_to_close, 6)),
                    "time_since_reopen_hours": round(since_reopen, 6),
                    "reopen_evidence_missing": True,
                    "wind_down": False, "forced_flatten": False}
        hint = reopen_evidence.hint_time_since_reopen_hours
        hint_disagrees = (hint is not None
                          and abs(hint - since_reopen) > 1.0)
        blackout = (
            since_reopen < policy["reopen_min_hours"]
            or reopen_evidence.closed_bars_since_reopen
            < policy["reopen_min_closed_bars"]
            or reopen_evidence.stability_checks_passed
            < policy["stability_consecutive_checks"])
        if blackout:
            return {**base, "state": "REOPEN_BLACKOUT",
                    "time_to_next_close_hours": (
                        None if hours_to_close is None
                        else round(hours_to_close, 6)),
                    "time_since_reopen_hours": round(since_reopen, 6),
                    "closed_bars_since_reopen":
                        reopen_evidence.closed_bars_since_reopen,
                    "stability_checks_passed":
                        reopen_evidence.stability_checks_passed,
                    "adapter_hint_disagrees": hint_disagrees,
                    "wind_down": False, "forced_flatten": False}
    if hours_to_close is not None:
        if hours_to_close <= policy["forced_flatten_hours"]:
            return {**base, "state": "FORCED_FLATTEN",
                    "time_to_next_close_hours": round(
                        hours_to_close, 6),
                    "time_since_reopen_hours": (
                        None if since_reopen is None
                        else round(since_reopen, 6)),
                    "wind_down": True, "forced_flatten": True}
        if hours_to_close <= policy["wind_down_hours"]:
            return {**base, "state": "WIND_DOWN",
                    "time_to_next_close_hours": round(
                        hours_to_close, 6),
                    "time_since_reopen_hours": (
                        None if since_reopen is None
                        else round(since_reopen, 6)),
                    "wind_down": True, "forced_flatten": False}
    return {**base, "state": "NORMAL_TRADING",
            "time_to_next_close_hours": (
                None if hours_to_close is None
                else round(hours_to_close, 6)),
            "time_since_reopen_hours": (
                None if since_reopen is None
                else round(since_reopen, 6)),
            "wind_down": False, "forced_flatten": False}


def overlay_action(policy: dict, state_block: dict,
                   exposure: ExposureFacts,
                   raw_action: Any, *,
                   classification: Optional[dict] = None
                   ) -> dict[str, Any]:
    """Deterministic action overlay recording raw action, mapped
    classification, overlay decision and final command SEPARATELY.

    ``classification`` lets a caller in a DIFFERENT action domain
    supply its own already-mapped classification (see
    classify_discrete_command) so the state-driven policy below stays
    the single authority instead of being reimplemented per domain.
    It is validated, never trusted blindly."""
    state = require_enum("state_block.state", state_block.get("state"),
                         STATES)
    if state == "EXPECTED_MARKET_CLOSED":
        # no actionable step exists; the action is not even mapped
        return {"raw_model_action": raw_action,
                "mapped_action": None, "session_state": state,
                "overlay": "no_actionable_step",
                "cancel_pending": False, "final_action": None}
    if classification is None:
        classification = classify_action(raw_action, exposure)
    else:
        if not isinstance(classification, dict):
            raise SessionEvidenceError(
                "classification must be a mapping")
        require_enum("classification.kind",
                     classification.get("kind"),
                     ("close", "hold_flat", "entry_from_flat",
                      "reversal", "enlargement", "reduction", "hold"))
        if not isinstance(classification.get("risk_increasing"), bool):
            raise SessionEvidenceError(
                "classification.risk_increasing must be a bool")
    decision = {"raw_model_action": float(raw_action),
                "mapped_action": classification,
                "session_state": state,
                "overlay": "pass_through",
                "cancel_pending": False,
                "final_action": float(raw_action)}
    if not policy["enabled"] or state == "NORMAL_TRADING":
        return decision
    if state == "WIND_DOWN":
        if policy["cancel_pending_on_wind_down"] and \
                exposure.entry_orders > 0:
            decision["cancel_pending"] = True
            decision["cancel_scope"] = "pending_entry_orders_only"
        if classification["risk_increasing"]:
            decision["overlay"] = "masked_risk_increase"
            decision["final_action"] = 0.0 if not \
                exposure.has_position else exposure.signed_exposure
        return decision
    if state == "FORCED_FLATTEN":
        decision["cancel_pending"] = exposure.entry_orders > 0
        if exposure.entry_orders > 0:
            decision["cancel_scope"] = "pending_entry_orders_only"
        if exposure.has_position:
            decision["overlay"] = "forced_close"
            decision["final_action"] = "CLOSE"
        elif classification["risk_increasing"]:
            decision["overlay"] = "masked_risk_increase"
            decision["final_action"] = 0.0
        return decision
    if state == "REOPEN_BLACKOUT":
        if classification["risk_increasing"]:
            decision["overlay"] = "masked_entry_during_blackout"
            decision["final_action"] = 0.0 if not \
                exposure.has_position else exposure.signed_exposure
        return decision
    raise SessionPolicyError(f"unhandled state {state}")


def reconciliation_gate(positions_total: Any, orders_total: Any,
                        evidence_age_seconds: Any,
                        max_age_seconds: Any = 120.0
                        ) -> dict[str, Any]:
    """FORCED_FLATTEN success requires FRESH DIRECT venue evidence of
    zero positions AND zero orders. Unavailable evidence is a TYPED
    refusal, never a raw TypeError and never a success."""
    positions = require_count("positions_total", positions_total,
                              minimum=0)
    orders = require_count("orders_total", orders_total, minimum=0)
    age = require_real("evidence_age_seconds", evidence_age_seconds,
                       nonnegative=True)
    limit = require_real("max_age_seconds", max_age_seconds,
                         positive=True)
    fresh = age <= limit
    flat = positions == 0 and orders == 0
    return {"flat_confirmed": bool(fresh and flat), "fresh": fresh,
            "positions": positions, "orders": orders,
            "incident": (None if fresh and flat else
                         "FORCED_FLATTEN_FAILED: "
                         + ("stale evidence" if not fresh
                            else "exposure remains"))}


# ---------------------------------------------------------------- #
# C3: watchdog with truthful precedence                             #
# ---------------------------------------------------------------- #

WATCHDOG_STATES = (
    "EXPECTED_MARKET_CLOSED", "FEED_STALE_DURING_OPEN_WINDOW",
    "TERMINAL_DISCONNECTED", "SESSION_EVIDENCE_UNAVAILABLE",
    "WIND_DOWN_EXPOSURE_PRESENT", "FORCED_FLATTEN_FAILED",
    "REOPEN_BLACKOUT_ACTIVE", "TRADING_SESSION_HEALTHY",
    "CARRIED_POSITION_RECOVERY_ACTIVE",
    "UNEXPECTED_EXPOSURE_DURING_CLOSURE",
    "ACCOUNT_OR_BRACKET_FAULT")


@dataclass(frozen=True)
class CarriedPositionMigration:
    """ONE-USE record for exposure that predates the policy. It never
    normalizes future weekend exposure (C3)."""

    migration_id: str
    venue: str
    account_fingerprint: str
    symbol: str
    position_identity: str
    opened_before: datetime
    covers_closure_started_at: datetime
    native_protection_confirmed: bool

    def __post_init__(self):
        for name in ("migration_id", "venue", "account_fingerprint",
                     "symbol", "position_identity"):
            require_identity(name, getattr(self, name))
        require_utc("opened_before", self.opened_before)
        require_utc("covers_closure_started_at",
                    self.covers_closure_started_at)
        if not isinstance(self.native_protection_confirmed, bool):
            raise SessionEvidenceError(
                "native_protection_confirmed must be a bool")
        if self.opened_before > self.covers_closure_started_at:
            raise SessionEvidenceError(
                "a carried position must predate the closure it is "
                "migrated across")

    @staticmethod
    def build(*, migration_id: str, venue: str,
              account_fingerprint: str, symbol: str,
              position_identity: str, opened_before: Any,
              covers_closure_started_at: Any,
              native_protection_confirmed: bool
              ) -> "CarriedPositionMigration":
        return CarriedPositionMigration(
            migration_id=migration_id, venue=venue,
            account_fingerprint=account_fingerprint, symbol=symbol,
            position_identity=position_identity,
            opened_before=require_utc("opened_before", opened_before),
            covers_closure_started_at=require_utc(
                "covers_closure_started_at",
                covers_closure_started_at),
            native_protection_confirmed=native_protection_confirmed)


def watchdog_state(state_block: dict, *, bars_fresh: bool,
                   terminal_connected: bool,
                   exposure: ExposureFacts,
                   brackets_ok: bool = True,
                   account_ok: bool = True,
                   flatten_incident: Optional[str] = None,
                   recovery_claim_active: bool = False,
                   now: Any = None) -> str:
    """Expected closure suppresses ONLY bar staleness. Terminal,
    account, bracket, pending-order and exposure-policy incidents
    take precedence (C3).

    STRICTLY READ-ONLY (D1): ``recovery_claim_active`` is a FACT read
    from durable custody (``MigrationCustody.is_active``), never an
    authorization performed here."""
    if not isinstance(bars_fresh, bool) or not isinstance(
            terminal_connected, bool):
        raise SessionEvidenceError(
            "bars_fresh and terminal_connected must be bools")
    if not terminal_connected:
        return "TERMINAL_DISCONNECTED"
    if not state_block.get("evidence_ok", True):
        return "SESSION_EVIDENCE_UNAVAILABLE"
    if not account_ok or not brackets_ok:
        return "ACCOUNT_OR_BRACKET_FAULT"
    if flatten_incident:
        return "FORCED_FLATTEN_FAILED"
    state = require_enum("state_block.state", state_block.get("state"),
                         STATES)
    exposed = exposure.has_position or exposure.pending_orders > 0
    if state == "EXPECTED_MARKET_CLOSED":
        if exposed:
            # D1: the watchdog OBSERVES; it never authorizes. The
            # recovery controller claims durable custody BEFORE any
            # action, and this read simply reports the resulting
            # state — repeatable and non-mutating.
            if not isinstance(recovery_claim_active, bool):
                raise SessionEvidenceError(
                    "recovery_claim_active must be a bool")
            if recovery_claim_active:
                return "CARRIED_POSITION_RECOVERY_ACTIVE"
            return "UNEXPECTED_EXPOSURE_DURING_CLOSURE"
        return "EXPECTED_MARKET_CLOSED"
    if exposed and state in ("WIND_DOWN", "FORCED_FLATTEN"):
        return "WIND_DOWN_EXPOSURE_PRESENT"
    if not bars_fresh:
        return "FEED_STALE_DURING_OPEN_WINDOW"
    if state == "REOPEN_BLACKOUT":
        return "REOPEN_BLACKOUT_ACTIVE"
    return "TRADING_SESSION_HEALTHY"
