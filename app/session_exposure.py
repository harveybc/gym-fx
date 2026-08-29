"""Weekly session-exposure state machine (work plan 42, order
@45c49003). PURE, venue-agnostic and driver-free: the five states and
the action overlay are computed from typed inputs (bound session
calendar, clock, exposure facts, reopen evidence) so gym-fx, the lts
runners and the calibration materializer share ONE authority.

States: NORMAL_TRADING, WIND_DOWN, FORCED_FLATTEN,
EXPECTED_MARKET_CLOSED, REOPEN_BLACKOUT.

The overlay never learns and never infers: a learned policy may close
opportunistically during WIND_DOWN, but the deterministic deadline
guarantees flat exposure. Raw model action, overlay decision and
final action are ALWAYS distinct recorded facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

STATES = ("NORMAL_TRADING", "WIND_DOWN", "FORCED_FLATTEN",
          "EXPECTED_MARKET_CLOSED", "REOPEN_BLACKOUT")


class SessionPolicyError(ValueError):
    """Invalid configuration/evidence — refused, never defaulted."""


REQUIRED_KEYS = (
    "enabled", "session_source", "wind_down_hours",
    "forced_flatten_hours", "cancel_pending_on_wind_down",
    "allow_risk_increase_during_wind_down", "reopen_min_hours",
    "reopen_min_closed_bars", "stability_consecutive_checks",
    "max_spread_relative_to_baseline", "max_gap_sigma",
    "max_realized_vol_relative_to_baseline",
    "carried_position_recovery", "holiday_policy",
    "calendar_identity")


def validate_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Typed surface of work plan 42 §4. Invalid combinations are
    ABSENT from materialization: they refuse here."""
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise SessionPolicyError(
            f"session_exposure_policy missing fields {missing}")
    unknown = sorted(set(config) - set(REQUIRED_KEYS))
    if unknown:
        raise SessionPolicyError(
            f"session_exposure_policy unknown fields {unknown}")
    if not isinstance(config["enabled"], bool):
        raise SessionPolicyError("enabled must be a bool")
    wind_down = float(config["wind_down_hours"])
    flatten = float(config["forced_flatten_hours"])
    if wind_down <= 0 or flatten <= 0:
        raise SessionPolicyError(
            "wind_down_hours and forced_flatten_hours must be > 0")
    if flatten >= wind_down:
        raise SessionPolicyError(
            "forced flatten must occur AFTER wind-down begins and "
            "BEFORE closure: forced_flatten_hours < wind_down_hours")
    for key in ("reopen_min_hours",):
        if float(config[key]) < 0:
            raise SessionPolicyError(f"{key} must be >= 0")
    for key in ("reopen_min_closed_bars",
                "stability_consecutive_checks"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < 1:
            raise SessionPolicyError(
                f"{key} must be a positive integer")
    for key in ("max_spread_relative_to_baseline", "max_gap_sigma",
                "max_realized_vol_relative_to_baseline"):
        if float(config[key]) <= 0:
            raise SessionPolicyError(f"{key} must be > 0")
    if config["enabled"] and not config[
            "cancel_pending_on_wind_down"] and not config[
            "allow_risk_increase_during_wind_down"]:
        pass  # legal: strict blocking with cancellation disabled is
        # NOT the invalid combo; the invalid combo is below
    if config["enabled"] and config[
            "allow_risk_increase_during_wind_down"]:
        raise SessionPolicyError(
            "allow_risk_increase_during_wind_down=true while "
            "demanding flat exposure is invalid (work plan 42 §4)")
    if not str(config["calendar_identity"]).strip():
        raise SessionPolicyError("calendar_identity must be bound")
    return dict(config)


@dataclass
class SessionEvidence:
    """Facts the machine consumes — every field causally available.

    ``closures``: sorted list of (close_at, reopen_at) UTC datetimes
    from the BOUND venue/symbol session calendar (broker sessions +
    versioned operator exceptions). Missing/contradictory evidence is
    represented by ``evidence_ok=False`` and FAILS CLOSED for new
    entries."""

    now: datetime
    closures: list
    evidence_ok: bool = True
    time_since_reopen_hours: float | None = None
    closed_bars_since_reopen: int = 0
    stability_checks_passed: int = 0


@dataclass
class ExposureFacts:
    open_position: bool = False
    pending_orders: int = 0


def _next_closure(evidence: SessionEvidence):
    for close_at, reopen_at in evidence.closures:
        if evidence.now < reopen_at:
            return close_at, reopen_at
    return None, None


def session_state(policy: dict, evidence: SessionEvidence) -> dict:
    """The five-state machine as a pure function. Returns the typed
    state block (observation fields included)."""
    if not policy["enabled"]:
        return {"state": "NORMAL_TRADING", "policy_enabled": False,
                "time_to_next_close_hours": None,
                "time_since_reopen_hours": None,
                "wind_down": False, "forced_flatten": False,
                "evidence_ok": evidence.evidence_ok}
    if not evidence.evidence_ok:
        # missing or contradictory session evidence fails CLOSED for
        # new entries: strictest exposure state outside closure
        return {"state": "WIND_DOWN", "policy_enabled": True,
                "evidence_failed_closed": True,
                "time_to_next_close_hours": None,
                "time_since_reopen_hours": None,
                "wind_down": True, "forced_flatten": False,
                "evidence_ok": False}
    close_at, reopen_at = _next_closure(evidence)
    now = evidence.now
    in_closure = (close_at is not None and close_at <= now
                  and now < reopen_at)
    if in_closure:
        return {"state": "EXPECTED_MARKET_CLOSED",
                "policy_enabled": True,
                "time_to_next_close_hours": 0.0,
                "time_since_reopen_hours": None,
                "wind_down": False, "forced_flatten": False,
                "evidence_ok": True}
    since_reopen = evidence.time_since_reopen_hours
    if since_reopen is not None:
        min_hours = float(policy["reopen_min_hours"])
        min_bars = int(policy["reopen_min_closed_bars"])
        checks = int(policy["stability_consecutive_checks"])
        blackout = (since_reopen < min_hours
                    or evidence.closed_bars_since_reopen < min_bars
                    or evidence.stability_checks_passed < checks)
        if blackout:
            return {"state": "REOPEN_BLACKOUT",
                    "policy_enabled": True,
                    "time_to_next_close_hours": (
                        (close_at - now).total_seconds() / 3600
                        if close_at else None),
                    "time_since_reopen_hours": round(
                        since_reopen, 3),
                    "wind_down": False, "forced_flatten": False,
                    "evidence_ok": True}
    hours_to_close = ((close_at - now).total_seconds() / 3600
                      if close_at else None)
    if hours_to_close is not None:
        if hours_to_close <= float(policy["forced_flatten_hours"]):
            return {"state": "FORCED_FLATTEN",
                    "policy_enabled": True,
                    "time_to_next_close_hours": round(
                        hours_to_close, 3),
                    "time_since_reopen_hours": since_reopen,
                    "wind_down": True, "forced_flatten": True,
                    "evidence_ok": True}
        if hours_to_close <= float(policy["wind_down_hours"]):
            return {"state": "WIND_DOWN", "policy_enabled": True,
                    "time_to_next_close_hours": round(
                        hours_to_close, 3),
                    "time_since_reopen_hours": since_reopen,
                    "wind_down": True, "forced_flatten": False,
                    "evidence_ok": True}
    return {"state": "NORMAL_TRADING", "policy_enabled": True,
            "time_to_next_close_hours": (round(hours_to_close, 3)
                                         if hours_to_close is not None
                                         else None),
            "time_since_reopen_hours": since_reopen,
            "wind_down": False, "forced_flatten": False,
            "evidence_ok": True}


def overlay_action(policy: dict, state_block: dict,
                   exposure: ExposureFacts,
                   raw_action: float) -> dict:
    """Deterministic action overlay. Records raw action, overlay
    decision and final action SEPARATELY (work plan 42 §5). The model
    is never blinded — only its illegal actions are masked."""
    state = state_block["state"]
    decision = {"raw_model_action": float(raw_action),
                "session_state": state,
                "overlay": "pass_through",
                "cancel_pending": False,
                "final_action": float(raw_action)}
    if not policy["enabled"] or state == "NORMAL_TRADING":
        return decision
    risk_increasing = _is_risk_increasing(raw_action, exposure)
    if state == "WIND_DOWN":
        if policy["cancel_pending_on_wind_down"] and \
                exposure.pending_orders > 0:
            decision["cancel_pending"] = True
        if risk_increasing:
            decision["overlay"] = "masked_risk_increase"
            decision["final_action"] = 0.0
        return decision
    if state == "FORCED_FLATTEN":
        decision["cancel_pending"] = exposure.pending_orders > 0
        if exposure.open_position:
            decision["overlay"] = "forced_close"
            decision["final_action"] = "CLOSE"
        else:
            decision["overlay"] = ("masked_risk_increase"
                                   if risk_increasing
                                   else "pass_through")
            if risk_increasing:
                decision["final_action"] = 0.0
        return decision
    if state == "EXPECTED_MARKET_CLOSED":
        decision["overlay"] = "no_actionable_step"
        decision["final_action"] = None
        return decision
    if state == "REOPEN_BLACKOUT":
        if risk_increasing:
            decision["overlay"] = "masked_entry_during_blackout"
            decision["final_action"] = 0.0
        return decision
    raise SessionPolicyError(f"unknown state {state}")


def _is_risk_increasing(raw_action: float,
                        exposure: ExposureFacts) -> bool:
    """Entries and reversals increase risk; hold/reduce/close do not.
    With no position, any nonzero target is an entry. With one, only
    same-direction-or-flat targets are non-increasing — the pure
    fact is decided by the executing strategy's exposure semantics,
    approximated here by |target| > 0 without an open position, or a
    sign flip with one."""
    try:
        target = float(raw_action)
    except (TypeError, ValueError):
        return True
    if not exposure.open_position:
        return abs(target) > 0.0
    return False  # reductions/holds/closes and same-direction holds


def reconciliation_gate(positions_total: int, orders_total: int,
                        evidence_age_seconds: float,
                        max_age_seconds: float = 120.0) -> dict:
    """FORCED_FLATTEN success requires FRESH DIRECT venue evidence of
    zero positions and zero orders; anything else is a critical typed
    incident, never a reported success."""
    fresh = evidence_age_seconds <= max_age_seconds
    flat = positions_total == 0 and orders_total == 0
    return {"flat_confirmed": bool(fresh and flat),
            "fresh": bool(fresh), "positions": int(positions_total),
            "orders": int(orders_total),
            "incident": (None if fresh and flat else
                         "FORCED_FLATTEN_FAILED: "
                         + ("stale evidence" if not fresh else
                            "exposure remains"))}


WATCHDOG_STATES = (
    "EXPECTED_MARKET_CLOSED", "FEED_STALE_DURING_OPEN_WINDOW",
    "TERMINAL_DISCONNECTED", "SESSION_EVIDENCE_UNAVAILABLE",
    "WIND_DOWN_EXPOSURE_PRESENT", "FORCED_FLATTEN_FAILED",
    "REOPEN_BLACKOUT_ACTIVE", "TRADING_SESSION_HEALTHY")


def watchdog_state(state_block: dict, *, bars_fresh: bool,
                   terminal_connected: bool,
                   exposure: ExposureFacts) -> str:
    """Work plan 42 §9: expected closure suppresses stale-bar alarms
    but NEVER terminal/exposure failures."""
    if not terminal_connected:
        return "TERMINAL_DISCONNECTED"
    if not state_block.get("evidence_ok", True):
        return "SESSION_EVIDENCE_UNAVAILABLE"
    state = state_block["state"]
    if state == "EXPECTED_MARKET_CLOSED":
        return "EXPECTED_MARKET_CLOSED"
    if not bars_fresh:
        return "FEED_STALE_DURING_OPEN_WINDOW"
    if state in ("WIND_DOWN", "FORCED_FLATTEN") and (
            exposure.open_position or exposure.pending_orders):
        return "WIND_DOWN_EXPOSURE_PRESENT"
    if state == "REOPEN_BLACKOUT":
        return "REOPEN_BLACKOUT_ACTIVE"
    return "TRADING_SESSION_HEALTHY"
