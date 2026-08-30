"""R3: durable custody for a pending FLATTEN OBLIGATION.

An obligation to close exposure before a session boundary must not be
able to disappear. The previous implementation held it only in
process memory, so ``reset()`` -- and a fortiori a process restart --
silently forgot an in-flight close and let the next episode start
clean. That is the opposite of recovery.

The obligation is therefore written durably at the moment it is
requested, using the SAME no-overwrite protocol the migration custody
already carries: ``O_EXCL`` creation, ``0600`` permissions, fsync of
file and parent directory, symlink refusal, and an exclusive
transition lock with expected-state identity for terminal moves.

After a restart only two outcomes are legal, and both are enforced
here rather than assumed by the caller:

* the pending close is resumed and VERIFIED against fresh evidence of
  zero positions and zero orders; or
* a typed RECOVERY state is entered which BLOCKS every risk increase
  until that evidence exists.

Forgetting the obligation is not among them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.migration_custody import (
    MigrationCustodyError, _atomic_exclusive_write,
    _atomic_terminal_transition, _sha_obj, _utc_now)

OPEN_STATES = ("flatten_requested", "flatten_in_flight")
TERMINAL_STATES = ("flatten_confirmed", "flatten_failed")
STATES = OPEN_STATES + TERMINAL_STATES


class FlattenObligationError(RuntimeError):
    """A durable flatten obligation was misused — typed refusal."""


class FlattenObligationStore:
    """One directory, one record per obligation identity."""

    def __init__(self, root: Any):
        self.root = Path(root)
        if self.root.is_symlink():
            raise FlattenObligationError(
                f"{self.root}: symlinked custody root refused")
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------
    def _path(self, obligation_id: str) -> Path:
        if not isinstance(obligation_id, str) or \
                not obligation_id.strip():
            raise FlattenObligationError(
                f"obligation id must be a nonempty string, got "
                f"{obligation_id!r}")
        if "/" in obligation_id or obligation_id.startswith("."):
            raise FlattenObligationError(
                f"unsafe obligation id {obligation_id!r}")
        return self.root / f"{obligation_id}.json"

    # -- lifecycle --------------------------------------------------
    def open_obligation(self, obligation_id: str, *, venue: str,
                        account_fingerprint: str, symbol: str,
                        position_identity: str,
                        signed_exposure: float,
                        requested_at_bar: int,
                        code_identity: str,
                        now: Optional[str] = None) -> dict:
        """Durably record a NEW obligation. ``O_EXCL`` on the final
        path means a second opener observes the first and refuses."""
        for name, value in (("venue", venue),
                            ("account_fingerprint",
                             account_fingerprint),
                            ("symbol", symbol),
                            ("position_identity", position_identity),
                            ("code_identity", code_identity)):
            if not isinstance(value, str) or not value.strip():
                raise FlattenObligationError(
                    f"{name} must be a nonempty string, got {value!r}")
        if isinstance(signed_exposure, bool) or not isinstance(
                signed_exposure, (int, float)):
            raise FlattenObligationError(
                f"signed_exposure must be a real number, got "
                f"{signed_exposure!r}")
        if isinstance(requested_at_bar, bool) or not isinstance(
                requested_at_bar, int) or requested_at_bar < 0:
            raise FlattenObligationError(
                f"requested_at_bar must be a nonnegative int, got "
                f"{requested_at_bar!r}")
        record = {
            "obligation_id": obligation_id,
            "state": "flatten_requested",
            "venue": venue,
            "account_fingerprint": account_fingerprint,
            "symbol": symbol,
            "position_identity": position_identity,
            "signed_exposure_at_request": float(signed_exposure),
            "requested_at_bar": int(requested_at_bar),
            "code_identity": code_identity,
            "opened_at": now or _utc_now(),
            "confirmed_at_bar": None,
            "reconciliation": None,
            "incident": None,
        }
        record["record_digest"] = _sha_obj(record)
        try:
            _atomic_exclusive_write(self._path(obligation_id), record)
        except MigrationCustodyError as exc:
            raise FlattenObligationError(
                f"{obligation_id}: {exc}") from exc
        return record

    def mark_in_flight(self, obligation_id: str, *,
                       bar_index: int) -> dict:
        record = self.read(obligation_id)
        if record["state"] == "flatten_in_flight":
            return record
        if record["state"] != "flatten_requested":
            raise FlattenObligationError(
                f"{obligation_id}: cannot go in flight from "
                f"{record['state']!r}")
        record = {**record, "state": "flatten_in_flight",
                  "in_flight_at_bar": int(bar_index)}
        self._overwrite_open(obligation_id, record)
        return record

    def confirm(self, obligation_id: str, *, reconciliation: dict,
                bar_index: int) -> dict:
        """Terminal, elected by an exclusive lock. A confirmation
        requires evidence that actually says flat; anything else is a
        refusal, never a quiet success."""
        if not isinstance(reconciliation, dict) or \
                reconciliation.get("flat_confirmed") is not True:
            raise FlattenObligationError(
                f"{obligation_id}: confirmation requires evidence "
                f"with flat_confirmed=True, got {reconciliation!r}")
        record = self.read(obligation_id)
        payload = {**record, "state": "flatten_confirmed",
                   "confirmed_at_bar": int(bar_index),
                   "reconciliation": dict(reconciliation),
                   "incident": None,
                   "closed_at": _utc_now()}
        self._terminal(obligation_id, record["state"], payload)
        return payload

    def fail(self, obligation_id: str, *, incident: str) -> dict:
        record = self.read(obligation_id)
        payload = {**record, "state": "flatten_failed",
                   "incident": str(incident),
                   "closed_at": _utc_now()}
        self._terminal(obligation_id, record["state"], payload)
        return payload

    # -- reads ------------------------------------------------------
    def read(self, obligation_id: str) -> dict:
        path = self._path(obligation_id)
        if not path.is_file():
            raise FlattenObligationError(
                f"{obligation_id}: no such obligation")
        return json.loads(path.read_text())

    def outstanding(self) -> tuple:
        """Every obligation that has NOT reached a terminal state.

        This is what a restart reads. An unreadable record counts as
        outstanding: a record that cannot be parsed is not evidence
        that the obligation was discharged."""
        found = []
        for path in sorted(self.root.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except Exception:
                found.append({"obligation_id": path.stem,
                              "state": "unreadable",
                              "incident": "record could not be read"})
                continue
            if record.get("state") in OPEN_STATES:
                found.append(record)
        return tuple(found)

    # -- internals --------------------------------------------------
    def _overwrite_open(self, obligation_id: str, record: dict
                        ) -> None:
        path = self._path(obligation_id)
        if path.is_symlink():
            raise FlattenObligationError(
                f"{obligation_id}: symlinked record refused")
        tmp = path.with_suffix(".json.upd")
        tmp.write_text(json.dumps(record, indent=1, default=str))
        tmp.replace(path)

    def _terminal(self, obligation_id: str, expected: str,
                  payload: dict) -> None:
        # A terminal record is IMMUTABLE. Passing the record's own
        # current state as the expected state let a confirmed
        # obligation be overwritten by a later fail(), because the
        # expected-state check trivially matched itself.
        if expected in TERMINAL_STATES:
            raise FlattenObligationError(
                f"{obligation_id}: already terminal in state "
                f"{expected!r} — a terminal obligation is immutable")
        if expected not in OPEN_STATES:
            raise FlattenObligationError(
                f"{obligation_id}: cannot close from unknown state "
                f"{expected!r}")
        path = self._path(obligation_id)
        try:
            _atomic_terminal_transition(path, expected, payload)
        except MigrationCustodyError as exc:
            raise FlattenObligationError(
                f"{obligation_id}: {exc}") from exc
        # migration_custody releases its lock only for ITS terminal
        # vocabulary; release ours explicitly once the record really
        # is terminal, so a stray lock cannot outlive the transition.
        if json.loads(path.read_text()).get("state") in \
                TERMINAL_STATES:
            lock = path.with_suffix(path.suffix + ".terminal.lock")
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
