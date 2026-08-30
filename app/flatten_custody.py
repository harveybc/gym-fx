"""Durable custody for a pending FLATTEN OBLIGATION (D1-D3).

An obligation to close exposure before a session boundary must not be
able to disappear, to be silently altered, or to be certified by
something that never observed the exposure it names.

**One protocol.** Every write -- creation and every transition alike
-- goes through the same durable path: symlink refusal on root,
record, temporary and lock; a temporary created with ``O_EXCL`` at
``0600``; write, ``fchmod``, ``fsync``; atomic rename; ``fsync`` of
the parent directory. Transitions are elected by an exclusive lock
and re-verify the expected state UNDER that lock. A failing ``fsync``
is never acknowledged as a transition. The root is ``0700``, records,
temporaries and locks are ``0600``, whatever the ambient umask.

**Integrity.** Every record carries a digest over its own content,
recomputed and verified on EVERY read. A record whose digest does not
match is not returned as data: it is a typed refusal, and it counts
as an unresolved obligation rather than as a discharge.

**Economic validity.** A confirmation must come from the SAME episode
identity that opened the obligation. A fresh simulated account is
born flat, and that emptiness says nothing about whether the previous
exposure was ever closed; it is refused. An episode abandoned while
an obligation is open terminates as ``interrupted_unresolved`` --
recorded, terminal, and explicitly NOT a close.

**Multiplicity.** Several open obligations under one root have no
safe automatic resolution, so the store refuses to choose. That is an
operator disposition, not a silent pick of the last one.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OPEN_STATES = ("flatten_requested", "flatten_in_flight")
TERMINAL_STATES = ("flatten_confirmed", "flatten_failed",
                   "interrupted_unresolved")
SUCCESS_STATES = ("flatten_confirmed",)
STATES = OPEN_STATES + TERMINAL_STATES

ROOT_MODE = 0o700
FILE_MODE = 0o600
DIGEST_FIELD = "record_digest"

IDENTITY_FIELDS = ("venue", "account_fingerprint", "symbol",
                   "position_identity", "episode_identity",
                   "code_identity")


class FlattenObligationError(RuntimeError):
    """A durable flatten obligation was misused — typed refusal."""


class FlattenIntegrityError(FlattenObligationError):
    """A record does not match its own digest — typed refusal."""


class FlattenDispositionRequired(FlattenObligationError):
    """Several open obligations: only an operator may dispose."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != DIGEST_FIELD}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def _refuse_symlink(path: Path, what: str) -> None:
    if path.is_symlink():
        raise FlattenObligationError(
            f"{what} {path.name}: symlinked path refused")


def _pending_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".unacknowledged")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_write(path: Path, payload: dict, *,
                   exclusive: bool) -> None:
    """The ONE write protocol. Any failure -- including a failing
    fsync of the file or of the parent directory -- leaves the final
    path untouched, so a transition is never acknowledged unless it
    reached stable storage."""
    _refuse_symlink(path.parent, "custody root")
    _refuse_symlink(path, "record")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    _refuse_symlink(tmp, "temporary")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                     FILE_MODE)
    except FileExistsError as exc:
        raise FlattenObligationError(
            f"{tmp.name}: concurrent write in progress") from exc
    try:
        os.write(fd, json.dumps(payload, indent=1,
                                default=str).encode())
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    os.close(fd)
    if exclusive:
        try:
            claim = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            FILE_MODE)
        except FileExistsError as exc:
            os.unlink(tmp)
            raise FlattenObligationError(
                f"{path.name}: already claimed") from exc
        os.close(claim)
    # An UNACKNOWLEDGED marker is planted BEFORE the rename and
    # removed only after the parent directory fsync succeeds. The
    # rename itself makes new content visible immediately, so without
    # this a failing directory fsync -- or a crash between the rename
    # and that fsync -- would leave a transition that looks complete
    # but was never made durable. While the marker exists the record
    # is refused on every read.
    pending = _pending_path(path)
    _refuse_symlink(pending, "unacknowledged marker")
    try:
        marker_fd = os.open(pending, os.O_WRONLY | os.O_CREAT |
                            os.O_EXCL, FILE_MODE)
    except FileExistsError as exc:
        os.unlink(tmp)
        raise FlattenObligationError(
            f"{path.name}: a previous write was never acknowledged; "
            "the record is not trustworthy") from exc
    try:
        os.write(marker_fd, str(os.getpid()).encode())
        os.fchmod(marker_fd, FILE_MODE)
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)
    try:
        _fsync_dir(path.parent)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    os.unlink(pending)
    _fsync_dir(path.parent)


class FlattenObligationStore:
    """One directory, one record per obligation identity."""

    def __init__(self, root: Any):
        self.root = Path(root)
        _refuse_symlink(self.root, "custody root")
        self.root.mkdir(parents=True, exist_ok=True, mode=ROOT_MODE)
        # mkdir honours the umask, so 0700 is ENFORCED afterwards
        os.chmod(self.root, ROOT_MODE)

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

    def _lock_path(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".lock")

    # -- lifecycle --------------------------------------------------
    def open_obligation(self, obligation_id: str, *, venue: str,
                        account_fingerprint: str, symbol: str,
                        position_identity: str,
                        episode_identity: str,
                        signed_exposure: float,
                        requested_at_bar: int,
                        code_identity: str,
                        checkpoint_identity: Optional[str] = None,
                        now: Optional[str] = None) -> dict:
        """Durably record a NEW obligation bound to a FULL identity."""
        values = {"venue": venue,
                  "account_fingerprint": account_fingerprint,
                  "symbol": symbol,
                  "position_identity": position_identity,
                  "episode_identity": episode_identity,
                  "code_identity": code_identity}
        for name, value in values.items():
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
            **values,
            "checkpoint_identity": checkpoint_identity,
            "signed_exposure_at_request": float(signed_exposure),
            "requested_at_bar": int(requested_at_bar),
            "opened_at": now or _utc_now(),
            "confirmed_at_bar": None,
            "reconciliation": None,
            "incident": None,
        }
        record[DIGEST_FIELD] = _digest(record)
        _durable_write(self._path(obligation_id), record,
                       exclusive=True)
        return record

    def mark_in_flight(self, obligation_id: str, *,
                       bar_index: int,
                       episode_identity: str) -> dict:
        return self._transition(
            obligation_id, expected=OPEN_STATES,
            episode_identity=episode_identity,
            require_same_episode=True,
            changes={"state": "flatten_in_flight",
                     "in_flight_at_bar": int(bar_index)})

    def confirm(self, obligation_id: str, *, reconciliation: dict,
                bar_index: int, episode_identity: str) -> dict:
        """D1: only the episode that OPENED the obligation may
        confirm it, and only with evidence that says flat.

        A fresh simulated account is born with zero positions and zero
        orders. That emptiness is a fact about the NEW episode and
        says nothing about whether the exposure named here was ever
        closed, so it can never discharge this obligation."""
        if not isinstance(reconciliation, dict) or \
                reconciliation.get("flat_confirmed") is not True:
            raise FlattenObligationError(
                f"{obligation_id}: confirmation requires evidence "
                f"with flat_confirmed=True, got {reconciliation!r}")
        return self._transition(
            obligation_id, expected=OPEN_STATES,
            episode_identity=episode_identity,
            require_same_episode=True,
            changes={"state": "flatten_confirmed",
                     "confirmed_at_bar": int(bar_index),
                     "reconciliation": dict(reconciliation),
                     "incident": None,
                     "closed_at": _utc_now()})

    def fail(self, obligation_id: str, *, incident: str,
             episode_identity: str) -> dict:
        return self._transition(
            obligation_id, expected=OPEN_STATES,
            episode_identity=episode_identity,
            changes={"state": "flatten_failed",
                     "incident": str(incident),
                     "closed_at": _utc_now()})

    def interrupt(self, obligation_id: str, *, reason: str,
                  episode_identity: str) -> dict:
        """D1: the episode was abandoned with the obligation open.

        Terminal and recorded, but explicitly NOT a close: nothing
        here claims the exposure was flattened. This transition is
        deliberately allowed from a DIFFERENT episode, because
        recording that an obligation was abandoned is not a claim
        about the exposure."""
        return self._transition(
            obligation_id, expected=OPEN_STATES,
            episode_identity=episode_identity,
            changes={"state": "interrupted_unresolved",
                     "incident": str(reason),
                     "closed_at": _utc_now(),
                     "closure_claimed": False})

    # -- reads ------------------------------------------------------
    def read(self, obligation_id: str) -> dict:
        """Verified read. A record that does not match its own digest
        is a typed refusal, never returned as data."""
        path = self._path(obligation_id)
        _refuse_symlink(self.root, "custody root")
        _refuse_symlink(path, "record")
        if not path.is_file():
            raise FlattenObligationError(
                f"{obligation_id}: no such obligation")
        if _pending_path(path).exists():
            raise FlattenIntegrityError(
                f"{obligation_id}: a write was never acknowledged "
                "(unacknowledged marker present) — the record is not "
                "trustworthy and the obligation stays unresolved")
        try:
            record = json.loads(path.read_text())
        except Exception as exc:
            raise FlattenIntegrityError(
                f"{obligation_id}: record is unreadable: {exc}") \
                from exc
        if not isinstance(record, dict):
            raise FlattenIntegrityError(
                f"{obligation_id}: record is not an object")
        stored = record.get(DIGEST_FIELD)
        expected = _digest(record)
        if stored != expected:
            raise FlattenIntegrityError(
                f"{obligation_id}: record digest mismatch — stored "
                f"{str(stored)[:12]}… but the content hashes to "
                f"{expected[:12]}… — the record was altered")
        if record.get("state") not in STATES:
            raise FlattenIntegrityError(
                f"{obligation_id}: unknown state "
                f"{record.get('state')!r}")
        return record

    def outstanding(self) -> tuple:
        """Every obligation NOT in a terminal state.

        A record that cannot be read, or whose digest does not match,
        counts as OUTSTANDING: a damaged or altered record is not
        evidence that the obligation was discharged. Symlinked
        records are refused rather than followed."""
        _refuse_symlink(self.root, "custody root")
        found = []
        for path in sorted(self.root.glob("*.json")):
            if path.is_symlink():
                found.append({"obligation_id": path.stem,
                              "state": "integrity_failed",
                              "incident": "symlinked record refused"})
                continue
            try:
                record = self.read(path.stem)
            except FlattenObligationError as exc:
                found.append({"obligation_id": path.stem,
                              "state": "integrity_failed",
                              "incident": str(exc)})
                continue
            if record.get("state") in OPEN_STATES:
                found.append(record)
        return tuple(found)

    def require_single_open(self) -> Optional[dict]:
        """D3: zero or one open obligation has a defined meaning.
        Several do NOT, and picking the last one silently is exactly
        how the earlier ones became irresolvable."""
        open_records = self.outstanding()
        if not open_records:
            return None
        if len(open_records) > 1:
            raise FlattenDispositionRequired(
                f"{len(open_records)} open flatten obligations under "
                f"{self.root}: "
                f"{[r.get('obligation_id') for r in open_records]} — "
                "no automatic resolution exists; an operator must "
                "dispose of them")
        return open_records[0]

    # -- internals --------------------------------------------------
    def _transition(self, obligation_id: str, *, expected: tuple,
                    episode_identity: str, changes: dict,
                    require_same_episode: bool = False) -> dict:
        if not isinstance(episode_identity, str) or \
                not episode_identity.strip():
            raise FlattenObligationError(
                f"episode_identity must be a nonempty string, got "
                f"{episode_identity!r}")
        path = self._path(obligation_id)
        lock = self._lock_path(path)
        _refuse_symlink(lock, "lock")
        try:
            lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT |
                              os.O_EXCL, FILE_MODE)
        except FileExistsError as exc:
            raise FlattenObligationError(
                f"{obligation_id}: a competing transition holds the "
                "lock — the winner is preserved, never overwritten") \
                from exc
        try:
            os.write(lock_fd, str(os.getpid()).encode())
            os.fchmod(lock_fd, FILE_MODE)
            os.fsync(lock_fd)
        finally:
            os.close(lock_fd)
        try:
            # expected state and integrity re-verified UNDER the lock
            record = self.read(obligation_id)
            if record["state"] in TERMINAL_STATES:
                raise FlattenObligationError(
                    f"{obligation_id}: already terminal in state "
                    f"{record['state']!r} — a terminal obligation is "
                    "immutable")
            if record["state"] not in expected:
                raise FlattenObligationError(
                    f"{obligation_id}: expected one of {list(expected)}"
                    f" but found {record['state']!r}")
            if require_same_episode and \
                    record["episode_identity"] != episode_identity:
                raise FlattenObligationError(
                    f"{obligation_id}: opened by episode "
                    f"{record['episode_identity']!r} and cannot be "
                    f"advanced by episode {episode_identity!r} — a "
                    "different account state never observed the "
                    "exposure this obligation names")
            if changes.get("state") == record["state"]:
                return record
            payload = {**record, **changes,
                       "transitioned_by_episode": episode_identity}
            payload[DIGEST_FIELD] = _digest(payload)
            _durable_write(path, payload, exclusive=False)
            return payload
        finally:
            try:
                os.unlink(lock)
            except FileNotFoundError:
                pass
