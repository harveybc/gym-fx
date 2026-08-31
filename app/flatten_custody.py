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


class FlattenTransitionObserved(FlattenObligationError):
    """F3: the transition was already WON by another process. The
    caller OBSERVED the winner; it did not win, and nothing was
    overwritten. The winner's record rides on the exception."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


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


def _ack_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".ack")


ACK_PENDING = b"PENDING"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _set_ack(ack: Path, value: bytes) -> None:
    """Update an EXISTING acknowledgement IN PLACE.

    An in-place content update needs only a file fsync: the directory
    entry does not change, so no parent fsync is involved and there is
    no later step whose failure could leave the update unacknowledged.
    That is precisely why the acknowledgement lives in its own file
    instead of in the record."""
    fd = os.open(ack, os.O_WRONLY | os.O_TRUNC, FILE_MODE)
    try:
        os.write(fd, value)
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_ack(ack: Path) -> None:
    _refuse_symlink(ack, "acknowledgement")
    if ack.exists():
        return
    fd = os.open(ack, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        os.write(fd, ACK_PENDING)
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(ack.parent)


def read_ack(path: Path) -> bytes:
    ack = _ack_path(path)
    _refuse_symlink(ack, "acknowledgement")
    if not ack.is_file():
        return b""
    return ack.read_bytes().strip()


def _durable_write(path: Path, payload: dict, *,
                   exclusive: bool) -> None:
    """The ONE write protocol, with a MONOTONE acknowledgement.

    A record is trustworthy only while its acknowledgement file holds
    that record's digest. The acknowledgement is set to PENDING before
    the record changes and to the new digest only after the record is
    durable, and both updates are IN-PLACE file writes whose
    durability needs no directory fsync.

    The previous version removed a marker and then fsynced the parent
    directory; when that final fsync failed the marker was already
    gone and a fresh reader accepted the new content -- exactly the
    partial acknowledgement the marker existed to prevent. There is no
    removal here: every failure leaves the acknowledgement holding
    PENDING or the PREVIOUS digest, and read() refuses on both."""
    _refuse_symlink(path.parent, "custody root")
    _refuse_symlink(path, "record")
    ack = _ack_path(path)
    if exclusive:
        # Claim the identity BEFORE touching the acknowledgement. My
        # first version set it to PENDING first, so a REFUSED
        # duplicate open invalidated the acknowledgement of a
        # perfectly good existing record and made it unreadable — a
        # rejected write must never damage the record it lost to.
        try:
            claim = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            FILE_MODE)
        except FileExistsError as exc:
            raise FlattenObligationError(
                f"{path.name}: already claimed") from exc
        os.close(claim)
    _ensure_ack(ack)
    _set_ack(ack, ACK_PENDING)

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
    try:
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    try:
        _set_ack(ack, payload[DIGEST_FIELD].encode())
    except Exception as exc:
        # the acknowledgement could not be made durable. Force it back
        # to PENDING so every reader refuses; if even that fails the
        # state is genuinely undefined and is reported as such.
        try:
            _set_ack(ack, ACK_PENDING)
        except Exception as restore:
            raise FlattenObligationError(
                f"{path.name}: acknowledgement could not be written "
                f"({exc}) and could not be reset ({restore}) — the "
                "record must be treated as unresolved") from exc
        raise


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
            # F3: a fresh generation binds every transition claim to
            # THIS incarnation of the obligation — an ABA record
            # cannot inherit another incarnation's claims
            "generation": os.urandom(16).hex(),
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
        acknowledgement = read_ack(path)
        if acknowledgement == ACK_PENDING or not acknowledgement:
            raise FlattenIntegrityError(
                f"{obligation_id}: the write was never acknowledged "
                f"(acknowledgement is "
                f"{acknowledgement.decode() or 'absent'!r}) — the "
                "record is not trustworthy and the obligation stays "
                "unresolved")
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
        if acknowledgement.decode() != expected:
            raise FlattenIntegrityError(
                f"{obligation_id}: acknowledgement names digest "
                f"{acknowledgement.decode()[:12]}… but the record "
                f"hashes to {expected[:12]}… — the last write was "
                "never acknowledged")
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
    def _claim_path(self, path: Path, record: dict) -> Path:
        """F3: the claim is keyed by the DIGEST of the exact record
        version being transitioned (which covers the generation), so
        it is a durable compare-and-swap token: one winner per
        version, never unlinked, useless against any other version
        or incarnation."""
        return path.with_suffix(
            f".claim.{record[DIGEST_FIELD][:32]}")

    def _transition(self, obligation_id: str, *, expected: tuple,
                    episode_identity: str, changes: dict,
                    require_same_episode: bool = False) -> dict:
        """F3 (order agent-multi@4ad4937b): the COMPLETE
        read/verify/expected-state/write/ack transaction is
        serialized by one durable generation-bound claim. Exactly
        one process wins a given transition; every loser gets a
        TYPED refusal carrying the winner (FlattenTransitionObserved)
        or a fail-closed integrity refusal — never a silent success
        and never an overwrite. Claims are never unlinked."""
        if not isinstance(episode_identity, str) or \
                not episode_identity.strip():
            raise FlattenObligationError(
                f"episode_identity must be a nonempty string, got "
                f"{episode_identity!r}")
        path = self._path(obligation_id)
        record = self.read(obligation_id)
        if record["state"] in TERMINAL_STATES:
            raise FlattenObligationError(
                f"{obligation_id}: already terminal in state "
                f"{record['state']!r} — a terminal obligation is "
                "immutable")
        if changes.get("state") == record["state"]:
            raise FlattenTransitionObserved(
                f"{obligation_id}: already in state "
                f"{record['state']!r} — the transition was won by "
                "another process and is observed, not won again",
                record)
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
        claim = self._claim_path(path, record)
        _refuse_symlink(claim, "transition claim")
        try:
            fd = os.open(claim, os.O_WRONLY | os.O_CREAT |
                         os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
        except FileExistsError:
            # a claim for THIS record version already exists: either
            # the winner has finished (state advanced -> observed) or
            # it is mid-flight/crashed (state unchanged -> unresolved)
            try:
                current = self.read(obligation_id)
            except FlattenObligationError as exc:
                raise FlattenIntegrityError(
                    f"{obligation_id}: a transition claim exists and "
                    f"the record is unreadable ({exc}) — unresolved, "
                    "an operator disposes") from exc
            if current[DIGEST_FIELD] != record[DIGEST_FIELD]:
                raise FlattenTransitionObserved(
                    f"{obligation_id}: the transition on this "
                    f"version was won by another process (now "
                    f"{current['state']!r}) — observed, not won",
                    current)
            raise FlattenIntegrityError(
                f"{obligation_id}: a transition claim exists for "
                "this exact record version but the transition never "
                "completed — the claim holder crashed or is "
                "mid-flight; unresolved, an operator disposes")
        try:
            os.write(fd, f"{os.getpid()}:{episode_identity}".encode())
            os.fchmod(fd, FILE_MODE)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(path.parent)
        # the claim is DURABLE and this process owns this version's
        # transition exclusively: re-verify under the claim, write.
        current = self.read(obligation_id)
        if current[DIGEST_FIELD] != record[DIGEST_FIELD]:
            raise FlattenIntegrityError(
                f"{obligation_id}: the record changed under a "
                "freshly won claim — refused, an operator disposes")
        payload = {**current, **changes,
                   "transitioned_by_episode": episode_identity,
                   "transition_claim": claim.name}
        payload[DIGEST_FIELD] = _digest(payload)
        _durable_write(path, payload, exclusive=False)
        return payload
