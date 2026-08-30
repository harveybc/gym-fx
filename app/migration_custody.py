"""Durable one-use custody for carried-position migrations
(order @c933da64 D1-D2). An in-memory dict is NOT custody.

Authority boundary (D1): this module OWNS the transition. The
watchdog never mutates it — monitoring is strictly read-only and
repeatable.

Durable state machine (D2): ``prepared -> active -> completed |
failed``. Terminal states are immutable and never reusable. Exactly
ONE claimant may move ``prepared -> active``, enforced by an atomic
no-overwrite file protocol: ``O_EXCL`` creation, restrictive
permissions applied before the record becomes visible, file AND
parent-directory fsync, symlink refusal, and process-level exclusion
through the same exclusive create.

Every record binds migration id, venue, account, symbol, position
identity, closure interval, native-protection evidence digest and
policy/code identity. A partially written record can never appear
authorized: the claim file is created exclusively, fully written and
fsynced BEFORE rename, and readers only accept a complete record.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STATES = ("prepared", "active", "completed", "failed")
TERMINAL_STATES = ("completed", "failed")


class MigrationCustodyError(RuntimeError):
    """Typed custody refusal — never a silent success."""


def _sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_exclusive_write(path: Path, payload: dict) -> None:
    """Create EXCLUSIVELY, write, fsync, chmod, rename, fsync parent.

    An interrupted write leaves only a temporary file that no reader
    accepts, so a partial record can never appear authorized."""
    if path.is_symlink() or path.parent.is_symlink():
        raise MigrationCustodyError(
            f"{path}: symlinked custody path refused")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MigrationCustodyError(
            f"{tmp.name}: concurrent write in progress") from exc
    try:
        os.write(fd, json.dumps(payload, indent=1,
                                default=str).encode())
        os.fsync(fd)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    try:
        # O_EXCL on the FINAL path is what elects a single claimant
        final_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                           0o600)
    except FileExistsError as exc:
        os.unlink(tmp)
        raise MigrationCustodyError(
            f"{path.name}: already claimed by another process") \
            from exc
    os.close(final_fd)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _durable_update(path: Path, payload: dict) -> None:
    """Update an EXISTING record durably (state transitions after the
    exclusive claim). The claim itself never takes this path."""
    tmp = path.with_suffix(path.suffix + f".upd.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(payload, indent=1,
                                default=str).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class MigrationCustody:
    """Durable custody root. One file per migration identity."""

    def __init__(self, root: Path):
        root = Path(root)
        if root.is_symlink():
            raise MigrationCustodyError(
                "custody root is a symlink — refused")
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        self.root = root

    def _path(self, migration_id: str) -> Path:
        safe = hashlib.sha256(migration_id.encode()).hexdigest()[:32]
        return self.root / f"migration_{safe}.json"

    def read(self, migration_id: str) -> Optional[dict]:
        """READ-ONLY, repeatable, never mutating (D1)."""
        path = self._path(migration_id)
        if not path.is_file():
            return None
        record = json.loads(path.read_text())
        if record.get("state") not in STATES or \
                "record_digest" not in record:
            raise MigrationCustodyError(
                f"{migration_id}: incomplete custody record")
        expected = _sha_obj({k: v for k, v in record.items()
                             if k != "record_digest"})
        if record["record_digest"] != expected:
            raise MigrationCustodyError(
                f"{migration_id}: custody record digest mismatch")
        return record

    def is_active(self, migration_id: str) -> bool:
        record = self.read(migration_id)
        return record is not None and record["state"] == "active"

    def claim(self, migration, state_block: dict,
              position_identity: str, *,
              native_protection_digest: str,
              policy_identity: str,
              code_identity: str) -> dict:
        """prepared -> active by EXACTLY ONE claimant (D2).

        Every identity must match the state block; a second claim, a
        terminal record, another closure/position/symbol/account/
        venue, or missing/stale protection evidence REFUSE."""
        from app.session_exposure import require_identity, require_utc

        require_identity("native_protection_digest",
                         native_protection_digest)
        require_identity("policy_identity", policy_identity)
        require_identity("code_identity", code_identity)
        require_identity("position_identity", position_identity)
        if not migration.native_protection_confirmed:
            raise MigrationCustodyError(
                f"{migration.migration_id}: native protection is not "
                "confirmed — recovery claim refuses")
        closure_started = state_block.get("closure_started_at")
        if closure_started is None:
            raise MigrationCustodyError(
                "claim outside a closure interval refuses")
        started = require_utc("closure_started_at",
                              datetime.fromisoformat(closure_started))
        if started != migration.covers_closure_started_at:
            raise MigrationCustodyError(
                f"{migration.migration_id}: record covers "
                f"{migration.covers_closure_started_at.isoformat()}, "
                f"not {started.isoformat()} — one-closure custody")
        if migration.opened_before > started:
            raise MigrationCustodyError(
                "the position does not predate the closure")
        for label, expected, actual in (
                ("symbol", migration.symbol,
                 state_block.get("symbol")),
                ("venue", migration.venue, state_block.get("venue")),
                ("account_fingerprint", migration.account_fingerprint,
                 state_block.get("account_fingerprint"))):
            if expected != actual:
                raise MigrationCustodyError(
                    f"{label} mismatch: record {expected!r} vs state "
                    f"{actual!r} — cross-{label} custody refuses")
        if position_identity != migration.position_identity:
            raise MigrationCustodyError(
                "position identity mismatch — custody refuses")
        existing = self.read(migration.migration_id)
        if existing is not None:
            if existing["state"] in TERMINAL_STATES:
                raise MigrationCustodyError(
                    f"{migration.migration_id}: record is terminal "
                    f"({existing['state']}) — one-use custody refuses "
                    "reuse")
            raise MigrationCustodyError(
                f"{migration.migration_id}: already claimed "
                f"(state={existing['state']}) — exactly one claimant")
        record = {
            "schema": "gym_fx.carried_position_migration_custody.v1",
            "migration_id": migration.migration_id,
            "venue": migration.venue,
            "account_fingerprint": migration.account_fingerprint,
            "symbol": migration.symbol,
            "position_identity": position_identity,
            "closure_started_at": started.isoformat(),
            "closure_reopens_at": state_block.get(
                "closure_reopens_at"),
            "opened_before": migration.opened_before.isoformat(),
            "native_protection_digest": native_protection_digest,
            "policy_identity": policy_identity,
            "code_identity": code_identity,
            "state": "active",
            "claimed_at": _utc_now(),
            "claimed_by_pid": os.getpid(),
        }
        record["record_digest"] = _sha_obj(record)
        _atomic_exclusive_write(self._path(migration.migration_id),
                                record)
        return record

    def finish(self, migration_id: str, terminal_state: str, *,
               reconciliation: dict) -> dict:
        """active -> completed | failed. Requires DIRECT FRESH
        reconciliation evidence (D3.9)."""
        if terminal_state not in TERMINAL_STATES:
            raise MigrationCustodyError(
                f"illegal terminal state {terminal_state!r}")
        record = self.read(migration_id)
        if record is None:
            raise MigrationCustodyError(
                f"{migration_id}: no custody record to finish")
        if record["state"] in TERMINAL_STATES:
            raise MigrationCustodyError(
                f"{migration_id}: already {record['state']} — "
                "terminal states are immutable")
        if not isinstance(reconciliation, dict) or \
                "flat_confirmed" not in reconciliation or \
                "fresh" not in reconciliation:
            raise MigrationCustodyError(
                f"{migration_id}: direct reconciliation evidence is "
                "required to finish custody")
        if terminal_state == "completed":
            if not reconciliation.get("flat_confirmed") or \
                    not reconciliation.get("fresh"):
                raise MigrationCustodyError(
                    f"{migration_id}: completion requires FRESH "
                    "evidence of zero exposure")
        updated = {k: v for k, v in record.items()
                   if k != "record_digest"}
        updated.update({"state": terminal_state,
                        "finished_at": _utc_now(),
                        "reconciliation": reconciliation})
        updated["record_digest"] = _sha_obj(updated)
        _durable_update(self._path(migration_id), updated)
        return updated
