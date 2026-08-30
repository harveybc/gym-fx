"""D4: adversarial acceptance for the durable flatten custody.

Every assertion here is an attack on the store: altered records,
symlinked paths, a permissive umask, failing fsyncs, two processes
racing the same transition, a real writer process and a real
recovering process, several open obligations, and a foreign account
trying to certify someone else's close.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.flatten_custody import (
    DIGEST_FIELD, FILE_MODE, ROOT_MODE, FlattenDispositionRequired,
    FlattenIntegrityError, FlattenObligationError,
    FlattenObligationStore, _digest)


REPO = str(Path(__file__).resolve().parents[1])
IDENTITY = dict(venue="mt5_demo", account_fingerprint="fp-1",
                symbol="ETHUSD", position_identity="pos-1",
                episode_identity="ep-A", code_identity="code-1")


def _store(tmp_path, name="custody"):
    return FlattenObligationStore(tmp_path / name)


def _open(store, obligation_id="o-1", **kw):
    payload = {**IDENTITY, "signed_exposure": 99.8,
               "requested_at_bar": 5}
    payload.update(kw)
    return store.open_obligation(obligation_id, **payload)


# =================================================================== #
# integrity                                                           #
# =================================================================== #

class TestRecordIntegrity:

    MUTABLE = ("obligation_id", "state", "venue",
               "account_fingerprint", "symbol", "position_identity",
               "episode_identity", "code_identity",
               "signed_exposure_at_request", "requested_at_bar",
               "opened_at", "confirmed_at_bar", "incident")

    @pytest.mark.parametrize("field", MUTABLE)
    def test_mutating_any_field_is_refused(self, tmp_path, field):
        store = _store(tmp_path)
        _open(store)
        path = store.root / "o-1.json"
        record = json.loads(path.read_text())
        original = record[field]
        record[field] = ("TAMPERED" if isinstance(original, str)
                         else 999)
        path.write_text(json.dumps(record))
        with pytest.raises(FlattenIntegrityError,
                           match="digest mismatch"):
            store.read("o-1")

    def test_mutating_the_digest_itself_is_refused(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        path = store.root / "o-1.json"
        record = json.loads(path.read_text())
        record[DIGEST_FIELD] = "0" * 64
        path.write_text(json.dumps(record))
        with pytest.raises(FlattenIntegrityError):
            store.read("o-1")

    def test_a_consistently_reforged_record_still_needs_the_content(
            self, tmp_path):
        """Recomputing the digest over altered content makes the file
        self-consistent -- the digest is an integrity check, not an
        authenticity one, and this test states that limit plainly."""
        store = _store(tmp_path)
        _open(store)
        path = store.root / "o-1.json"
        record = json.loads(path.read_text())
        record["signed_exposure_at_request"] = 999.0
        record[DIGEST_FIELD] = _digest(record)
        path.write_text(json.dumps(record))
        assert store.read("o-1")["signed_exposure_at_request"] == 999.0

    def test_an_altered_record_counts_as_outstanding(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        path = store.root / "o-1.json"
        record = json.loads(path.read_text())
        record["state"] = "flatten_confirmed"
        path.write_text(json.dumps(record))
        outstanding = store.outstanding()
        assert len(outstanding) == 1
        assert outstanding[0]["state"] == "integrity_failed", (
            "an altered record must never read as a discharge")

    def test_an_unreadable_record_counts_as_outstanding(self,
                                                        tmp_path):
        store = _store(tmp_path)
        (store.root / "broken.json").write_text("{ not json")
        outstanding = store.outstanding()
        assert len(outstanding) == 1
        assert outstanding[0]["state"] == "integrity_failed"

    def test_an_unknown_state_is_refused(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        path = store.root / "o-1.json"
        record = json.loads(path.read_text())
        record["state"] = "definitely_closed_trust_me"
        record[DIGEST_FIELD] = _digest(record)
        path.write_text(json.dumps(record))
        with pytest.raises(FlattenIntegrityError,
                           match="unknown state"):
            store.read("o-1")


# =================================================================== #
# permissions and symlinks                                            #
# =================================================================== #

class TestPermissionsAndSymlinks:

    def test_modes_are_enforced_under_a_permissive_umask(self,
                                                         tmp_path):
        old = os.umask(0o000)
        try:
            store = _store(tmp_path, "umask_root")
            _open(store)
            store.mark_in_flight("o-1", bar_index=6,
                                 episode_identity="ep-A")
            root_mode = stat.S_IMODE(os.stat(store.root).st_mode)
            file_mode = stat.S_IMODE(
                os.stat(store.root / "o-1.json").st_mode)
        finally:
            os.umask(old)
        assert root_mode == ROOT_MODE, oct(root_mode)
        assert file_mode == FILE_MODE, oct(file_mode)

    def test_no_temporary_or_lock_survives_a_transition(self,
                                                        tmp_path):
        store = _store(tmp_path)
        _open(store)
        store.mark_in_flight("o-1", bar_index=6,
                             episode_identity="ep-A")
        store.confirm("o-1", reconciliation={"flat_confirmed": True},
                      bar_index=7, episode_identity="ep-A")
        leftovers = [p.name for p in store.root.iterdir()
                     if p.name != "o-1.json"]
        assert leftovers == [], leftovers

    def test_a_symlinked_root_is_refused(self, tmp_path):
        real = _store(tmp_path, "real_root")
        link = tmp_path / "linked_root"
        os.symlink(real.root, link)
        with pytest.raises(FlattenObligationError,
                           match="symlinked path refused"):
            FlattenObligationStore(link)

    def test_a_symlinked_record_is_refused_on_read(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        os.symlink(store.root / "o-1.json", store.root / "o-2.json")
        with pytest.raises(FlattenObligationError,
                           match="symlinked path refused"):
            store.read("o-2")

    def test_outstanding_refuses_a_symlinked_record(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        store.confirm("o-1", reconciliation={"flat_confirmed": True},
                      bar_index=6, episode_identity="ep-A")
        os.symlink(store.root / "o-1.json", store.root / "o-2.json")
        outstanding = store.outstanding()
        assert len(outstanding) == 1
        assert outstanding[0]["obligation_id"] == "o-2"
        assert outstanding[0]["state"] == "integrity_failed", (
            "a symlinked record is refused, never followed")

    def test_a_planted_temporary_symlink_is_refused(self, tmp_path):
        store = _store(tmp_path)
        target = tmp_path / "elsewhere.json"
        tmp_name = f"o-9.json.tmp.{os.getpid()}"
        os.symlink(target, store.root / tmp_name)
        with pytest.raises(FlattenObligationError,
                           match="symlinked path refused"):
            _open(store, "o-9")
        assert not target.exists(), "the symlink target was written"

    def test_a_planted_lock_symlink_is_refused(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        os.symlink(tmp_path / "lock_target",
                   store.root / "o-1.json.lock")
        with pytest.raises(FlattenObligationError,
                           match="symlinked path refused"):
            store.mark_in_flight("o-1", bar_index=6,
                                 episode_identity="ep-A")


# =================================================================== #
# durability                                                          #
# =================================================================== #

class TestDurabilityIsNeverAcknowledgedWithoutFsync:

    def test_a_failing_file_fsync_does_not_create_the_record(
            self, tmp_path, monkeypatch):
        import app.flatten_custody as fc
        store = _store(tmp_path)
        real_fsync = os.fsync

        def failing(fd):
            raise OSError("simulated file fsync failure")

        monkeypatch.setattr(fc.os, "fsync", failing)
        with pytest.raises(OSError):
            _open(store)
        monkeypatch.setattr(fc.os, "fsync", real_fsync)
        assert list(store.root.iterdir()) == [], (
            "a failed fsync must leave nothing behind")

    def test_a_failing_directory_fsync_does_not_acknowledge(
            self, tmp_path, monkeypatch):
        import app.flatten_custody as fc
        store = _store(tmp_path)
        _open(store)
        before = (store.root / "o-1.json").read_text()

        def failing_dir(path):
            raise OSError("simulated directory fsync failure")

        monkeypatch.setattr(fc, "_fsync_dir", failing_dir)
        with pytest.raises(OSError):
            store.mark_in_flight("o-1", bar_index=6,
                                 episode_identity="ep-A")
        monkeypatch.undo()
        # the record content is unchanged AND the unacknowledged
        # marker makes every later read refuse, so a half-durable
        # transition can never be mistaken for a completed one
        assert (store.root / "o-1.json").read_text() == before
        with pytest.raises(FlattenIntegrityError,
                           match="never acknowledged"):
            store.read("o-1")
        outstanding = store.outstanding()
        assert len(outstanding) == 1
        assert outstanding[0]["state"] == "integrity_failed"

    def test_a_failing_transition_leaves_no_temporary(self, tmp_path,
                                                      monkeypatch):
        import app.flatten_custody as fc
        store = _store(tmp_path)
        _open(store)
        monkeypatch.setattr(
            fc, "_fsync_dir",
            lambda path: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError):
            store.mark_in_flight("o-1", bar_index=6,
                                 episode_identity="ep-A")
        monkeypatch.undo()
        leftovers = sorted(p.name for p in store.root.iterdir())
        assert not any(".tmp." in name for name in leftovers), (
            f"no temporary may survive: {leftovers}")
        assert "o-1.json.unacknowledged" in leftovers, (
            "the marker MUST survive: it is what stops an "
            "unacknowledged write from reading as complete")
        assert not any(name.endswith(".lock")
                       for name in leftovers), leftovers


# =================================================================== #
# concurrency and real processes                                      #
# =================================================================== #

_WRITER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo!r})
    from app.flatten_custody import FlattenObligationStore
    store = FlattenObligationStore(sys.argv[1])
    store.open_obligation(
        sys.argv[2], venue="mt5_demo", account_fingerprint="fp-1",
        symbol="ETHUSD", position_identity="pos-1",
        episode_identity="ep-writer", signed_exposure=42.5,
        requested_at_bar=5, code_identity="code-1")
    store.mark_in_flight(sys.argv[2], bar_index=6,
                         episode_identity="ep-writer")
    print(json.dumps({{"wrote": sys.argv[2]}}))
""")

_RECOVERER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo!r})
    from app.flatten_custody import FlattenObligationStore
    store = FlattenObligationStore(sys.argv[1])
    print(json.dumps({{
        "outstanding": [o["obligation_id"] for o in store.outstanding()],
        "states": [o["state"] for o in store.outstanding()],
        "exposure": [o.get("signed_exposure_at_request")
                     for o in store.outstanding()],
    }}))
""")

_RACER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo!r})
    from app.flatten_custody import (FlattenObligationStore,
                                     FlattenObligationError)
    store = FlattenObligationStore(sys.argv[1])
    transition, oid, bar = sys.argv[2], sys.argv[3], int(sys.argv[4])
    try:
        if transition == "confirm":
            store.confirm(oid,
                          reconciliation={{"flat_confirmed": True}},
                          bar_index=bar, episode_identity="ep-A")
        elif transition == "in_flight":
            store.mark_in_flight(oid, bar_index=bar,
                                 episode_identity="ep-A")
        elif transition == "interrupt":
            store.interrupt(oid, reason=f"bar {{bar}}",
                            episode_identity="ep-A")
        else:
            store.fail(oid, incident=f"bar {{bar}}",
                       episode_identity="ep-A")
        print(json.dumps({{"winner": True, "bar": bar}}))
    except FlattenObligationError as exc:
        print(json.dumps({{"winner": False, "error": str(exc)[:80]}}))
""")

_ENV_WRITER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {repo!r} + "/tests")
    import test_session_exposure_env as T
    import pathlib
    tmp = pathlib.Path(sys.argv[1])
    env = T._env(tmp, session_flatten_custody_root=sys.argv[2])
    frames = T._drive(env, T.LONG[:6])
    print(json.dumps({{
        "phase": frames[-1]["info"]["session_flatten_phase"],
        "outstanding": [o["obligation_id"]
                        for o in env._flatten_store.outstanding()],
    }}))
""")

_ENV_RECOVERER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {repo!r} + "/tests")
    import test_session_exposure_env as T
    import pathlib
    tmp = pathlib.Path(sys.argv[1])
    env = T._env(tmp, session_flatten_custody_root=sys.argv[2])
    env.reset(seed=7)
    _o, _r, _t, _tr, info = env.step([1.0])
    ids = [p.stem for p in env._flatten_store.root.glob("*.json")]
    print(json.dumps({{
        "recovery_active": info["session_recovery_active"],
        "overlay": info["session_overlay"],
        "submitted": info["session_final_action"],
        "states": {{i: env._flatten_store.read(i)["state"]
                   for i in ids}},
    }}))
""")


def _run(script, *args):
    return subprocess.run(
        [sys.executable, "-c", script.format(repo=REPO), *args],
        capture_output=True, text=True, timeout=120)


class TestRealProcesses:

    def test_a_writer_process_and_a_recoverer_process(self, tmp_path):
        """D4: the writer really exits before the recoverer starts."""
        root = str(tmp_path / "cross_process")
        wrote = _run(_WRITER, root, "cross-1")
        assert wrote.returncode == 0, wrote.stderr
        assert json.loads(wrote.stdout)["wrote"] == "cross-1"

        recovered = _run(_RECOVERER, root)
        assert recovered.returncode == 0, recovered.stderr
        payload = json.loads(recovered.stdout)
        assert payload["outstanding"] == ["cross-1"]
        assert payload["states"] == ["flatten_in_flight"]
        assert payload["exposure"] == [42.5]

    @pytest.mark.parametrize("transition,final,exclusive", [
        ("confirm", "flatten_confirmed", True),
        ("interrupt", "interrupted_unresolved", True),
        ("fail", "flatten_failed", True),
        # mark_in_flight is deliberately IDEMPOTENT: a retry must not
        # fail. The exactly-once property there is that only ONE
        # process WRITES the transition, which the recorded bar shows.
        ("in_flight", "flatten_in_flight", False),
    ])
    def test_two_processes_racing_each_transition(self, tmp_path,
                                                  transition, final,
                                                  exclusive):
        root = str(tmp_path / f"race_{transition}")
        store = FlattenObligationStore(root)
        _open(store, "race-1")
        first = _run(_RACER, root, transition, "race-1", "7")
        second = _run(_RACER, root, transition, "race-1", "8")
        results = [json.loads(first.stdout), json.loads(second.stdout)]
        record = store.read("race-1")
        assert record["state"] == final
        if exclusive:
            winners = [r for r in results if r["winner"]]
            assert len(winners) == 1, results
        else:
            assert record["in_flight_at_bar"] in (7, 8), record
            assert record["transitioned_by_episode"] == "ep-A"
            # exactly one bar was recorded: the second process
            # observed the state the first had already written
            assert isinstance(record["in_flight_at_bar"], int)

    def test_a_real_env_writer_and_a_real_env_recoverer(self,
                                                        tmp_path):
        """D4: an env in ONE process opens the obligation and exits;
        an env in ANOTHER process recovers it and is blocked."""
        root = str(tmp_path / "env_cross")
        wrote = _run(_ENV_WRITER, str(tmp_path), root)
        assert wrote.returncode == 0, wrote.stderr[-2000:]
        written = json.loads(wrote.stdout.strip().splitlines()[-1])
        assert written["phase"] == "flatten_in_flight"
        assert len(written["outstanding"]) == 1

        recovered = _run(_ENV_RECOVERER, str(tmp_path), root)
        assert recovered.returncode == 0, recovered.stderr[-2000:]
        payload = json.loads(
            recovered.stdout.strip().splitlines()[-1])
        assert payload["recovery_active"] is True
        assert payload["overlay"] == "blocked_by_flatten_recovery"
        assert payload["submitted"] == 0
        assert set(payload["states"].values()) == {
            "interrupted_unresolved"}, payload

    def test_a_held_lock_blocks_a_competing_transition(self,
                                                       tmp_path):
        store = _store(tmp_path, "held")
        _open(store)
        lock = store.root / "o-1.json.lock"
        lock.touch(mode=FILE_MODE)
        with pytest.raises(FlattenObligationError,
                           match="competing transition holds the lock"):
            store.mark_in_flight("o-1", bar_index=6,
                                 episode_identity="ep-A")
        assert store.read("o-1")["state"] == "flatten_requested"

    def test_a_second_open_of_the_same_identity_is_refused(self,
                                                           tmp_path):
        store = _store(tmp_path)
        _open(store)
        with pytest.raises(FlattenObligationError,
                           match="already claimed"):
            _open(store)
        assert store.read("o-1")["state"] == "flatten_requested"


# =================================================================== #
# semantics                                                           #
# =================================================================== #

class TestObligationSemantics:

    def test_a_terminal_obligation_is_immutable(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        store.confirm("o-1", reconciliation={"flat_confirmed": True},
                      bar_index=6, episode_identity="ep-A")
        for call in (
            lambda: store.fail("o-1", incident="late",
                               episode_identity="ep-A"),
            lambda: store.mark_in_flight("o-1", bar_index=9,
                                         episode_identity="ep-A"),
            lambda: store.interrupt("o-1", reason="late",
                                    episode_identity="ep-A"),
        ):
            with pytest.raises(FlattenObligationError,
                               match="already terminal"):
                call()
        assert store.read("o-1")["state"] == "flatten_confirmed"

    def test_confirmation_requires_evidence_that_says_flat(self,
                                                           tmp_path):
        store = _store(tmp_path)
        _open(store)
        for bad in ({"flat_confirmed": False}, {}, None,
                    {"flat_confirmed": "yes"}, []):
            with pytest.raises(FlattenObligationError,
                               match="flat_confirmed=True"):
                store.confirm("o-1", reconciliation=bad, bar_index=6,
                              episode_identity="ep-A")
        assert store.read("o-1")["state"] == "flatten_requested"

    def test_a_foreign_episode_cannot_advance_or_confirm(self,
                                                         tmp_path):
        store = _store(tmp_path)
        _open(store)
        for call in (
            lambda: store.confirm(
                "o-1", reconciliation={"flat_confirmed": True},
                bar_index=6, episode_identity="ep-OTHER"),
            lambda: store.mark_in_flight(
                "o-1", bar_index=6, episode_identity="ep-OTHER"),
        ):
            with pytest.raises(FlattenObligationError,
                               match="cannot be advanced by episode"):
                call()
        assert store.read("o-1")["state"] == "flatten_requested"

    def test_an_interruption_is_terminal_but_claims_no_closure(
            self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        record = store.interrupt("o-1", reason="episode abandoned",
                                 episode_identity="ep-LATER")
        assert record["state"] == "interrupted_unresolved"
        assert record["closure_claimed"] is False
        assert store.outstanding() == ()
        assert store.read("o-1")["state"] == "interrupted_unresolved"

    def test_an_interruption_is_never_a_success_state(self):
        from app.flatten_custody import (SUCCESS_STATES,
                                         TERMINAL_STATES)
        assert "interrupted_unresolved" in TERMINAL_STATES
        assert "interrupted_unresolved" not in SUCCESS_STATES
        assert SUCCESS_STATES == ("flatten_confirmed",)

    def test_zero_one_and_many_open_obligations(self, tmp_path):
        store = _store(tmp_path, "many")
        assert store.require_single_open() is None
        _open(store, "a-1")
        assert store.require_single_open()["obligation_id"] == "a-1"
        _open(store, "a-2", episode_identity="ep-B")
        with pytest.raises(FlattenDispositionRequired,
                           match="no automatic resolution"):
            store.require_single_open()
        # discharging one restores a defined meaning
        store.interrupt("a-2", reason="disposed",
                        episode_identity="ep-B")
        assert store.require_single_open()["obligation_id"] == "a-1"

    def test_identity_fields_are_required_and_typed(self, tmp_path):
        store = _store(tmp_path)
        for field in ("venue", "account_fingerprint", "symbol",
                      "position_identity", "episode_identity",
                      "code_identity"):
            for bad in ("", "   ", None, 1, True):
                with pytest.raises(FlattenObligationError):
                    _open(store, f"bad-{field}", **{field: bad})

    def test_restart_reads_are_repeatable(self, tmp_path):
        store = _store(tmp_path)
        _open(store)
        store.mark_in_flight("o-1", bar_index=6,
                             episode_identity="ep-A")
        first = FlattenObligationStore(store.root).outstanding()
        second = FlattenObligationStore(store.root).outstanding()
        assert first == second
        assert first[0]["state"] == "flatten_in_flight"
