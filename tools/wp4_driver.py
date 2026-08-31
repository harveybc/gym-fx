"""WP4 shared driver, corrected under order agent-multi@051ef265
(WP4-C1..C8). CPU mechanics and bounded evidence only — ZERO economic
authority, no training, no GPU.

C1 — complete executable identity: every frozen gym-fx file this
driver actually consumes is digest-verified before any step, and a
run must present the materialization MANIFEST plus its expected
digest: a cell runs only if the manifest's cell index names its id
and digest, so a consistently re-digested cell, a substituted
manifest, or a missing/extra cell refuses. The WP3 executor files are
bound in the WP4.0 freeze but are NOT consumed by this CPU driver;
they are enumerated as bound-unconsumed rather than falsely claimed
verified-in-use.

C2 — genuinely paired treatment: actions come from a pre-generated
tape derived from the SEED ONLY, persisted with its digest and shared
across every cell of a pair/family. Nothing about the cell may steer
the tape.

C3 — truthful session evidence: fixtures come from HASHED HISTORICAL
bars (HistData EURUSD 4h via the financial-data lake, provenance
sha256 verified before use), preserving the real missing intervals.
Closure intervals are derived mechanically from the observed gaps and
declared as such. The spread and feature channels are instrumentation
(the historical source carries neither) and are stamped
non-historical. EXPECTED_MARKET_CLOSED cannot appear as an env step —
there are correctly no bars inside a closure — so it is probed
separately through the pure session authority at timestamps inside
each gap: the actionable env trajectory contains four states and the
authority probe covers the fifth.

C5 — conservation is DERIVED from executed artifacts (the per-step
record rows, the authoritative closed-trade stream and the summary),
never asserted; the underlying rows are persisted with digests.
"""
from __future__ import annotations

import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.env import GymFxEnv
from app.session_exposure import (SessionCalendar, require_utc,
                                  session_state)
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from metrics_plugins.default_metrics import Plugin as Metrics
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor)
from reward_plugins.pnl_reward import Plugin as Reward
from strategy_plugins.shared_execution_envelope import (
    Plugin as Envelope)
from tools.wp4_materializer import (canonical_bytes, sha256_hex,
                                    verify_cell)

# ------------------------------------------------------------------ #
# C1: complete executable identity                                   #
# ------------------------------------------------------------------ #

# every WP4.0-frozen gym-fx file this driver CONSUMES (all six are
# imported through app.env / app.session_exposure)
FROZEN_AUTHORITY_SHA256 = {
    "app/session_exposure.py":
        "0a33065805b3304372a3b58a018506214aecfc3879d5a556b029e30b8"
        "27132e2",
    "app/flatten_custody.py":
        "689a3e5c54692bf3273ea269fb94dd46f0bd7f5f9506e2d81e0d237bf"
        "f5c7012",
    "app/oanda_calendar.py":
        "f433d5a0ddcabbed6ba7bcaf26c101012b575524971b51372551a42cc"
        "a332e11",
    "app/env.py":
        "994c83888a1831b9164c0dab2a408828d7463ddcf85b05c97c56ed959"
        "10c70b1",
    "app/direct_evidence.py":
        "afdcb3a57393dd6a0552a253a27e711a7b5df3c222682dc6f60d610e6"
        "bb66ec6",
    "app/migration_custody.py":
        "f3703f3a1c9a1b22b31883e14560991b44ab656b39a46698960abae6b"
        "5b6cefe",
}

# WP4.0-frozen identities this CPU driver does NOT consume: bound in
# the materialization identity block, enumerated here so nobody reads
# the verification list as the whole freeze.
BOUND_UNCONSUMED_IDENTITIES = {
    "lts@83dff62": ("app/effect_executor.py",
                    "app/venue_direct_evidence.py",
                    "app/session_authority_adapter.py",
                    "app/live_flatten_custody.py",
                    "app/session_watchdog.py"),
}


class Wp4IdentityError(RuntimeError):
    """The executing authority or binding is not the frozen one."""


def verify_frozen_identity(repo_root: Path) -> dict:
    seen = {}
    for rel, expected in FROZEN_AUTHORITY_SHA256.items():
        actual = hashlib.sha256(
            (Path(repo_root) / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise Wp4IdentityError(
                f"{rel}: sha256 {actual[:16]}… does not match the "
                f"frozen WP4.0 identity {expected[:16]}… — the "
                "driver refuses to run unreviewed authority")
        seen[rel] = actual
    return seen


def verify_manifest(manifest: dict,
                    expected_manifest_digest: str) -> dict:
    body = {k: v for k, v in manifest.items() if k != "digest"}
    if manifest.get("digest") != sha256_hex(canonical_bytes(body)):
        raise Wp4IdentityError(
            "manifest self-digest mismatch — a substituted or "
            "altered manifest binds nothing")
    if manifest["digest"] != expected_manifest_digest:
        raise Wp4IdentityError(
            f"manifest digest {manifest['digest'][:16]}… is not the "
            f"reviewed {expected_manifest_digest[:16]}… — a "
            "substituted manifest is refused")
    return manifest


def verify_cell_binding(cell: dict, manifest: dict,
                        expected_manifest_digest: str) -> dict:
    """C1: the manifest, not the cell, is the external authority. A
    consistently altered and re-digested cell fails here because the
    manifest's cell index no longer names its digest."""
    verify_manifest(manifest, expected_manifest_digest)
    verify_cell(cell)
    index = manifest["cell_index"]
    cell_id = cell["cell_id"]
    if cell_id not in index:
        raise Wp4IdentityError(
            f"cell {cell_id!r} is not in the reviewed manifest — an "
            "extra or renamed cell runs nothing")
    if index[cell_id] != cell["digest"]:
        raise Wp4IdentityError(
            f"cell {cell_id!r}: digest {cell['digest'][:16]}… does "
            f"not match the manifest binding "
            f"{index[cell_id][:16]}… — a re-digested cell runs "
            "nothing")
    return cell


def verify_manifest_matches_dir(manifest: dict,
                                cells_dir: Path) -> None:
    """Missing or extra cell files against the manifest refuse."""
    on_disk = {p.stem for p in Path(cells_dir).glob("*.json")} \
        - {"manifest", "rejection_ledger"}
    named = set(manifest["cell_index"])
    missing = sorted(named - on_disk)
    extra = sorted(on_disk - named)
    if missing or extra:
        raise Wp4IdentityError(
            f"materialization does not match the manifest — "
            f"missing {missing}, extra {extra}")


# ------------------------------------------------------------------ #
# C2: the seed-only action tape                                      #
# ------------------------------------------------------------------ #

def action_tape(seed: int, length: int) -> dict:
    """Pre-generated actions from the SEED ONLY. Every cell of a
    pair/family receives this identical tape for a given seed; the
    tape digest is recorded into each run so drift is checkable."""
    rng = np.random.default_rng(int(seed))
    actions = []
    for index in range(int(length)):
        roll = rng.random()
        if roll < 0.15:
            actions.append(0.0)            # hold
        elif roll < 0.60:
            actions.append(1.0)            # long pressure
        elif roll < 0.90:
            actions.append(-1.0)           # short pressure
        else:
            actions.append(float(rng.uniform(-1.0, 1.0)))
    tape = {"schema": "gymfx.wp4.action_tape.v1", "seed": int(seed),
            "length": int(length), "actions": actions}
    tape["digest"] = sha256_hex(canonical_bytes(tape))
    return tape


# ------------------------------------------------------------------ #
# C3: hashed historical fixture                                      #
# ------------------------------------------------------------------ #

HISTORICAL_SOURCE = {
    "path": ("/home/harveybc/Documents/GitHub/financial-data/"
             "market_data/forex/g10/eurusd/4h.parquet"),
    "sha256": ("359bd825708dd6906ccd0b2359e7f8dc7313fdcecf09629ee14"
               "e609ba6ecbd25"),
    "role": ("historical market bars — HistData EURUSD, resampled "
             "to 4h in the financial-data lake; read-only source "
             "of truth for timestamps and prices"),
}
WINDOW_START = "2023-12-12 00:00:00"
WINDOW_END = "2024-01-10 00:00:00"
INSTRUMENTATION = {
    "SPREAD": ("constant 0.0002 — NOT historical: the source "
               "carries no spread; declared instrumentation for "
               "the stability gate"),
    "feat": ("normalized close — derived from historical closes, "
             "not an independent historical channel"),
}


def load_historical_window(tmp_dir: Path, *,
                           bar_hours: float = 4.0) -> dict:
    """Slice the bounded window from the hashed historical source,
    PRESERVING the real missing intervals, derive the closure
    intervals mechanically from the observed gaps, and classify
    weekend versus holiday/exception closures."""
    source = Path(HISTORICAL_SOURCE["path"])
    if not source.is_file():
        raise Wp4IdentityError(
            f"historical source missing: {source} — evidence is "
            "not fabricated in its absence")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != HISTORICAL_SOURCE["sha256"]:
        raise Wp4IdentityError(
            f"historical source sha256 {actual[:16]}… does not "
            "match the published provenance — refused")
    frame = pd.read_parquet(source)
    ts = pd.to_datetime(frame["datetime"], utc=True)
    mask = (ts >= pd.Timestamp(WINDOW_START, tz="UTC")) & \
           (ts <= pd.Timestamp(WINDOW_END, tz="UTC"))
    window = frame.loc[mask].reset_index(drop=True)
    stamps = pd.to_datetime(window["datetime"], utc=True)
    bar = pd.Timedelta(hours=bar_hours)
    closures = []
    diffs = stamps.diff()
    for i in range(1, len(stamps)):
        if diffs.iloc[i] > bar:
            start = stamps.iloc[i - 1] + bar
            end = stamps.iloc[i]
            hours = (end - start).total_seconds() / 3600.0
            kind = ("weekend" if 40.0 <= hours <= 56.0
                    else "holiday_or_exception")
            closures.append({"start": str(start), "end": str(end),
                             "hours": hours, "kind": kind})
    weekends = [c for c in closures if c["kind"] == "weekend"]
    holidays = [c for c in closures
                if c["kind"] == "holiday_or_exception"]
    holiday_status = ("present" if holidays
                      else "HOLIDAY_EVIDENCE_UNAVAILABLE")
    n = len(window)
    closes = window["close"].astype(float).to_numpy()
    fixture = pd.DataFrame({
        "DATE_TIME": stamps.dt.tz_localize(None),
        "OPEN": window["open"].astype(float),
        "HIGH": window["high"].astype(float),
        "LOW": window["low"].astype(float),
        "CLOSE": closes,
        "VOLUME": 1000.0,
        "SPREAD": np.full(n, 0.0002),
        "feat": (closes - closes.min())
                / max(closes.max() - closes.min(), 1e-12),
    })
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    csv = Path(tmp_dir) / "wp4_historical_window.csv"
    fixture.to_csv(csv, index=False)
    meta = {
        "source": HISTORICAL_SOURCE,
        "instrumentation_channels": INSTRUMENTATION,
        "window": {"start": str(stamps.iloc[0]),
                   "end": str(stamps.iloc[-1]), "bars": n},
        "closures_derived_from_observed_gaps": closures,
        "weekend_closures": len(weekends),
        "holiday_or_exception_closures": len(holidays),
        "holiday_evidence": holiday_status,
        "calendar_note": ("closure intervals derived MECHANICALLY "
                          "from the observed missing intervals "
                          "(gap > one bar); the broker's historical "
                          "calendar is not available, and this "
                          "derivation is declared rather than "
                          "presented as venue authority"),
        "fixture_csv_sha256": hashlib.sha256(
            csv.read_bytes()).hexdigest(),
    }
    meta["digest"] = sha256_hex(canonical_bytes(meta))
    (Path(tmp_dir) / "wp4_historical_window.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True))
    return {"csv": csv, "meta": meta,
            "intervals": [[c["start"], c["end"]] for c in closures],
            "bars": n}


def expected_market_closed_probe(cell: dict, window: dict) -> dict:
    """The FIFTH state, probed through the pure session authority at
    timestamps inside each derived closure — the env trajectory
    correctly contains no step there because no bar exists there."""
    from datetime import timezone as _tz

    def _utc(value):
        # mirror GymFxEnv._session_utc: a pandas Timestamp allocates
        # a new object under astimezone and the calendar refuses it
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.to_pydatetime().astimezone(_tz.utc)

    policy = cell["session_exposure_policy"]
    calendar = SessionCalendar.build(
        venue="mt5_demo", account_fingerprint="fp-wp4",
        symbol="EURUSD",
        calendar_digest=str(policy["calendar_identity"]),
        intervals=[(_utc(a), _utc(b))
                   for a, b in window["intervals"]])
    probes = []
    for a, b in window["intervals"]:
        midpoint = _utc(pd.Timestamp(a) + (pd.Timestamp(b)
                                           - pd.Timestamp(a)) / 2)
        state = session_state(policy, now=midpoint,
                              calendar=calendar,
                              expected_venue="mt5_demo",
                              expected_account_fingerprint="fp-wp4",
                              expected_symbol="EURUSD")
        probes.append({"inside": str(midpoint),
                       "state": state["state"]})
    return {
        "note": ("the actionable env trajectory contains four "
                 "states; EXPECTED_MARKET_CLOSED is covered by this "
                 "authority probe because no bar exists inside a "
                 "closure"),
        "probes": probes,
        "all_expected_market_closed": all(
            p["state"] == "EXPECTED_MARKET_CLOSED"
            for p in probes),
    }


# ------------------------------------------------------------------ #
# env construction and the recorded run                              #
# ------------------------------------------------------------------ #

def build_env(cell: dict, window: dict, *, tmp_dir: Path,
              calendar_intervals="from_window",
              envelope_cls=Envelope, leverage_cap: float = 1.0,
              symbol: str = "EURUSD") -> GymFxEnv:
    policy = cell["session_exposure_policy"]
    config = {
        "input_data_file": str(window["csv"]),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "include_price_window": False,
        "window_size": 4, "initial_cash": 10000.0,
        "position_size": 1.0, "min_equity": 0.0,
        "env_mode": "training", "commission": 0.0, "leverage": 1.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        "execution_envelope": {
            "envelope_mode": "fixed_fraction",
            "sl_fraction": 0.50, "tp_fraction": 0.50,
            "leverage_cap": leverage_cap},
    }
    if policy["enabled"]:
        config.update({
            "session_exposure_enabled": True,
            "session_exposure_policy": dict(policy),
            "session_venue": "mt5_demo",
            "session_account_fingerprint": "fp-wp4",
            "session_symbol": symbol,
            "session_spread_column": "SPREAD",
            "session_flatten_custody_root": str(
                Path(tmp_dir) / f"{cell['cell_id']}_custody"),
        })
        if calendar_intervals == "from_window":
            config["session_calendar_intervals"] = window["intervals"]
        elif calendar_intervals is not None:
            config["session_calendar_intervals"] = calendar_intervals
    return GymFxEnv(config, DataFeed(config), Broker(config),
                    envelope_cls(config), Preprocessor(config),
                    Reward(config), Metrics(config))


def _obs_digest(obs, *, shared_only: bool) -> str:
    parts = []
    for key in sorted(obs):
        if shared_only and key.startswith("session_"):
            continue
        parts.append(key.encode())
        parts.append(np.asarray(obs[key]).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def recorded_run(cell: dict, manifest: dict,
                 expected_manifest_digest: str, tape: dict,
                 window: dict, *, tmp_dir: Path, repo_root: Path,
                 seed: int = 7) -> dict:
    """One bounded CPU mechanics run: identity- and manifest-bound,
    tape-driven, per-step rows persisted with a digest, conservation
    DERIVED afterwards."""
    identity = verify_frozen_identity(repo_root)
    verify_cell_binding(cell, manifest, expected_manifest_digest)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    env = build_env(cell, window, tmp_dir=tmp_dir)
    started = time.perf_counter()
    obs, _info = env.reset(seed=seed)
    rows = []
    for index, action in enumerate(tape["actions"]):
        obs, reward, term, trunc, info = env.step([float(action)])
        rows.append({
            "index": index,
            "action": float(action),
            "reward": float(reward),
            "pnl": float(info.get("pnl", 0.0)),
            "equity": float(getattr(env.bridge, "equity", 0.0)),
            "session_state": info.get("session_state"),
            "session_overlay": info.get("session_overlay"),
            "session_final_action": info.get("session_final_action"),
            "signed_exposure": info.get("session_signed_exposure"),
            "cancel_refs": list(
                info.get("session_cancel_requested_refs") or ()),
            "obs_digest_shared": _obs_digest(obs, shared_only=True),
            "obs_digest_full": _obs_digest(obs, shared_only=False),
        })
        if term or trunc:
            break
    wall = time.perf_counter() - started
    summary = env.summary()
    stream = [dict(e) for e in getattr(env.bridge,
                                       "closed_trade_stream", [])]
    conservation = derive_conservation(env, rows, window, summary,
                                       stream)
    rows_path = Path(tmp_dir) / f"{cell['cell_id']}_rows.json"
    rows_path.write_text(json.dumps(rows, indent=1))
    stream_path = Path(tmp_dir) / f"{cell['cell_id']}_trades.json"
    stream_path.write_text(json.dumps(stream, indent=1,
                                      default=str))
    result = {
        "cell_id": cell["cell_id"],
        "family": cell["family"],
        "cell_digest": cell["digest"],
        "manifest_digest": manifest["digest"],
        "tape_digest": tape["digest"],
        "window_digest": window["meta"]["digest"],
        "identity_verified": identity,
        "bound_unconsumed_identities": {
            k: list(v) for k, v in
            BOUND_UNCONSUMED_IDENTITIES.items()},
        "seed": seed,
        "steps": len(rows),
        "wall_seconds": round(wall, 4),
        "peak_rss_kb":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "session_state_counts": _count(rows, "session_state"),
        "overlay_counts": _count(rows, "session_overlay"),
        "rows_sha256": hashlib.sha256(
            rows_path.read_bytes()).hexdigest(),
        "trades_sha256": hashlib.sha256(
            stream_path.read_bytes()).hexdigest(),
        "conservation": conservation,
    }
    return result


def _count(rows, key):
    counts = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return counts


# ------------------------------------------------------------------ #
# C5: conservation DERIVED from executed artifacts                   #
# ------------------------------------------------------------------ #

def derive_conservation(env, rows, window, summary, stream) -> dict:
    stamps = pd.to_datetime(
        pd.read_csv(window["csv"])["DATE_TIME"]).dt.tz_localize(
            "UTC")
    inside = 0
    for a, b in window["intervals"]:
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        inside += int(((stamps >= start) & (stamps < end)).sum())
    suppressed = [r["index"] for r in rows
                  if r["reward"] == 0.0 and abs(r["pnl"]) > 1e-9]
    trades_total = summary.get("trades_total", 0)
    won = summary.get("trades_won", 0)
    lost = summary.get("trades_lost", 0)
    breakeven = summary.get("trades_breakeven", 0)
    gross_net_violations = [
        e for e in stream
        if abs((float(e["gross_pnl"]) - float(e["costs"]))
               - float(e["net_pnl"])) > 1e-6]
    net_total = sum(float(e["net_pnl"]) for e in stream)
    initial = float(env.initial_cash)
    final = float(getattr(env.bridge, "equity", initial))
    open_at_end = bool(summary.get("open_position_at_end"))
    equity_gap = final - (initial + net_total)
    reasons = summary.get("close_reason_counts", {})
    return {
        "bar_timestamps_inside_closures": inside,
        "suppressed_reward_steps": suppressed,
        "close_event_conservation": {
            "trades_total": trades_total, "won": won, "lost": lost,
            "breakeven": breakeven,
            "holds": trades_total == won + lost + breakeven},
        "gross_minus_costs_equals_net": {
            "violations": len(gross_net_violations)},
        "equity_reconciliation": {
            "initial": initial, "final": final,
            "net_pnl_total": round(net_total, 6),
            "open_position_at_end": open_at_end,
            "gap": round(equity_gap, 6),
            "holds_when_flat": (not open_at_end
                                and abs(equity_gap) < 1e-4)
                               or open_at_end},
        "close_reason_counts": reasons,
        "trade_costs_total": summary.get("trade_costs_total"),
        "holds": (inside == 0 and not suppressed
                  and trades_total == won + lost + breakeven
                  and not gross_net_violations),
    }


# ------------------------------------------------------------------ #
# C2: the paired-prefix assertion                                    #
# ------------------------------------------------------------------ #

def assert_identical_prefix(rows_a, rows_b, prefix_end) -> None:
    """Refuse ANY drift in the shared channels before the first
    treatment-dependent state."""
    drift = []
    for i in range(prefix_end):
        ra, rb = rows_a[i], rows_b[i]
        for key in ("action", "reward", "pnl", "equity",
                    "obs_digest_shared"):
            if ra[key] != rb[key]:
                drift.append({"index": i, "key": key,
                              "a": ra[key], "b": rb[key]})
    if drift:
        raise Wp4IdentityError(
            f"paired prefix drift BEFORE the first treatment-"
            f"dependent state: {drift[:3]}")


def paired_prefix_check(cell_a: dict, cell_b: dict, manifest: dict,
                        expected_manifest_digest: str, tape: dict,
                        *, tmp_dir: Path, repo_root: Path,
                        seed: int = 7) -> dict:
    """Run BOTH cells on the identical tape, window and seed. Every
    shared channel must be bit-identical until the first
    treatment-dependent state; any earlier drift refuses. Structural
    session_* observation fields differ by treatment definition and
    are excluded from the shared comparison, which is declared."""
    tmp_dir = Path(tmp_dir)
    window = load_historical_window(tmp_dir / "window")
    runs = {}
    raw_rows = {}
    for tag, cell in (("a", cell_a), ("b", cell_b)):
        run_dir = tmp_dir / tag
        result = recorded_run(cell, manifest,
                              expected_manifest_digest, tape,
                              window, tmp_dir=run_dir,
                              repo_root=repo_root, seed=seed)
        runs[tag] = result
        raw_rows[tag] = json.loads(
            (run_dir / f"{cell['cell_id']}_rows.json").read_text())
    if runs["a"]["tape_digest"] != runs["b"]["tape_digest"] or \
            runs["a"]["window_digest"] != runs["b"]["window_digest"]:
        raise Wp4IdentityError(
            "paired runs drifted in tape or window — refused")
    INTERVENTIONS = ("masked_risk_increase", "forced_close",
                     "masked_entry_during_blackout")
    first_treatment = None
    for i, (ra, rb) in enumerate(zip(raw_rows["a"], raw_rows["b"])):
        states = (ra.get("session_state"), rb.get("session_state"))
        overlays = (ra.get("session_overlay"),
                    rb.get("session_overlay"))
        # pass_through is the NO-treatment overlay; treatment begins
        # when either arm leaves NORMAL_TRADING or actually
        # intervenes in an action
        if any(s not in (None, "NORMAL_TRADING") for s in states) \
                or any(o in INTERVENTIONS for o in overlays):
            first_treatment = i
            break
    prefix_end = first_treatment if first_treatment is not None \
        else min(len(raw_rows["a"]), len(raw_rows["b"]))
    assert_identical_prefix(raw_rows["a"], raw_rows["b"],
                            prefix_end)
    return {
        "pair": (cell_a["cell_id"], cell_b["cell_id"]),
        "tape_digest": tape["digest"],
        "window_digest": runs["a"]["window_digest"],
        "first_treatment_dependent_step": first_treatment,
        "identical_prefix_steps": prefix_end,
        "shared_channels_compared": ["action", "reward", "pnl",
                                     "equity",
                                     "obs_digest_shared"],
        "excluded_by_treatment_definition":
            "session_* observation fields",
        "runs": {k: {kk: v[kk] for kk in
                     ("steps", "session_state_counts",
                      "overlay_counts")}
                 for k, v in runs.items()},
    }


# ------------------------------------------------------------------ #
# C8: benchmark — median and dispersion, env throughput only         #
# ------------------------------------------------------------------ #

def benchmark(cell: dict, manifest: dict,
              expected_manifest_digest: str, tape: dict,
              window: dict, *, tmp_dir: Path, repo_root: Path,
              repeats: int = 7) -> dict:
    walls, rates = [], []
    base = None
    for i in range(repeats):
        result = recorded_run(cell, manifest,
                              expected_manifest_digest, tape,
                              window, tmp_dir=Path(tmp_dir) / str(i),
                              repo_root=repo_root)
        walls.append(result["wall_seconds"])
        rates.append(result["steps"] / result["wall_seconds"])
        base = result
    return {
        "cell_id": cell["cell_id"],
        "repeats": repeats,
        "steps_per_run": base["steps"],
        "wall_seconds_median": round(statistics.median(walls), 4),
        "wall_seconds_iqr": [
            round(np.percentile(walls, 25), 4),
            round(np.percentile(walls, 75), 4)],
        "env_steps_per_second_median": round(
            statistics.median(rates), 2),
        "env_steps_per_second_iqr": [
            round(np.percentile(rates, 25), 2),
            round(np.percentile(rates, 75), 2)],
        "scope": ("environment mechanics throughput ONLY — SAC "
                  "update throughput is unmeasured and no GPU "
                  "hours are extrapolated from this number"),
        "session_state_counts": base["session_state_counts"],
        "conservation_holds": base["conservation"]["holds"],
    }
