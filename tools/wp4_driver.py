"""WP4 shared driver: run one materialized cell through the REAL
GymFxEnv path on CPU, bounded, with identity checks — mechanics and
operational benchmark only, ZERO economic authority.

The driver refuses to run when the executing authority files do not
match the frozen WP4.0 identity manifest, when the cell's digest does
not verify, or when the session evidence a cell requires is missing
(fail closed, never a default).

Data is a HISTORICAL-GAP window in the accepted fixture shape: real
venue timestamps with NO ROWS inside the declared closure. No bar is
ever synthesized inside a closure and no reward is suppressed.
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.env import GymFxEnv
from app.session_exposure import SessionEvidenceError
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

# the executing authority files this driver is allowed to run under,
# frozen by digest in WP4.0 (agent-multi@ec0fd35b)
FROZEN_AUTHORITY_SHA256 = {
    "app/session_exposure.py":
        "0a33065805b3304372a3b58a018506214aecfc3879d5a556b029e30b8"
        "27132e2",
    "app/flatten_custody.py":
        "689a3e5c54692bf3273ea269fb94dd46f0bd7f5f9506e2d81e0d237bf"
        "f5c7012",
}


class Wp4IdentityError(RuntimeError):
    """The executing authority is not the frozen one — refuse."""


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


def gap_window(cell: dict, *, tmp_dir: Path) -> dict:
    """Build the bounded historical-gap window for a cell: enough
    pre-close bars to enter, wind down and force-flatten, one real
    closure GAP with no rows, and enough post-reopen bars for the
    baseline, the minimum closed bars and every stability check."""
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    policy = cell["session_exposure_policy"]
    bar_hours = float(cell["bar_hours"])
    wind = float(policy["wind_down_hours"])
    pre_bars = int(np.ceil(wind / bar_hours)) + 8
    post_bars = (policy["reopen_baseline_bars"]
                 + policy["reopen_min_closed_bars"]
                 + policy["stability_consecutive_checks"]
                 + int(np.ceil(float(policy["reopen_min_hours"])
                               / bar_hours)) + 6)
    close_at = pd.Timestamp("2024-01-05 00:00:00", tz="UTC")
    reopen_at = close_at + pd.Timedelta(hours=48)
    before = pd.date_range(
        close_at - pd.Timedelta(hours=bar_hours * pre_bars),
        periods=pre_bars, freq=f"{bar_hours:g}h")
    after = pd.date_range(reopen_at, periods=post_bars,
                          freq=f"{bar_hours:g}h")
    stamps = before.append(after).tz_localize(None)
    n = len(stamps)
    closes = 100.0 + 0.20 * np.sin(np.arange(n, dtype=float))
    frame = pd.DataFrame({
        "DATE_TIME": stamps,
        "OPEN": closes, "HIGH": closes * 1.0005,
        "LOW": closes * 0.9995, "CLOSE": closes,
        "VOLUME": 1000.0,
        "SPREAD": np.full(n, 0.0002),
        "feat": np.linspace(0.0, 1.0, n),
    })
    path = Path(tmp_dir) / f"{cell['cell_id']}_bars.csv"
    frame.to_csv(path, index=False)
    return {"csv": path, "close_at": close_at,
            "reopen_at": reopen_at, "bars": n,
            "pre_bars": pre_bars, "post_bars": post_bars}


def build_env(cell: dict, window: dict, *, tmp_dir: Path,
              calendar_intervals="from_window") -> GymFxEnv:
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
            "leverage_cap": 1.0},
    }
    if policy["enabled"]:
        config.update({
            "session_exposure_enabled": True,
            "session_exposure_policy": dict(policy),
            "session_venue": "mt5_demo",
            "session_account_fingerprint": "fp-wp4",
            "session_symbol": "ETHUSD",
            "session_spread_column": "SPREAD",
            "session_flatten_custody_root": str(
                Path(tmp_dir) / f"{cell['cell_id']}_custody"),
        })
        if calendar_intervals == "from_window":
            config["session_calendar_intervals"] = [
                [str(window["close_at"]), str(window["reopen_at"])]]
        elif calendar_intervals is not None:
            config["session_calendar_intervals"] = calendar_intervals
        # calendar_intervals=None: the missing-evidence refusal case
    return GymFxEnv(config, DataFeed(config), Broker(config),
                    Envelope(config), Preprocessor(config),
                    Reward(config), Metrics(config))


def mechanics_smoke(cell: dict, *, tmp_dir: Path, repo_root: Path,
                    seed: int = 7) -> dict:
    """Bounded CPU mechanics run: enter early, attempt entries in
    every later state, and collect the operational metrics the order
    names. Deterministic by seed and cell digest."""
    identity = verify_frozen_identity(repo_root)
    verify_cell(cell)
    window = gap_window(cell, tmp_dir=tmp_dir)
    env = build_env(cell, window, tmp_dir=tmp_dir)
    rng = np.random.default_rng(
        seed + int(cell["digest"][:8], 16) % 1000)
    started = time.perf_counter()
    obs, _info = env.reset(seed=seed)
    states, overlays, finals = [], [], []
    blocked = forced = cancels = steps = 0
    rewards, pnls = [], []
    terminated = False
    for index in range(window["bars"] + 4):
        action = 1.0 if index % 3 != 2 else -1.0
        if rng.random() < 0.10:
            action = 0.0
        obs, reward, term, trunc, info = env.step([float(action)])
        steps += 1
        rewards.append(float(reward))
        pnls.append(float(info.get("pnl", 0.0)))
        state = info.get("session_state")
        if state is not None:
            states.append(state)
            overlays.append(info.get("session_overlay"))
            finals.append(info.get("session_final_action"))
            if info.get("session_overlay") == \
                    "masked_risk_increase":
                blocked += 1
            if info.get("session_overlay") == "forced_close":
                forced += 1
            if info.get("session_cancel_pending"):
                cancels += 1
        if term or trunc:
            terminated = term
            break
    wall = time.perf_counter() - started
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    trades = getattr(env, "trades", None)
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
    result = {
        "cell_id": cell["cell_id"],
        "family": cell["family"],
        "cell_digest": cell["digest"],
        "identity_verified": identity,
        "bars_in_window": window["bars"],
        "steps": steps,
        "wall_seconds": round(wall, 4),
        "steps_per_second": round(steps / wall, 2) if wall else None,
        "peak_rss_kb": peak_kb,
        "terminated": terminated,
        "session_state_counts": state_counts,
        "reopen_blocked_decisions": blocked,
        "forced_closes": forced,
        "cancellation_decisions": cancels,
        "reward_sum": round(float(np.sum(rewards)), 8),
        "pnl_sum": round(float(np.sum(pnls)), 8),
        "no_bar_inside_closure": True,
    }
    return result


def replay_is_deterministic(cell: dict, *, tmp_dir: Path,
                            repo_root: Path) -> bool:
    first = mechanics_smoke(cell, tmp_dir=Path(tmp_dir) / "a",
                            repo_root=repo_root)
    second = mechanics_smoke(cell, tmp_dir=Path(tmp_dir) / "b",
                             repo_root=repo_root)
    keys = ("steps", "session_state_counts",
            "reopen_blocked_decisions", "forced_closes",
            "cancellation_decisions", "reward_sum", "pnl_sum")
    return all(first[k] == second[k] for k in keys)
