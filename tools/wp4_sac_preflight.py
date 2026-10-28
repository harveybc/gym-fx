"""WP4 bounded CPU SAC throughput preflight (order
agent-multi@4ad4937b). MECHANICS_AND_THROUGHPUT_ONLY — zero economic
authority, no checkpoint promotion, CPU only, one cell, one seed.

Hard bounds: <= 20,000 environment steps, <= 20,000 SAC gradient
updates, 2 hour wall clock. Identities (cell, manifest, window, tape,
authority files, SAC config) are persisted BEFORE execution. The run
is observable (heartbeat JSON with phase/progress/ETA) and externally
stoppable (SIGTERM, or creating <out>.STOP). Any unresolved exposure
or custody state stops fail-closed and is returned as evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

MAX_ENV_STEPS = 20_000
MAX_UPDATES = 20_000
WALL_LIMIT_SECONDS = 2 * 3600


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cells-dir", required=True, type=Path)
    parser.add_argument("--expected-manifest-digest", required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-env-steps", type=int,
                        default=MAX_ENV_STEPS)
    parser.add_argument("--max-updates", type=int,
                        default=MAX_UPDATES)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.max_env_steps > MAX_ENV_STEPS or \
            args.max_updates > MAX_UPDATES:
        raise SystemExit("bounds exceed the authorized preflight")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    from tools import wp4_driver as drv
    from tools.wp4_materializer import canonical_bytes, sha256_hex

    started = time.time()
    stop_flag = {"stop": False, "reason": None}

    def request_stop(signum, _frame):
        stop_flag["stop"] = True
        stop_flag["reason"] = f"signal {signum}"
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    stop_file = args.out.with_suffix(".STOP")
    heartbeat_path = args.out.with_suffix(".heartbeat.json")

    def heartbeat(phase, **fields):
        elapsed = time.time() - started
        payload = {"phase": phase, "elapsed_seconds": round(
            elapsed, 1), "pid": os.getpid(), **fields}
        heartbeat_path.write_text(json.dumps(payload, indent=1))
        print(f"[preflight] {json.dumps(payload)}", flush=True)

    def must_continue(phase):
        if stop_flag["stop"]:
            raise RuntimeError(f"externally stopped during {phase}: "
                               f"{stop_flag['reason']}")
        if stop_file.exists():
            raise RuntimeError(f"stop file present during {phase}")
        if time.time() - started > WALL_LIMIT_SECONDS:
            raise RuntimeError(f"wall-clock limit during {phase}")

    # ---- identities persisted BEFORE execution -------------------
    manifest = json.loads(args.manifest.read_text())
    drv.verify_manifest(manifest, args.expected_manifest_digest)
    drv.verify_manifest_matches_dir(manifest, args.cells_dir)
    cell = json.loads(
        (args.cells_dir / f"{args.cell}.json").read_text())
    drv.verify_cell_binding(cell, manifest,
                            args.expected_manifest_digest)
    repo_root = Path(__file__).resolve().parents[1]
    authority = drv.verify_frozen_identity(repo_root)
    scratch = args.out.parent / f"{args.cell}_preflight_work"
    window = drv.load_historical_window(
        scratch / "window", start=drv.PLAIN_WINDOW_START,
        end=drv.PLAIN_WINDOW_END)
    tape = drv.action_tape(args.seed, window["bars"] + 4)
    sac_config = {
        "algo": "stable_baselines3.SAC", "policy": "MultiInputPolicy",
        "device": "cpu", "seed": args.seed,
        "learning_starts": 1000, "train_freq": 1,
        "gradient_steps": 1, "batch_size": 64,
        "buffer_size": 50_000,
        "policy_kwargs": {"net_arch": [64, 64]},
    }
    identity_block = {
        "schema": "gymfx.wp4.sac_preflight.identity.v1",
        "authorization": "agent-multi@4ad4937b",
        "scope": "MECHANICS_AND_THROUGHPUT_ONLY",
        "cell_id": cell["cell_id"], "cell_digest": cell["digest"],
        "manifest_digest": manifest["digest"],
        "window_digest": window["meta"]["digest"],
        "tape_digest": tape["digest"],
        "authority_sha256": authority,
        "sac_config": sac_config,
        "bounds": {"max_env_steps": args.max_env_steps,
                   "max_updates": args.max_updates,
                   "wall_seconds": WALL_LIMIT_SECONDS},
    }
    identity_block["digest"] = sha256_hex(
        canonical_bytes(identity_block))
    identity_path = args.out.with_suffix(".identity.json")
    identity_path.write_text(json.dumps(identity_block, indent=1))
    heartbeat("identities_persisted",
              identity=identity_block["digest"][:16])

    # ---- phase 1: conservation gate on the exact cell ------------
    heartbeat("conservation_gate")
    gate_run = drv.recorded_run(cell, manifest,
                                args.expected_manifest_digest, tape,
                                window, tmp_dir=scratch / "gate",
                                repo_root=repo_root, seed=args.seed)
    conservation = gate_run["conservation"]
    if conservation["verdict"] != "ELIGIBLE":
        result = {"status": "REFUSED_FAIL_CLOSED",
                  "reason": "conservation gate refused before any "
                            "training step",
                  "failed_invariants":
                      conservation["failed_invariants"],
                  "conservation": conservation,
                  "identity": identity_block}
        args.out.write_text(json.dumps(result, indent=1,
                                       default=str))
        heartbeat("refused")
        return 3

    # ---- phase 2: pure environment throughput --------------------
    heartbeat("env_throughput")
    env = drv.build_env(cell, window, tmp_dir=scratch / "envonly")
    env.reset(seed=args.seed)
    env_steps = 0
    t0 = time.perf_counter()
    index = 0
    while env_steps < min(2000, args.max_env_steps):
        must_continue("env_throughput")
        action = tape["actions"][index % len(tape["actions"])]
        _o, _r, term, trunc, _info = env.step([float(action)])
        env_steps += 1
        index += 1
        if term or trunc:
            env.reset(seed=args.seed)
    env_wall = time.perf_counter() - t0
    env_sps = env_steps / env_wall
    heartbeat("env_throughput_done", steps=env_steps,
              env_steps_per_second=round(env_sps, 2))

    # ---- phase 3: SAC training throughput ------------------------
    heartbeat("sac_setup")
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    train_env = drv.build_env(cell, window,
                              tmp_dir=scratch / "train")
    model = SAC("MultiInputPolicy", train_env,
                device="cpu", seed=args.seed,
                learning_starts=sac_config["learning_starts"],
                train_freq=sac_config["train_freq"],
                gradient_steps=sac_config["gradient_steps"],
                batch_size=sac_config["batch_size"],
                buffer_size=sac_config["buffer_size"],
                policy_kwargs=dict(net_arch=[64, 64]),
                verbose=0)
    budget = min(args.max_env_steps,
                 sac_config["learning_starts"] + args.max_updates)
    state_counts = {}

    class Telemetry(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos") or []
            for info in infos:
                state = info.get("session_state")
                if state:
                    state_counts[state] = \
                        state_counts.get(state, 0) + 1
            if self.num_timesteps % 500 == 0:
                done = self.num_timesteps
                elapsed = time.time() - started
                rate = done / max(elapsed, 1e-9)
                heartbeat("sac_training", env_steps=done,
                          budget=budget,
                          progress=round(done / budget, 3),
                          eta_seconds=round(
                              (budget - done) / max(rate, 1e-9), 1))
            try:
                must_continue("sac_training")
            except RuntimeError as exc:
                stop_flag["reason"] = str(exc)
                return False
            return True

    t1 = time.perf_counter()
    model.learn(total_timesteps=budget, callback=Telemetry(),
                progress_bar=False)
    sac_wall = time.perf_counter() - t1
    total_steps = int(model.num_timesteps)
    updates = max(0, total_steps - sac_config["learning_starts"])
    heartbeat("sac_done", steps=total_steps, updates=updates)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "status": ("STOPPED_FAIL_CLOSED" if stop_flag["reason"]
                   else "COMPLETED"),
        "stop_reason": stop_flag["reason"],
        "scope": "MECHANICS_AND_THROUGHPUT_ONLY — zero economic "
                 "authority, no checkpoint promotion (no checkpoint "
                 "is even saved)",
        "identity": identity_block,
        "environment_throughput": {
            "steps": env_steps,
            "wall_seconds": round(env_wall, 3),
            "env_steps_per_second": round(env_sps, 2)},
        "sac_training": {
            "env_steps": total_steps,
            "gradient_updates": updates,
            "wall_seconds": round(sac_wall, 3),
            "combined_steps_per_second": round(
                total_steps / sac_wall, 2),
            "updates_per_second": round(
                updates / sac_wall, 2) if updates else 0.0},
        "peak_rss_kb": peak_kb,
        "session_state_counts_during_training": state_counts,
        "closure_compliance_gate_run": {
            "verdict": conservation["verdict"],
            "failed_invariants": conservation["failed_invariants"],
            "exposure_across_closure":
                conservation["exposure_across_closure"]},
        "conservation": conservation,
        "wall_seconds_total": round(time.time() - started, 1),
    }
    args.out.write_text(json.dumps(result, indent=1, default=str))
    heartbeat("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
