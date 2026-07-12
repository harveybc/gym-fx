#!/usr/bin/env python3
"""Verify isolated Nautilus evaluator processes produce identical evidence."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from decimal import Decimal


def _worker(profile_path: str) -> str:
    from simulation_engines.bakeoff import build_multi_asset_fixture
    from simulation_engines.bakeoff import build_rollover_rate_fixture
    from simulation_engines.contracts import load_execution_cost_profile
    from simulation_engines.nautilus_adapter import NautilusReplayAdapter

    profile = load_execution_cost_profile(profile_path)
    instruments, frames, actions = build_multi_asset_fixture()
    result = NautilusReplayAdapter(profile).run(
        instrument_specs=instruments,
        frames=frames,
        actions=actions,
        initial_cash=Decimal("100000"),
        financing_rate_data=(
            build_rollover_rate_fixture() if profile.financing_enabled else None
        ),
    )
    return result["result_hash"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 2 or args.tasks < args.workers:
        raise ValueError("use at least two workers and one task per worker")
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        hashes = pool.map(_worker, [args.profile] * args.tasks)
    evidence = {
        "workers": args.workers,
        "tasks": args.tasks,
        "unique_hashes": sorted(set(hashes)),
        "deterministic_across_processes": len(set(hashes)) == 1,
    }
    print(json.dumps(evidence, indent=2))
    if not evidence["deterministic_across_processes"]:
        raise SystemExit("parallel evaluator hashes differ")


if __name__ == "__main__":
    main()
