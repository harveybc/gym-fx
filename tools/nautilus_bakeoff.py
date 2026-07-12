#!/usr/bin/env python3
"""Run and persist the deterministic NautilusTrader engine bake-off fixture."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from simulation_engines.bakeoff import build_multi_asset_fixture
from simulation_engines.bakeoff import reconcile_fills
from simulation_engines.bakeoff import build_rollover_rate_fixture
from simulation_engines.bakeoff import export_execution_reports
from simulation_engines.contracts import load_execution_cost_profile
from simulation_engines.nautilus_adapter import NautilusReplayAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 for determinism evidence")
    profile = load_execution_cost_profile(args.profile)
    instruments, frames, actions = build_multi_asset_fixture()
    initial_cash = Decimal("100000")
    rate_data = build_rollover_rate_fixture() if profile.financing_enabled else None
    runs = []
    for _ in range(args.repeat):
        run = NautilusReplayAdapter(profile).run(
            instrument_specs=instruments,
            frames=frames,
            actions=actions,
            initial_cash=initial_cash,
            financing_rate_data=rate_data,
        )
        runs.append(run)

    hashes = {run["result_hash"] for run in runs}
    evidence = {
        "schema_version": "simulation_engine_bakeoff.v1",
        "engine": "nautilus_trader",
        "engine_version": NautilusReplayAdapter.ENGINE_VERSION,
        "profile_path": str(Path(args.profile).resolve()),
        "repeat_count": args.repeat,
        "deterministic": len(hashes) == 1,
        "result_hashes": sorted(hashes),
        "reconciliation": reconcile_fills(
            runs[0],
            instruments,
            profile,
            initial_cash=initial_cash,
        ),
        "execution_reports": export_execution_reports(
            runs[0], instruments, profile
        ),
        "run": runs[0],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({k: evidence[k] for k in ("engine", "engine_version", "deterministic", "reconciliation")}, indent=2))
    if not evidence["deterministic"]:
        raise SystemExit("Nautilus replay was not deterministic")


if __name__ == "__main__":
    main()
