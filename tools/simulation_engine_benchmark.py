#!/usr/bin/env python3
"""Compare fresh-process-style replay overhead for Backtrader and Nautilus."""

from __future__ import annotations

import argparse
import json
import resource
import time
from decimal import Decimal
from pathlib import Path

import backtrader as bt
import pandas as pd

from simulation_engines.bakeoff import build_multi_asset_fixture
from simulation_engines.bakeoff import build_rollover_rate_fixture
from simulation_engines.contracts import load_execution_cost_profile
from simulation_engines.nautilus_adapter import NautilusReplayAdapter


class _BacktraderTargets(bt.Strategy):
    params = (("targets", ()),)

    def next(self):
        target = dict(self.p.targets).get(len(self.data))
        if target is not None:
            self.order_target_size(target=target)


def _run_backtrader(profile, frames, actions) -> None:
    eur_frames = [frame for frame in frames if frame.instrument_id == "EUR/USD.SIM"]
    targets_by_time = {
        action.ts_event_ns: float(action.target_units)
        for action in actions
        if action.instrument_id == "EUR/USD.SIM"
    }
    rows = []
    targets = []
    for index, frame in enumerate(eur_frames, start=1):
        rows.append(
            {
                "datetime": pd.Timestamp(frame.ts_event_ns, tz="UTC").tz_localize(None),
                "open": float(frame.open),
                "high": float(frame.high),
                "low": float(frame.low),
                "close": float(frame.close),
                "volume": float(frame.volume),
                "openinterest": 0.0,
            }
        )
        if frame.ts_event_ns in targets_by_time:
            targets.append((index, targets_by_time[frame.ts_event_ns]))
    dataframe = pd.DataFrame(rows).set_index("datetime")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=dataframe))
    cerebro.addstrategy(_BacktraderTargets, targets=tuple(targets))
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=float(profile.commission_rate_per_side))
    adverse = float(profile.quote_adverse_rate_per_side)
    if adverse:
        cerebro.broker.set_slippage_perc(
            perc=adverse,
            slip_open=True,
            slip_limit=True,
            slip_match=True,
        )
    cerebro.run()


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "runs": len(samples),
        "total_seconds": sum(samples),
        "mean_seconds": sum(samples) / len(samples),
        "median_seconds": ordered[len(ordered) // 2],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "runs_per_second": len(samples) / sum(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.runs < 3:
        raise ValueError("--runs must be at least 3")

    profile = load_execution_cost_profile(args.profile)
    instruments, frames, actions = build_multi_asset_fixture()
    rate_data = build_rollover_rate_fixture() if profile.financing_enabled else None
    backtrader_samples = []
    nautilus_samples = []
    for _ in range(args.runs):
        started = time.perf_counter()
        _run_backtrader(profile, frames, actions)
        backtrader_samples.append(time.perf_counter() - started)

        started = time.perf_counter()
        NautilusReplayAdapter(profile).run(
            instrument_specs=instruments,
            frames=frames,
            actions=actions,
            initial_cash=Decimal("100000"),
            financing_rate_data=rate_data,
        )
        nautilus_samples.append(time.perf_counter() - started)

    evidence = {
        "schema_version": "simulation_engine_benchmark.v1",
        "workload_note": (
            "Backtrader runs the EUR/USD subset; Nautilus runs the full two-instrument "
            "shared-account fixture. This measures fresh-run overhead, not normalized "
            "per-event throughput."
        ),
        "profile_id": profile.profile_id,
        "backtrader": _summary(backtrader_samples),
        "nautilus_trader": _summary(nautilus_samples),
        "process_max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
