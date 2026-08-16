# AGENTS.md — gym-fx

Instructions for AI coding agents working in this repository.
Human-facing documentation is in [`README.md`](README.md).

## Project overview

gym-fx is a Gymnasium-compatible trading environment for FX and crypto time
series. It wraps a backtrader simulation behind the standard `gym.Env`
`reset`/`step` interface, so an agent can trade a historical CSV price feed with
order execution, protected stop-loss / take-profit brackets, margin and solvency
accounting, execution costs and a session/event calendar. Data feed, broker,
strategy (order execution), preprocessor, reward and metrics are all plugins
selected by name from a flat JSON config.

It does **not** train agents (that is agent-multi), does not predict prices
(predictor / feature-eng), and has no live brokerage connectivity or real order
routing. Everything runs offline over CSV files.

## Agent quickstart (install → run → show the user results)

Verified end to end on Python 3.12.13 with gymnasium 1.3.0, backtrader
release 1.9.78 build 123, pandas, numpy.

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate     # or: conda create -n gymfx python=3.12
pip install -e .
```

`setup.py` requires `pandas`, `backtrader`, `gymnasium`, `numpy`, `requests`.
The optional NautilusTrader engine is `pip install -e ".[nautilus]"`
(`nautilus_trader==1.230.0`).

The install also registers this repository's plugin entry points, which the
config resolves by name. Use a dedicated environment: sibling repositories in
this stack ship a top-level `app` package and also publish into the shared
`preprocessor.plugins` entry-point group, so co-installation can shadow names.

*Not verified:* a from-scratch `pip install -e .` in an empty environment. All
commands below were run in an environment that already had this package
installed editable.

### 2. Smoke test

```bash
python -m pytest tests -q
```

Expected scale: **84 passed** in about 2 seconds (48 warnings). The suite covers
protected order execution, solvency modes, continuous-action thresholds,
execution-cost context, event-context overlays, the feature-window
preprocessor, trading metrics, the OANDA calendar and the Nautilus engine
contracts.

### 3. Representative run — a short episode with a trivial policy

`examples/config/random_driver.json` drives 490 steps of uniformly random
discrete actions over the bundled 500-row EURUSD minute sample
(`examples/data/eurusd_sample.csv`):

```bash
python app/main.py --load_config examples/config/random_driver.json
```

Runs in a few seconds and prints the end-of-episode summary as JSON. Observed:
490 steps, ~10000.01 final equity from 10000 initial cash, 0 closed trades
(the random driver flips position constantly), and `action_diagnostics` counting
hold/long/short actions. Exact counts vary — the random driver is not seeded.

Deterministic alternatives owned by the repo:
`examples/config/buy_hold.json` (one long entry, then hold — a fixed
constant-action policy) and `examples/config/nautilus_gym_smoke.json` (needs
the `nautilus` extra).

### 4. Analytics the repo actually supports

```bash
python tools/check_gym_compliance.py   # gymnasium.utils.env_checker on the default env
python tools/smoke_test.py             # flat / buy_hold / seeded-reset / return-identity assertions
```

Both exit 0. `check_gym_compliance.py` prints
`[check_gym_compliance] env passes gymnasium.env_checker` (with two benign
warnings about infinite observation-space bounds). `smoke_test.py` **rewrites**
`examples/results/flat_summary.json` and `examples/results/buy_hold_summary.json`;
the committed copies predate the diagnostics fields, so expect a git diff that
adds `action_diagnostics` / `execution_diagnostics` blocks. Revert with
`git checkout -- examples/results/` unless the update is intended.

Other tools (`tools/simulation_engine_benchmark.py`,
`tools/nautilus_bakeoff.py`, `tools/nautilus_parallel_smoke.py`,
`tools/oom_smoke_test.py`) are heavier and were not run here.

There is **no per-step trace file**. Per-step values are exposed only through
the `info` dict returned by `env.step()`; only the end-of-episode summary is
persisted. To get an equity curve you must record `info` yourself in a loop
around `GymFxEnv` (see `_run_env` in `app/main.py` for the loop shape).

### 5. Final message to the user

> The episode finished. Results are in
> `examples/results/random_driver_summary.json` (episode summary: final equity,
> total return, max drawdown, trade counts, `action_diagnostics`,
> `execution_diagnostics`), and the fully merged effective config was written to
> `examples/config/random_driver_out.json` so the run can be replayed exactly.
> There is no UI and no per-step trace file.
>
> Suggested first analysis: bar-chart the `action_diagnostics` hold/long/short
> counts from the summary JSON and compare the random driver against
> `examples/config/buy_hold.json` — it shows how much of the equity difference
> comes from trading activity rather than from the price path. For an equity
> curve you have to write a small loop over `GymFxEnv` that appends
> `info["equity"]` each step; the CLI does not emit one.

## Build, test and lint commands

```bash
pip install -e .                        # editable install (registers plugin entry points)
pip install -e ".[nautilus]"            # optional NautilusTrader engine
python -m pytest tests -q               # 84 passed, ~2s
python -m pytest tests --collect-only -q # collection only
python app/main.py --load_config <cfg>  # run one episode from a JSON config
gym-fx-env --load_config <cfg>          # console script installed by setup.py (same entry point)
```

No linter or formatter is configured in this repository (no ruff/flake8/black
config, no pre-commit, no CI workflow). Do not introduce one without asking.

## Layout

| Path | Contents |
|---|---|
| `app/` | Env core: `env.py` (`GymFxEnv`), `bt_bridge.py` (actions → backtrader orders), `main.py` (CLI runner), `config.py`, `cli.py`, `config_merger.py`, `data_handler.py`, `oanda_calendar.py` |
| `gym_fx/` | Thin package re-export of the environment |
| `data_feed_plugins/` | CSV data feeds (`default_data_feed`) |
| `broker_plugins/` | Simulated brokers (`default_broker`, `oanda_broker`) |
| `strategy_plugins/` | Order-execution plugins (`default_strategy`, `direct_fixed_sltp`, `direct_atr_sltp`) |
| `preprocessor_plugins/` | Observation builders (`default_preprocessor`, `feature_window_preprocessor`) |
| `reward_plugins/` | `pnl_reward`, `sharpe_reward`, `dd_penalized_reward` |
| `metrics_plugins/` | `default_metrics`, `trading_metrics` |
| `simulation_engines/` | Alternative engines and contracts, incl. the optional NautilusTrader adapter |
| `examples/` | `config/` JSON configs, `data/` sample CSVs, `results/` example summaries |
| `tests/` | pytest suite (84 tests) |
| `tools/` | Smoke tests, gym-compliance check, benchmarks |
| `scripts/` | `bootstrap_nautilus_env.sh` |

## Conventions and constraints

- **Config-driven.** One flat JSON merged over `DEFAULT_VALUES` in
  `app/config.py`; CLI flags override file values (`app/config_merger.py`).
  Unknown `--flags` are merged too, so any config key is settable from the CLI.
- **Plugin architecture.** Six entry-point groups declared in `setup.py`
  (`data_feed.plugins`, `broker.plugins`, `strategy.plugins`,
  `preprocessor.plugins`, `reward.plugins`, `metrics.plugins`). Plugins are
  resolved by name via `importlib.metadata`, so a new plugin needs a `setup.py`
  entry and a reinstall. A plugin exposes `plugin_params` (defaults merged into
  the config) and is constructed by `app/plugin_loader.py`.
- **`preprocessor.plugins` is a shared group** across sibling repositories,
  including a same-named `default_preprocessor` from predictor. One environment
  per application.
- **Protected execution.** With `"require_protected_entries": true` the bridge
  refuses to start unless the strategy plugin implements `apply_action()`, and a
  runtime plugin failure rejects and counts the entry rather than silently
  downgrading it to a naked market order. Do not add a fallback path that
  weakens this.
- **Solvency modes.** `normal_realistic` (default; margin breach terminates the
  episode) and `easy_chronological_continuation` (train-only; the env raises if
  it is used outside training mode). Evaluation always runs `normal_realistic`.
- **Data contract.** Input CSV needs a date column and a price column named by
  `date_column` / `price_column` (`DATE_TIME` / `CLOSE` in the samples), with
  `headers: true`. Outputs: `results_file` (episode summary JSON) and
  `save_config` (merged effective config).
- **Determinism.** Deterministic drivers (`buy_hold`, fixed feed) reproduce
  identical summaries; the `random` driver does not seed its policy.
- No credentials belong in configs. The default feed and broker are offline CSV
  simulations; there is no live-trading path.

## Do not touch

- `build/`, `gym_fx.egg-info/`, `__pycache__/`, `.pytest_cache/` — generated.
- `examples/data/` and `examples/data_downsampled/` — committed input fixtures;
  tests and configs depend on their exact contents.
- `examples/results/*.json` — committed reference summaries. Runs overwrite
  them; revert unless the change is intended.
- Sibling repositories (agent-multi, predictor, preprocessor, doin-*). Changes
  that alter the env's observed behavior break their evidence trails — flag them
  instead of silently adjusting semantics.
- Do not weaken the protected-entry or solvency-mode invariants to make a test
  pass.
