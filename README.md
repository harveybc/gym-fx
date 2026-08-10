# gym-fx

Gymnasium-compatible trading environment for FX and crypto time series. It
wraps a [backtrader](https://www.backtrader.com/) simulation behind a standard
`gym.Env` step/reset interface so reinforcement-learning agents can trade a
historical price feed with realistic order execution, protected stop-loss /
take-profit brackets, margin and solvency accounting, execution-cost models and
session/event context. Every moving part — data feed, broker, strategy
(order execution), preprocessor, reward and metrics — is a plugin selected by
name from a JSON config.

## Status

**Lifecycle: ACTIVE-CORE.** This environment is the execution layer consumed by
the [agent-multi](https://github.com/harveybc/agent-multi) RL training and
optimization pipelines. It is actively maintained; its behavior is part of the
evidence contract of ongoing experiments.

> **Disclaimer:** everything in this repository operates on historical or
> synthetic data in simulation/backtest mode. Nothing here executes real-capital
> trades, and none of the examples or configurations are financial advice.

## Role and non-responsibilities

**Role:** provide the trading *environment only* — observation construction,
order simulation, reward computation and per-episode diagnostics behind the
Gymnasium API (`GymFxEnv` in [`app/env.py`](app/env.py)).

**Not responsible for:**

- Training agents or optimizing hyperparameters — that is
  [agent-multi](https://github.com/harveybc/agent-multi).
- Price prediction or feature/label engineering — see
  [predictor](https://github.com/harveybc/predictor) and
  [feature-eng](https://github.com/harveybc/feature-eng).
- Distributed optimization — gym-fx has no DOIN integration of its own; it is
  driven indirectly when [doin-node](https://github.com/harveybc/doin-node)
  runs agent-multi experiments.
- Live brokerage connectivity or real order routing.

## Architecture

```
JSON config ──> app/main.py ──> GymFxEnv (app/env.py)
                                   │ observation: price/returns windows,
                                   │ position, equity, margin, event context
                                   ▼
                            BTBridge (app/bt_bridge.py)
                                   │ discrete/continuous actions ──> orders
                                   ▼
                       backtrader Cerebro simulation
              (alternative engines in simulation_engines/,
               incl. an optional NautilusTrader adapter)
```

Key mechanisms, each covered by tests:

- **Protected SL/TP execution** — with `"require_protected_entries": true`
  the bridge refuses to start unless the strategy plugin implements
  `apply_action()` (bracket orders), and if the plugin fails at runtime the
  entry action is *rejected and counted*, never silently downgraded to a naked
  market order. Risk-reducing closes remain available. See
  [`app/bt_bridge.py`](app/bt_bridge.py) and
  [`tests/test_protected_order_execution.py`](tests/test_protected_order_execution.py).
- **Solvency modes** — `"solvency_mode"`: `normal_realistic` (default; a margin
  breach terminates the episode) or `easy_chronological_continuation`
  (train-only, enforced by the env: the would-be margin call is recorded, the
  loss is retained, operational capital is recapitalized as debt and the
  chronological episode continues). See
  [`tests/test_solvency_modes.py`](tests/test_solvency_modes.py).
- **Diagnostics for evidence trails** — the observation includes price and
  returns windows; per-step `info` carries raw event-context and execution
  values, and the end-of-episode summary contains `action_diagnostics` and
  `execution_diagnostics` counters (entries seen, protected rejections,
  forced-flat orders, deadband actions, ...).
- **Execution-cost and event context** — spread/slippage/financing profiles
  under [`examples/config/execution_cost_profiles/`](examples/config/execution_cost_profiles/)
  and an OANDA-style session/event calendar
  ([`app/oanda_calendar.py`](app/oanda_calendar.py)) with entry-blocking and
  forced-flat windows.

### Relationship to sibling repositories

- [agent-multi](https://github.com/harveybc/agent-multi) consumes this package
  through its `gym_fx_env` environment plugin; agent-multi's SAC/PPO/DQN
  pipelines and curricula are the primary users of the solvency modes and
  protected-execution semantics.
- [predictor](https://github.com/harveybc/predictor) and
  [prediction_provider](https://github.com/harveybc/prediction_provider) live
  upstream on the forecasting side and are not imported here.

## Prerequisites

From [`setup.py`](setup.py): `pandas`, `backtrader`, `gymnasium`, `numpy`,
`requests`; optional extra `nautilus` pins `nautilus_trader==1.230.0`. No
`python_requires` is declared; the environment is exercised in practice on
Python 3.12 (verified below with Python 3.12.13, gymnasium 1.3.0).

## Installation

```bash
git clone https://github.com/harveybc/gym-fx.git
cd gym-fx
pip install -e .
# optional NautilusTrader engine:
pip install -e ".[nautilus]"
```

*Unverified in a clean environment* — the commands above are the standard
editable install; they were not re-executed from scratch for this README. The
package imports and the example below were verified in an existing Python
3.12.13 environment with this repository installed editable.

## Smallest working example

Runs a buy-and-hold driver for 490 steps over the bundled EURUSD sample
(verified: exit code 0):

```bash
python app/main.py --load_config examples/config/buy_hold.json
```

Observed result: prints per-step diagnostics and writes
`examples/results/buy_hold_summary.json` with `final_equity: 9999.99121`,
`trades_total: 0`, plus `action_diagnostics` (`steps: 490`,
`hold_actions: 489`, `long_actions: 1`) and `execution_diagnostics`. The
console entry point `gym-fx-env` (installed by `setup.py`) invokes the same
`app.main:main`.

Other repository-owned configs: [`examples/config/random_driver.json`](examples/config/random_driver.json),
[`examples/config/nautilus_gym_smoke.json`](examples/config/nautilus_gym_smoke.json).

## Distributed / DOIN usage

None directly. gym-fx is a local library; distributed campaigns orchestrate it
only through agent-multi (see the
[agent-multi README](https://github.com/harveybc/agent-multi)).

## Configuration and plugins

Configuration is a flat JSON merged over defaults in
[`app/config.py`](app/config.py) (CLI flags in [`app/cli.py`](app/cli.py)
override file values). Plugins are resolved by
[`app/plugin_loader.py`](app/plugin_loader.py) from `importlib.metadata` entry
points declared in [`setup.py`](setup.py):

| Entry-point group | Plugins (this package) |
|---|---|
| `data_feed.plugins` | `default_data_feed` |
| `broker.plugins` | `default_broker`, `oanda_broker` |
| `strategy.plugins` | `default_strategy`, `direct_fixed_sltp`, `direct_atr_sltp` |
| `preprocessor.plugins` | `default_preprocessor`, `feature_window_preprocessor` |
| `reward.plugins` | `pnl_reward`, `sharpe_reward`, `dd_penalized_reward` |
| `metrics.plugins` | `default_metrics`, `trading_metrics` |

**Note on `preprocessor.plugins`:** the preprocessors under
[`preprocessor_plugins/`](preprocessor_plugins/) are local to this repository
but registered into the *shared* `preprocessor.plugins` entry-point group also
used by sibling packages ([preprocessor](https://github.com/harveybc/preprocessor),
[predictor](https://github.com/harveybc/predictor)). If several of those
packages are installed in one environment, the group contains entries from all
of them — including a same-named `default_preprocessor` from predictor — so
prefer a dedicated environment per application when plugin names matter.

## Tests

```bash
python -m pytest tests --collect-only -q   # observed: "84 tests collected in 0.77s"
python -m pytest tests                     # full run: unverified for this README
```

The suite covers protected order execution, solvency modes, continuous-action
thresholds, execution-cost context, event-context overlays, the
feature-window preprocessor, trading metrics, the OANDA calendar and the
Nautilus engine contracts. Smoke/benchmark utilities live in
[`tools/`](tools/) (e.g. [`tools/smoke_test.py`](tools/smoke_test.py),
[`tools/check_gym_compliance.py`](tools/check_gym_compliance.py)).

## Outputs and reproducibility

- `results_file` (JSON): episode summary — equity, return, drawdown, trade
  counts, `action_diagnostics`, `execution_diagnostics`.
- `save_config` (JSON): the fully merged effective config, written back so a
  run can be reproduced exactly from its emitted config.
- Example outputs land under [`examples/results/`](examples/results/).

Determinism depends on the driver: the bundled deterministic drivers
(`buy_hold`, fixed data feed) reproduce bit-identical summaries; RL agents on
top add their own seeding (managed by agent-multi).

## Safety and credentials

The default data feed and broker are offline simulations over CSV files; no
network access or credentials are needed. The `oanda_broker` plugin and the
OANDA calendar can consume broker-formatted data, but this repository ships no
credentials and no live-trading path; do not commit API keys into configs.
Simulation results do not guarantee live performance.

## Limitations and migration notes

- `easy_chronological_continuation` is train-only by design; the env raises if
  it is enabled outside training mode. Evaluation always runs
  `normal_realistic`.
- No `python_requires`/version pinning in packaging metadata yet.
- Top-level package names (`app`, `*_plugins`) are shared conventions across
  sibling repositories; installing several of them into one environment can
  shadow same-named modules. Use one environment per application, or run from
  the repository root so local packages win.
- The NautilusTrader adapter ([`simulation_engines/`](simulation_engines/)) is
  optional and only exercised when the `nautilus` extra is installed.

## Related repositories

- [agent-multi](https://github.com/harveybc/agent-multi) — RL trainer/optimizer consuming this env
- [doin-node](https://github.com/harveybc/doin-node) — decentralized optimization runtime (drives agent-multi, not gym-fx directly)
- [preprocessor](https://github.com/harveybc/preprocessor) — standalone CSV preprocessing app sharing the `preprocessor.plugins` group
- [predictor](https://github.com/harveybc/predictor) — phased deep-learning price prediction

## License

This repository does not currently include a LICENSE file; no license terms
are published. Contact the owner before reusing the code.
