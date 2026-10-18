# RL preparation review

Satoshi inspected base `e287ea2`; this review also checks the later static-input
branch. Component tests are not full environment acceptance.

## Corrected

`trading_metrics` no longer converts unknown/nonfinite/negative drawdown or
nonfinite returns into finite zero-risk RAP. Derived metrics are null with
`risk_metric_status=NOT_ESTIMABLE`; measured zero remains a legitimate zero.
Risk penalty must be finite and nonnegative. The raw analyzer evidence stays
unchanged. Seven new tests failed before and pass after; the focused simulator
suite totals 198 passed. This does not repair separate agent checkpoint-ranking
code that also coerces missing scores.

## Independently reproduced transition defect

Real Backtrader, existing shared-envelope test fixture, constant price 100,
initial equity 10000, commission 0.001, seed 7. No financial data or fit.
Strict window is 13 observations, stride 2; context consumes 25 actual bars.

| Returned step | Completed bar count | Position units | Equity | Reward |
|---|---:|---:|---:|---:|
| reset | 25 | 0 | 10000 | n/a |
| BUY | 26 | 0 | 10000 | 0 |
| HOLD | 27 | 99.8 | 9990.02 | -0.000998 |
| HOLD | 28 | 99.8 | 9990.02 | 0 |

Without strict warmup, reset and the first returned step both report bar 1.
The action issued using bar 25 is applied at bar 26 and fills at bar 27;
its cost is returned with the following HOLD action. Delayed execution is not
automatically invalid RL, but this is not the specified next-open transition,
and pending-action state/credit must not be silently misrepresented.

Reproduction: use `tests/test_shared_execution_envelope.py::_env` with 40
constant closes, `commission=0.001`, `window_size=13`,
`observation_stride_bars=2`, `require_full_observation_window=True`; reset at
seed 7, then step `[1.0]`, `[0.0]`, `[0.0]`, reading bridge `bar_index`,
`position_units`, `equity` and returned reward; always close the environment.
The bridge was NOT patched speculatively in this review. Next work is a
test-led next-open transition repair, including session-close timing, terminal
settlement, cancellations and causal perturbation across the real engine.

## Still open

- Backtrader accepts financing-enabled configuration without debiting it;
  finance/rollover accounting and cost assumptions require an explicit contract.
- Raw fractional sizing can enlarge positions without the discrete weekly
  overlay seeing increased risk. This is not universal: the shared envelope
  holds same-direction positions unchanged. The static protected-path quota
  does not certify every legacy sizing path.
- Full fill-price spread/slippage and margin acceptance remain pending.
- Default test evaluation and missing-score handling in agent-multi need
  campaign-specific validation before financial training.

No deployed service, DOIN node, financial fit or accepted dataset changed.
