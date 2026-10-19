# Next-open execution repair

Scope: explicit `execution_transition=next_open_v2`, used by the curated static
RL comparison. Existing configurations retain `legacy_delayed_v1`; this change
does not silently reinterpret historical runs or launch training.

The policy receives completed bar t (including its close), submits an action,
and Backtrader executes it at the next open. The returned observation/reward
contains that fill and costs at the next completed bar. Entry-bar protection
settles before publication. Every step advances one observed bar. A callback
failure raises to the caller instead of masquerading as successful termination.

For session protection, `execution_max_delay_hours` is mandatory with v2.
Only the already-known calendar is projected by this bound to request flattening
in time; prices/spread/stability evidence use at most the completed bar. This
does not guarantee execution across an unannounced halt. Real admission must
verify bar timestamps, calendar and the execution-delay bound.

## Traceability

| Requirement | Evidence |
|---|---|
| Correct transition, both directions, strict 13x2 warmup | `tests/test_next_open_transition.py`: action-step cost, gap-open fill, one-bar advance |
| Causality, reset and error handling | Future perturbation, fresh reset and injected callback failure in the same test file |
| Forced-flat and reopening restrictions | Real-engine synthetic week with omitted closure bars and 12-hour buffers, both directions |
| Net economics | Entry-bar stop: net PnL equals equity change; entry/exit commissions included; leverage 1 and 5 |
| Financing debit | `tests/test_financing_debit.py`: long/short two-day hand ledger; malformed/absent enabled contract rejected |
| Compatibility | Existing quota, windows, protection, session, reconciliation, solvency, foundation and cost-context suites |
| Submitted protection cancellation | `tests/test_submitted_order_cancellation.py`: native bracket/OCO cancellation before acceptance, entry-bar policy close in both directions |

## Submitted-order cancellation repair

A real TRAIN replay exposed protective orders surviving an immediate policy
close. Native BackBroker.cancel searches only its pending queue, not newly
submitted orders. Re-leveled SL/TP could therefore execute after the position
was closed, eventually tripping the unchanged cross-closure exposure guard.
The broker adapter now honors cancellation from the submitted queue using
native cancellation notifications and bracket/OCO propagation. This does not
advance the broker, accept or execute orders early, move prices or relax any
session guard. PRE: both immediate bracket/OCO cancellation probes filled
orders despite cancellation. Failed campaigns remain failed; new training
requires a new engine pin and campaign identity.

PRE: financing tests 7 failed; entry-bar net-cost tests 4 failed. POST: focused
suite 247 passed, no skips. These are synthetic engine tests, not financial
performance or complete simulator certification.

## Financing semantics

Only if `financing_enabled=true`, require both
`financing_model="backtrader_annual_notional_daily_v1"` and a finite nonnegative
`financing_annual_rate`. Backtrader debits
`days * rate / 365 * abs(units) * entry_price` for both directions at changes of
the data clock's date. The rate is never guessed or defaulted to zero when enabled.
This analytical scenario is NOT a broker-verified FX swap, NY rollover, triple-swap
calendar or side-specific positive/negative carry model. Those remain required
before declaring broker-realistic costs. Cash/equity reflect these daily debits;
Backtrader attaches accrued interest to closing trade costs in its normal path.

The v2 synthetic entry-bar settlement requires stocklike accounting with
`shortcash=true` and authenticated parent-fill commission evidence. It returns
entry capital with leverage using Backtrader's accounting convention. Legacy
settlement is unchanged. Alternate broker/margin models and broker-measured
spread/slippage acceptance remain separate work.

## Repeated-close accounting correction

The synthetic entry-bar settlement previously flattened broker position/cash
but left Backtrader's native Trade open. A subsequent entry was merged into the
old trade, and an ordinary protective close was missing from the trade ledger.
PRE: four long/short x leverage 1/5 round-trip tests failed with one close event
instead of two. The v2 path now closes the native Trade lifecycle with the actual
settlement price/commission and notifies its analyzers once, without submitting
another broker order or duplicating the bridge event. A net PnL of exactly zero
also remains zero rather than falling back to gross PnL.

POST: nine new tests pass, including repeated synthetic/native closes and a
hand ledger with nonzero slippage, commissions and two days of analytical
financing. The consolidated focused suite is now 256/256, zero skips. This does
not certify every margin model or substitute for actual broker cost evidence.

## Automated verification

From the matching agent-multi checkout/environment:

```bash
python -m tools.verify_rl_static_engineering \
  --gym-root <gym-fx-checkout> --output-root <new-private-report-directory>
python -m tools.verify_rl_static_engineering \
  --output-root <same-report-directory> --status
```

The runner forces CPU, bounds each stage, keeps logs/JUnit and updates status.
It does not start an RL campaign, change services or grant data admission.

## Completed H1 availability labels

Explicit opt-in for admitted raw H1 data (legacy behavior is unchanged):

```json
{
  "execution_transition": "next_open_v2",
  "timestamp_semantics": "completed_bar_available_at",
  "timeframe": "1h",
  "execution_max_delay_hours": 1
}
```

`DATE_TIME` is the completed bar END / availability time. OHLC belongs to
`[DATE_TIME - 1h, DATE_TIME)`. Action time is this availability time, while a
Market order fills at the next row's OPEN, whose physical time is that next
row's label minus one hour. Neither prices, row labels nor features are shifted.
Unsupported timeframes/transitions, unordered or unaligned labels refuse.

Bind `session_calendar_intervals` to the admitted **CLOSURE** export, not the
open-session export. Closures are `[close, reopen)`: a bar ending exactly at
close is valid; a bar starting exactly at reopen is valid. A bar overlapping
a closure refuses before reset. At the last completed bar ending at close,
`EXPECTED_MARKET_CLOSED` masks every action to HOLD. Advancing across a known
closure requires zero positions and zero orders; otherwise a typed
`CROSS_CLOSURE_EXPOSURE_OR_ORDER` refusal prevents the next broker cycle.
The existing calendar-only delay projection and 12-hour flatten/reopen
policy remain in force. Missing spread/stability evidence still fails closed.

Step info separates `action_time`, `engine_bar_time` and
`observation_available_at`. `bridge.fill_events` records executed price,
order role, row label and fill interval. Market `fill_time` is exact OPEN;
intrabar stop/limit time is unknown and is explicitly null, not fabricated.
Both quota-enabled strategies charge entries using the fill's containing H1
interval, whose NY date is unambiguous on the required UTC-hour grid.
Submission quota remains conservatively charged on action day as before.

### Bounded Acceptance

`tests/test_completed_bar_clock.py` first reproduced the Friday 22Z exception
and next-END-label midnight quota bug. It covers both quota strategies, long
and short, NY winter/DST dates, invalid intervals, and surviving order/position
refusals. Its optional real-data tests read admitted parquet without changing
OHLC, replay an actual January 2024 week/month through GymFxEnv + Backtrader,
and bind the published v3 closure export (including the preceding holiday).

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
GYM_FX_ADMITTED_EURUSD_PARQUET=<admitted-v3-directory>/eurusd.parquet \
python -m pytest tests/test_completed_bar_clock.py -q
```

`GYM_FX_CLOSURE_INTERVALS` may override the sibling `CLOSURE_INTERVALS.json`.
Without the parquet environment variable only real-data tests skip. Two
separate scenarios cover absent spread (reopening remains blocked) and
`ASSUMED_SPREAD=0.00010` with nonzero assumed commission `0.00005`. The latter
is explicitly an analytical assumption, **not observed or broker-verified**.
Checks cover untouched OHLC, exact next-OPEN prices/times, real closure gaps,
12-hour flat/reopen restrictions and at most one filled entry per NY day.
This certifies bounded clock behavior, not broker costs or financial fitness.

The source-fidelity regression also renders all 37,309 admitted v3 rows using
`%.17g` and compares each OHLC float64 byte string after the real feed loads
it with `require_curated_rl_input=true`. It passed on pandas 3.0.3; a direct
feed check on pandas 2.2.2 also found zero changed values. Accordingly no CSV
parser change was necessary for this source. The regression is exact, not
`allclose`, and will expose precision loss on another runtime.

Verified 2026-10-19 on CPU: 295 passed, zero skips in 55.76 seconds across
the new clock/fidelity tests and existing next-open, session, quota, protected
execution, reconciliation, financing, strided observation, foundation,
solvency and execution-cost-context suites. No service or training was started.
