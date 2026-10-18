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
