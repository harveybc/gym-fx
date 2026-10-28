# Static RL input and daily-entry additions

Component work for the 2026-10-18 RL-first plan. No financial training or
live service was started. Configuration defaults preserve existing experiments.

- `feature_window_preprocessor`: `observation_stride_bars=2`, `window_size=13`
  selects row indices `step-25, step-23, ..., step-1`. The bridge's `step` is
  the number of completed bars, so `step-1` is the known current tick, not a
  lag. The underlying admitted grid must already distinguish expected closures
  from missing open-session data. No smoothing or interpolation is done.
- `require_full_observation_window=true`: reset consumes actual context bars
  with HOLD before exposing the first policy observation. No padded history,
  policy rewards or replay transitions during this warmup. Nonfinite input or
  scaling history fails. Provide prior context in each evaluation split.
- Both protected strategies accept `max_entries_per_day` (positive integer;
  default null preserves legacy behavior) and `entry_day_timezone` (default
  America/New_York). One pending/held position; submissions consume quota and
  delayed/partial entry fills are additionally charged once on their fill day.
  Protective children and exits do not consume quota. In quota mode the shared
  envelope closes before a later action may reverse. Failed/cancelled entry
  submissions still consume the daily allowance; no hidden retry churn.
- The observation includes `daily_entry_budget_available` when enabled. The
  summary records submissions, filled entries, fill dates and daily maxima.
- Net winning-trade percentage and profit factor in `GymFxEnv.summary()` use
  `closed_trade_stream_v2`, including synthetic entry-bar settlements. Analyzer
  figures are diagnostic only; undefined ratios remain null.

Tests and the data receipt interface are documented in agent-multi
`docs/work_plan/44_CURATED_RL_INPUT_INTERFACE.md`. Acceptance remains scoped to
Backtrader; the Nautilus execution path has not acquired the daily quota here.
Do not advertise this cap for that engine or launch it under this protocol.
Full source/calendar, evaluation freeze and accounting certification precede
financial fits. No strategy-specific exception may bypass the business mask.
