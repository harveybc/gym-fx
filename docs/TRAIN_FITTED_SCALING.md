# Frozen TRAIN Scaling

Candidate implementation, not deployed to an active scientific campaign.
Owner request2026-10-21: replace moving feature statistics with per-feature
statistics fitted on TRAIN and reused unchanged for later periods.

`preprocessor_plugins.train_fitted_scaler.fit_train` uses sklearn StandardScaler.
Inputs: finite TRAIN dataframe, ordered feature names, timezone-aware fit bounds,
declared source SHA256 and optional explicitly declared passthrough columns.
The owner's baseline uses no passthrough exceptions. The helper rejects a
validation/test role, rows outside the fit bounds, empty/nonfinite inputs and
duplicate/unsorted/naive timestamps. Constant columns use StandardScaler's
scale1 policy. The caller must authenticate the source manifest/file and role;
this helper cannot infer provenance from arbitrary in-memory rows or prove that
a caller's declared SHA256 is truthful.

`save` stores immutable JSON with names/order, means/scales/variances and fit
period/row/source identity. `load` checks artifact digest, source identity,
column order and finite positive scales. Digests are integrity bindings, not
signatures or independent custody. Optional sklearn dependency is `gym-fx[scaling]`
for fitting only; inference does not import sklearn.

The existing feature-window plugin's additive `train_zscore` mode requires
explicit full-window policy, clip setting and artifact/source bindings:

```json
{
  "feature_scaling": "train_zscore",
  "feature_scaler_file": "<frozen-scaler.json>",
  "feature_scaler_sha256": "<artifact-sha256>",
  "feature_scaler_train_sha256": "<TRAIN-source-sha256>",
  "feature_clip": 0,
  "require_full_observation_window": true
}
```

No statistics are fitted during observation construction, no rolling history
is used, no reward/target/execution prices are modified, and nonfinite outputs
are rejected rather than replaced with zeros. clip0 disables clipping; a
positive clip must be explicitly configured. Historical modes/defaults are
unchanged, so the active pinned campaign is not silently reinterpreted.

PRE21 expected failures. POST48 passed,9 skipped across new scaler, existing
feature-window and completed-bar-clock tests; an additional33/33 focused run
passed. Evidence covers fixed-scale formula equality with StandardScaler,
constant columns, immutable artifact/reload, validation/future perturbations,
price preservation, explicit optional cycle passthrough, invalid inputs and
tamper/column/source/clip-policy refusals. No model fit, GPU claim or full
agent-multi selected-input adapter integration is established by these tests.
That integration requires a new sealed request/scaler/data binding; old v12
contracts deliberately remain rolling-only and must not be weakened in place.
