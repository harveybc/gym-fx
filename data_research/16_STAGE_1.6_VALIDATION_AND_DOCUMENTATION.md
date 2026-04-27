# Stage 1.6 — Validation and Documentation Audit

**Stage goal:** Final validation pass over ENTIRE data lake. Audit completeness, fix any gaps, produce comprehensive Phase 1 completion report.

**Inputs:** Stages 1.3 + 1.4 + 1.5 complete.

**Outputs:**
- All folders pass documentation audit (README + data_dictionary + provenance.json)
- All datasets pass validation tests
- `PHASE_1_COMPLETION_REPORT.md` — comprehensive Phase 1 summary
- Data lake ready for Phase 2 use

**Machine:** Omega (audit work, light compute).

---

## 1. Stage 1.6 Procedure

Three-pass audit:

1. **Pass 1 — Folder Audit:** every folder has required documentation
2. **Pass 2 — Validation Audit:** every dataset passes statistical tests
3. **Pass 3 — Coverage Audit:** verify no data gaps vs catalog

After all 3 passes complete, produce master completion report.

---

## 2. Pass 1: Folder Documentation Audit

Script: `_scripts/audit_documentation.py`

```python
import os
import json
from pathlib import Path

DATA_LAKE = Path.home() / "Documents/financial_data"

# Skip these (project-level folders)
SKIP_FOLDERS = {"_templates", "_metadata", "_scripts"}

audit_results = {
    "total_folders": 0,
    "folders_complete": 0,
    "folders_missing_readme": [],
    "folders_missing_data_dictionary": [],
    "folders_missing_provenance": [],
    "folders_completed_correctly": [],
}

# Walk all subfolders
for folder in DATA_LAKE.rglob("*"):
    if not folder.is_dir():
        continue
    if any(skip in folder.parts for skip in SKIP_FOLDERS):
        continue
    
    audit_results["total_folders"] += 1
    
    has_readme = (folder / "README.md").exists()
    has_dict = (folder / "data_dictionary.md").exists()
    has_prov = (folder / "provenance.json").exists()
    
    # Determine if folder should have all 3
    # Heuristic: contains data files (.csv, .parquet, .jsonl)
    has_data = any(folder.glob("*.csv")) or any(folder.glob("*.parquet")) or any(folder.glob("*.jsonl"))
    
    if has_data:
        # Should have all 3
        if not has_readme:
            audit_results["folders_missing_readme"].append(str(folder))
        if not has_dict:
            audit_results["folders_missing_data_dictionary"].append(str(folder))
        if not has_prov:
            audit_results["folders_missing_provenance"].append(str(folder))
        if has_readme and has_dict and has_prov:
            audit_results["folders_completed_correctly"].append(str(folder))
            audit_results["folders_complete"] += 1
    else:
        # Sub-organizing folder, only needs README
        if not has_readme:
            audit_results["folders_missing_readme"].append(str(folder))
        else:
            audit_results["folders_complete"] += 1

# Save audit results
with open(DATA_LAKE / "_metadata/audit_documentation.json", "w") as f:
    json.dump(audit_results, f, indent=2)

# Print summary
print(f"Total folders: {audit_results['total_folders']}")
print(f"Complete: {audit_results['folders_complete']}")
print(f"Missing README: {len(audit_results['folders_missing_readme'])}")
print(f"Missing data_dictionary: {len(audit_results['folders_missing_data_dictionary'])}")
print(f"Missing provenance: {len(audit_results['folders_missing_provenance'])}")
```

If any folders missing required files: agent generates missing files using templates from Stage 1.1, then re-runs audit.

ALL folders must pass before proceeding.

---

## 3. Pass 2: Validation Audit

Script: `_scripts/audit_validation.py`

For each time-series dataset acquired:

```python
def audit_dataset_validation(folder):
    """Run Stage II-0b 6-test battery on time-series dataset."""
    
    # Find data files
    csv_files = list(folder.glob("*.csv"))
    parquet_files = list(folder.glob("*.parquet"))
    
    # Skip if no data
    if not csv_files and not parquet_files:
        return {"status": "no_data", "tests": {}}
    
    # Load data
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
    else:
        df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
    
    # Determine asset type from path
    asset_type = infer_asset_type(folder)  # "fx", "equity", "crypto", "macro", "fundamental", etc.
    
    # Run 6-test battery (skip irrelevant tests)
    tests = {}
    
    if asset_type in ["fx", "equity", "crypto", "commodity", "bond"]:
        # Time-series price data: full battery
        tests["bar_count"] = check_bar_count_realistic(df, asset_type)
        tests["weekend_gaps"] = check_weekend_gaps(df) if asset_type != "crypto" else "N/A"
        tests["fat_tails"] = check_kurtosis(df) > 4
        tests["vol_clustering"] = check_squared_returns_acf(df) > 0.05
        tests["return_acf"] = abs(check_return_acf(df)) > 0.001
        tests["no_gbm"] = check_no_gbm_fingerprint(df)
    elif asset_type == "macro":
        # Macro time series: subset of tests
        tests["non_empty"] = len(df) > 0
        tests["no_extreme_outliers"] = check_outliers(df)
        tests["coverage_complete"] = check_no_long_gaps(df)
    elif asset_type == "fundamental":
        # Fundamental data: schema + non-empty checks
        tests["schema_valid"] = check_fundamental_schema(df)
        tests["non_empty"] = len(df) > 0
    elif asset_type == "alternative":
        # Alternative data: source-specific
        tests["non_empty"] = len(df) > 0
        tests["expected_columns"] = check_expected_columns(df, source=folder.name)
    
    # Update provenance.json with validation results
    update_provenance(folder, tests)
    
    return {"status": "tested", "tests": tests}

# Walk all dataset folders
for folder in DATA_LAKE.rglob("*"):
    if not folder.is_dir():
        continue
    if any(skip in folder.parts for skip in SKIP_FOLDERS):
        continue
    
    has_data = any(folder.glob("*.csv")) or any(folder.glob("*.parquet"))
    if has_data:
        result = audit_dataset_validation(folder)
        # Aggregate results
```

Output: `_metadata/audit_validation.json` with:
- Total datasets tested
- Datasets passing all tests
- Datasets with advisory failures (e.g., autocorrelation borderline)
- Datasets failing validation (must be investigated)

---

## 4. Pass 3: Coverage Audit

Script: `_scripts/audit_coverage.py`

Compare actual data lake contents against catalog from Stage 1.2:

```python
import json

# Load catalog
with open(DATA_LAKE / "_metadata/data_catalog.json") as f:
    catalog = json.load(f)

# For each catalog entry, check if data exists at target folder
coverage = {
    "expected": [],
    "acquired": [],
    "missing": [],
    "partially_acquired": [],
}

for entry in catalog["entries"]:
    target = DATA_LAKE / entry["target_folder"]
    if not target.exists():
        coverage["missing"].append(entry)
        continue
    
    has_data = any(target.glob("*.csv")) or any(target.glob("*.parquet"))
    if not has_data:
        coverage["missing"].append(entry)
        continue
    
    # Check coverage period
    actual_range = get_actual_coverage(target)
    expected_range = (entry["start_date"], entry["end_date"])
    
    if covers_full_range(actual_range, expected_range):
        coverage["acquired"].append(entry)
    else:
        coverage["partially_acquired"].append({
            **entry,
            "actual_range": actual_range,
            "gap": compute_gap(actual_range, expected_range),
        })
```

Output: `_metadata/audit_coverage.json` with:
- Acquired (full coverage)
- Partially acquired (with gap details)
- Missing (not yet acquired or failed)
- Reasons for any missing/partial

If any HIGH-priority sources missing → ESCALATION + retry acquisition.

If some LOW-priority sources missing → document as known gap, proceed.

---

## 5. Top-Level Inventory Document

Script: `_scripts/generate_inventory.py`

Walks entire data lake, produces master inventory:

`~/Documents/financial_data/INVENTORY.md`:

```markdown
# Financial Data Lake — Master Inventory

Generated: YYYY-MM-DD

## Summary

- Total folders: [N]
- Total datasets: [N]  
- Total file count: [N]
- Total disk usage: [X] GB
- Date range covered: [earliest] to [latest]

## Datasets by Category

### Market Data — Equities
| Dataset | Symbols | Timeframes | History | Size |
|---------|---------|-----------|---------|------|
| US Indices | 5 | daily | 1985+ | 5MB |
| US Individual | 500 | daily, 1m | Variable | 25GB |
| ... | | | | |

### Market Data — Forex
[similar table]

### Market Data — Crypto
[similar table]

### ... etc per category

## Highest-Coverage Datasets (priorities for Phase 2)

1. BTC/USDT all timeframes 2017-present
2. ETH/USDT all timeframes 2017-present
3. EUR/USD all timeframes 2005-present
4. ... (top 20)

## Known Gaps

| Dataset | Reason | Priority | Workaround |
|---------|--------|----------|------------|
| ... | ... | ... | ... |

## Subscription Status

- Total active: [N] subscriptions
- Total cost: $[X]/month
- Cost ceiling: $500/month

## Source Distribution

- Free sources: [N]
- Paid sources: [N]
- Manual download required: [N]
```

---

## 6. Phase 1 Completion Report

Master deliverable: `PHASE_1_COMPLETION_REPORT.md`

```markdown
# Phase 1 Completion Report

## Date: YYYY-MM-DD

## Executive Summary

Phase 1 (Data Acquisition) of Project 3 is complete. Comprehensive multi-asset, multi-source financial data lake established at `~/Documents/financial_data/` with full documentation and validation.

[Brief paragraph on what was accomplished]

## Quantitative Summary

| Metric | Value |
|--------|-------|
| Total folders | [N] |
| Total datasets | [N] |
| Total data files | [N] |
| Total disk usage | [X] GB |
| Categories covered | 8 (market_data, macro, alt_data, microstructure, derivatives, fundamental, reference, _metadata) |
| Active subscriptions | [N] |
| Monthly subscription cost | $[X] |
| Date range | [earliest] to [latest] |

## Stage Completions

- [x] Stage 1.1: Storage Architecture
- [x] Stage 1.2: Data Catalog
- [x] Stage 1.3: Free Data Acquisition
- [x] Stage 1.4: Registrations and Keys
- [x] Stage 1.5: Paid Data Acquisition
- [x] Stage 1.6: Validation and Documentation

## Audit Results

### Documentation Audit
- Folders complete: [N]/[N] (100%)
- Folders incomplete: 0

### Validation Audit
- Datasets passing all tests: [N]
- Datasets with advisory failures: [N] (documented, not blocking)
- Datasets failing validation: [N] (resolved or excluded with reason)

### Coverage Audit
- Catalog entries fully acquired: [N]
- Partially acquired: [N]
- Missing: [N] (with reasons)

## Subscriptions Active

| Service | Tier | Cost/mo | Started |
|---------|------|---------|---------|
| Glassnode | Standard | $30 | YYYY-MM-DD |
| CryptoQuant | Standard | $39 | YYYY-MM-DD |
| Polygon.io | Developer | $79 | YYYY-MM-DD |
| FMP | Starter | $14 | YYYY-MM-DD |
| Twitter (if opted) | Basic | $100 | YYYY-MM-DD |

Total: $[X]/month

## Lessons Learned

[Any issues encountered, how resolved]

## Known Gaps and Reasons

| Gap | Reason | Mitigation |
|-----|--------|------------|
| [missing data] | [reason] | [what we use instead] |

## Phase 2 Readiness

Phase 2 (Feature Engineering) requires:
- [x] All raw data acquired and validated
- [x] Per-folder documentation complete
- [x] Inventory generated
- [x] Validation results documented per dataset

Phase 2 may proceed.

## Files

- Master inventory: `~/Documents/financial_data/INVENTORY.md`
- Data catalog: `~/Documents/financial_data/_metadata/data_catalog.json`
- Documentation audit: `~/Documents/financial_data/_metadata/audit_documentation.json`
- Validation audit: `~/Documents/financial_data/_metadata/audit_validation.json`
- Coverage audit: `~/Documents/financial_data/_metadata/audit_coverage.json`
- Subscriptions: `~/Documents/financial_data/_metadata/subscriptions.json`
- Acquisition log: `~/Documents/financial_data/_metadata/acquisition_log.csv`

## User Gate

Awaiting user approval to proceed to Phase 2.
```

---

## 7. Stage 1.6 Deliverable

The `PHASE_1_COMPLETION_REPORT.md` IS the deliverable for Stage 1.6 (it serves both as Stage 1.6 output and Phase 1 completion).

Plus `STAGE_1.6_DELIVERABLE.md` (brief, references main report):

```markdown
# Stage 1.6 Deliverable — Validation and Documentation Audit

## Status: COMPLETE

See `PHASE_1_COMPLETION_REPORT.md` for full details.

## Audit Summary

| Audit | Pass Rate |
|-------|-----------|
| Documentation | 100% |
| Validation | XX% |
| Coverage | XX% |

## User Gate

Phase 1 is complete. User approves Phase 2 start.
```

---

## 8. User Gate

User reviews Phase 1 Completion Report. Approves Phase 2 start.

This is the most critical user gate of Phase 1 — confirms the data lake is production-ready before any feature engineering or model experiments.
