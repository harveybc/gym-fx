# Stage 1.1 — Storage Architecture

**Stage goal:** Create the complete folder structure for the financial data lake. Set up README templates, data dictionary templates, and provenance.json templates.

**Inputs:** None (this is foundation stage).

**Outputs:**
- Complete empty folder structure at `~/Documents/financial_data/`
- Templates in `~/Documents/financial_data/_templates/`
- `STAGE_1.1_DELIVERABLE.md` confirming structure created

**Machine:** Omega (local).

---

## 1. Folder Structure (CREATE EXACTLY THIS)

Execute these commands in order on Omega:

```bash
cd ~/Documents
mkdir -p financial_data
cd financial_data

# Top-level categories
mkdir -p _templates
mkdir -p _metadata
mkdir -p market_data
mkdir -p macro_economic
mkdir -p alternative_data
mkdir -p microstructure
mkdir -p derivatives
mkdir -p fundamental
mkdir -p reference_data

# Market data subcategories
mkdir -p market_data/equities
mkdir -p market_data/equities/us_indices
mkdir -p market_data/equities/us_individual
mkdir -p market_data/equities/eu_indices
mkdir -p market_data/equities/eu_individual
mkdir -p market_data/equities/asia_indices
mkdir -p market_data/equities/asia_individual
mkdir -p market_data/equities/emerging
mkdir -p market_data/equities/etfs

mkdir -p market_data/forex
mkdir -p market_data/forex/g10
mkdir -p market_data/forex/emerging_markets
mkdir -p market_data/forex/exotic_pairs

mkdir -p market_data/crypto
mkdir -p market_data/crypto/spot_top50
mkdir -p market_data/crypto/perpetuals
mkdir -p market_data/crypto/funding_rates

mkdir -p market_data/commodities
mkdir -p market_data/commodities/precious_metals
mkdir -p market_data/commodities/energy
mkdir -p market_data/commodities/agriculture
mkdir -p market_data/commodities/industrial_metals

mkdir -p market_data/bonds
mkdir -p market_data/bonds/us_treasuries
mkdir -p market_data/bonds/sovereign_global
mkdir -p market_data/bonds/corporate

# Macro economic data
mkdir -p macro_economic/fred
mkdir -p macro_economic/oecd
mkdir -p macro_economic/imf
mkdir -p macro_economic/world_bank
mkdir -p macro_economic/eurostat
mkdir -p macro_economic/boj
mkdir -p macro_economic/ecb
mkdir -p macro_economic/central_banks_other
mkdir -p macro_economic/inflation_expectations
mkdir -p macro_economic/yield_curves
mkdir -p macro_economic/economic_surprise

# Alternative data
mkdir -p alternative_data/news_sentiment
mkdir -p alternative_data/social_sentiment
mkdir -p alternative_data/google_trends
mkdir -p alternative_data/wikipedia_traffic
mkdir -p alternative_data/sec_filings
mkdir -p alternative_data/insider_trading
mkdir -p alternative_data/earnings_estimates
mkdir -p alternative_data/analyst_recommendations
mkdir -p alternative_data/short_interest
mkdir -p alternative_data/etf_flows
mkdir -p alternative_data/cot_reports
mkdir -p alternative_data/options_flow
mkdir -p alternative_data/dark_pool

# Crypto-specific alternative data
mkdir -p alternative_data/onchain_btc
mkdir -p alternative_data/onchain_eth
mkdir -p alternative_data/onchain_other
mkdir -p alternative_data/exchange_flows
mkdir -p alternative_data/whale_movements
mkdir -p alternative_data/defi_metrics

# Microstructure
mkdir -p microstructure/order_book_snapshots
mkdir -p microstructure/trade_volume_profile
mkdir -p microstructure/spread_data

# Derivatives
mkdir -p derivatives/options_chains
mkdir -p derivatives/options_greeks
mkdir -p derivatives/futures_curves
mkdir -p derivatives/vix_term_structure
mkdir -p derivatives/implied_volatility_surfaces

# Fundamental
mkdir -p fundamental/earnings
mkdir -p fundamental/balance_sheet
mkdir -p fundamental/income_statement
mkdir -p fundamental/cash_flow
mkdir -p fundamental/ratios

# Reference data
mkdir -p reference_data/calendars
mkdir -p reference_data/holidays
mkdir -p reference_data/index_constituents
mkdir -p reference_data/sector_classifications
mkdir -p reference_data/symbol_mappings
```

This creates ~70 folders. Do not create additional folders unless approved.

---

## 2. Templates Directory Contents

Create these template files at `~/Documents/financial_data/_templates/`:

### Template 1: README.md template

File: `~/Documents/financial_data/_templates/README_template.md`

```markdown
# [Folder Name]

## Purpose

[1-2 sentences: what this folder contains and why it exists]

## Contents

[List of subfolders or files with brief description each]

## Update Frequency

[How often is this data refreshed? e.g., "Daily", "Monthly", "One-time historical pull"]

## Last Updated

[YYYY-MM-DD]

## Related Folders

[Links to related folders if applicable]

## Notes

[Any caveats, known issues, version info]
```

### Template 2: data_dictionary.md template

File: `~/Documents/financial_data/_templates/data_dictionary_template.md`

```markdown
# Data Dictionary: [Dataset Name]

## Source

- **Provider:** [e.g., FRED, Binance, Polygon.io]
- **URL:** [base URL]
- **License:** [e.g., Public Domain, CC-BY, Proprietary - Subscription]
- **Acquisition method:** [API, bulk download, manual download]

## Coverage

- **Date range:** [YYYY-MM-DD to YYYY-MM-DD]
- **Frequency:** [tick, 1m, 5m, 1h, 1d, weekly, monthly]
- **Total observations:** [number]
- **Coverage gaps:** [any known gaps in data]

## Schema

| Column | Type | Description | Units | Example | Notes |
|--------|------|-------------|-------|---------|-------|
| timestamp | datetime | UTC timestamp of observation | ISO 8601 | 2024-01-01T00:00:00Z | |
| ... | | | | | |

## Known Issues

- [Any quirks, errors, or anomalies]

## Validation Status

- [ ] Bar count realistic
- [ ] Schema matches specification
- [ ] No duplicate timestamps
- [ ] No future timestamps
- [ ] Statistical validation (Stage II-0b 6 tests)

## Provenance

See `provenance.json` in same folder.
```

### Template 3: provenance.json template

File: `~/Documents/financial_data/_templates/provenance_template.json`

```json
{
  "dataset_name": "DATASET_NAME_HERE",
  "source": {
    "provider": "PROVIDER_NAME",
    "url": "https://provider.example.com",
    "api_endpoint": "/v1/endpoint",
    "api_version": "v1"
  },
  "acquisition": {
    "acquired_date": "YYYY-MM-DD",
    "acquired_by": "Project 3 Stage 1.X",
    "acquisition_method": "API|bulk_download|manual",
    "script_used": "scripts/fetch_X.py",
    "script_git_sha": "GIT_HASH_HERE"
  },
  "coverage": {
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "frequency": "1h",
    "total_observations": 0
  },
  "license": {
    "type": "Public Domain|CC-BY|Proprietary-Subscription|Other",
    "details": "License details if relevant"
  },
  "cost": {
    "type": "free|subscription",
    "monthly_cost_usd": 0,
    "subscription_provider": null
  },
  "validation": {
    "validated_date": null,
    "tests_passed": [],
    "tests_failed": [],
    "validation_report_path": null
  },
  "checksum": {
    "algorithm": "sha256",
    "value": null
  }
}
```

---

## 3. Top-Level README

Create file: `~/Documents/financial_data/README.md`

```markdown
# Financial Data Lake — Project 3

## Purpose

Comprehensive multi-asset, multi-source financial data repository for systematic RL trading research. Built during Project 3 (2026).

## Organization

| Top-level folder | Contents |
|------------------|----------|
| `_templates/` | README, data dictionary, provenance templates |
| `_metadata/` | Project-wide metadata, acquisition logs, audit trails |
| `market_data/` | Price + volume data: equities, forex, crypto, commodities, bonds |
| `macro_economic/` | Macroeconomic indicators: rates, inflation, employment, GDP |
| `alternative_data/` | News, sentiment, on-chain, COT, options flow, etc. |
| `microstructure/` | Order book, trade volume profile, spreads |
| `derivatives/` | Options chains, futures curves, IV surfaces |
| `fundamental/` | Earnings, balance sheets, income statements, ratios |
| `reference_data/` | Calendars, holidays, index constituents, classifications |

## Standards

Every data folder MUST contain:
- `README.md` — Folder description (use _templates/README_template.md)
- `data_dictionary.md` — Schema description (if folder contains datasets)
- `provenance.json` — Machine-readable source tracking

## Update procedures

Each data folder's README documents its update procedure (one-time historical pull vs ongoing daily/weekly update).

## See also

- Project 3 master plan: `[path to project root]/docs/00_PROJECT_3_MASTER_PLAN.md`
- Data catalog: `[path to project root]/docs/12_STAGE_1.2_DATA_CATALOG.md`
```

---

## 4. Metadata Files

Create these initial metadata files at `~/Documents/financial_data/_metadata/`:

### File 1: `acquisition_log.csv`

```
timestamp,stage,dataset,source,status,notes
```

(Empty CSV with headers. Each acquisition appends one row.)

### File 2: `subscriptions.json`

```json
{
  "active_subscriptions": [],
  "total_monthly_cost_usd": 0,
  "monthly_budget_cap_usd": 500,
  "last_updated": null
}
```

### File 3: `project_3_metadata.json`

```json
{
  "project": "Project 3",
  "project_phase": "Phase 1 - Data Acquisition",
  "current_stage": "1.1 - Storage Architecture",
  "started_date": null,
  "data_lake_root": "/home/harveybc/Documents/financial_data",
  "owner": "harveybc",
  "machine_local": "Omega"
}
```

---

## 5. Validation

After creating all folders and templates, agent runs:

```bash
cd ~/Documents/financial_data
find . -type d | wc -l  # should be ~70 folders
find _templates -type f | wc -l  # should be 3 files
ls -la README.md  # must exist
ls -la _metadata/*.{csv,json}  # all 3 files exist
```

Document outputs in deliverable.

---

## 6. Stage 1.1 Deliverable

File: `STAGE_1.1_DELIVERABLE.md`

Content:

```markdown
# Stage 1.1 Deliverable — Storage Architecture

## Date: YYYY-MM-DD

## Folder Structure

Total folders created: [N]
Tree structure:
[paste output of `tree -L 3 ~/Documents/financial_data/`]

## Templates Created

- ~/Documents/financial_data/_templates/README_template.md ✓
- ~/Documents/financial_data/_templates/data_dictionary_template.md ✓
- ~/Documents/financial_data/_templates/provenance_template.json ✓

## Metadata Files Created

- ~/Documents/financial_data/_metadata/acquisition_log.csv ✓
- ~/Documents/financial_data/_metadata/subscriptions.json ✓
- ~/Documents/financial_data/_metadata/project_3_metadata.json ✓

## Top-level README

Created at ~/Documents/financial_data/README.md ✓

## Verification

- [x] All 70 folders exist
- [x] All 3 templates present
- [x] All 3 metadata files present
- [x] Top-level README exists

## User Gate

Awaiting user approval to proceed to Stage 1.2 (Data Catalog).
```

---

## 7. User Gate

User reviews deliverable, confirms folder structure, approves Stage 1.2 start.
