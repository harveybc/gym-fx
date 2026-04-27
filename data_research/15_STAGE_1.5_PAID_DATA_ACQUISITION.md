# Stage 1.5 — Paid Data Acquisition

**Stage goal:** Acquire all paid subscription data sources using credentials from Stage 1.4. Same documentation/validation rigor as Stage 1.3.

**Inputs:** Stage 1.4 complete with all credentials validated. Stage 1.3 completed or in parallel.

**Outputs:** All paid data sources downloaded to target folders with full documentation.

**Machine assignment:**
- **Dragon:** Glassnode + CryptoQuant (crypto-focused, high data volume)
- **Gamma:** Polygon (US equities + options + news)
- **Omega:** FMP fundamentals + Twitter sentiment + Quandl

---

## 1. Pre-Flight Checks

```bash
source ~/Documents/financial_data/_metadata/.env

# Verify all paid credentials present
[[ -n "$GLASSNODE_API_KEY" ]] || (echo "Glassnode key missing" && exit 1)
[[ -n "$CRYPTOQUANT_API_KEY" ]] || (echo "CryptoQuant key missing" && exit 1)
[[ -n "$POLYGON_API_KEY" ]] || (echo "Polygon key missing" && exit 1)
[[ -n "$FMP_API_KEY" ]] || (echo "FMP key missing" && exit 1)
echo "All paid credentials present"
```

---

## 2. Task 1.5.A: Glassnode comprehensive (Dragon)

Script: `_scripts/fetch_glassnode_paid.py`

Glassnode Standard tier provides 100+ metrics across categories:
- Market: price, volume, market cap variants
- Supply: circulating, free float, illiquid supply, age bands
- Addresses: active, new, sending, receiving counts
- Transactions: count, mean, median, fee
- Mining: hash rate, difficulty, miner revenue, miner outflows
- Indicators: SOPR, MVRV, NUPL, NVT, Stock-to-Flow, Mayer Multiple
- ETH-specific: gas, staking, supply burned

```python
import requests, os
import pandas as pd
from datetime import datetime

GLASSNODE_KEY = os.environ['GLASSNODE_API_KEY']
BASE = "https://api.glassnode.com/v1/metrics"

ASSETS = ["BTC", "ETH"]

# Comprehensive metric list (Glassnode Standard tier)
METRICS_BY_CATEGORY = {
    "market": [
        "market/price_usd_close",
        "market/marketcap_usd",
        "market/marketcap_realized_usd",
        "market/mvrv",
        "market/mvrv_z_score",
        "market/price_usd_ohlc",
    ],
    "supply": [
        "supply/current",
        "supply/issued",
        "supply/inflation_rate",
        "supply/active_more_1y_percent",
        "supply/loss_sum",
        "supply/profit_sum",
        "supply/profit_relative",
    ],
    "addresses": [
        "addresses/active_count",
        "addresses/new_non_zero_count",
        "addresses/sending_count",
        "addresses/receiving_count",
        "addresses/min_1k_count",
        "addresses/min_10k_count",
        "addresses/min_100k_count",
    ],
    "transactions": [
        "transactions/count",
        "transactions/transfers_volume_sum",
        "transactions/size_mean",
        "transactions/transfers_to_exchanges_count",
        "transactions/transfers_from_exchanges_count",
    ],
    "fees": [
        "fees/volume_sum",
        "fees/volume_mean",
        "fees/gas_used_mean",
    ],
    "mining": [
        "mining/hash_rate_mean",
        "mining/difficulty_mean",
        "mining/revenue_sum",
        "mining/revenue_from_fees",
    ],
    "indicators": [
        "indicators/sopr",
        "indicators/sopr_adjusted",
        "indicators/cdd",
        "indicators/cyd",
        "indicators/asol",
        "indicators/msol",
        "indicators/nvt",
        "indicators/nvts",
        "indicators/velocity",
        "indicators/stock_to_flow_ratio",
        "indicators/mayer_multiple",
        "indicators/realized_profit",
        "indicators/realized_loss",
        "indicators/net_unrealized_profit_loss",
    ],
    "derivatives": [
        "derivatives/futures_open_interest_sum",
        "derivatives/futures_volume_daily_sum",
        "derivatives/futures_funding_rate_perpetual",
    ],
    "institutions": [
        "institutions/grayscale_holdings_sum",
        "institutions/microstrategy_holdings_sum",
    ],
}

# ETH-specific extras
ETH_EXTRA_METRICS = [
    "supply/eth/burned",
    "supply/eth/issued_pos",
    "addresses/non_zero_count",
    "eth2/staking_total_volume_sum",
    "eth2/staking_validators_count",
]

target_dir_btc = os.path.expanduser("~/Documents/financial_data/alternative_data/onchain_btc/glassnode/")
target_dir_eth = os.path.expanduser("~/Documents/financial_data/alternative_data/onchain_eth/glassnode/")

os.makedirs(target_dir_btc, exist_ok=True)
os.makedirs(target_dir_eth, exist_ok=True)

for asset in ASSETS:
    target_dir = target_dir_btc if asset == "BTC" else target_dir_eth

    for category, metrics in METRICS_BY_CATEGORY.items():
        cat_dir = os.path.join(target_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        for metric_path in metrics:
            url = f"{BASE}/{metric_path}"
            params = {
                "a": asset,
                "i": "24h",
                "api_key": GLASSNODE_KEY,
            }

            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        df = pd.DataFrame(data)
                        df["t"] = pd.to_datetime(df["t"], unit="s")
                        df = df.set_index("t")

                        slug = metric_path.replace("/", "_")
                        df.to_csv(os.path.join(cat_dir, f"{slug}.csv"))
                else:
                    # Some metrics may not be in Standard tier — log and continue
                    log_skip(asset, metric_path, r.status_code)
            except Exception as e:
                log_error(asset, metric_path, str(e))

    # Per-asset README, data_dictionary, provenance
    write_glassnode_documentation(target_dir, asset, METRICS_BY_CATEGORY)
```

---

## 3. Task 1.5.B: CryptoQuant comprehensive (Dragon)

Script: `_scripts/fetch_cryptoquant_paid.py`

CryptoQuant Standard tier focuses on exchange flows + whales + miners:

```python
CQ_KEY = os.environ['CRYPTOQUANT_API_KEY']
BASE = "https://api.cryptoquant.com/v1"

CATEGORIES = {
    "exchange_flows": {
        "btc": [
            "btc/exchange-flows/inflow",
            "btc/exchange-flows/outflow",
            "btc/exchange-flows/netflow",
            "btc/exchange-flows/in-house-flow",
        ],
        "eth": [
            "eth/exchange-flows/inflow",
            "eth/exchange-flows/outflow",
            "eth/exchange-flows/netflow",
        ],
    },
    "miner_flows": {
        "btc": [
            "btc/miner-flows/all-miners-inflow",
            "btc/miner-flows/all-miners-outflow",
            "btc/miner-flows/all-miners-reserve",
        ],
    },
    "whale_metrics": {
        "btc": ["btc/exchange-flows/transactions-count"],
        "eth": ["eth/exchange-flows/transactions-count"],
    },
    "market_indicator": {
        "btc": [
            "btc/market-indicator/sopr",
            "btc/market-indicator/mpi",
            "btc/market-indicator/coin-days-destroyed",
        ],
    },
    "stable_coins": {
        "stables": [
            "usdt-erc20/exchange-flows/inflow",
            "usdt-erc20/exchange-flows/outflow",
            "usdc/exchange-flows/inflow",
            "usdc/exchange-flows/outflow",
        ],
    },
}

# Fetch all
for category, asset_metrics in CATEGORIES.items():
    for asset, endpoints in asset_metrics.items():
        for endpoint in endpoints:
            url = f"{BASE}/{endpoint}"
            r = requests.get(url, params={"window": "day"},
                            headers={"Authorization": f"Bearer {CQ_KEY}"})
            # Save data
```

Target folder: `~/Documents/financial_data/alternative_data/exchange_flows/cryptoquant/`

---

## 4. Task 1.5.C: Polygon US equities + options + news (Gamma)

#  TODO: news?  uw news estamos usando? como obtenemos datos alimentables a nuestros modelos de rl desde news? tendríamos que hacer algúntipos de sentiment analysis o algo, pero esto es un overkill, creo que eso lo podemos probar en otro proyecto, no usemos news, pero si podemos usar calendario financiero y sy=us sresultados.

Polygon Developer tier ($79/mo) includes:
- Stocks: 5+ years of 1m, 5m, 15m, 1h, daily aggregates for all US stocks
- Options: 5 years of options chain data
- Forex: same
- Crypto: same
- News: financial news headlines + content

### 4.1 Polygon stocks aggregates

Script: `_scripts/fetch_polygon_stocks.py`

```python
POLYGON_KEY = os.environ['POLYGON_API_KEY']

# Get full S&P 500 list (with historical changes)
sp500_tickers = load_sp500_constituents()  # from reference_data

# Plus current top 1000 by market cap
top1000 = get_top_1000_by_marketcap()  # via Polygon /v3/reference/tickers

ALL_TICKERS = sorted(set(sp500_tickers + top1000))

TIMEFRAMES = [
    ("1", "minute"),
    ("5", "minute"),
    ("15", "minute"),
    ("1", "hour"),
    ("1", "day"),
]

for ticker in ALL_TICKERS:
    for multiplier, timespan in TIMEFRAMES:
        # Polygon paginated fetch
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        # Save as parquet (efficient)
        save_parquet(...)
```

Estimated volume: 1000 tickers × 5 timeframes = 5000 files. 1m data largest.

### 4.2 Polygon options chains

Script: `_scripts/fetch_polygon_options.py`

```python
# For SPY, QQQ, and major individual stocks
OPTIONS_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "META", "GOOGL"]

for underlying in OPTIONS_TICKERS:
    # Fetch all options contracts (active + historical)
    contracts = polygon_options_contracts(underlying)
    
    for contract in contracts:
        # Fetch daily aggs for each contract
        # Compute Greeks (delta, gamma, theta, vega, rho)
        save(...)
```

Target: `~/Documents/financial_data/derivatives/options_chains/polygon/`

### 4.3 Polygon financial news

Script: `_scripts/fetch_polygon_news.py`

```python
# Polygon news API
url = "https://api.polygon.io/v2/reference/news"
# Paginated fetch all news 2015-present
# Filter by tickers if needed
```

Target: `~/Documents/financial_data/alternative_data/news_sentiment/polygon_news/`

---

## 5. Task 1.5.D: FMP fundamentals (Omega)

Script: `_scripts/fetch_fmp_fundamentals.py`

```python
FMP_KEY = os.environ['FMP_API_KEY']
BASE = "https://financialmodelingprep.com/api/v3"

# For each S&P 500 ticker
for ticker in SP500_TICKERS:
    # Income statement (annual + quarterly)
    income_q = requests.get(f"{BASE}/income-statement/{ticker}?period=quarter&limit=120&apikey={FMP_KEY}").json()
    income_a = requests.get(f"{BASE}/income-statement/{ticker}?period=annual&limit=30&apikey={FMP_KEY}").json()
    
    # Balance sheet
    balance_q = requests.get(f"{BASE}/balance-sheet-statement/{ticker}?period=quarter&limit=120&apikey={FMP_KEY}").json()
    
    # Cash flow
    cashflow_q = requests.get(f"{BASE}/cash-flow-statement/{ticker}?period=quarter&limit=120&apikey={FMP_KEY}").json()
    
    # Ratios
    ratios = requests.get(f"{BASE}/ratios/{ticker}?limit=120&apikey={FMP_KEY}").json()
    
    # Save all to fundamental/fmp/<ticker>/
```

---

## 6. Task 1.5.E: Twitter sentiment (Omega) [if user opted in]

If Twitter $100/mo subscribed, fetch financial sentiment:

```python
import tweepy

client = tweepy.Client(bearer_token=os.environ['TWITTER_BEARER_TOKEN'])

# Search queries
QUERIES = [
    "$BTC OR #Bitcoin -is:retweet lang:en",
    "$ETH OR #Ethereum -is:retweet lang:en",
    "$SPY OR S&P500 -is:retweet lang:en",
    "Federal Reserve OR FOMC -is:retweet lang:en",
    # ... financial queries
]

# Note: Twitter Basic tier has 10K tweet/month read limit
# Use sparingly, prioritize most valuable searches
```

Target: `~/Documents/financial_data/alternative_data/social_sentiment/twitter/`

If user opted out: skip this task entirely.

---

## 7. Task 1.5.F: Quandl/Nasdaq Data Link selected datasets (Omega)

Script: `_scripts/fetch_quandl.py`

Free + selected paid datasets:

```python
import nasdaqdatalink as ndl
ndl.ApiConfig.api_key = os.environ['NASDAQ_DATA_LINK_KEY']

# Free datasets useful for our research
FREE_DATASETS = [
    "WIKI/AAPL",  # historical EOD data (prior to dataset deprecation)
    "FRED/GDP",   # FRED via Quandl
    "OPEC/ORB",   # OPEC oil prices
    "LBMA/GOLD",  # LBMA gold fixings
    "LBMA/SILVER",
    # ... more free datasets
]

# Selected paid that fit budget
PAID_DATASETS_UNDER_BUDGET = [
    # Add specific datasets if their value justifies cost
    # Example: Sharadar Core US Fundamentals (~$50/mo) — competitive with FMP
]

for dataset in FREE_DATASETS:
    df = ndl.get(dataset)
    save(...)
```

---

## 8. Validation

Same approach as Stage 1.3. Validate each dataset acquired.

For paid sources specifically, verify:
- Coverage matches subscription tier promised
- No 401/403 errors during fetches
- Sample data sanity-check (no all-zeros, no all-NaN, realistic ranges)

---

## 9. Cost Tracking

After Stage 1.5 completes, update `subscriptions.json`:

```json
{
  "active_subscriptions": [...],
  "total_monthly_cost_usd": 162,  // or 262 if Twitter included
  "monthly_budget_cap_usd": 500,
  "budget_remaining_usd": 338,
  "annual_projected_cost_usd": 1944,
  "last_updated": "2026-04-22"
}
```

---

## 10. Compute Distribution

| Task | Machine | Estimated complexity |
|------|---------|---------------------|
| 1.5.A Glassnode | Dragon | 3-5 hours (rate-limited) |
| 1.5.B CryptoQuant | Dragon | 2-3 hours |
| 1.5.C Polygon stocks | Gamma | 24-72 hours (1000 tickers × 5 TFs) |
| 1.5.C Polygon options | Gamma | 12-24 hours |
| 1.5.C Polygon news | Gamma | 4-8 hours |
| 1.5.D FMP | Omega | 2-4 hours (rate-limited) |
| 1.5.E Twitter | Omega | Variable, conserve quota |
| 1.5.F Quandl | Omega | 1-2 hours |

Polygon stocks dominates compute time. Run Polygon as background task on Gamma; other tasks complete in parallel.

---

## 11. Stage 1.5 Deliverable

`STAGE_1.5_DELIVERABLE.md`:

```markdown
# Stage 1.5 Deliverable — Paid Data Acquisition

## Summary

- Tasks completed: [N]/6
- Total datasets acquired: [N]
- Total data size: [X] GB
- Active subscriptions: [N]
- Monthly cost: $[X]/month

## Per-task status

| Task | Source | Status | Data size | Notes |
|------|--------|--------|-----------|-------|
| 1.5.A | Glassnode | DONE | X GB | All metrics in Standard tier acquired |
| 1.5.B | CryptoQuant | DONE | X GB | |
| 1.5.C | Polygon stocks | DONE | X GB | 1000 tickers × 5 timeframes |
| 1.5.C | Polygon options | DONE | X GB | |
| 1.5.C | Polygon news | DONE | X GB | |
| 1.5.D | FMP | DONE | X GB | |
| 1.5.E | Twitter | DONE/SKIPPED | | |
| 1.5.F | Quandl | DONE | X GB | |

## Subscription cost summary

| Service | Cost/month | Active |
|---------|-----------|--------|
| Glassnode Standard | $30 | YES |
| CryptoQuant Standard | $39 | YES |
| Polygon Developer | $79 | YES |
| FMP Starter | $14 | YES |
| Twitter Basic | $100 | YES/NO |
| **Total** | **$162-$262** | |

## Documentation audit

- All paid data folders have README: [PASS/FAIL]
- All paid data folders have data_dictionary: [PASS/FAIL]
- All paid data folders have provenance.json: [PASS/FAIL]

## Validation audit

[Per dataset validation results]

## User Gate

User reviews. If approved, proceed to Stage 1.6 (Validation and Documentation Audit).
```

---

## 12. User Gate

User reviews paid data acquisition results. Approves Stage 1.6.
