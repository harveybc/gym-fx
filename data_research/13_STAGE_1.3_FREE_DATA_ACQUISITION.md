# Stage 1.3 — Free Data Acquisition

**Stage goal:** Acquire ALL free data sources from the catalog (Stage 1.2). Validate, document, organize per Stage 1.1 templates.

**Inputs:** Stages 1.1 + 1.2 complete. User has approved catalog.

**Outputs:** All free data sources downloaded to their target folders with READMEs, data dictionaries, provenance.json files.

**Machine assignment:**
- **Omega:** Coordination + light tasks (FRED, yfinance equity indices, BTC/ETH spot)
- **Dragon:** Heavy crypto fetches (top 50 spot + perpetuals across multiple timeframes)
- **Gamma:** Macro + alternative data (FRED comprehensive, OECD, GDELT, on-chain)

Tasks parallelizable across machines per assignment.

---

## 1. Pre-Flight Checks

Before starting any acquisition, verify on each machine:

```bash
source /home/harveybc/anaconda3/etc/profile.d/conda.sh && conda activate tensorflow

# Required Python libraries
pip install --quiet yfinance fredapi pytrends requests pandas numpy pyarrow \
                    sec-edgar-downloader pmaw exchange_calendars python-holidays

# Verify
python -c "import yfinance, fredapi, pytrends, pandas as pd; print('OK')"

# Verify network access
curl -sI https://api.binance.com/api/v3/ping | head -1
curl -sI https://api.stlouisfed.org/fred/ | head -1
```

Document outputs in deliverable.

---

## 2. Acquisition Procedures

For each data source, agent follows this procedure:

1. **Create target folder** (already exists from Stage 1.1, but verify)
2. **Build acquisition script** at `~/Documents/financial_data/_scripts/fetch_<source>.py`
3. **Execute acquisition** with progress logging
4. **Validate data** using Stage II-0b 6-test battery for time series data
5. **Write README.md** in target folder using template
6. **Write data_dictionary.md** using template
7. **Write provenance.json** using template
8. **Append row to acquisition_log.csv**
9. **Compute SHA-256 checksum** of acquired files, record in provenance.json

If any step fails: HALT, produce `ESCALATION_<source>.md`, do not proceed.

---

## 3. Task Decomposition

### Task 1.3.A: Setup acquisition scripts directory (Omega)

```bash
mkdir -p ~/Documents/financial_data/_scripts
mkdir -p ~/Documents/financial_data/_scripts/lib  # shared utilities
```

Create shared utilities at `~/Documents/financial_data/_scripts/lib/`:

- `validation.py` — Stage II-0b 6-test battery as reusable functions
- `provenance.py` — provenance.json builder
- `readme_writer.py` — README and data_dictionary writer using templates
- `acquisition_log.py` — appends to acquisition_log.csv

These utilities are shared across all fetch scripts.

### Task 1.3.B: FRED comprehensive macro pull (Gamma)

Script: `_scripts/fetch_fred_comprehensive.py`

```python
"""
Fetches all FRED series listed in catalog.
Categories: inflation, employment, GDP, money, rates, consumer, housing,
            industrial, trade, fx_indices, stress, recession.
"""
import os
from fredapi import Fred

FRED_API_KEY = os.environ['FRED_API_KEY']  # from .env
fred = Fred(api_key=FRED_API_KEY)

SERIES_BY_CATEGORY = {
    "inflation": ["CPIAUCSL", "CPILFESL", "PPIACO", "PCEPILFE", "PCECTPI",
                  "MEDCPIM158SFRBCLE", "TRMMEANCPIM159SFRBCLE", "FPCPITOTLZGUSA"],
    "employment": ["UNRATE", "PAYEMS", "CIVPART", "U6RATE", "EMRATIO",
                   "AHETPI", "CES0500000003", "ICSA", "CCSA"],
    "gdp": ["GDP", "GDPC1", "A191RL1Q225SBEA", "GDPNOW", "GDPPOT"],
    "money": ["M1SL", "M2SL", "BOGMBASE", "WALCL", "WTREGEN"],
    "rates": ["DFF", "FEDFUNDS", "DPRIME", "DGS10", "DGS2", "DGS5", "DGS30",
              "DGS3MO", "TB3MS", "DTB3", "T10Y2Y", "T10Y3M"],
    "consumer": ["UMCSENT", "CSCICP03USM665S", "RSAFS", "PCEC", "PSAVERT",
                 "DSPI", "DSPIC96"],
    "housing": ["HOUST", "EXHOSLUSM495S", "CSUSHPISA", "MORTGAGE30US",
                "PERMIT", "PERMIT1", "MSACSR"],
    "industrial": ["INDPRO", "CAPUTLB50001SQ", "NAPM", "NAPMNOI", "BUSINV"],
    "trade": ["BOPGSTB", "IEAMTNQ", "IMPCH", "EXPCH"],
    "fx_indices": ["DTWEXBGS", "DTWEXAFEGS", "DTWEXEMEGS"],
    "stress": ["STLFSI4", "NFCI", "ANFCI", "TEDRATE"],
    "recession": ["USREC", "USRECP", "USRECQ", "USRECDM"],
    "inflation_expectations": ["T5YIE", "T5YIFR", "T10YIE", "MICH"],
}

for category, series_ids in SERIES_BY_CATEGORY.items():
    target_dir = f"~/Documents/financial_data/macro_economic/fred/{category}/"
    os.makedirs(os.path.expanduser(target_dir), exist_ok=True)

    for series_id in series_ids:
        try:
            data = fred.get_series(series_id)
            data.to_csv(f"{target_dir}{series_id}.csv")
            # ... write provenance, append log
        except Exception as e:
            # log error, continue (not all series available)
            pass

    # Write README and data_dictionary for this category folder
    write_readme(target_dir, category, series_ids)
    write_data_dictionary(target_dir, series_ids)
```

Folder structure produced:

```
macro_economic/fred/
├── inflation/
│   ├── README.md
│   ├── data_dictionary.md
│   ├── provenance.json
│   ├── CPIAUCSL.csv
│   ├── CPILFESL.csv
│   └── ...
├── employment/
│   └── ...
└── ...
```

Estimated time: 30-60 min (rate-limited to 120 requests/min).

### Task 1.3.C: Yahoo Finance equity indices (Omega)

Script: `_scripts/fetch_yfinance_equities.py`

For each ticker in equity indices list (S&P 500, NASDAQ, DAX, Nikkei, etc.):

```python
import yfinance as yf

# Fetch maximum available history
ticker_obj = yf.Ticker(ticker)
data = ticker_obj.history(period="max", interval="1d")

# Save
data.to_csv(f"market_data/equities/{region}/{symbol}/daily.csv")

# Provenance
write_provenance(...)
write_readme(...)
write_data_dictionary(...)
```

Tickers list:

```python
TICKERS = {
    "us_indices": {
        "spx": "^GSPC",
        "ndx": "^NDX",
        "dji": "^DJI",
        "rut": "^RUT",
        "vix": "^VIX",
    },
    "eu_indices": {
        "ftse100": "^FTSE",
        "dax": "^GDAXI",
        "cac40": "^FCHI",
        "stoxx50": "^STOXX50E",
        "ibex35": "^IBEX",
        "ftse_mib": "FTSEMIB.MI",
    },
    "asia_indices": {
        "nikkei225": "^N225",
        "hsi": "^HSI",
        "sse_comp": "000001.SS",
        "kospi": "^KS11",
        "nifty50": "^NSEI",
        "asx200": "^AXJO",
        "twii": "^TWII",
    },
    "emerging": {
        "bvsp": "^BVSP",
        "ipc_mx": "^MXX",
        "bist100": "XU100.IS",
        "jse40": "^J400.JO",
        "moex": "IMOEX.ME",
    },
}
```

After completion: 25 indices × 1 daily file each = 25 files + folder structure.

### Task 1.3.D: Yahoo Finance commodities + bonds (Omega)

Script: `_scripts/fetch_yfinance_commodities.py`

```python
COMMODITIES = {
    "precious_metals": {"gold": "GC=F", "silver": "SI=F", "platinum": "PL=F", "palladium": "PA=F"},
    "energy": {"wti_crude": "CL=F", "brent_crude": "BZ=F", "natural_gas": "NG=F",
               "heating_oil": "HO=F", "gasoline": "RB=F"},
    "agriculture": {"corn": "ZC=F", "wheat": "ZW=F", "soybeans": "ZS=F",
                    "sugar": "SB=F", "coffee": "KC=F", "cotton": "CT=F"},
    "industrial_metals": {"copper": "HG=F"},
}

# Same pattern as equity indices
```

### Task 1.3.E: Yahoo Finance ETFs (Omega)

Script: `_scripts/fetch_yfinance_etfs.py`

```python
ETFS = {
    "sector_spdrs": ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"],
    "country": ["EWJ", "EWZ", "FXI", "EEM", "EFA", "VWO", "EWG", "EWU", "INDA", "EWY"],
    "themes": ["ARKK", "SOXX", "ICLN", "TAN", "GDX", "GDXJ", "JETS"],
    "bonds": ["TLT", "IEF", "SHY", "HYG", "LQD", "EMB", "BND", "AGG"],
    "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "DBB", "PDBC"],
}
```

### Task 1.3.F: Yahoo Finance forex emerging markets (Omega)

```python
FX_EM = {
    "usdmxn": "USDMXN=X",
    "usdbrl": "USDBRL=X",
    "usdzar": "USDZAR=X",
    "usdtry": "USDTRY=X",
    "usdinr": "USDINR=X",
    "usdcny": "USDCNY=X",
    "usdrub": "USDRUB=X",
    "eurusd": "EURUSD=X",  # daily for cross-validation only; primary 1m from HistData
    "usdjpy": "USDJPY=X",
}
```

### Task 1.3.G: HistData FX bulk (USER manual + Omega processing)

User responsibility: download yearly 1-minute zip files for missing FX pairs.

User has already downloaded: EUR/USD, USD/JPY (5m via HistData; verify if 1m bulk needed).

User must download for these additional pairs (2005-present):
- GBP/USD
- USD/CHF
- AUD/USD
- USD/CAD
- NZD/USD
- EUR/GBP
- EUR/JPY
- GBP/JPY
- EUR/CHF
- AUD/JPY

Per pair, download all yearly zips (2005-2024 = 20 zips per pair × 10 pairs = 200 zips).

User downloads to `~/Downloads/histdata/<pair>/` (will be moved to data lake by agent).

REQUEST_USER document agent produces:

```
# REQUEST_USER: HistData additional FX pairs

User has previously downloaded EUR/USD and USD/JPY HistData.

Now agent needs additional 10 FX pairs of 1-minute ASCII data, 2005-2024:
GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF, AUD/JPY

Per pair:
1. Visit https://www.histdata.com/download-free-forex-data/
2. Click pair, select "1 Minute Bar Quotes" + "Generic ASCII"
3. Download each year 2005-2024 (20 zips per pair)
4. Place in: /home/harveybc/Downloads/histdata/<pair>/
   (e.g., /home/harveybc/Downloads/histdata/gbpusd/, etc.)

Total: 200 zip files across 10 directories.

Reply to chat when complete:
"Completed: histdata additional pairs
- gbpusd: <count> zips
- usdchf: <count> zips
- ...
"

Agent will process all zips and copy organized to data lake.
```

Agent's processing script: `_scripts/fetch_histdata_additional.py`

```python
# Process each pair's zips:
# 1. Unzip 20 yearly files
# 2. Concatenate to single 1m DataFrame
# 3. Resample to 5m, 15m, 1h, 4h, daily, weekly
# 4. Save to market_data/forex/g10/<pair>/{1m,5m,15m,1h,4h,1d,1w}.csv
# 5. Validate (Stage II-0b 6 tests)
# 6. README + data_dictionary + provenance
```

### Task 1.3.H: Binance crypto comprehensive (Dragon)

Script: `_scripts/fetch_binance_crypto.py`

Phase 1: get top 50 cryptocurrencies by market cap dynamically:

```python
# CoinGecko free API
import requests
r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                 params={"vs_currency": "usd", "order": "market_cap_desc",
                         "per_page": 50, "page": 1})
top50_ids = [coin["id"] for coin in r.json()]
```

Phase 2: for each Binance-traded symbol in top 50, fetch klines:

```python
# Each timeframe: 5m, 15m, 1h, 4h, 1d
# Each symbol's full Binance history
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

for symbol in BINANCE_SYMBOLS_FROM_TOP50:
    for tf in TIMEFRAMES:
        # Paginated fetch (Binance returns max 1000 bars per request)
        all_klines = fetch_full_history(symbol, tf)
        save_to_parquet(all_klines, f"market_data/crypto/spot_top50/{symbol}/{tf}.parquet")
```

Estimated volume: 50 symbols × 5 timeframes = 250 files. 5m files largest (~700K bars per symbol over 7 years).

Phase 3: perpetual futures + funding rates for top 20:

```python
# Binance Futures public API
PERP_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
PERP_SYMBOLS = TOP_20_PERPETUALS

# Klines
for symbol in PERP_SYMBOLS:
    for tf in PERP_TIMEFRAMES:
        save_to_parquet(...)

# Funding rates (8h frequency)
for symbol in PERP_SYMBOLS:
    save_funding_rate_history(...)
```

### Task 1.3.I: TrueFX cross-validation (USER + Dragon)

REQUEST_USER for TrueFX login + manual download (user already registered).

Per pair (EUR/USD, USD/JPY at minimum, others if user has bandwidth):
- Monthly zip files 2009-present
- Place in `~/Downloads/truefx/<pair>/`
- Agent processes and stores in `market_data/forex/g10/<pair>/_truefx_validation/`

### Task 1.3.J: OANDA cross-validation (Dragon)

# TODO: porque bajamos los pares de fx dos veces, una desde histfiles (mi trabajo manual) y otra desde oanda?

User must provide OANDA credentials (already pending from Project 2 setup; verify).

Script: `_scripts/fetch_oanda_validation.py`

For G10 pairs, fetch recent 5 years of 1h data from OANDA v20 API.

Stored in `market_data/forex/g10/<pair>/_oanda_validation/`.

### Task 1.3.K: Dukascopy cross-validation (Dragon)

Script: `_scripts/fetch_dukascopy.py`

Use `dukascopy-python` library or direct HTTP fetching of Dukascopy historical archive.

Free, no registration. Tick-level data.

For G10 pairs: fetch 2003-present.

### Task 1.3.L: GDELT news event database (Gamma)

Script: `_scripts/fetch_gdelt.py`

GDELT has 2 datasets:
- GDELT 1.0 (1979-2013, daily updates)
- GDELT 2.0 (2015-present, every 15 min)

Both are bulk downloads of CSV files. Massive volume (~TB if everything; we filter relevant events).

```python
# GDELT events filtered to financial/economic codes
RELEVANT_EVENT_CODES = [
    "0211", "0212", "021", "0411", "0412",  # economic cooperation/conflict
    "0231", "0233", "0241",  # economic sanctions, aid
    # ... all relevant CAMEO codes
]

# Daily files: gkg.gdelt.org or direct .zip downloads
# Filter at parse time, save filtered subset
```

Save to `alternative_data/news_sentiment/gdelt/{daily_csv_files}/`.

### Task 1.3.M: Reddit historical (Gamma)

Script: `_scripts/fetch_reddit.py`

Use `pmaw` (Pushshift wrapper) for historical comments/posts:

```python
from pmaw import PushshiftAPI

api = PushshiftAPI()

SUBREDDITS = ["wallstreetbets", "investing", "stocks", "cryptocurrency",
              "Bitcoin", "ethfinance", "options", "Daytrading"]

for subreddit in SUBREDDITS:
    # Fetch posts and comments 2008-present (pmaw handles pagination)
    posts = api.search_submissions(subreddit=subreddit, before=int(now), after=int(2008_start))
    save_jsonl(posts, f"alternative_data/social_sentiment/reddit/{subreddit}/posts.jsonl")
```

Note: Pushshift access has been intermittent in 2023-2024. If primary fails, alternative scrapers to be documented.

### Task 1.3.N: Google Trends (Gamma)

Script: `_scripts/fetch_google_trends.py`

Use `pytrends`:

```python
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)

KEYWORDS = [
    # Financial/market terms
    "bitcoin", "ethereum", "stock market crash", "recession", "inflation",
    "federal reserve", "interest rates", "buy stocks", "sell stocks",
    # Asset names
    "S&P 500", "NASDAQ", "gold price", "oil price", "USD",
    # Crypto-specific
    "buy bitcoin", "ethereum price", "crypto crash", "Binance",
    # Sentiment
    "stock market", "market crash", "buy now", "sell now",
]

# Pytrends has rate limits; respect them
for keyword in KEYWORDS:
    df = pytrends.build_payload([keyword], timeframe='2004-01-01 ' + today)
    save_to_csv(df, f"alternative_data/google_trends/{slug(keyword)}.csv")
```

### Task 1.3.O: Wikipedia traffic (Gamma)

Script: `_scripts/fetch_wikipedia_traffic.py`

Wikimedia REST API:

```python
ARTICLES = [
    "Bitcoin", "Ethereum", "Stock_market", "Gold", "Federal_Reserve",
    "S%26P_500", "NASDAQ", "Recession", "Inflation",
    # ... more articles
]

for article in ARTICLES:
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/20150701/{today}"
    data = requests.get(url).json()
    save(data, ...)
```

### Task 1.3.P: SEC EDGAR (Gamma)

Script: `_scripts/fetch_sec_edgar.py`

Use `sec-edgar-downloader`:

```python
from sec_edgar_downloader import Downloader

dl = Downloader("ProjectName", "harveybc@example.com",
                "/home/harveybc/Documents/financial_data/alternative_data/sec_filings/edgar/")

# For each S&P 500 ticker (and ideally more)
for ticker in SP500_TICKERS:
    dl.get("10-K", ticker, after="2003-01-01")
    dl.get("10-Q", ticker, after="2003-01-01")
    dl.get("8-K", ticker, after="2003-01-01")
    dl.get("4", ticker, after="2003-01-01")  # insider transactions
```

Significant data volume. Estimated 100-500GB.

### Task 1.3.Q: CFTC COT Reports (Omega)

Script: `_scripts/fetch_cftc_cot.py`

Reuse Stage II-0 logic. Bulk annual zips:

```python
for year in range(2000, 2026):
    url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
    download_and_extract(url, target=f"alternative_data/cot_reports/cftc/{year}/")

# Parse all and create unified dataframes per asset class
```

### Task 1.3.R: CoinMetrics Community on-chain (Gamma)

Script: `_scripts/fetch_coinmetrics_community.py` (extend Project 2 version)

Free metrics for BTC, ETH, and top 20 cryptocurrencies:

```python
ASSETS = ["btc", "eth", "ltc", "bch", "xmr", "doge", "ada", "dot", "sol", "atom",
          "near", "matic", "avax", "trx", "xlm", "etc", "fil", "icp", "uni", "link"]

COMMUNITY_METRICS = ["AdrActCnt", "TxCnt", "HashRate", "DiffMean", "BlkCnt",
                     "FeeMeanUSD", "TxTfrCnt", "TxTfrValAdjUSD"]

for asset in ASSETS:
    for metric in COMMUNITY_METRICS:
        # CoinMetrics community API (free, no key)
        url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets={asset}&metrics={metric}&start_time=2009-01-01&end_time={today}"
        # Save
```

### Task 1.3.S: Blockchain.com BTC metrics (Gamma)

Reuse Project 2 script.

### Task 1.3.T: Etherscan ETH metrics (Gamma)

Free tier, requires API key.

```python
# user must provide ETHERSCAN_API_KEY in .env
ETHERSCAN_KEY = os.environ['ETHERSCAN_API_KEY']

ENDPOINTS = [
    "/api?module=stats&action=ethsupply",
    "/api?module=stats&action=ethprice",
    "/api?module=stats&action=tokensupply",
    # ... daily snapshots tracked over time
]
```

### Task 1.3.U: DeFiLlama (Gamma)

Script: `_scripts/fetch_defillama.py`

DeFiLlama free public API:

```python
import requests

# TVL by chain
chains = requests.get("https://api.llama.fi/chains").json()

# TVL by protocol  
protocols = requests.get("https://api.llama.fi/protocols").json()

# Historical TVL per chain
for chain in MAJOR_CHAINS:
    history = requests.get(f"https://api.llama.fi/v2/historicalChainTvl/{chain}").json()
    save(history, f"alternative_data/defi_metrics/defillama/{chain}.json")
```

### Task 1.3.V: FINRA short interest (Gamma)

Bulk downloads from FINRA. Bi-weekly reports.

### Task 1.3.W: Trading calendars + holidays (Omega)

Use `exchange_calendars` and `python-holidays` libraries:

```python
import exchange_calendars as xcals

# Get all major exchange calendars
EXCHANGES = ["NYSE", "NASDAQ", "LSE", "XTKS", "XSHG", "XHKG", "XPAR", "XFRA"]

for exch in EXCHANGES:
    cal = xcals.get_calendar(exch)
    sessions = cal.sessions_in_range("1990-01-01", "2026-12-31")
    save_csv(sessions, f"reference_data/calendars/exchange_calendars/{exch}.csv")
```

### Task 1.3.X: BLS, BEA, Treasury (Gamma)

Government data sources beyond FRED:

- BLS: detailed employment series via FRED proxy or direct API
- BEA: GDP components via FRED proxy
- Treasury: TIC capital flows, Treasury Direct historical auctions

Most are wrapped by FRED. For non-FRED-wrapped:

```python
# Treasury TIC
treasury_tic_url = "https://home.treasury.gov/data/treasury-international-capital-tic-system"
# ... scrape or API access
```

### Task 1.3.Y: TradingEconomics free tier (Omega)

Limited free tier. Useful for economic calendar:

```python
# trading-economics public widget data
# OR fxstreet economic calendar (more accessible)
```

### Task 1.3.Z: BIS, World Bank, IMF (Gamma)

- BIS: international banking statistics, FX turnover surveys
- World Bank: Open Data API (development indicators, debt)
- IMF: SDMX API (balance of payments, IFS)

```python
# BIS bulk data: https://www.bis.org/statistics/full_data_sets.htm
# WB: pip install wbgapi
# IMF: pip install sdmx
```

---

## 4. Validation per Dataset

After each acquisition, run validation. For time series price/volume data:

```python
# Stage II-0b 6 tests adapted
def validate_timeseries(df, asset_type):
    tests = {}
    tests["bar_count"] = check_bar_count_realistic(df, expected_per_year=...)
    tests["weekend_gaps"] = check_weekend_gaps(df) if asset_type in ["fx", "equity"] else "N/A"
    tests["fat_tails"] = check_kurtosis(df) > 4
    tests["vol_clustering"] = check_squared_returns_acf(df) > 0.05
    tests["return_acf"] = abs(check_return_acf(df)) > 0.001
    tests["no_gbm"] = check_no_gbm_fingerprint(df)
    return tests
```

For non-timeseries data (events, news, fundamentals): different validation per type.

---

## 5. Documentation Standards

Per acquired dataset, generate:

**README.md** using template from Stage 1.1:
- Purpose: 1-2 sentences
- Contents: list of files
- Update frequency
- Last updated
- Related folders

**data_dictionary.md** using template:
- Source details
- Coverage
- Schema (column-by-column)
- Known issues
- Validation status

**provenance.json** using template:
- Source URLs
- Acquisition date
- License
- Cost
- Validation results
- Checksum

---

## 6. Acquisition Log

Every acquisition appends one row to `~/Documents/financial_data/_metadata/acquisition_log.csv`:

```csv
timestamp,stage,dataset,source,status,size_mb,duration_sec,notes
2026-04-22T10:00:00Z,1.3.B,fred_inflation_CPIAUCSL,FRED,success,0.5,2,
2026-04-22T10:00:05Z,1.3.B,fred_inflation_CPILFESL,FRED,success,0.4,1,
...
```

---

## 7. Compute Distribution

| Machine | Tasks |
|---------|-------|
| **Omega** | 1.3.A (setup), 1.3.C (yfinance equities), 1.3.D (commodities), 1.3.E (ETFs), 1.3.F (FX EM), 1.3.G (HistData processing), 1.3.Q (CFTC), 1.3.W (calendars), 1.3.Y (econ calendar) |
| **Dragon** | 1.3.H (Binance crypto comprehensive), 1.3.I (TrueFX), 1.3.J (OANDA), 1.3.K (Dukascopy) |
| **Gamma** | 1.3.B (FRED comprehensive), 1.3.L (GDELT), 1.3.M (Reddit), 1.3.N (Google Trends), 1.3.O (Wikipedia), 1.3.P (SEC EDGAR), 1.3.R (CoinMetrics), 1.3.S (Blockchain.com), 1.3.T (Etherscan), 1.3.U (DeFiLlama), 1.3.V (FINRA), 1.3.X (BLS/BEA/Treasury), 1.3.Z (BIS/WB/IMF) |

Tasks parallelizable; use `tmux` or background execution per machine. Periodic agent check-ins to monitor progress.

---

## 8. Stage 1.3 Deliverable

`STAGE_1.3_DELIVERABLE.md`:

```markdown
# Stage 1.3 Deliverable — Free Data Acquisition

## Summary

- **Tasks completed:** [N]/26 (1.3.A through 1.3.Z)
- **Total datasets acquired:** [N]
- **Total data size:** [X] GB
- **Total acquisition time:** [Y] hours
- **Failed acquisitions:** [N] (with reasons)

## Acquisition Log

[Reference acquisition_log.csv]

## Per-Task Status

| Task | Source | Status | Notes |
|------|--------|--------|-------|
| 1.3.A | Setup | DONE | |
| 1.3.B | FRED | DONE | 200 series across 12 categories |
| 1.3.C | yfinance equities | DONE | 25 indices |
| ... | ... | ... | ... |

## Documentation Audit

- All folders have README.md: [PASS/FAIL]
- All dataset folders have data_dictionary.md: [PASS/FAIL]
- All dataset folders have provenance.json: [PASS/FAIL]
- Acquisition log complete: [PASS/FAIL]

## Validation Audit

- Datasets passed all validation tests: [N]
- Datasets with advisory failures: [N]
- Datasets failed validation: [N]
- Failed validation details: [list]

## Pending Manual Tasks (REQUEST_USER)

If any tasks require user actions (HistData additional pairs, TrueFX downloads):
- [list of pending REQUEST_USER docs]

## User Gate

Awaiting user approval to proceed to Stage 1.5 (Paid Data) once Stage 1.4 (Registrations) complete.
```

---

## 9. User Gate

User reviews deliverable. Confirms acquisition complete. Approves moving to Stage 1.5.
