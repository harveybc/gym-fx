# Stage 1.2 — Data Catalog

**Stage goal:** Document EXHAUSTIVELY every data source we will acquire in Phase 1. This is the master inventory. Stages 1.3 (free) and 1.5 (paid) execute against this catalog.

**Inputs:** Stage 1.1 complete (folder structure exists).

**Outputs:**
- `~/Documents/financial_data/_metadata/data_catalog.json` — machine-readable
- `STAGE_1.2_DELIVERABLE.md` — human-readable
- User-approved subscription list

**Machine:** Omega (catalog construction is reading + writing, no compute).

---

## 1. Catalog Categories

The catalog is organized by data category. Each entry specifies:
- Data type
- Source provider
- Cost (free / subscription with $/mo)
- Coverage (timeframe, history depth)
- Acquisition method
- Target folder
- Priority (HIGH / MEDIUM / LOW)
- Justification (why we want this)

---

## 2. EQUITIES (Market Data)

### 2.1 US Equity Indices

| Source | Provider | Cost | Coverage | Method | Folder | Priority | Justification |
|--------|----------|------|----------|--------|--------|----------|---------------|
| S&P 500 daily | yfinance | Free | 1993-present, daily | API | `market_data/equities/us_indices/spx/` | HIGH | Reference index |
| NASDAQ-100 daily | yfinance | Free | 1985-present, daily | API | `market_data/equities/us_indices/ndx/` | HIGH | Tech-heavy reference |
| Dow Jones daily | yfinance | Free | 1992-present, daily | API | `market_data/equities/us_indices/dji/` | HIGH | Industrial reference |
| Russell 2000 daily | yfinance | Free | 1987-present, daily | API | `market_data/equities/us_indices/rut/` | HIGH | Small-cap reference |
| VIX daily | yfinance + FRED | Free | 1990-present, daily | API | `market_data/equities/us_indices/vix/` | HIGH | Volatility regime |
| SPX intraday 1m | Polygon.io | $30/mo Starter | 5+ years intraday | API | `market_data/equities/us_indices/spx_intraday/` | HIGH | Higher frequency |
| SPY ETF tick | Polygon.io Starter | $30/mo | 5+ years tick | API | `market_data/equities/etfs/spy_tick/` | MEDIUM | Microstructure |

### 2.2 US Individual Stocks

| Source | Provider | Cost | Coverage | Method | Folder | Priority | Justification |
|--------|----------|------|----------|--------|--------|----------|---------------|
| S&P 500 components daily | yfinance | Free | Variable per stock, daily | API | `market_data/equities/us_individual/sp500/` | HIGH | Cross-sectional features |
| NASDAQ-100 components daily | yfinance | Free | Variable, daily | API | `market_data/equities/us_individual/ndx100/` | HIGH | Tech sector |
| Top 1000 by market cap intraday | Polygon.io Developer | $79/mo | 5+ years 1m bars | API | `market_data/equities/us_individual/top1000_1m/` | MEDIUM | If budget allows |

### 2.3 European Equity Indices

| Source | Provider | Cost | Coverage | Method | Folder | Priority | Justification |
|--------|----------|------|----------|--------|--------|----------|---------------|
| FTSE 100 daily | yfinance | Free | 1984-present | API | `market_data/equities/eu_indices/ftse100/` | HIGH | UK reference |
| DAX daily | yfinance | Free | 1990-present | API | `market_data/equities/eu_indices/dax/` | HIGH | German reference |
| CAC 40 daily | yfinance | Free | 1990-present | API | `market_data/equities/eu_indices/cac40/` | HIGH | French reference |
| EURO STOXX 50 daily | yfinance | Free | 1986-present | API | `market_data/equities/eu_indices/stoxx50/` | HIGH | Eurozone reference |
| IBEX 35 daily | yfinance | Free | 1993-present | API | `market_data/equities/eu_indices/ibex35/` | MEDIUM | Spanish |
| FTSE MIB daily | yfinance | Free | 1997-present | API | `market_data/equities/eu_indices/ftse_mib/` | MEDIUM | Italian |

### 2.4 Asia Equity Indices

| Source | Provider | Cost | Coverage | Method | Folder | Priority | Justification |
|--------|----------|------|----------|--------|--------|----------|---------------|
| Nikkei 225 daily | yfinance | Free | 1965-present | API | `market_data/equities/asia_indices/nikkei225/` | HIGH | Japan reference |
| Hang Seng daily | yfinance | Free | 1986-present | API | `market_data/equities/asia_indices/hsi/` | HIGH | HK reference |
| Shanghai Composite daily | yfinance | Free | 1990-present | API | `market_data/equities/asia_indices/sse_comp/` | HIGH | China reference |
| KOSPI daily | yfinance | Free | 1996-present | API | `market_data/equities/asia_indices/kospi/` | HIGH | Korea reference |
| Nifty 50 daily | yfinance | Free | 2007-present | API | `market_data/equities/asia_indices/nifty50/` | HIGH | India reference |
| ASX 200 daily | yfinance | Free | 1992-present | API | `market_data/equities/asia_indices/asx200/` | MEDIUM | Australia |
| Taiwan TWII daily | yfinance | Free | 1997-present | API | `market_data/equities/asia_indices/twii/` | MEDIUM | Taiwan |

### 2.5 Emerging Markets

| Source | Provider | Cost | Coverage | Method | Folder | Priority | Justification |
|--------|----------|------|----------|--------|--------|----------|---------------|
| Bovespa (Brazil) daily | yfinance | Free | 1993-present | API | `market_data/equities/emerging/bvsp/` | HIGH | Latin America |
| Mexico IPC daily | yfinance | Free | 1991-present | API | `market_data/equities/emerging/ipc_mx/` | MEDIUM | |
| Turkey BIST 100 daily | yfinance | Free | 1990-present | API | `market_data/equities/emerging/bist100/` | MEDIUM | Frontier |
| South Africa JSE 40 daily | yfinance | Free | 1995-present | API | `market_data/equities/emerging/jse40/` | MEDIUM | Africa |
| Russia MOEX daily | yfinance | Free | 1997-present (suspended 2022) | API | `market_data/equities/emerging/moex/` | LOW | Geopolitical risk |

### 2.6 ETFs

| Source | Provider | Cost | Coverage | Method | Folder | Priority | Justification |
|--------|----------|------|----------|--------|--------|----------|---------------|
| Major sector SPDRs (XLF, XLK, XLE, etc.) daily | yfinance | Free | 1998-present | API | `market_data/equities/etfs/sector_spdrs/` | HIGH | Sector rotation features |
| Country ETFs (EWJ, EWZ, FXI, EEM, etc.) daily | yfinance | Free | Variable | API | `market_data/equities/etfs/country/` | HIGH | International exposure |
| Theme ETFs (ARKK, SOXX, etc.) daily | yfinance | Free | Variable | API | `market_data/equities/etfs/themes/` | MEDIUM | |
| Bond ETFs (TLT, HYG, LQD, etc.) daily | yfinance | Free | Variable | API | `market_data/equities/etfs/bonds/` | HIGH | Cross-asset features |
| Commodity ETFs (GLD, USO, DBA, etc.) daily | yfinance | Free | Variable | API | `market_data/equities/etfs/commodities/` | HIGH | |

---

## 3. FOREX (Market Data)

### 3.1 G10 Major Pairs

# TODO: do not use 1m, nor weekly, nor daily, we neet the other periodicities tho


For ALL of the following, acquire 1m, 5m, 15m, 1h, 4h, daily, weekly:

| Pair | Source | Cost | Coverage | Method | Folder | Priority |
|------|--------|------|----------|--------|--------|----------|
| EUR/USD | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/eurusd/` | HIGH |
| USD/JPY | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/usdjpy/` | HIGH |
| GBP/USD | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/gbpusd/` | HIGH |
| USD/CHF | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/usdchf/` | HIGH |
| AUD/USD | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/audusd/` | HIGH |
| USD/CAD | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/usdcad/` | HIGH |
| NZD/USD | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/nzdusd/` | HIGH |
| EUR/GBP | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/eurgbp/` | HIGH |
| EUR/JPY | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/eurjpy/` | HIGH |
| GBP/JPY | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/gbpjpy/` | HIGH |
| EUR/CHF | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/eurchf/` | MEDIUM |
| AUD/JPY | HistData | Free | 2005-present, 1m | bulk download | `market_data/forex/g10/audjpy/` | MEDIUM |

User has already downloaded EUR/USD and USD/JPY 5m. Stage 1.3 procedure: agent verifies and supplements; user re-downloads remaining pairs as needed.

### 3.2 Emerging Market FX

| Pair | Source | Cost | Coverage | Method | Folder | Priority |
|------|--------|------|----------|--------|--------|----------|
| USD/MXN | yfinance | Free | 1996-present, daily | API | `market_data/forex/emerging_markets/usdmxn/` | MEDIUM |
| USD/BRL | yfinance | Free | 2003-present, daily | API | `market_data/forex/emerging_markets/usdbrl/` | MEDIUM |
| USD/ZAR | yfinance | Free | 2003-present, daily | API | `market_data/forex/emerging_markets/usdzar/` | MEDIUM |
| USD/TRY | yfinance | Free | 2003-present, daily | API | `market_data/forex/emerging_markets/usdtry/` | MEDIUM |
| USD/INR | yfinance | Free | 2003-present, daily | API | `market_data/forex/emerging_markets/usdinr/` | MEDIUM |
| USD/CNY | yfinance | Free | 1981-present, daily | API | `market_data/forex/emerging_markets/usdcny/` | MEDIUM |
| USD/RUB | yfinance | Free | 1996-present (suspended 2022) | API | `market_data/forex/emerging_markets/usdrub/` | LOW |

### 3.3 Tick-Level FX (Cross-validation)

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| TrueFX EUR/USD, USD/JPY | TrueFX | Free w/ registration | 2009-present | manual download | `market_data/forex/g10/eurusd/_truefx_validation/` | HIGH |
| OANDA EUR/USD, USD/JPY | OANDA Demo API | Free | Recent ~5 years | API | `market_data/forex/g10/eurusd/_oanda_validation/` | MEDIUM |
| Dukascopy historical tick | Dukascopy | Free | 2003-present | bulk download | `market_data/forex/g10/_dukascopy_validation/` | MEDIUM |

---

## 4. CRYPTO (Market Data)

### 4.1 Spot Top 50 by Market Cap

For ALL: acquire 5m, 15m, 1h, 4h, daily.

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| BTC/USDT | Binance public API | Free | 2017-present | API | `market_data/crypto/spot_top50/btc_usdt/` | HIGH |
| ETH/USDT | Binance public API | Free | 2017-present | API | `market_data/crypto/spot_top50/eth_usdt/` | HIGH |
| BNB/USDT | Binance public API | Free | 2017-present | API | `market_data/crypto/spot_top50/bnb_usdt/` | HIGH |
| SOL/USDT | Binance public API | Free | 2020-present | API | `market_data/crypto/spot_top50/sol_usdt/` | HIGH |
| XRP/USDT | Binance public API | Free | 2018-present | API | `market_data/crypto/spot_top50/xrp_usdt/` | HIGH |
| ADA/USDT | Binance public API | Free | 2018-present | API | `market_data/crypto/spot_top50/ada_usdt/` | HIGH |
| AVAX/USDT | Binance public API | Free | 2020-present | API | `market_data/crypto/spot_top50/avax_usdt/` | HIGH |
| DOGE/USDT | Binance public API | Free | 2019-present | API | `market_data/crypto/spot_top50/doge_usdt/` | HIGH |
| DOT/USDT | Binance public API | Free | 2020-present | API | `market_data/crypto/spot_top50/dot_usdt/` | HIGH |
| MATIC/USDT | Binance public API | Free | 2019-present | API | `market_data/crypto/spot_top50/matic_usdt/` | HIGH |
| TRX/USDT | Binance public API | Free | 2018-present | API | `market_data/crypto/spot_top50/trx_usdt/` | HIGH |
| LINK/USDT | Binance public API | Free | 2019-present | API | `market_data/crypto/spot_top50/link_usdt/` | HIGH |
| LTC/USDT | Binance public API | Free | 2017-present | API | `market_data/crypto/spot_top50/ltc_usdt/` | MEDIUM |
| BCH/USDT | Binance public API | Free | 2017-present | API | `market_data/crypto/spot_top50/bch_usdt/` | MEDIUM |
| ATOM/USDT | Binance public API | Free | 2019-present | API | `market_data/crypto/spot_top50/atom_usdt/` | MEDIUM |
| ... (top 50 by current market cap) | Binance | Free | Variable | API | `market_data/crypto/spot_top50/<symbol>/` | MEDIUM |

Agent fetches list of top 50 by market cap dynamically via CoinGecko free API at acquisition time.

### 4.2 Perpetual Futures

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| BTCUSDT perp | Binance Futures public API | Free | 2019-09 to present | API | `market_data/crypto/perpetuals/btcusdt_perp/` | HIGH |
| ETHUSDT perp | Binance Futures public API | Free | 2019-11 to present | API | `market_data/crypto/perpetuals/ethusdt_perp/` | HIGH |
| Top 20 perpetuals | Binance Futures | Free | Variable | API | `market_data/crypto/perpetuals/<symbol>_perp/` | MEDIUM |

### 4.3 Funding Rates

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| Funding rates Binance perpetuals | Binance Futures public API | Free | 2019-09 to present, 8h freq | API | `market_data/crypto/funding_rates/binance/` | HIGH |
| Funding rates Bybit | Bybit public API | Free | 2019-present | API | `market_data/crypto/funding_rates/bybit/` | MEDIUM |
| Funding rates OKX | OKX public API | Free | 2019-present | API | `market_data/crypto/funding_rates/okx/` | MEDIUM |

---

## 5. COMMODITIES

### 5.1 Precious Metals

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| Gold (GC=F, XAU/USD) daily | yfinance | Free | 2000-present | API | `market_data/commodities/precious_metals/gold/` | HIGH |
| Silver (SI=F, XAG/USD) daily | yfinance | Free | 2000-present | API | `market_data/commodities/precious_metals/silver/` | HIGH |
| Platinum (PL=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/precious_metals/platinum/` | MEDIUM |
| Palladium (PA=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/precious_metals/palladium/` | MEDIUM |
| LBMA daily fix | LBMA | Free | 1968-present | bulk download | `market_data/commodities/precious_metals/lbma_fix/` | MEDIUM |

### 5.2 Energy

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| WTI Crude (CL=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/energy/wti_crude/` | HIGH |
| Brent Crude (BZ=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/energy/brent_crude/` | HIGH |
| Natural Gas (NG=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/energy/natural_gas/` | HIGH |
| Heating Oil (HO=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/energy/heating_oil/` | MEDIUM |
| RBOB Gasoline (RB=F) daily | yfinance | Free | 2005-present | API | `market_data/commodities/energy/gasoline/` | MEDIUM |
| EIA inventories | EIA API | Free | 1982-present, weekly | API | `market_data/commodities/energy/eia_inventories/` | HIGH |

### 5.3 Agriculture

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| Corn (ZC=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/agriculture/corn/` | MEDIUM |
| Wheat (ZW=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/agriculture/wheat/` | MEDIUM |
| Soybeans (ZS=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/agriculture/soybeans/` | MEDIUM |
| Sugar (SB=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/agriculture/sugar/` | LOW |
| Coffee (KC=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/agriculture/coffee/` | LOW |
| Cotton (CT=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/agriculture/cotton/` | LOW |
| USDA WASDE reports | USDA | Free | Monthly | bulk | `market_data/commodities/agriculture/wasde/` | LOW |

### 5.4 Industrial Metals

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| Copper (HG=F) daily | yfinance | Free | 2000-present | API | `market_data/commodities/industrial_metals/copper/` | HIGH |
| Aluminum (LMAHDS03 LME) | Quandl Nasdaq Data Link | $20/mo | 1989-present | API | `market_data/commodities/industrial_metals/aluminum/` | LOW |
| Iron Ore | LBMA / Various | Free | Variable | bulk | `market_data/commodities/industrial_metals/iron_ore/` | LOW |

---

## 6. BONDS

### 6.1 US Treasuries

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| US 10Y Yield | FRED | Free | 1962-present | API | `market_data/bonds/us_treasuries/dgs10/` | HIGH |
| US 2Y Yield | FRED | Free | 1976-present | API | `market_data/bonds/us_treasuries/dgs2/` | HIGH |
| US 5Y Yield | FRED | Free | 1962-present | API | `market_data/bonds/us_treasuries/dgs5/` | HIGH |
| US 30Y Yield | FRED | Free | 1977-present | API | `market_data/bonds/us_treasuries/dgs30/` | HIGH |
| US 3M T-Bill | FRED | Free | 1934-present | API | `market_data/bonds/us_treasuries/tb3ms/` | HIGH |
| Yield curve full term structure | FRED daily | Free | 1990-present | API | `market_data/bonds/us_treasuries/yield_curve/` | HIGH |
| TIPS yields | FRED | Free | 2003-present | API | `market_data/bonds/us_treasuries/tips/` | HIGH |
| Breakeven inflation | FRED | Free | 2003-present | API | `market_data/bonds/us_treasuries/breakeven/` | HIGH |

### 6.2 Sovereign Global

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| German 10Y Bund | OECD / FRED | Free | 1990-present | API | `market_data/bonds/sovereign_global/de_10y/` | HIGH |
| UK 10Y Gilt | FRED / BoE | Free | 1980-present | API | `market_data/bonds/sovereign_global/uk_10y/` | HIGH |
| Japan 10Y JGB | FRED | Free | 1989-present | API | `market_data/bonds/sovereign_global/jp_10y/` | HIGH |
| France 10Y OAT | OECD | Free | 1990-present | API | `market_data/bonds/sovereign_global/fr_10y/` | MEDIUM |
| Italy 10Y BTP | OECD | Free | 1990-present | API | `market_data/bonds/sovereign_global/it_10y/` | MEDIUM |
| Spain 10Y | OECD | Free | 1990-present | API | `market_data/bonds/sovereign_global/es_10y/` | MEDIUM |

### 6.3 Corporate Spreads

| Asset | Source | Cost | Coverage | Method | Folder | Priority |
|-------|--------|------|----------|--------|--------|----------|
| US Investment Grade OAS (BAMLC0A0CM) | FRED | Free | 1996-present | API | `market_data/bonds/corporate/ig_oas/` | HIGH |
| US High Yield OAS (BAMLH0A0HYM2) | FRED | Free | 1996-present | API | `market_data/bonds/corporate/hy_oas/` | HIGH |
| AAA-BAA spread | FRED | Free | 1919-present | API | `market_data/bonds/corporate/aaa_baa_spread/` | HIGH |

---

## 7. MACRO ECONOMIC DATA

### 7.1 FRED (Federal Reserve Economic Data)

Primary macro source. Free, comprehensive, gold-standard.

| Series Group | Series IDs | Coverage | Folder | Priority |
|--------------|-----------|----------|--------|----------|
| Inflation | CPIAUCSL, CPILFESL, PPIACO, PCEPILFE | 1947-present | `macro_economic/fred/inflation/` | HIGH |
| Employment | UNRATE, PAYEMS, CIVPART, U6RATE | Variable | `macro_economic/fred/employment/` | HIGH |
| GDP | GDP, GDPC1, GDPNOW (Atlanta Fed) | 1947-present | `macro_economic/fred/gdp/` | HIGH |
| Money supply | M1SL, M2SL, BOGMBASE | Variable | `macro_economic/fred/money/` | HIGH |
| Interest rates | DFF, FEDFUNDS, DPRIME | 1954-present | `macro_economic/fred/rates/` | HIGH |
| Consumer | UMCSENT, CSCICP03USM665S, RSAFS | Variable | `macro_economic/fred/consumer/` | HIGH |
| Housing | HOUST, EXHOSLUSM495S, CSUSHPISA | Variable | `macro_economic/fred/housing/` | HIGH |
| Industrial | INDPRO, CAPUTLB50001SQ, NAPM | Variable | `macro_economic/fred/industrial/` | HIGH |
| Trade | BOPGSTB, IEAMTNQ | Variable | `macro_economic/fred/trade/` | MEDIUM |
| Exchange rates | DTWEXBGS, DTWEXAFEGS | Variable | `macro_economic/fred/fx_indices/` | HIGH |
| Stress indices | STLFSI4, NFCI, ANFCI | Variable | `macro_economic/fred/stress/` | HIGH |
| Recession indicators | USREC, USRECP | Variable | `macro_economic/fred/recession/` | HIGH |

Comprehensive FRED pull: agent gets ALL series in above categories. Approximate count: ~200 individual series.

### 7.2 OECD

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Composite Leading Indicators | OECD API | Free | 1955-present | API | `macro_economic/oecd/cli/` | HIGH |
| Business Confidence | OECD API | Free | 1960-present | API | `macro_economic/oecd/business_confidence/` | HIGH |
| Consumer Confidence | OECD API | Free | 1960-present | API | `macro_economic/oecd/consumer_confidence/` | HIGH |

### 7.3 ECB

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| ECB main rate | ECB SDW | Free | 1999-present | API | `macro_economic/ecb/main_rate/` | HIGH |
| Eurozone HICP inflation | ECB SDW | Free | 1990-present | API | `macro_economic/ecb/hicp/` | HIGH |
| Eurozone unemployment | ECB SDW | Free | 1990-present | API | `macro_economic/ecb/unemployment/` | HIGH |

### 7.4 BoJ

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Tankan survey | BoJ | Free | Quarterly, 1974-present | bulk | `macro_economic/boj/tankan/` | MEDIUM |
| Japan policy rate | BoJ / FRED | Free | 1972-present | API | `macro_economic/boj/policy_rate/` | HIGH |

### 7.5 Inflation Expectations

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| 5Y5Y forward inflation expectations | FRED (T5YIFR) | Free | 2003-present | API | `macro_economic/inflation_expectations/usd_5y5y/` | HIGH |
| Michigan 1Y inflation expectations | FRED | Free | 1978-present | API | `macro_economic/inflation_expectations/michigan/` | HIGH |
| Survey of Professional Forecasters | Philly Fed | Free | 1968-present | bulk | `macro_economic/inflation_expectations/spf/` | MEDIUM |

### 7.6 Economic Surprise Indices

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Citi Economic Surprise Index | Bloomberg / Citi | NOT FREE | NA | NA | NA | EXCLUDED |
| Custom-built economic surprise index | Calculated from FRED + actual vs estimate | Free | Custom | computed | `macro_economic/economic_surprise/custom/` | HIGH |

---

## 8. ALTERNATIVE DATA

### 8.1 News Sentiment

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| GDELT (Global news event database) | GDELT Project | Free | 1979-present | bulk download | `alternative_data/news_sentiment/gdelt/` | HIGH |
| Common Crawl news | Common Crawl | Free | 2008-present | bulk | `alternative_data/news_sentiment/common_crawl/` | LOW (volume too large) |
| Financial news headlines historical | Polygon.io News API | $79/mo Developer | 2015-present | API | `alternative_data/news_sentiment/polygon_news/` | HIGH |
| StockTwits historical | StockTwits API | Free tier limited | 2008-present | API | `alternative_data/news_sentiment/stocktwits/` | MEDIUM |

### 8.2 Social Sentiment

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Twitter/X historical | X API v2 Basic | $100/mo | 2010-present (limited) | API | `alternative_data/social_sentiment/twitter/` | MEDIUM |
| Reddit (specific subs: WSB, cryptocurrency, investing) | Pushshift / pmaw | Free | 2008-present | bulk | `alternative_data/social_sentiment/reddit/` | HIGH |

### 8.3 Search Trends

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Google Trends | pytrends library | Free | 2004-present | API | `alternative_data/google_trends/` | HIGH |
| Wikipedia page traffic | Wikimedia REST API | Free | 2008-present | API | `alternative_data/wikipedia_traffic/` | HIGH |

### 8.4 SEC Filings

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| EDGAR all filings | SEC EDGAR | Free | 1993-present | API | `alternative_data/sec_filings/edgar/` | HIGH |
| 10-K, 10-Q parsed | SEC + sec-edgar-downloader | Free | 1993-present | API + parsing | `alternative_data/sec_filings/parsed/` | HIGH |
| 8-K filings | SEC EDGAR | Free | 1993-present | API | `alternative_data/sec_filings/8k/` | HIGH |

### 8.5 Insider Trading

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| SEC Form 4 (insider transactions) | SEC EDGAR | Free | 2003-present | API | `alternative_data/insider_trading/form4/` | HIGH |

### 8.6 Earnings Estimates and Analyst Recommendations

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Yahoo Finance analyst data | yfinance | Free (limited) | Recent | API | `alternative_data/earnings_estimates/yfinance/` | MEDIUM |
| FMP (Financial Modeling Prep) Premium | FMP | $14/mo Starter | 5+ years | API | `alternative_data/analyst_recommendations/fmp/` | HIGH |

### 8.7 Short Interest

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| FINRA short interest reports | FINRA | Free | 2007-present, biweekly | bulk | `alternative_data/short_interest/finra/` | HIGH |
| NYSE short interest | NYSE | Free | bi-monthly | bulk | `alternative_data/short_interest/nyse/` | MEDIUM |

### 8.8 ETF Flows

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| ETF.com daily flow data | ETF.com | Free (delayed) | Recent | scraping | `alternative_data/etf_flows/etfcom/` | MEDIUM |

### 8.9 COT Reports

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| CFTC Commitments of Traders | CFTC | Free | 2000-present, weekly | bulk download | `alternative_data/cot_reports/cftc/` | HIGH |

### 8.10 Options Flow / Dark Pool

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| FINRA TRACE | FINRA | Free | Recent | API | `alternative_data/dark_pool/finra/` | LOW |
| Polygon options data | Polygon.io Developer | Already in $79/mo | 2015-present | API | `derivatives/options_chains/polygon/` | HIGH (in derivatives) |

### 8.11 Crypto On-Chain BTC

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| CoinMetrics Community | CoinMetrics | Free | 2009-present | API | `alternative_data/onchain_btc/coinmetrics_community/` | HIGH |
| Glassnode Standard Tier | Glassnode | $30/mo | 2009-present | API | `alternative_data/onchain_btc/glassnode/` | HIGH |
| Blockchain.com API | Blockchain.com | Free | 2009-present | API | `alternative_data/onchain_btc/blockchain_com/` | HIGH |
| BitInfoCharts | BitInfoCharts | Free | 2010-present | scraping | `alternative_data/onchain_btc/bitinfocharts/` | MEDIUM |
| Mempool.space | Mempool.space | Free | 2014-present | API | `alternative_data/onchain_btc/mempool_space/` | HIGH |

### 8.12 Crypto On-Chain ETH

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Etherscan API | Etherscan | Free tier | 2015-present | API | `alternative_data/onchain_eth/etherscan/` | HIGH |
| Glassnode ETH | Glassnode | Already in $30/mo | 2015-present | API | `alternative_data/onchain_eth/glassnode/` | HIGH |
| Dune Analytics public dashboards | Dune | Free | Recent | API | `alternative_data/onchain_eth/dune/` | MEDIUM |
| The Graph protocol | The Graph | Free | 2018-present | API | `alternative_data/onchain_eth/the_graph/` | LOW |

### 8.13 Crypto Exchange Flows

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| CryptoQuant Standard | CryptoQuant | $39/mo | 2017-present | API | `alternative_data/exchange_flows/cryptoquant/` | HIGH |
| Whale Alert | Whale Alert | Free tier | Recent | API | `alternative_data/whale_movements/whale_alert/` | MEDIUM |

### 8.14 DeFi Metrics

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| DeFiLlama | DeFiLlama | Free | 2019-present | API | `alternative_data/defi_metrics/defillama/` | HIGH |
| DefiLab / TokenTerminal | TokenTerminal | $200/mo or free tier | Recent | API | `alternative_data/defi_metrics/tokenterminal/` | LOW |

---

## 9. MICROSTRUCTURE

### 9.1 Order Book Snapshots

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Binance order book snapshots | Binance public | Free | Real-time only (no historical for free) | WebSocket capture | `microstructure/order_book_snapshots/binance/` | LOW (need real-time capture) |
| LOBSTER NASDAQ order book | LOBSTER | $$$ | Variable | manual | NA | EXCLUDED (cost-prohibitive academic) |

### 9.2 Trade Volume Profile

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Binance trade history | Binance public | Free | Recent (12 months) | API | `microstructure/trade_volume_profile/binance/` | MEDIUM |
| Polygon trade ticks (US equities) | Polygon Developer | Already in $79 | 5 years | API | `microstructure/trade_volume_profile/polygon/` | HIGH |

### 9.3 Spread Data

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Computed from tick data | Internal | Free | Variable | computation | `microstructure/spread_data/computed/` | MEDIUM |

---

## 10. DERIVATIVES

### 10.1 Options Chains

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Polygon options chains | Polygon Developer | Already in $79 | 2015-present | API | `derivatives/options_chains/polygon/` | HIGH |
| Yahoo options chain (current only) | yfinance | Free | Current snapshot only | API | `derivatives/options_chains/yfinance/` | LOW |
| CBOE settlement values | CBOE | Free | 2010-present | bulk | `derivatives/options_chains/cboe_settlements/` | MEDIUM |

### 10.2 Options Greeks (computed)

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Computed from options chains | Internal | Free | matches options data | computation | `derivatives/options_greeks/computed/` | HIGH |

### 10.3 Futures Curves

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| WTI futures curve (CL, multiple expiries) | yfinance | Free | 2000-present | API | `derivatives/futures_curves/wti/` | HIGH |
| Gold futures curve | yfinance | Free | 2000-present | API | `derivatives/futures_curves/gold/` | MEDIUM |
| Copper futures curve | yfinance | Free | Variable | API | `derivatives/futures_curves/copper/` | MEDIUM |
| Treasury futures curve (ZN, ZB, ZF, ZT) | yfinance | Free | 2000-present | API | `derivatives/futures_curves/treasuries/` | HIGH |

### 10.4 VIX Term Structure

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| VIX, VIX9D, VIX3M, VIX6M, VIX1Y | yfinance | Free | Variable | API | `derivatives/vix_term_structure/cboe/` | HIGH |
| VVIX (vol of vol) | yfinance | Free | 2007-present | API | `derivatives/vix_term_structure/vvix/` | HIGH |

### 10.5 Implied Volatility Surfaces

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Computed from options chains | Internal (Polygon-backed) | Free (computation cost only) | matches options | computation | `derivatives/implied_volatility_surfaces/computed/` | HIGH |

---

## 11. FUNDAMENTAL DATA

### 11.1 Earnings, Balance Sheet, Income Statement, Cash Flow, Ratios

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Financial Modeling Prep Starter | FMP | $14/mo | 30+ years for major stocks | API | `fundamental/fmp/` | HIGH |
| yfinance fundamentals (limited) | yfinance | Free | Recent quarters | API | `fundamental/yfinance/` | MEDIUM |
| SEC EDGAR XBRL | SEC | Free | 2009-present | API + parsing | `fundamental/edgar_xbrl/` | HIGH |

---

## 12. REFERENCE DATA

### 12.1 Calendars

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Trading calendars all major exchanges | exchange_calendars Python lib | Free | Comprehensive | API | `reference_data/calendars/exchange_calendars/` | HIGH |
| Economic calendar | TradingEconomics free / FXStreet | Free tier | Recent | API | `reference_data/calendars/economic_events/` | HIGH |

### 12.2 Holidays

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| Holidays per country | python-holidays | Free | Comprehensive | library | `reference_data/holidays/python_holidays/` | HIGH |

### 12.3 Index Constituents

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| S&P 500 historical constituents | Wikipedia + manual | Free | Variable | scraping | `reference_data/index_constituents/sp500_historical/` | HIGH |
| NASDAQ-100 historical | Wikipedia | Free | Variable | scraping | `reference_data/index_constituents/ndx_historical/` | MEDIUM |

### 12.4 Sector Classifications

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| GICS sector classifications | yfinance / FMP | Free / Subscription | Current | API | `reference_data/sector_classifications/gics/` | HIGH |

### 12.5 Symbol Mappings

| Source | Provider | Cost | Coverage | Method | Folder | Priority |
|--------|----------|------|----------|--------|--------|----------|
| ISIN/CUSIP/Ticker mappings | OpenFIGI (Bloomberg free) | Free | Comprehensive | API | `reference_data/symbol_mappings/openfigi/` | MEDIUM |

---

## 13. SUBSCRIPTION DECISION SUMMARY

Total monthly cost if all subscriptions taken: $30 + $30 + $39 + $14 + $79 + $100 = **$292/month**, well under $500 cap.

**Recommended subscriptions (all under $500/mo cap):**

| Service | Cost | Justification |
|---------|------|---------------|
| Glassnode Standard | $30/mo | Premier crypto on-chain metrics, no good free alternative |
| CryptoQuant Standard | $39/mo | Exchange flows, whale data, complementary to Glassnode |
| Polygon.io Developer | $79/mo | US equities intraday + options + news, single best paid source for stocks |
| FMP Starter | $14/mo | Fundamental data, 30+ years coverage |
| Twitter/X API Basic | $100/mo | Social sentiment, optional but valuable |

**Total: $262/mo** committed core subscriptions.

**Optional add-ons up to $500 cap:**
- Quandl/Nasdaq Data Link selected datasets (variable cost)
- Polygon.io upgrade to higher tier if needed
- TokenTerminal if DeFi metrics critical

**Excluded (cost-prohibitive vs benefit):**
- Bloomberg Terminal ($24K/yr) — no clear advantage over Polygon + free sources
- Refinitiv Eikon ($22K/yr) — same
- LOBSTER NASDAQ tick data — academic-priced but order book reconstruction overkill for our timeframes
- Bloomberg/Citi Economic Surprise Index — we'll compute equivalent from FRED data ourselves

---

## 14. Stage 1.2 Deliverable

File: `STAGE_1.2_DELIVERABLE.md`

Content:

```markdown
# Stage 1.2 Deliverable — Data Catalog

## Summary

- **Total data sources cataloged:** [N] sources across [M] categories
- **Free sources:** [N_free]
- **Paid subscriptions recommended:** [N_paid]
- **Total monthly subscription cost:** $[X]/month (within $500/mo cap)

## Catalog by Priority

- HIGH priority sources: [count]
- MEDIUM priority sources: [count]
- LOW priority sources: [count]
- EXCLUDED sources: [count] (with reasons documented)

## Storage Estimate

- Estimated total raw data size after acquisition: [X] GB / [Y] TB

## Categories Covered

[Checklist of all categories in document]

## Subscription Decisions Required from User

User must decide whether to subscribe to:
- Glassnode Standard ($30/mo)
- CryptoQuant Standard ($39/mo)
- Polygon.io Developer ($79/mo)
- FMP Starter ($14/mo)
- Twitter/X API Basic ($100/mo) — OPTIONAL

User responds via chat with approved subscription list.

## User Gate

Awaiting user approval of:
1. Catalog completeness (any missing sources to add?)
2. Subscription decisions (which paid sources to acquire?)
3. Approval to proceed to Stage 1.3 (Free Data) and Stage 1.4 (Registrations)
```

The catalog itself (this entire document content) is also written to `~/Documents/financial_data/_metadata/data_catalog.json` (machine-readable) and referenced from deliverable.

---

## 15. User Gate

User reviews catalog. Decides:
1. Any sources to add or remove
2. Which subscriptions to approve (within $500/mo cap)
3. Approves Stage 1.3 + 1.4 parallel start
