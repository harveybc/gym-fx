# Stage 1.4 — Registrations and Keys

**Stage goal:** User performs all manual subscription registrations and provides API keys / credentials. Agent waits for user inputs, then validates each credential works before Stage 1.5.

**Inputs:** Stage 1.2 complete with subscription decisions approved.

**Outputs:**
- All approved subscriptions active
- All API keys provided to agent via secure mechanism
- All credentials validated working
- `STAGE_1.4_DELIVERABLE.md` confirming readiness for Stage 1.5

**Machine:** None for user tasks. Omega for credential validation.

---

## 1. Stage 1.4 Procedure

This stage is mostly USER MANUAL WORK. Agent's job here is:
1. Generate REQUEST_USER document for each registration
2. Wait for user to complete and provide credentials
3. Validate each credential works
4. Document credential storage location (`.env` files)
5. Confirm Stage 1.5 can begin

---

## 2. Master REQUEST_USER document

Agent produces `REQUEST_USER_subscriptions_and_keys.md`:

```markdown
# REQUEST_USER: Subscriptions and API Keys for Project 3

Agent needs the following manual setup completed by user. Items can be done in any order. Reply to chat with credentials as each is completed.

## Required (per user-approved Stage 1.2 catalog):

### A. Glassnode Standard Subscription ($30/mo)
1. Visit https://glassnode.com/pricing
2. Subscribe to Standard tier ($30/month)
3. Generate API key in account settings
4. Reply to chat: "Glassnode key: <key>"

### B. CryptoQuant Standard Subscription ($39/mo)
1. Visit https://cryptoquant.com/products/data
2. Subscribe to Standard ($39/month)
3. Generate API key
4. Reply: "CryptoQuant key: <key>"

### C. Polygon.io Developer ($79/mo)
1. Visit https://polygon.io/pricing
2. Subscribe to Developer tier ($79/month)
3. Get API key from dashboard
4. Reply: "Polygon key: <key>"

### D. Financial Modeling Prep Starter ($14/mo)
1. Visit https://site.financialmodelingprep.com/developer/docs
2. Subscribe to Starter ($14/month)
3. Get API key
4. Reply: "FMP key: <key>"

### E. Twitter/X API Basic ($100/mo) [OPTIONAL]
If you opted in:
1. Visit https://developer.x.com/en/portal/products/basic
2. Subscribe to Basic ($100/month)
3. Generate Bearer Token
4. Reply: "Twitter Bearer Token: <token>"

## Free registrations needed:

### F. Etherscan API key (free)
1. Visit https://etherscan.io/myapikey
2. Register account, generate free API key (5 calls/sec free tier)
3. Reply: "Etherscan key: <key>"

### G. NewsAPI.org (free tier)
1. Visit https://newsapi.org/register
2. Free tier: 100 requests/day
3. Reply: "NewsAPI key: <key>"

### H. CoinMarketCap (free tier)
1. Visit https://pro.coinmarketcap.com/account
2. Free tier: 333 calls/day
3. Reply: "CoinMarketCap key: <key>"

### I. Alpha Vantage (free tier)
1. Visit https://www.alphavantage.co/support/#api-key
2. Free instant key
3. Reply: "AlphaVantage key: <key>"

### J. Quandl/Nasdaq Data Link (free tier)
1. Visit https://data.nasdaq.com/sign-up
2. Free tier sufficient for selected datasets
3. Reply: "Nasdaq Data Link key: <key>"

## Already have (verify):

### K. FRED API key
Already provided in Project 2: f7ad2f1190b5cfd083e5b4c8ee7e5140
Verify still valid by:
- Visiting https://fred.stlouisfed.org/docs/api/api_key.html
- Confirming key still listed in your account
Reply: "FRED key still valid: yes/no [if regenerated, new key]"

### L. OANDA Demo (Project 2 pending)
If you completed OANDA demo registration in Project 2:
Reply with: token, account_id, environment="practice"

If not yet completed:
1. Visit https://www.oanda.com/apply/?pt=demo
2. Apply for DEMO account (free, no funding)
3. Generate v20 API token, note Account ID
4. Reply: "OANDA token: <token>, account: <id>, env: practice"

### M. TrueFX (Project 2 already)
Already provided: harveybc / truefx.1
Verify login still works at https://www.truefx.com/
Reply: "TrueFX login verified: yes/no [or new credentials]"

## Status: WAITING

Reply each item as completed. Agent stores all in /home/harveybc/Documents/financial_data/_metadata/.env (NOT committed to git).
```

---

## 3. Credential Storage

Agent creates `~/Documents/financial_data/_metadata/.env`:

```bash
# Project 3 credentials — DO NOT COMMIT
# Auto-loaded by all acquisition scripts via python-dotenv

# Free APIs
export FRED_API_KEY="f7ad2f1190b5cfd083e5b4c8ee7e5140"
export ETHERSCAN_API_KEY=""  # to be filled
export NEWSAPI_KEY=""
export COINMARKETCAP_API_KEY=""
export ALPHAVANTAGE_API_KEY=""
export NASDAQ_DATA_LINK_KEY=""

# Paid subscriptions
export GLASSNODE_API_KEY=""
export CRYPTOQUANT_API_KEY=""
export POLYGON_API_KEY=""
export FMP_API_KEY=""
export TWITTER_BEARER_TOKEN=""  # if opted in

# Already have
export TRUEFX_USER="harveybc"
export TRUEFX_PASS="truefx.1"

# OANDA
export OANDA_TOKEN=""
export OANDA_ACCOUNT_ID=""
export OANDA_ENV="practice"
```

`.env` permissions: `chmod 600`

`.gitignore` includes `.env` at top of `~/Documents/financial_data/.gitignore`.

---

## 4. Credential Validation Procedures

After user provides each credential, agent validates by making a test API call:

### Glassnode validation

```python
import requests, os
key = os.environ['GLASSNODE_API_KEY']
r = requests.get("https://api.glassnode.com/v1/metrics/market/price_usd_close",
                 params={"a": "BTC", "api_key": key, "i": "24h", "s": 1577836800})
assert r.status_code == 200, f"Glassnode failed: {r.status_code}"
assert isinstance(r.json(), list)
```

### CryptoQuant validation

```python
key = os.environ['CRYPTOQUANT_API_KEY']
r = requests.get("https://api.cryptoquant.com/v1/btc/exchange-flows/inflow",
                 params={"window": "day"}, headers={"Authorization": f"Bearer {key}"})
assert r.status_code == 200
```

### Polygon validation

```python
key = os.environ['POLYGON_API_KEY']
r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05",
                 params={"apiKey": key})
assert r.status_code == 200
```

### FMP validation

```python
key = os.environ['FMP_API_KEY']
r = requests.get(f"https://financialmodelingprep.com/api/v3/profile/AAPL?apikey={key}")
assert r.status_code == 200
```

### Twitter validation

```python
import tweepy
client = tweepy.Client(bearer_token=os.environ['TWITTER_BEARER_TOKEN'])
# Just check authentication works
me = client.get_me()
```

### Etherscan, NewsAPI, CoinMarketCap, AlphaVantage, Nasdaq Data Link

Similar simple test calls per their API docs.

### OANDA validation

```python
from oandapyV20 import API
api = API(access_token=os.environ['OANDA_TOKEN'], environment=os.environ['OANDA_ENV'])
# Get account info
from oandapyV20.endpoints.accounts import AccountSummary
r = AccountSummary(accountID=os.environ['OANDA_ACCOUNT_ID'])
api.request(r)
assert r.response['account']['id'] == os.environ['OANDA_ACCOUNT_ID']
```

### TrueFX validation

```python
import requests
session = requests.Session()
r = session.post("https://www.truefx.com/dev/data/", auth=(os.environ['TRUEFX_USER'], os.environ['TRUEFX_PASS']))
# 200 means login worked
```

### FRED validation

```python
from fredapi import Fred
fred = Fred(api_key=os.environ['FRED_API_KEY'])
test = fred.get_series('CPIAUCSL', limit=5)
assert len(test) > 0
```

---

## 5. Validation results documentation

Per credential, agent records:

```json
{
  "service": "Glassnode",
  "credential_type": "API key",
  "validation_date": "YYYY-MM-DD",
  "test_call": "GET /v1/metrics/market/price_usd_close",
  "result": "PASS",
  "notes": "Standard tier confirmed active"
}
```

Stored at `~/Documents/financial_data/_metadata/credential_validation.json`.

---

## 6. Subscription Tracking

Update `~/Documents/financial_data/_metadata/subscriptions.json`:

```json
{
  "active_subscriptions": [
    {
      "service": "Glassnode",
      "tier": "Standard",
      "monthly_cost_usd": 30,
      "started_date": "2026-04-22",
      "billing_cycle": "monthly",
      "auto_renew": true,
      "purpose": "Crypto on-chain metrics",
      "credentials_path": "_metadata/.env"
    },
    ...
  ],
  "total_monthly_cost_usd": 162,
  "monthly_budget_cap_usd": 500,
  "budget_remaining_usd": 338,
  "last_updated": "2026-04-22"
}
```

---

## 7. Stage 1.4 Deliverable

`STAGE_1.4_DELIVERABLE.md`:

```markdown
# Stage 1.4 Deliverable — Registrations and Keys

## Summary

- Total credentials requested from user: 13
- Credentials received: [N]
- Credentials validated working: [N]
- Pending: [N] (list)

## Per-credential status

| Service | Type | Received | Validated | Notes |
|---------|------|----------|-----------|-------|
| FRED | Free API | YES | PASS | Existing from Project 2 |
| Glassnode | Paid $30/mo | YES/PENDING | PASS/FAIL | |
| CryptoQuant | Paid $39/mo | YES/PENDING | PASS/FAIL | |
| Polygon | Paid $79/mo | YES/PENDING | PASS/FAIL | |
| FMP | Paid $14/mo | YES/PENDING | PASS/FAIL | |
| Twitter (optional) | Paid $100/mo | OPTIONAL | | |
| Etherscan | Free | YES/PENDING | PASS/FAIL | |
| NewsAPI | Free | YES/PENDING | PASS/FAIL | |
| CoinMarketCap | Free | YES/PENDING | PASS/FAIL | |
| AlphaVantage | Free | YES/PENDING | PASS/FAIL | |
| Nasdaq Data Link | Free | YES/PENDING | PASS/FAIL | |
| OANDA | Free demo | YES/PENDING | PASS/FAIL | |
| TrueFX | Free | YES (existing) | PASS | |

## Total monthly subscription cost

$[X]/month (out of $500/month cap)

## Storage

All credentials stored at: `/home/harveybc/Documents/financial_data/_metadata/.env`
Permissions: 600
Listed in .gitignore: YES

## Pending items

[list of items still pending from user]

## User Gate

When all credentials received and validated, user approves moving to Stage 1.5 (Paid Data Acquisition).
```

---

## 8. User Gate

User confirms all credentials provided OR explicitly opts out of any optional ones (Twitter $100/mo). Agent confirms validation. User approves Stage 1.5.
