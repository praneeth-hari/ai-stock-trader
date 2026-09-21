# GitHub Secrets To Add

Go to your GitHub repo:
Settings → Secrets → Actions → New Secret

Add these secrets:

- `TELEGRAM_BOT_TOKEN` = your bot token
- `TELEGRAM_CHAT_ID` = your chat id
- `GEMINI_API_KEY` = your gemini key
- `ALPHA_VANTAGE_API_KEY` = your key
- `FRED_API_KEY` = your key (optional)
- `DATABASE_URL` = your supabase url

---

## Quick Step-by-Step Setup Guide

### 1. Get Free Cloud PostgreSQL on Supabase (No Credit Card)
1. Go to [supabase.com](https://supabase.com) and click **Start your project** (Sign up with GitHub).
2. Create a new organization and project (e.g. `ai-stock-trader`).
3. Choose a strong database password and keep note of it.
4. Once created, go to **Project Settings** → **Database**.
5. Under **Connection string**, select **URI** (or Transaction / Session Pooler).
6. Copy the connection string format:
   ```text
   postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
   ```
7. Replace `[YOUR-PASSWORD]` with your actual project database password.

### 2. Configure GitHub Repository Secrets
1. In your GitHub repository, navigate to **Settings** → **Secrets and variables** → **Actions**.
2. Click **New repository secret** for each of the following:

| Secret Name | Value Description | Mandatory? |
|---|---|---|
| `DATABASE_URL` | Supabase PostgreSQL URI from Step 1 | **Yes** (persists trade data across runs) |
| `TELEGRAM_BOT_TOKEN` | Your Telegram Bot token from @BotFather | Recommended |
| `TELEGRAM_CHAT_ID` | Your personal or group Telegram Chat ID | Recommended |
| `GEMINI_API_KEY` | Google Gemini API key for AI summaries/analyst | Optional |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key for market data fallback | Optional |
| `FRED_API_KEY` | Federal Reserve Economic Data API key | Optional |

### 3. Automated Workflows Running 100% Free in GitHub Actions
Once secrets are saved and code is pushed, the following workflows execute automatically in the cloud:

1. **Daily Trading Pipeline** (`.github/workflows/daily_trading.yml`):
   - **Trigger**: Every weekday Monday–Friday at 2:00 AM IST (8:30 PM UTC previous day).
   - **Manual Trigger**: Can also be manually run anytime via **Actions** → **Daily Trading Pipeline** → **Run workflow**.
   - **Operation**: Pulls data, generates signals, executes simulated paper orders, saves snapshots to Supabase, and dispatches Telegram notifications.

2. **Weekly Summary** (`.github/workflows/weekly_summary.yml`):
   - **Trigger**: Every Sunday at 9:00 AM IST (3:30 AM UTC).
   - **Operation**: Aggregates the past week's performance from Supabase and sends a portfolio digest.

3. **Monthly Model Retrain** (`.github/workflows/monthly_retrain.yml`):
   - **Trigger**: First Saturday of each month at 10:00 AM IST (4:30 AM UTC).
   - **Operation**: Evaluates model performance and retrains machine learning models.
