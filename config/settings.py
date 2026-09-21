"""
config/settings.py — Centralised configuration for the AI Stock Trader (V1).

RULE: This is the ONLY module that reads environment variables or the .env
file.  No other module may import os.environ, load_dotenv, or read secrets
directly.  All §1 locked values live here as defaults; they are overridable
via .env for testing but must never be changed without an explicit decision.

V1 is paper-trading only.  No real-money or broker settings are defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Internal helper ───────────────────────────────────────────────────────────

def _default_tickers() -> List[str]:
    """Return the default US ticker universe across all 8 sectors."""
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "VZ", "CMCSA",
        "AMZN", "TSLA", "HD", "JPM", "V", "BAC", "UNH", "JNJ",
        "LLY", "CAT", "GE", "UPS", "PG", "COST", "KO", "XOM", "CVX", "COP"
    ]


class Settings(BaseSettings):
    """
    Single source of truth for every tunable value in the system.

    Load order (later wins):
      1. Defaults defined below  ← the §1 locked values
      2. Variables from .env     ← local overrides (git-ignored)
      3. Real environment variables (highest priority)

    NOTE on tickers: pydantic-settings 2.x tries to JSON-decode List fields
    read from .env, which fails for a plain comma-separated string.  We
    therefore declare `tickers` as a plain `str` at the pydantic-settings
    layer and convert it to List[str] in `parse_tickers` before validation.
    Callers always receive List[str].
    """

    model_config = SettingsConfigDict(
        # Look for .env in the project root (one level above this file).
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        # Ignore any extra keys in .env that are not defined here.
        extra="ignore",
    )

    # ── Capital ───────────────────────────────────────────────────────────────
    initial_capital: float = Field(
        default=10000.0,
        gt=0,
        description="Starting paper-trading capital in USD.",
    )

    # ── Universe ──────────────────────────────────────────────────────────────
    # Declared as str so pydantic-settings does NOT attempt JSON-decode on the
    # comma-separated .env value.  parse_tickers converts it to List[str].
    tickers: str = Field(
        default="AAPL,MSFT,NVDA,GOOGL,META,VZ,CMCSA,AMZN,TSLA,HD,JPM,V,BAC,UNH,JNJ,LLY,CAT,GE,UPS,PG,COST,KO,XOM,CVX,COP",
        description=(
            "US ticker symbols to trade (standard format, no suffix). "
            "Comma-separated string in .env; accessed as a list via the "
            "ticker_list property."
        ),
    )

    benchmark: str = Field(
        default="SPY",
        description=(
            "Benchmark ticker.  Used for (1) the regime filter and "
            "(2) comparison in every backtest report."
        ),
    )

    # ── Prediction label (§1.3) ───────────────────────────────────────────────
    prediction_horizon_days: int = Field(
        default=5,
        gt=0,
        description="Forward-looking horizon in trading days for the label.",
    )
    win_threshold: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Minimum fractional return to label a move a 'win' "
            "(0.01 = +1%).  Set above round-trip cost so a predicted "
            "win is a profitable win, not noise."
        ),
    )

    # ── Entry / exit thresholds (§1.4) ────────────────────────────────────────
    buy_bar: float = Field(
        default=0.60,
        gt=0,
        lt=1,
        description="Minimum model probability to trigger a buy.",
    )
    signal_exit: float = Field(
        default=0.45,
        gt=0,
        lt=1,
        description=(
            "Exit (signal exit) when probability drops below this value. "
            "The dead zone [signal_exit, buy_bar) means: hold, no action."
        ),
    )
    stop_loss: float = Field(
        default=0.08,
        gt=0,
        description=(
            "Hard stop-loss: exit if a position falls this fraction from "
            "entry (0.08 = −8%).  Checked FIRST in exit priority."
        ),
    )
    take_profit: float = Field(
        default=0.15,
        gt=0,
        description=(
            "Hard take-profit: exit if a position gains this fraction from "
            "entry (0.15 = +15%).  Checked second in exit priority."
        ),
    )

    # ── Position / sizing rules (§1.4) ────────────────────────────────────────
    max_positions: int = Field(
        default=3,
        gt=0,
        description="Maximum number of concurrent open long positions.",
    )
    cash_reserve: float = Field(
        default=0.15,
        gt=0,
        lt=1,
        description=(
            "Fraction of total portfolio value always kept in cash "
            "(0.15 = 15%).  Dry powder + risk buffer."
        ),
    )
    min_trade_size: float = Field(
        default=10.0,
        gt=0,
        description="Minimum dollar value of any single simulated trade.",
    )

    # ── Costs (§1.6) ──────────────────────────────────────────────────────────
    simulated_cost_per_trade: float = Field(
        default=0.002,
        ge=0,
        description=(
            "Simulated cost applied to every fill as a fraction of trade "
            "value (0.002 = 0.2%).  Round-trip cost is therefore 0.4%. "
            "Always deducted — results are always NET."
        ),
    )

    # ── Regime filter (§1.8) ──────────────────────────────────────────────────
    regime_ma_window: int = Field(
        default=200,
        gt=0,
        description=(
            "Moving-average look-back window (trading days) for the regime "
            "filter.  When the benchmark closes below this MA: no new buys."
        ),
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    data_raw_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory where raw, unmodified market-data pulls are saved.",
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for cleaned / processed data.",
    )
    data_models_dir: Path = Field(
        default=Path("data/models"),
        description="Directory where trained model artefacts are saved.",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    db_url: str = Field(
        default="sqlite:///data/processed/trader.db",
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL", "db_url"),
        description=(
            "SQLAlchemy connection URL for the project database. "
            "Defaults to SQLite at data/processed/trader.db — no server needed. "
            "Upgrade to PostgreSQL by setting DATABASE_URL in .env: "
            "postgresql+psycopg2://user:pass@host/dbname."
        ),
    )

    # ── Feature engineering ───────────────────────────────────────────────────
    warmup_bars: int = Field(
        default=200,
        gt=0,
        description=(
            "Number of trading days of extra look-back history to fetch when "
            "a date-range pull is requested (start_date / end_date mode). "
            "Ensures that rolling-window features at the analysis start_date "
            "are not NaN. The longest feature window is 200 (ma_200), so the "
            "default of 200 is the minimum safe value. "
            "200 trading days ≈ 290 calendar days; fetch_ticker_data adds a "
            "15-day buffer, pulling ~305 calendar days before start_date."
        ),
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    # ── Automation / Scheduling (§1.9) ─────────────────────────────────────────
    schedule_hour: int = Field(
        default=9,
        ge=0,
        le=23,
        description="Hour (0-23) to run daily pipeline in schedule_timezone.",
    )
    schedule_minute: int = Field(
        default=0,
        ge=0,
        le=59,
        description="Minute (0-59) to run daily pipeline in schedule_timezone.",
    )
    schedule_timezone: str = Field(
        default="America/New_York",
        description="Timezone for the daily market schedule (e.g. America/New_York).",
    )

    # ── Market Hours & Data Acquisition ───────────────────────────────────────
    market_close_hour: int = Field(
        default=16,
        ge=0,
        le=23,
        description="Market close check hour in market_hours_timezone (16 = 4 PM).",
    )
    market_close_minute: int = Field(
        default=30,
        ge=0,
        le=59,
        description="Market close check minute in market_hours_timezone (30 min -> 4:30 PM).",
    )
    market_hours_timezone: str = Field(
        default="America/New_York",
        description="Timezone for market hours awareness (uses zoneinfo America/New_York).",
    )
    data_fetch_max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts on market data fetch failure.",
    )
    data_fetch_retry_wait_sec: float = Field(
        default=60.0,
        ge=0,
        description="Wait time in seconds between fetch retries.",
    )

    # ── Dynamic Rejection Thresholds (§1.7) ───────────────────────────────────
    dynamic_spike_std_multiplier: float = Field(
        default=10.0,
        gt=0,
        description="Multiplier on stock's historical average daily move for overnight price jump rejection.",
    )
    suspicious_price_jump_pct: float = Field(
        default=0.25,
        gt=0,
        description="Secondary price jump threshold (25%+) requiring volume spike validation.",
    )
    suspicious_volume_spike_threshold: float = Field(
        default=1.5,
        gt=0,
        description="Minimum volume ratio relative to 20-day mean required to validate a 25%+ price jump.",
    )

    # ── Section 2: Risk & Safety Rules ────────────────────────────────────────
    # Adaptive conviction thresholds (Item 5)
    vix_low_threshold: float = Field(
        default=15.0,
        gt=0,
        description="VIX / realized vol low threshold (%, e.g. 15.0 or 12% realized).",
    )
    vix_high_threshold: float = Field(
        default=25.0,
        gt=0,
        description="VIX / realized vol high threshold (%, e.g. 25.0 or 20% realized).",
    )
    adaptive_buy_bar_low: float = Field(
        default=0.60,
        description="Conviction buy bar in low volatility regime.",
    )
    adaptive_signal_exit_low: float = Field(
        default=0.45,
        description="Conviction signal exit in low volatility regime.",
    )
    adaptive_buy_bar_normal: float = Field(
        default=0.63,
        description="Conviction buy bar in normal volatility regime.",
    )
    adaptive_signal_exit_normal: float = Field(
        default=0.47,
        description="Conviction signal exit in normal volatility regime.",
    )
    adaptive_buy_bar_high: float = Field(
        default=0.67,
        description="Conviction buy bar in high volatility regime.",
    )
    adaptive_signal_exit_high: float = Field(
        default=0.50,
        description="Conviction signal exit in high volatility regime.",
    )

    # Minimum holding period (Item 6)
    min_holding_days: int = Field(
        default=5,
        ge=0,
        description="Minimum trading days a position must be held before signal exit is permitted.",
    )

    # Macro crash circuit breakers (Item 8)
    macro_cb_5d_drop_pct: float = Field(
        default=0.07,
        gt=0,
        description="SPY 5-day drop threshold for circuit breaker (0.07 = 7%).",
    )
    macro_cb_5d_halt_days: int = Field(
        default=5,
        ge=1,
        description="Number of trading days to halt new buys after a 5-day crash.",
    )
    macro_cb_20d_drop_pct: float = Field(
        default=0.15,
        gt=0,
        description="SPY 20-day drop threshold for circuit breaker (0.15 = 15%).",
    )
    macro_cb_20d_halt_days: int = Field(
        default=10,
        ge=1,
        description="Number of trading days to halt new buys after a 20-day crash.",
    )

    # PSI drift protocol (Item 9)
    psi_monitor_threshold: float = Field(
        default=0.10,
        ge=0,
        description="PSI threshold indicating moderate feature/prediction drift.",
    )
    psi_monitor_size_multiplier: float = Field(
        default=0.50,
        gt=0,
        le=1.0,
        description="Allocation multiplier when moderate PSI drift is detected (0.50 = cut in half).",
    )
    psi_drift_alert_threshold: float = Field(
        default=0.25,
        gt=0,
        description="PSI threshold indicating severe drift requiring buy signal pause.",
    )

    # ── Section 3: ML & Training Settings ─────────────────────────────────────
    training_history_start_date: str = Field(
        default="2008-01-01",
        description="Historical training window start date (January 2008 for crisis coverage).",
    )
    walk_forward_train_years: float = Field(
        default=2.0,
        gt=0,
        description="Walk-forward training window duration in years (default: 2.0).",
    )
    walk_forward_test_months: float = Field(
        default=6.0,
        gt=0,
        description="Walk-forward test window duration in months (default: 6.0).",
    )

    # ── Section 4: Operational & Slippage Settings (Item 15) ──────────────────
    slippage_tier_low_pct: float = Field(
        default=0.0005,
        ge=0,
        description="Slippage rate for <= 1% ADV participation (0.05%).",
    )
    slippage_tier_mid_pct: float = Field(
        default=0.0015,
        ge=0,
        description="Slippage rate for 1% - 5% ADV participation (0.15%).",
    )
    slippage_tier_high_pct: float = Field(
        default=0.0030,
        ge=0,
        description="Slippage rate for > 5% ADV participation (0.30%).",
    )

    # ── Section 5: Intelligence Upgrades Settings ─────────────────────────────
    # FRED API Key for Macro Economic Indicators
    fred_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FRED_API_KEY", "fred_api_key"),
        description="Federal Reserve Economic Data (FRED) API key.",
    )
    news_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("NEWS_API_KEY", "news_api_key"),
        description="Optional NewsAPI.org API key.",
    )

    # News Sentiment Analysis thresholds (Item 1)
    sentiment_veto_threshold: float = Field(
        default=-0.60,
        description="Sentiment score below which a candidate is vetoed entirely.",
    )
    sentiment_positive_threshold: float = Field(
        default=0.30,
        description="Sentiment score above which a +3% score boost is applied.",
    )
    sentiment_negative_threshold: float = Field(
        default=-0.30,
        description="Sentiment score below which a -5% score penalty is applied.",
    )
    sentiment_boost_pct: float = Field(
        default=0.03,
        description="Score boost percentage for positive sentiment (+3%).",
    )
    sentiment_penalty_pct: float = Field(
        default=0.05,
        description="Score penalty percentage for negative sentiment (-5%).",
    )

    # Earnings Calendar Awareness (Item 2)
    earnings_blackout_days: int = Field(
        default=2,
        ge=1,
        description="Days until earnings within which new buys are blocked (1-2 days).",
    )
    earnings_caution_min_days: int = Field(
        default=3,
        ge=1,
        description="Minimum days until earnings for cautious 50% position sizing.",
    )
    earnings_caution_max_days: int = Field(
        default=5,
        ge=1,
        description="Maximum days until earnings for cautious 50% position sizing.",
    )
    earnings_caution_size_multiplier: float = Field(
        default=0.50,
        gt=0,
        le=1.0,
        description="Position size multiplier when earnings are 3-5 days away (0.50).",
    )

    # Sector Rotation Intelligence (Item 3)
    sector_rotation_window: int = Field(
        default=20,
        ge=5,
        description="Return lookback window in trading days for sector rotation (20 days).",
    )
    sector_boost_pct: float = Field(
        default=0.04,
        description="Score modifier for top 2 sectors (+4%).",
    )
    sector_penalty_pct: float = Field(
        default=0.04,
        description="Score penalty for bottom 3 sectors (-4%).",
    )

    # Macro Economic Indicators (Item 4)
    macro_neutral_size_multiplier: float = Field(
        default=0.80,
        gt=0,
        le=1.0,
        description="Position size multiplier under NEUTRAL macro regime (0.80 = -20%).",
    )
    macro_restrictive_size_multiplier: float = Field(
        default=0.60,
        gt=0,
        le=1.0,
        description="Position size multiplier under RESTRICTIVE macro regime (0.60 = -40%).",
    )
    macro_restrictive_buy_bar_shift: float = Field(
        default=0.03,
        ge=0,
        description="Buy threshold increase under RESTRICTIVE macro regime (+0.03 = +3%).",
    )

    # ── Section 7: Notification & Alert Channels ───────────────────────────────
    alert_email_sender: str = Field(
        default="",
        validation_alias=AliasChoices("ALERT_EMAIL_SENDER", "alert_email_sender"),
        description="Gmail address sending pipeline email alerts via SMTP.",
    )
    alert_email_password: str = Field(
        default="",
        validation_alias=AliasChoices("ALERT_EMAIL_PASSWORD", "alert_email_password"),
        description="Google App Password for alert_email_sender.",
    )
    alert_email_recipient: str = Field(
        default="",
        validation_alias=AliasChoices("ALERT_EMAIL_RECIPIENT", "alert_email_recipient"),
        description="Recipient email address receiving pipeline alert notifications.",
    )
    alert_email_smtp_host: str = Field(
        default="smtp.gmail.com",
        description="SMTP server host for email alerts.",
    )
    alert_email_smtp_port: int = Field(
        default=587,
        description="SMTP server port for email alerts (STARTTLS, default 587).",
    )

    # ── Section 7 Item 2: Telegram Notifications ───────────────────────────────
    telegram_bot_token: str = Field(
        default="",
        validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "telegram_bot_token"),
        description="Telegram Bot token from @BotFather. Leave blank to disable Telegram alerts.",
    )
    telegram_chat_id: str = Field(
        default="",
        validation_alias=AliasChoices("TELEGRAM_CHAT_ID", "telegram_chat_id"),
        description="Telegram Chat/User ID from @userinfobot. Leave blank to disable Telegram alerts.",
    )

    # ── Section 8 Item 1: Multiple Data Sources ───────────────────────────────
    alpha_vantage_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ALPHA_VANTAGE_API_KEY", "alpha_vantage_api_key"),
        description="Free API key from alphavantage.co for backup market data.",
    )
    alpha_vantage_daily_limit: int = Field(
        default=25,
        description="Daily API call limit for Alpha Vantage free tier.",
    )

    # ── Section 8 Item 2: Position Sizing By Confidence ───────────────────────
    confidence_half_position_max: float = Field(
        default=0.65,
        description="Upper threshold for 50% half position sizing [0.60, 0.65).",
    )
    confidence_three_quarter_max: float = Field(
        default=0.75,
        description="Upper threshold for 75% three-quarter position sizing [0.65, 0.75).",
    )

    # ── Section 8 Item 3: Trailing Stop Loss ──────────────────────────────────
    trailing_stop_pct: float = Field(
        default=0.08,
        description="Trailing stop-loss percentage (default 0.08 = 8%).",
    )

    # ── Section 8b: Correlation Filter ───────────────────────────────────────
    correlation_lookback_days: int = Field(
        default=60,
        description="Lookback window in trading days for price return correlation.",
    )
    correlation_block_threshold: float = Field(
        default=0.85,
        description="Correlation threshold above which a buy is blocked (default 0.85).",
    )
    correlation_warn_threshold: float = Field(
        default=0.70,
        description="Correlation threshold above which a warning is logged (default 0.70).",
    )
    correlation_different_sector_threshold: float = Field(
        default=0.90,
        description="Block threshold when candidate and held stock are in different sectors (default 0.90).",
    )

    # ── Section 9 Item 1: Hyperparameter Auto-Tuning ─────────────────────────
    auto_tune_on_retrain: bool = Field(
        default=False,
        description="When True, run Optuna hyperparameter tuning before each retraining run.",
    )
    optuna_trials: int = Field(
        default=50,
        description="Number of Optuna trials per model during hyperparameter search.",
    )
    optuna_timeout_seconds: int = Field(
        default=600,
        description="Maximum wall-clock seconds per model during Optuna tuning (default 10 min).",
    )
    random_seed: int = Field(
        default=42,
        description="Global random seed for reproducibility of training and Optuna runs.",
    )

    # ── Section 9 Item 3: Automatic Retraining Schedule ──────────────────────
    auto_retrain_enabled: bool = Field(
        default=True,
        description="Master flag to enable automatic monthly model retraining.",
    )
    auto_retrain_day: str = Field(
        default="saturday",
        description="Day of week for monthly retraining job (default: saturday).",
    )
    auto_retrain_hour: int = Field(
        default=10,
        description="Hour of day (0-23) for monthly retraining job (default: 10 AM).",
    )
    auto_promote_min_improvement: float = Field(
        default=0.01,
        description="Minimum ROC-AUC improvement required to auto-promote a new model (default +1.0%).",
    )
    auto_retrain_keep_backup: bool = Field(
        default=True,
        description="When True, preserve a backup copy of previous active model before promotion.",
    )

    # ── Section 10 Item 1: Paper Trading Leaderboard ──────────────────────────
    leaderboard_enabled: bool = Field(
        default=True,
        description="Master flag to enable parallel strategy variant leaderboard tracking.",
    )
    leaderboard_strategies: int = Field(
        default=3,
        description="Number of default strategy variants to run concurrently.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("db_url", mode="before")
    @classmethod
    def normalise_db_url(cls, v: object) -> str:
        s = str(v).strip()
        if s.startswith("postgres://"):
            return "postgresql+psycopg2://" + s[len("postgres://"):]
        return s

    @field_validator("tickers", mode="before")
    @classmethod
    def normalise_tickers(cls, v: object) -> str:
        """
        Normalise the raw tickers value to an uppercase comma-separated string.
        Accepts a plain str (from .env) or a list (from Python / tests).
        """
        if isinstance(v, list):
            return ",".join(str(t).strip().upper() for t in v if str(t).strip())
        if isinstance(v, str):
            parts = [t.strip().upper() for t in v.split(",") if t.strip()]
            return ",".join(parts)
        raise ValueError(f"Cannot normalise tickers from {v!r}")

    @field_validator("benchmark", mode="before")
    @classmethod
    def upper_benchmark(cls, v: object) -> str:
        return str(v).strip().upper()

    @field_validator("buy_bar")
    @classmethod
    def buy_bar_above_signal_exit(cls, v: float) -> float:
        if v <= 0.45:
            raise ValueError("buy_bar must be > signal_exit (0.45) to create a dead zone.")
        return v

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key for chatbot.",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name for chatbot.",
    )
    suppress_duplicate_alerts: bool = Field(
        default=True,
        description="When True, deduplicates alerts so the same alert key is sent only once per day.",
    )
    email_alerts_enabled: bool = Field(
        default=False,
        description="Master toggle to enable or disable Gmail email alerts. Set EMAIL_ALERTS_ENABLED=true in .env to enable.",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def upper_log_level(cls, v: object) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        val = str(v).strip().upper()
        if val not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {val!r}")
        return val

    # ── Derived helpers (read-only properties) ────────────────────────────────

    @property
    def prediction_horizon(self) -> int:
        """Alias for prediction_horizon_days (Phase 4 label horizon)."""
        return self.prediction_horizon_days

    @property
    def ticker_list(self) -> List[str]:
        """Return tickers as a list.  Use this everywhere in the codebase."""
        return [t for t in self.tickers.split(",") if t]

    @property
    def round_trip_cost(self) -> float:
        """Total cost for a buy+sell round trip (both legs)."""
        return self.simulated_cost_per_trade * 2

    @property
    def investable_fraction(self) -> float:
        """Fraction of capital that may be deployed into positions."""
        return 1.0 - self.cash_reserve

    @property
    def position_weight(self) -> float:
        """Target weight per position assuming equal sizing."""
        return self.investable_fraction / self.max_positions


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this everywhere: `from config.settings import settings`
settings = Settings()


# ── Sector Mapping for 25-Stock Universe ──────────────────────────────────────
TICKER_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Communications",
    "META": "Communications",
    "VZ": "Communications",
    "CMCSA": "Communications",
    "AMZN": "Consumer Cyclical",
    "TSLA": "Consumer Cyclical",
    "HD": "Consumer Cyclical",
    "JPM": "Financials",
    "V": "Financials",
    "BAC": "Financials",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "LLY": "Healthcare",
    "CAT": "Industrials",
    "GE": "Industrials",
    "UPS": "Industrials",
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
}


def get_ticker_sector(ticker: str) -> str:
    """Return the assigned sector for a ticker, or 'Other' if unmapped."""
    return TICKER_SECTOR_MAP.get(str(ticker).strip().upper(), "Other")

