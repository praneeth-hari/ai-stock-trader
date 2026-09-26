"""
src/db/models.py — SQLAlchemy ORM table definitions for AI Stock Trader (V1).

All seven entities live here. No other file defines table schemas.
The repository module (repository.py) is the only caller of these models.

Table design notes:
- market_data, features, predictions: keyed by (date, ticker) pairs.
- features stores the entire feature vector as a JSON blob per (date, ticker)
  row. This avoids having to know all feature names at schema-definition time,
  and keeps the schema stable as features are added/removed in Phase 3.
- portfolio captures a full snapshot of cash + positions per run date.
- orders records every decision; trades records every simulated fill.
- event_log provides the structured audit trail required by CLAUDE.md rule 8.

Upgrade path to PostgreSQL:
  Change settings.db_url — no schema changes needed. SQLAlchemy handles the
  dialect difference. JSON columns map to JSONB on PostgreSQL automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MarketDataRow(Base):
    """
    Validated OHLCV market data — output of Phase 1 + Phase 2 pipeline.
    Only validated data is written here (never raw pulls).
    """
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)   # 'YYYY-MM-DD'
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    data_as_of: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_market_data_date_ticker"),
        Index("ix_market_data_ticker_date", "ticker", "date"),
    )

    def __repr__(self) -> str:
        return f"<MarketDataRow {self.ticker} {self.date} close={self.close}>"


class FeatureRow(Base):
    """
    Feature vector per (date, ticker), stored as a JSON blob.
    Keyed by (date, ticker); the 'features' column holds a dict of
    {feature_name: value} computed by Phase 3 feature engineering.
    """
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)  # {name: value}
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_features_date_ticker"),
        Index("ix_features_ticker_date", "ticker", "date"),
    )

    def __repr__(self) -> str:
        n = len(self.features) if self.features else 0
        return f"<FeatureRow {self.ticker} {self.date} ({n} features)>"


class PredictionRow(Base):
    """
    Model probability output: P(stock rises >+1% over next 5 trading days).
    One row per (date, ticker) per model version.
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "ticker", "model_version", name="uq_predictions_date_ticker_model"),
        Index("ix_predictions_ticker_date", "ticker", "date"),
    )

    def __repr__(self) -> str:
        return f"<PredictionRow {self.ticker} {self.date} p={self.probability:.4f}>"


class PortfolioSnapshot(Base):
    """
    Full portfolio state captured at the end of each daily pipeline run.
    'positions' is a JSON dict: {ticker: {quantity, avg_cost, current_price, value}}.
    """
    __tablename__ = "portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[str] = mapped_column(String(10), nullable=False)
    market: Mapped[str] = mapped_column(String(10), nullable=False, default="US")
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    total_value: Mapped[float] = mapped_column(Float, nullable=False)
    total_slippage_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    highest_price_since_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trailing_stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    positions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Orders queued for the next open, committed in the same row as cash/positions so a run's
    # effects are all-or-nothing. NULL only on snapshots written before this column existed.
    pending_orders: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("run_date", "market", name="uq_portfolio_run_date_market"),
    )

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot {self.market} {self.run_date} cash={self.cash:.2f} total={self.total_value:.2f}>"


class OrderRow(Base):
    """
    Every order decision made by the pipeline (buy/sell/hold).
    Records the intended action before it becomes a fill.
    'reason' documents which rule triggered the decision (§1.4 exit priority,
    regime filter, signal, etc.) for full audit trail.
    """
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[str] = mapped_column(String(10), nullable=False)
    market: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, default="US")
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)   # BUY / SELL / HOLD
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_orders_run_date_ticker", "run_date", "ticker"),
    )

    def __repr__(self) -> str:
        return f"<OrderRow {self.market} {self.run_date} {self.action} {self.ticker} qty={self.quantity:.4f}>"


class TradeRow(Base):
    """
    Every simulated fill executed by the paper broker.
    'cost' = 0.2% of fill_price * quantity (§1.6 always applied, results always NET).
    'net_pnl' = realized P&L after costs for sells; 0.0 for buys (open position).
    """
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[str] = mapped_column(String(10), nullable=False)
    market: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, default="US")
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)   # BUY / SELL
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)         # simulated trading cost
    slippage_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # market impact slippage (Item 15)
    net_pnl: Mapped[float] = mapped_column(Float, nullable=False)      # 0.0 for buys
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_trades_run_date_ticker", "run_date", "ticker"),
    )

    def __repr__(self) -> str:
        return (
            f"<TradeRow {self.market} {self.run_date} {self.action} {self.ticker} "
            f"qty={self.quantity:.4f} fill={self.fill_price:.2f} pnl={self.net_pnl:.4f}>"
        )


class DecisionLogRow(Base):
    """
    Observability only: one row per ranked ticker per trading cycle, explaining the decision.
    Written after the day is committed; never read by any trading decision.
    """
    __tablename__ = "decision_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[str] = mapped_column(String(10), nullable=False)
    market: Mapped[str] = mapped_column(String(10), nullable=False, default="US")
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # NULL: held but not ranked
    sector: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    sector_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sector_modifier: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    macro_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    regime_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    correlation: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    final_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    buy_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    size_tier: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    decision: Mapped[str] = mapped_column(String(10), nullable=False)      # BUY / HOLD / SELL / REJECT
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    order_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # why an actual order won its slot
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("run_date", "market", "ticker", name="uq_decision_log_run_market_ticker"),
    )


class BenchmarkSnapshotRow(Base):
    """
    Shadow benchmark portfolio (SPY held only while SPY >= its 200-day SMA), one row per trading
    cycle. Completely separate from the V1 portfolio table; never read by any trading decision.
    """
    __tablename__ = "benchmark_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark: Mapped[str] = mapped_column(String(30), nullable=False)
    run_date: Mapped[str] = mapped_column(String(10), nullable=False)
    decision_bar_date: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    spy_shares: Mapped[float] = mapped_column(Float, nullable=False)
    spy_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sma_200: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    regime_risk_on: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    decision: Mapped[str] = mapped_column(String(30), nullable=False)
    pending_action: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    daily_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cumulative_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("benchmark", "run_date", name="uq_benchmark_snapshot_run"),
    )


class EventLog(Base):
    """
    Structured audit log for every prediction, decision, skip, and error.
    Satisfies CLAUDE.md rule 8: 'Fail loud, log everything.'
    'component' identifies which engine emitted the event (e.g. 'validation',
    'risk_engine', 'paper_broker').
    'details' is a JSON blob for structured metadata (ticker, values, rule names).
    """
    __tablename__ = "event_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    level: Mapped[str] = mapped_column(String(10), nullable=False)    # DEBUG/INFO/WARNING/ERROR/CRITICAL
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_event_log_timestamp", "timestamp"),
        Index("ix_event_log_level", "level"),
    )

    def __repr__(self) -> str:
        return f"<EventLog [{self.level}] {self.component}: {self.message[:60]}>"


# ── Section 5: Intelligence Tables ────────────────────────────────────────────

class SentimentScoreRow(Base):
    """
    Headline sentiment scores per (date, ticker) scored via VADER (Item 1).
    """
    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)   # 'YYYY-MM-DD'
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    headline_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sentiment_label: Mapped[str] = mapped_column(String(20), nullable=False, default="NEUTRAL")
    data_source: Mapped[str] = mapped_column(String(50), nullable=False, default="YahooFinance")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_sentiment_scores_ticker_date", "ticker", "date"),
    )

    def __repr__(self) -> str:
        return f"<SentimentScoreRow {self.ticker} {self.date} score={self.composite_score:.2f} ({self.sentiment_label})>"


class EarningsCalendarRow(Base):
    """
    Upcoming corporate earnings announcement dates and countdowns (Item 2).
    """
    __tablename__ = "earnings_calendar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    earnings_date: Mapped[str] = mapped_column(String(30), nullable=False)
    days_until_earnings: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fetched_date: Mapped[str] = mapped_column(String(10), nullable=False)  # 'YYYY-MM-DD'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_earnings_calendar_ticker_fetched", "ticker", "fetched_date"),
    )

    def __repr__(self) -> str:
        return f"<EarningsCalendarRow {self.ticker} date={self.earnings_date} days={self.days_until_earnings}>"


class SectorRankingRow(Base):
    """
    Daily 20-day return ranking across all 8 market sectors (Item 3).
    """
    __tablename__ = "sector_rankings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)   # 'YYYY-MM-DD'
    sector: Mapped[str] = mapped_column(String(50), nullable=False)
    avg_20d_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)      # 1 to 8
    rotation_multiplier: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_sector_rankings_date_rank", "date", "rank"),
    )

    def __repr__(self) -> str:
        return f"<SectorRankingRow {self.date} {self.sector} rank={self.rank} ret={self.avg_20d_return:.4f}>"


class MacroIndicatorRow(Base):
    """
    Federal Reserve (FRED) macro indicators & regime classification (Item 4).
    """
    __tablename__ = "macro_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)   # 'YYYY-MM-DD'
    fed_funds_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpi_yoy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unemployment_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    treasury_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macro_regime: Mapped[str] = mapped_column(String(20), nullable=False, default="FAVORABLE")
    fetched_date: Mapped[str] = mapped_column(String(10), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_macro_indicators_date", "date"),
    )

    def __repr__(self) -> str:
        return f"<MacroIndicatorRow {self.date} regime={self.macro_regime} fed_funds={self.fed_funds_rate}>"


class CorrelationMatrixRow(Base):
    """
    Daily pairwise correlation matrix entries for watchlist stocks (Section 8b).
    """
    __tablename__ = "correlation_matrix"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    ticker_a: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker_b: Mapped[str] = mapped_column(String(10), nullable=False)
    correlation: Mapped[float] = mapped_column(Float, nullable=False)
    lookback_days: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_correlation_matrix_date_tickers", "date", "ticker_a", "ticker_b"),
    )

    def __repr__(self) -> str:
        return f"<CorrelationMatrixRow {self.date} {self.ticker_a}-{self.ticker_b} r={self.correlation:.4f}>"


class FeatureImportanceRow(Base):
    """
    Per-retrain feature importance scores for each model type (Section 9 Item 2).
    Tracks how each of the 19 features has changed in importance over the last N retrains.
    """
    __tablename__ = "feature_importance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(30), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(80), nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_feature_importance_date_model", "date", "model_type"),
    )

    def __repr__(self) -> str:
        return f"<FeatureImportanceRow {self.date} {self.model_type} {self.feature_name}={self.importance_score:.4f}>"


# ── Section 10 Item 1: Paper Trading Leaderboard ──────────────────────────────

class StrategyVariantRow(Base):
    """
    Parallel strategy variants running on paper money (Section 10 Item 1).
    Stores strategy name, settings_json, active status, starting capital.
    """
    __tablename__ = "strategy_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    settings_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_date: Mapped[str] = mapped_column(String(10), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    starting_capital: Mapped[float] = mapped_column(Float, default=10000.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<StrategyVariantRow {self.id}:{self.name}>"


class StrategySnapshotRow(Base):
    """
    Daily portfolio snapshots for each strategy variant (Section 10 Item 1).
    """
    __tablename__ = "strategy_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(Integer, ForeignKey("strategy_variants.id"), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    positions_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    daily_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_strategy_snapshot_id_date", "strategy_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<StrategySnapshotRow strat={self.strategy_id} {self.date} val={self.portfolio_value:.2f}>"


class StrategyTradeRow(Base):
    """
    Executed trades for each strategy variant (Section 10 Item 1).
    """
    __tablename__ = "strategy_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(Integer, ForeignKey("strategy_variants.id"), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY or SELL
    price: Mapped[float] = mapped_column(Float, nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<StrategyTradeRow strat={self.strategy_id} {self.date} {self.action} {self.ticker}>"


class WalkForwardResultRow(Base):
    """
    Stores fold-level walk-forward validation results across retrains.
    """
    __tablename__ = "walk_forward_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    fold_number: Mapped[int] = mapped_column(Integer, nullable=False)
    train_start: Mapped[str] = mapped_column(String(10), nullable=False)
    train_end: Mapped[str] = mapped_column(String(10), nullable=False)
    test_start: Mapped[str] = mapped_column(String(10), nullable=False)
    test_end: Mapped[str] = mapped_column(String(10), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=False)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_walk_forward_model_fold", "model_type", "fold_number"),
    )

    def __repr__(self) -> str:
        return f"<WalkForwardResultRow model={self.model_type} fold={self.fold_number} roc_auc={self.roc_auc:.4f}>"

