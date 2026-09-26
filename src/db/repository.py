"""
src/db/repository.py — Single source-of-truth database access layer (Phase 2b).

ARCHITECTURE RULE:
Every other module in the system reads and writes through THIS file only.
No raw SQL, no SQLAlchemy sessions, and no DB imports are allowed anywhere
else in the codebase. This thin layer is the only place that knows about
the database schema and connection.

UPGRADE PATH:
Defaults to SQLite (zero infrastructure, no server). Switching to PostgreSQL
requires only changing settings.db_url — no code changes here.

PUBLIC API SURFACE (one function per logical operation):

  Engine / schema
    get_engine()          → the SA engine singleton
    create_all_tables()   → create tables if they don't exist

  Market data (validated OHLCV only — never raw)
    save_market_data(df)
    get_market_data(ticker, start_date, end_date) → DataFrame

  Features
    save_features(df)               df must have columns: date, ticker, + feature cols
    get_features(ticker, start, end) → DataFrame

  Predictions
    save_predictions(df)            df must have: date, ticker, probability[, model_version]
    get_predictions(ticker, start, end) → DataFrame

  Portfolio
    save_portfolio_snapshot(run_date, cash, total_value, positions)
    get_portfolio_snapshot(run_date) → dict | None
    get_latest_portfolio_snapshot()  → dict | None

  Orders
    save_order(run_date, ticker, action, quantity, price, reason)
    get_orders(run_date) → list[dict]

  Trades
    save_trade(run_date, ticker, action, quantity, fill_price, cost, net_pnl)
    get_trades(run_date) → list[dict]

  Event log
    log_event(level, component, message, details)
    get_events(level, limit) → list[dict]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import (
    Base,
    CorrelationMatrixRow,
    EarningsCalendarRow,
    EventLog,
    FeatureImportanceRow,
    FeatureRow,
    MacroIndicatorRow,
    MarketDataRow,
    OrderRow,
    PortfolioSnapshot,
    PredictionRow,
    SectorRankingRow,
    SentimentScoreRow,
    StrategySnapshotRow,
    StrategyTradeRow,
    StrategyVariantRow,
    TradeRow,
    WalkForwardResultRow,
    _utcnow,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Safeguard: Prevent tests/demos from mutating production database ───────────

def _assert_safe_write_target() -> None:
    """
    Safeguard ensuring tests, demos, or automation scripts can NEVER write
    to the production database (trader.db) unless explicitly authorised.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PREVENT_PROD_DB_WRITE") == "1":
        clean_url = str(settings.db_url).replace("\\", "/")
        if "data/processed/trader.db" in clean_url and os.environ.get("ALLOW_PROD_TEST_WRITE") != "1":
            raise RuntimeError(
                f"PRODUCTION DATABASE WRITE BLOCKED: Execution context is a test or guarded demo "
                f"({os.environ.get('PYTEST_CURRENT_TEST', 'PREVENT_PROD_DB_WRITE=1')}), but settings.db_url points "
                f"to production database '{settings.db_url}'. Tests must patch settings.db_url to an isolated temporary database."
            )

# ── Engine singleton ───────────────────────────────────────────────────────────

_engine = None

# ── Stateless-mode guard ───────────────────────────────────────────────────────
# When the database is unreachable (e.g. GitHub Actions IPv6/Supabase issue),
# this flag is flipped to False by _probe_db_connection(). All repository
# functions then silently no-op / return empty values so the pipeline can
# continue without a DB: fetch data → run AI → make decisions → Telegram alerts.
_db_available: bool = True


def _probe_db_connection() -> bool:
    """
    Attempt a lightweight connection to confirm the database is reachable.
    Returns True if the DB is available, False otherwise.
    On failure, sets the module-level _db_available = False and logs a warning.
    """
    global _db_available
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        _db_available = True
        return True
    except Exception as exc:
        _db_available = False
        logger.warning(
            "DATABASE_UNAVAILABLE: Could not connect to database (%s). "
            "Running in stateless mode — data will NOT be persisted.",
            exc,
        )
        # Propagate to settings so callers can inspect
        try:
            settings.stateless_mode = True
        except Exception:
            pass
        return False



def get_engine(db_url: Optional[str] = None):
    """
    Return the SQLAlchemy engine singleton (or an engine for an explicit db_url).

    The engine is created once and reused. Connection URL comes from settings.db_url
    (or DATABASE_URL) by default.
    - For SQLite: check_same_thread=False is enabled.
    - For PostgreSQL: connection pre-ping and connection pooling are enabled.
    """
    global _engine
    target_url = db_url or settings.db_url
    if db_url is not None:
        if target_url.startswith("sqlite"):
            return create_engine(target_url, connect_args={"check_same_thread": False}, echo=False)
        return create_engine(target_url, pool_pre_ping=True, pool_size=5, max_overflow=10, echo=False)

    if _engine is None:
        if target_url.startswith("sqlite"):
            _engine = create_engine(
                target_url,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            _engine = create_engine(
                target_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                echo=False,
            )
        logger.info("Database engine created: %s", target_url)
    return _engine


def create_all_tables() -> None:
    """
    Create all tables defined in models.py if they do not already exist.
    Safe to call on every startup — idempotent. Also migrates missing columns.
    If the database is unreachable, activates stateless mode and returns without crashing.
    """
    global _db_available
    # First check connectivity; bail out gracefully if unreachable
    if not _probe_db_connection():
        return

    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)

        # Lightweight schema migration for newly added columns
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        with engine.connect() as conn:
            if "market_data" in table_names:
                cols = {c["name"] for c in inspector.get_columns("market_data")}
                if "data_as_of" not in cols:
                    conn.execute(text("ALTER TABLE market_data ADD COLUMN data_as_of VARCHAR(30)"))
                    conn.commit()
                    logger.info("Migrated schema: added market_data.data_as_of")

            if "portfolio" in table_names:
                cols = {c["name"] for c in inspector.get_columns("portfolio")}
                if "market" not in cols:
                    conn.execute(text("ALTER TABLE portfolio ADD COLUMN market VARCHAR(10) DEFAULT 'US'"))
                    conn.commit()
                    logger.info("Migrated schema: added portfolio.market")

                # Check if SQLite table has legacy UNIQUE (run_date) constraint
                if "sqlite" in str(engine.url):
                    try:
                        res = conn.execute(text("SELECT sql FROM sqlite_master WHERE type='table' AND name='portfolio'")).scalar_one_or_none()
                        if res and "UNIQUE (run_date)" in res and "UNIQUE (run_date, market)" not in res:
                            logger.info("Migrating SQLite portfolio table to composite UNIQUE (run_date, market)...")
                            conn.execute(text("""
                                CREATE TABLE portfolio_new (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    run_date VARCHAR(10) NOT NULL,
                                    market VARCHAR(10) DEFAULT 'US',
                                    cash FLOAT NOT NULL,
                                    total_value FLOAT NOT NULL,
                                    total_slippage_cost FLOAT DEFAULT 0.0,
                                    highest_price_since_entry FLOAT,
                                    trailing_stop_price FLOAT,
                                    positions JSON,
                                    created_at DATETIME,
                                    CONSTRAINT uq_portfolio_snapshot_run_date_market UNIQUE (run_date, market)
                                )
                            """))
                            conn.execute(text("""
                                INSERT INTO portfolio_new (id, run_date, market, cash, total_value, total_slippage_cost, highest_price_since_entry, trailing_stop_price, positions, created_at)
                                SELECT id, run_date, COALESCE(market, 'US'), cash, total_value, COALESCE(total_slippage_cost, 0.0), highest_price_since_entry, trailing_stop_price, positions, created_at FROM portfolio
                            """))
                            conn.execute(text("DROP TABLE portfolio"))
                            conn.execute(text("ALTER TABLE portfolio_new RENAME TO portfolio"))
                            conn.commit()
                            logger.info("SQLite portfolio table migration complete.")
                    except Exception as _mig_exc:
                        logger.warning("SQLite portfolio table migration notice: %s", _mig_exc)

                cols = {c["name"] for c in inspector.get_columns("portfolio")}
                if "total_slippage_cost" not in cols:
                    conn.execute(text("ALTER TABLE portfolio ADD COLUMN total_slippage_cost FLOAT DEFAULT 0.0"))
                    conn.commit()
                    logger.info("Migrated schema: added portfolio.total_slippage_cost")
                if "highest_price_since_entry" not in cols:
                    conn.execute(text("ALTER TABLE portfolio ADD COLUMN highest_price_since_entry FLOAT"))
                    conn.commit()
                    logger.info("Migrated schema: added portfolio.highest_price_since_entry")
                if "trailing_stop_price" not in cols:
                    conn.execute(text("ALTER TABLE portfolio ADD COLUMN trailing_stop_price FLOAT"))
                    conn.commit()
                    logger.info("Migrated schema: added portfolio.trailing_stop_price")

            if "orders" in table_names:
                cols = {c["name"] for c in inspector.get_columns("orders")}
                if "market" not in cols:
                    conn.execute(text("ALTER TABLE orders ADD COLUMN market VARCHAR(10)"))
                    conn.commit()
                    logger.info("Migrated schema: added orders.market")
                conn.execute(text("UPDATE orders SET market = 'INDIA' WHERE (market IS NULL OR market = 'US') AND (ticker LIKE '%.NS%' OR ticker LIKE '%.BO%')"))
                conn.execute(text("UPDATE orders SET market = 'US' WHERE market IS NULL AND NOT (ticker LIKE '%.NS%' OR ticker LIKE '%.BO%')"))
                conn.commit()

            if "trades" in table_names:
                cols = {c["name"] for c in inspector.get_columns("trades")}
                if "market" not in cols:
                    conn.execute(text("ALTER TABLE trades ADD COLUMN market VARCHAR(10)"))
                    conn.commit()
                    logger.info("Migrated schema: added trades.market")
                if "slippage_cost" not in cols:
                    conn.execute(text("ALTER TABLE trades ADD COLUMN slippage_cost FLOAT DEFAULT 0.0"))
                    conn.commit()
                    logger.info("Migrated schema: added trades.slippage_cost")
                conn.execute(text("UPDATE trades SET market = 'INDIA' WHERE (market IS NULL OR market = 'US') AND (ticker LIKE '%.NS%' OR ticker LIKE '%.BO%')"))
                conn.execute(text("UPDATE trades SET market = 'US' WHERE market IS NULL AND NOT (ticker LIKE '%.NS%' OR ticker LIKE '%.BO%')"))
                conn.commit()

        logger.info("All database tables created/verified.")
    except Exception as exc:
        _db_available = False
        logger.warning(
            "DATABASE_UNAVAILABLE: Table creation failed (%s). Running in stateless mode.", exc
        )
        try:
            settings.stateless_mode = True
        except Exception:
            pass


# ── Market data ────────────────────────────────────────────────────────────────

def save_market_data(df: pd.DataFrame) -> int:
    """
    Upsert validated OHLCV rows into the market_data table.

    Skips rows that already exist for the same (date, ticker) pair rather
    than raising. Returns the number of rows inserted.
    In stateless mode (DB unavailable), silently returns 0.

    Args:
        df: Validated DataFrame with columns [date, open, high, low, close, volume, ticker].

    Returns:
        Count of rows inserted (existing rows are skipped).
    """
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if df.empty:
        return 0

    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"save_market_data: DataFrame missing columns {missing}")

    inserted = 0
    with Session(get_engine()) as session:
        for _, row in df.iterrows():
            # Check for existing row to avoid unique-constraint errors.
            existing = session.execute(
                select(MarketDataRow).where(
                    MarketDataRow.date == row["date"],
                    MarketDataRow.ticker == row["ticker"],
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(MarketDataRow(
                    date=str(row["date"]),
                    ticker=str(row["ticker"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    data_as_of=str(row["data_as_of"]) if ("data_as_of" in row and pd.notna(row["data_as_of"])) else datetime.now(timezone.utc).isoformat(),
                ))
                inserted += 1

        session.commit()

    logger.info("save_market_data: inserted %d rows (skipped existing).", inserted)
    return inserted


def get_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """
    Retrieve validated OHLCV rows for a ticker, optionally filtered by date range.
    In stateless mode (DB unavailable), returns an empty DataFrame.
    """
    if not _db_available:
        cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        if include_metadata:
            cols.append("data_as_of")
        return pd.DataFrame(columns=cols)
    with Session(get_engine()) as session:
        stmt = select(MarketDataRow).where(MarketDataRow.ticker == ticker.upper())
        if start_date:
            stmt = stmt.where(MarketDataRow.date >= start_date)
        if end_date:
            stmt = stmt.where(MarketDataRow.date <= end_date)
        stmt = stmt.order_by(MarketDataRow.date)
        rows = session.execute(stmt).scalars().all()

    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if include_metadata:
        cols.append("data_as_of")

    if not rows:
        return pd.DataFrame(columns=cols)

    data = []
    for r in rows:
        row_dict = {
            "date": r.date, "ticker": r.ticker,
            "open": r.open, "high": r.high, "low": r.low,
            "close": r.close, "volume": r.volume,
        }
        if include_metadata:
            row_dict["data_as_of"] = r.data_as_of
        data.append(row_dict)

    return pd.DataFrame(data)


# ── Features ──────────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame) -> int:
    """
    Upsert feature vectors into the features table.
    In stateless mode (DB unavailable), silently returns 0.
    """
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if df.empty:
        return 0

    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("save_features: DataFrame must have 'date' and 'ticker' columns.")

    feature_cols = [c for c in df.columns if c not in ("date", "ticker")]
    upserted = 0

    with Session(get_engine()) as session:
        for _, row in df.iterrows():
            feature_dict = {c: row[c] for c in feature_cols}
            existing = session.execute(
                select(FeatureRow).where(
                    FeatureRow.date == row["date"],
                    FeatureRow.ticker == row["ticker"],
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(FeatureRow(
                    date=str(row["date"]),
                    ticker=str(row["ticker"]),
                    features=feature_dict,
                ))
            else:
                existing.features = feature_dict
            upserted += 1

        session.commit()

    logger.info("save_features: upserted %d rows.", upserted)
    return upserted


def get_features(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve feature vectors for a ticker. Returns empty DataFrame in stateless mode."""
    if not _db_available:
        return pd.DataFrame(columns=["date", "ticker"])
    with Session(get_engine()) as session:
        stmt = select(FeatureRow).where(FeatureRow.ticker == ticker.upper())
        if start_date:
            stmt = stmt.where(FeatureRow.date >= start_date)
        if end_date:
            stmt = stmt.where(FeatureRow.date <= end_date)
        stmt = stmt.order_by(FeatureRow.date)
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return pd.DataFrame(columns=["date", "ticker"])

    records = []
    for r in rows:
        record = {"date": r.date, "ticker": r.ticker}
        record.update(r.features or {})
        records.append(record)

    return pd.DataFrame(records)


# ── Predictions ────────────────────────────────────────────────────────────────

def save_predictions(df: pd.DataFrame) -> int:
    """
    Upsert model probability predictions into the predictions table.
    In stateless mode (DB unavailable), silently returns 0.
    """
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if df.empty:
        return 0

    required = ["date", "ticker", "probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"save_predictions: DataFrame missing columns {missing}")

    upserted = 0
    with Session(get_engine()) as session:
        for _, row in df.iterrows():
            model_version = str(row.get("model_version", "v1"))
            existing = session.execute(
                select(PredictionRow).where(
                    PredictionRow.date == row["date"],
                    PredictionRow.ticker == row["ticker"],
                    PredictionRow.model_version == model_version,
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(PredictionRow(
                    date=str(row["date"]),
                    ticker=str(row["ticker"]),
                    probability=float(row["probability"]),
                    model_version=model_version,
                ))
            else:
                existing.probability = float(row["probability"])
            upserted += 1

        session.commit()

    logger.info("save_predictions: upserted %d rows.", upserted)
    return upserted


def get_predictions(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Returns predictions for a ticker as a DataFrame. Returns empty in stateless mode."""
    if not _db_available:
        return pd.DataFrame(columns=["date", "ticker", "probability", "model_version"])
    with Session(get_engine()) as session:
        stmt = select(PredictionRow).where(PredictionRow.ticker == ticker.upper())
        if start_date:
            stmt = stmt.where(PredictionRow.date >= start_date)
        if end_date:
            stmt = stmt.where(PredictionRow.date <= end_date)
        stmt = stmt.order_by(PredictionRow.date)
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "probability", "model_version"])

    return pd.DataFrame([{
        "date": r.date, "ticker": r.ticker,
        "probability": r.probability, "model_version": r.model_version,
    } for r in rows])


# ── Portfolio ─────────────────────────────────────────────────────────────────

def save_portfolio_snapshot(
    run_date: str,
    cash: float,
    total_value: float,
    positions: Optional[Dict[str, Any]] = None,
    total_slippage_cost: float = 0.0,
    market: str = "US",
) -> None:
    """Upsert a portfolio snapshot tagged by market ('US' or 'INDIA'). Silently skips in stateless mode."""
    if not _db_available:
        return
    _assert_safe_write_target()
    market_clean = str(market).strip().upper()
    with Session(get_engine()) as session:
        existing = session.execute(
            select(PortfolioSnapshot).where(
                PortfolioSnapshot.run_date == run_date,
                PortfolioSnapshot.market == market_clean,
            )
        ).scalar_one_or_none()

        if existing is None:
            session.add(PortfolioSnapshot(
                run_date=run_date,
                market=market_clean,
                cash=cash,
                total_value=total_value,
                total_slippage_cost=float(total_slippage_cost),
                positions=positions or {},
            ))
        else:
            existing.cash = cash
            existing.total_value = total_value
            existing.total_slippage_cost = float(total_slippage_cost)
            existing.positions = positions or {}

        session.commit()

    logger.info("save_portfolio_snapshot [%s]: %s cash=%.2f total=%.2f slippage=%.4f.", market_clean, run_date, cash, total_value, total_slippage_cost)


def get_portfolio_snapshot(run_date: str, market: str = "US") -> Optional[Dict[str, Any]]:
    """Returns the portfolio snapshot for a specific run date and market, or None."""
    if not _db_available:
        return None
    market_clean = str(market).strip().upper()
    with Session(get_engine()) as session:
        row = session.execute(
            select(PortfolioSnapshot).where(
                PortfolioSnapshot.run_date == run_date,
                PortfolioSnapshot.market == market_clean,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "run_date": row.run_date,
            "market": getattr(row, "market", market_clean),
            "cash": row.cash,
            "total_value": row.total_value,
            "total_slippage_cost": getattr(row, "total_slippage_cost", 0.0) or 0.0,
            "positions": row.positions,
        }


def get_latest_portfolio_snapshot(market: str = "US") -> Optional[Dict[str, Any]]:
    """Returns the most recent portfolio snapshot for the given market ('US' or 'INDIA')."""
    if not _db_available:
        return None
    market_clean = str(market).strip().upper()
    with Session(get_engine()) as session:
        row = session.execute(
            select(PortfolioSnapshot).where(PortfolioSnapshot.market == market_clean).order_by(PortfolioSnapshot.run_date.desc()).limit(1)
        ).scalar_one_or_none()
        if row is None and market_clean == "US":
            # Fallback for un-tagged legacy rows
            row = session.execute(
                select(PortfolioSnapshot).where(PortfolioSnapshot.market.is_(None)).order_by(PortfolioSnapshot.run_date.desc()).limit(1)
            ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "run_date": row.run_date,
            "market": getattr(row, "market", "US"),
            "cash": row.cash,
            "total_value": row.total_value,
            "total_slippage_cost": getattr(row, "total_slippage_cost", 0.0) or 0.0,
            "positions": row.positions,
        }


def get_portfolio_snapshots(limit: int = 500, market: str = "US") -> List[Dict[str, Any]]:
    """Returns portfolio snapshots ordered chronologically for the specified market."""
    if not _db_available:
        return []
    market_clean = str(market).strip().upper()
    with Session(get_engine()) as session:
        rows = session.execute(
            select(PortfolioSnapshot).where(PortfolioSnapshot.market == market_clean).order_by(PortfolioSnapshot.run_date.asc()).limit(limit)
        ).scalars().all()
        if not rows and market_clean == "US":
            rows = session.execute(
                select(PortfolioSnapshot).order_by(PortfolioSnapshot.run_date.asc()).limit(limit)
            ).scalars().all()
        return [{
            "run_date": r.run_date,
            "market": getattr(r, "market", "US"),
            "cash": r.cash,
            "total_value": r.total_value,
            "total_slippage_cost": getattr(r, "total_slippage_cost", 0.0) or 0.0,
            "positions": r.positions,
        } for r in rows]


# ── Orders ────────────────────────────────────────────────────────────────────

def save_order(
    run_date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    reason: str,
    market: str = "US",
) -> int:
    """Append an order decision record with explicit market context. Returns 0 in stateless mode (DB unavailable)."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    market_clean = str(market).strip().upper()
    with Session(get_engine()) as session:
        row = OrderRow(
            run_date=run_date,
            market=market_clean,
            ticker=ticker.upper(),
            action=action.upper(),
            quantity=quantity,
            price=price,
            reason=reason,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        row_id = row.id

    logger.info("save_order [%s]: %s %s %s qty=%.4f reason=%s", market_clean, run_date, action, ticker, quantity, reason[:80])
    return row_id


def get_orders(run_date: Optional[str] = None, limit: int = 100, market: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns orders for a given run date and market context. Returns empty list in stateless mode."""
    if not _db_available:
        return []
    from sqlalchemy import and_, or_
    with Session(get_engine()) as session:
        stmt = select(OrderRow)
        if market:
            market_clean = str(market).strip().upper()
            if market_clean == "INDIA":
                stmt = stmt.where(
                    or_(
                        OrderRow.market == "INDIA",
                        and_(
                            OrderRow.market.is_(None),
                            OrderRow.ticker.like("%.NS%")
                        )
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        OrderRow.market == market_clean,
                        and_(
                            OrderRow.market.is_(None),
                            ~OrderRow.ticker.like("%.NS%")
                        )
                    )
                )
        if run_date is not None:
            stmt = stmt.where(OrderRow.run_date == run_date).order_by(OrderRow.id)
        else:
            stmt = stmt.order_by(OrderRow.id.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [{
            "id": r.id, "run_date": r.run_date, "market": getattr(r, "market", "US") or "US", "ticker": r.ticker,
            "action": r.action, "quantity": r.quantity,
            "price": r.price, "reason": r.reason,
        } for r in rows]


# ── Trades ────────────────────────────────────────────────────────────────────

def save_trade(
    run_date: str,
    ticker: str,
    action: str,
    quantity: float,
    fill_price: float,
    cost: float,
    net_pnl: float,
    slippage_cost: float = 0.0,
    market: str = "US",
) -> int:
    """Record a simulated paper-broker fill with explicit market context. Returns 0 in stateless mode (DB unavailable)."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    market_clean = str(market).strip().upper()
    with Session(get_engine()) as session:
        row = TradeRow(
            run_date=run_date,
            market=market_clean,
            ticker=ticker.upper(),
            action=action.upper(),
            quantity=quantity,
            fill_price=fill_price,
            cost=cost,
            net_pnl=net_pnl,
            slippage_cost=slippage_cost,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        row_id = row.id

    logger.info(
        "save_trade [%s]: %s %s %s fill=%.2f cost=%.4f pnl=%.4f slippage=%.4f",
        market_clean, run_date, action, ticker, fill_price, cost, net_pnl, slippage_cost,
    )
    return row_id


def get_trades(run_date: Optional[str] = None, limit: int = 100, market: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns trades for a given run date and market context. Returns empty list in stateless mode."""
    if not _db_available:
        return []
    from sqlalchemy import and_, or_
    with Session(get_engine()) as session:
        stmt = select(TradeRow)
        if market:
            market_clean = str(market).strip().upper()
            if market_clean == "INDIA":
                stmt = stmt.where(
                    or_(
                        TradeRow.market == "INDIA",
                        and_(
                            TradeRow.market.is_(None),
                            TradeRow.ticker.like("%.NS%")
                        )
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        TradeRow.market == market_clean,
                        and_(
                            TradeRow.market.is_(None),
                            ~TradeRow.ticker.like("%.NS%")
                        )
                    )
                )
        if run_date is not None:
            stmt = stmt.where(TradeRow.run_date == run_date).order_by(TradeRow.id)
        else:
            stmt = stmt.order_by(TradeRow.id.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [{
            "id": r.id, "run_date": r.run_date, "market": getattr(r, "market", "US") or "US", "ticker": r.ticker,
            "action": r.action, "quantity": r.quantity,
            "fill_price": r.fill_price, "cost": r.cost,
            "net_pnl": r.net_pnl, "slippage_cost": getattr(r, "slippage_cost", 0.0) or 0.0,
        } for r in rows]





# ── Event log ─────────────────────────────────────────────────────────────────

def log_event(
    level: str,
    component: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a structured event to the audit log table.
    In stateless mode (DB unavailable), falls back to Python logger only — never crashes.
    """
    if not _db_available:
        # Degrade gracefully: emit to Python logger instead of crashing
        logger.info("[STATELESS EVENT] [%s] %s: %s", level.upper(), component, message)
        return
    try:
        with Session(get_engine()) as session:
            session.add(EventLog(
                level=level.upper(),
                component=component,
                message=message,
                details=details,
            ))
            session.commit()
    except Exception as exc:
        logger.warning("log_event: DB write failed (%s) — logging to stderr only.", exc)
        logger.info("[FALLBACK EVENT] [%s] %s: %s", level.upper(), component, message)


def get_events(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Retrieve recent event log entries, newest first. Returns empty list in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        stmt = select(EventLog).order_by(EventLog.timestamp.desc()).limit(limit)
        if level:
            stmt = stmt.where(EventLog.level == level.upper())
        if component:
            stmt = stmt.where(EventLog.component == component)
        rows = session.execute(stmt).scalars().all()
        return [{
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "level": r.level,
            "component": r.component,
            "message": r.message,
            "details": r.details,
        } for r in rows]


# ── Section 5: Intelligence Persistence ───────────────────────────────────────

def save_sentiment_scores(scores: List[Dict[str, Any]]) -> int:
    """Save headline sentiment scores per (ticker, date). Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if not scores:
        return 0
    with Session(get_engine()) as session:
        count = 0
        for s in scores:
            session.add(SentimentScoreRow(
                date=str(s["date"]),
                ticker=str(s["ticker"]).upper(),
                headline_count=int(s.get("headline_count", 0)),
                composite_score=float(s.get("composite_score", 0.0)),
                sentiment_label=str(s.get("sentiment_label", "NEUTRAL")),
                data_source=str(s.get("data_source", "YahooFinance")),
            ))
            count += 1
        session.commit()
        return count


def get_latest_sentiment_scores(as_of_date: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Retrieve latest sentiment scores for each ticker. Returns {} in stateless mode."""
    if not _db_available:
        return {}
    with Session(get_engine()) as session:
        stmt = select(SentimentScoreRow).order_by(SentimentScoreRow.date.desc(), SentimentScoreRow.id.desc())
        if as_of_date:
            stmt = stmt.where(SentimentScoreRow.date <= as_of_date)
        rows = session.execute(stmt).scalars().all()
        result: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if r.ticker not in result:
                result[r.ticker] = {
                    "date": r.date,
                    "ticker": r.ticker,
                    "headline_count": r.headline_count,
                    "composite_score": r.composite_score,
                    "sentiment_label": r.sentiment_label,
                    "data_source": r.data_source,
                }
        return result


def save_earnings_calendar(events: List[Dict[str, Any]]) -> int:
    """Save upcoming earnings dates per ticker. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if not events:
        return 0
    with Session(get_engine()) as session:
        count = 0
        for e in events:
            session.add(EarningsCalendarRow(
                ticker=str(e["ticker"]).upper(),
                earnings_date=str(e["earnings_date"]),
                days_until_earnings=int(e["days_until_earnings"]) if e.get("days_until_earnings") is not None else None,
                fetched_date=str(e["fetched_date"]),
            ))
            count += 1
        session.commit()
        return count


def get_latest_earnings_calendar(as_of_date: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Retrieve latest earnings record per ticker. Returns {} in stateless mode."""
    if not _db_available:
        return {}
    with Session(get_engine()) as session:
        stmt = select(EarningsCalendarRow).order_by(EarningsCalendarRow.fetched_date.desc(), EarningsCalendarRow.id.desc())
        if as_of_date:
            stmt = stmt.where(EarningsCalendarRow.fetched_date <= as_of_date)
        rows = session.execute(stmt).scalars().all()
        result: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if r.ticker not in result:
                result[r.ticker] = {
                    "ticker": r.ticker,
                    "earnings_date": r.earnings_date,
                    "days_until_earnings": r.days_until_earnings,
                    "fetched_date": r.fetched_date,
                }
        return result


def save_sector_rankings(rankings: List[Dict[str, Any]]) -> int:
    """Save daily 20-day return ranking for all sectors. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if not rankings:
        return 0
    with Session(get_engine()) as session:
        count = 0
        for r in rankings:
            session.add(SectorRankingRow(
                date=str(r["date"]),
                sector=str(r["sector"]),
                avg_20d_return=float(r.get("avg_20d_return", 0.0)),
                rank=int(r["rank"]),
                rotation_multiplier=float(r.get("rotation_multiplier", 0.0)),
            ))
            count += 1
        session.commit()
        return count


def get_latest_sector_rankings(as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve the latest sector rankings. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        stmt = select(SectorRankingRow).order_by(SectorRankingRow.date.desc(), SectorRankingRow.rank.asc())
        if as_of_date:
            stmt = stmt.where(SectorRankingRow.date <= as_of_date)
        rows = session.execute(stmt).scalars().all()
        if not rows:
            return []
        latest_date = rows[0].date
        return [
            {
                "date": r.date,
                "sector": r.sector,
                "avg_20d_return": r.avg_20d_return,
                "rank": r.rank,
                "rotation_multiplier": r.rotation_multiplier,
            }
            for r in rows if r.date == latest_date
        ]


def save_macro_indicators(ind: Dict[str, Any]) -> None:
    """Save macroeconomic indicators & regime score. No-op in stateless mode."""
    if not _db_available:
        return
    _assert_safe_write_target()
    with Session(get_engine()) as session:
        session.add(MacroIndicatorRow(
            date=str(ind["date"]),
            fed_funds_rate=float(ind["fed_funds_rate"]) if ind.get("fed_funds_rate") is not None else None,
            cpi_yoy=float(ind["cpi_yoy"]) if ind.get("cpi_yoy") is not None else None,
            unemployment_rate=float(ind["unemployment_rate"]) if ind.get("unemployment_rate") is not None else None,
            treasury_10y=float(ind["treasury_10y"]) if ind.get("treasury_10y") is not None else None,
            macro_regime=str(ind.get("macro_regime", "FAVORABLE")),
            fetched_date=str(ind.get("fetched_date", ind["date"])),
        ))
        session.commit()


def get_latest_macro_indicators(as_of_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the latest macroeconomic indicator snapshot. Returns None in stateless mode."""
    if not _db_available:
        return None
    with Session(get_engine()) as session:
        stmt = select(MacroIndicatorRow).order_by(MacroIndicatorRow.date.desc(), MacroIndicatorRow.id.desc())
        if as_of_date:
            stmt = stmt.where(MacroIndicatorRow.date <= as_of_date)
        row = session.execute(stmt).scalars().first()
        if not row:
            return None
        return {
            "date": row.date,
            "fed_funds_rate": row.fed_funds_rate,
            "cpi_yoy": row.cpi_yoy,
            "unemployment_rate": row.unemployment_rate,
            "treasury_10y": row.treasury_10y,
            "macro_regime": row.macro_regime,
            "fetched_date": row.fetched_date,
        }


def save_correlation_matrix(records: List[Dict[str, Any]]) -> int:
    """Saves pairwise correlation matrix entries. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if not records:
        return 0

    inserted = 0
    with Session(get_engine()) as session:
        for rec in records:
            dt = str(rec["date"])
            t_a = str(rec["ticker_a"]).upper()
            t_b = str(rec["ticker_b"]).upper()
            existing = session.execute(
                select(CorrelationMatrixRow).where(
                    CorrelationMatrixRow.date == dt,
                    CorrelationMatrixRow.ticker_a == t_a,
                    CorrelationMatrixRow.ticker_b == t_b,
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(CorrelationMatrixRow(
                    date=dt,
                    ticker_a=t_a,
                    ticker_b=t_b,
                    correlation=float(rec["correlation"]),
                    lookback_days=int(rec.get("lookback_days", 60)),
                ))
                inserted += 1
            else:
                existing.correlation = float(rec["correlation"])
                existing.lookback_days = int(rec.get("lookback_days", 60))

        session.commit()
    logger.info("save_correlation_matrix: saved %d correlation records.", len(records))
    return inserted


def get_correlation_matrix(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves pairwise correlation matrix entries. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        if date_str is None:
            latest_row = session.execute(
                select(CorrelationMatrixRow).order_by(CorrelationMatrixRow.date.desc())
            ).scalars().first()
            if not latest_row:
                return []
            date_str = latest_row.date

        rows = session.execute(
            select(CorrelationMatrixRow).where(CorrelationMatrixRow.date == date_str)
        ).scalars().all()

        return [{
            "date": r.date,
            "ticker_a": r.ticker_a,
            "ticker_b": r.ticker_b,
            "correlation": r.correlation,
            "lookback_days": r.lookback_days,
        } for r in rows]


# ── Feature Importance (Section 9 Item 2) ─────────────────────────────────────

def save_feature_importances(date_str: str, model_type: str, importances: Dict[str, float]) -> int:
    """Upsert feature importance scores. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    if not importances:
        return 0

    inserted = 0
    with Session(get_engine()) as session:
        for feat, score in importances.items():
            existing = session.execute(
                select(FeatureImportanceRow).where(
                    FeatureImportanceRow.date == date_str,
                    FeatureImportanceRow.model_type == model_type,
                    FeatureImportanceRow.feature_name == feat,
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(FeatureImportanceRow(
                    date=date_str,
                    model_type=model_type,
                    feature_name=feat,
                    importance_score=float(score),
                ))
                inserted += 1
            else:
                existing.importance_score = float(score)

        session.commit()

    logger.info(
        "save_feature_importances: saved %d importance records for model_type=%s date=%s.",
        len(importances), model_type, date_str,
    )
    return inserted


def get_feature_importances(model_type: str, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load feature importance scores. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        q = select(FeatureImportanceRow).where(FeatureImportanceRow.model_type == model_type)
        if date_str:
            q = q.where(FeatureImportanceRow.date == date_str)
        else:
            # Find the latest date
            latest = session.execute(
                select(FeatureImportanceRow.date)
                .where(FeatureImportanceRow.model_type == model_type)
                .order_by(FeatureImportanceRow.date.desc())
                .limit(1)
            ).scalar_one_or_none()
            if latest is None:
                return []
            q = q.where(FeatureImportanceRow.date == latest)

        rows = session.execute(q.order_by(FeatureImportanceRow.importance_score.desc())).scalars().all()
        return [
            {
                "date": r.date,
                "model_type": r.model_type,
                "feature_name": r.feature_name,
                "importance_score": r.importance_score,
            }
            for r in rows
        ]


def get_feature_importance_history(
    model_type: str,
    top_n_features: int = 5,
    last_n_retrains: int = 5,
) -> List[Dict[str, Any]]:
    """Return feature importance time-series. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        # Get distinct dates for this model_type, most recent first
        dates_q = (
            select(FeatureImportanceRow.date)
            .where(FeatureImportanceRow.model_type == model_type)
            .distinct()
            .order_by(FeatureImportanceRow.date.desc())
            .limit(last_n_retrains)
        )
        date_rows = session.execute(dates_q).scalars().all()
        if not date_rows:
            return []

        # Find top-N features by importance on the latest date
        latest_date = date_rows[0]
        top_feats_q = (
            select(FeatureImportanceRow.feature_name)
            .where(
                FeatureImportanceRow.model_type == model_type,
                FeatureImportanceRow.date == latest_date,
            )
            .order_by(FeatureImportanceRow.importance_score.desc())
            .limit(top_n_features)
        )
        top_features = session.execute(top_feats_q).scalars().all()
        if not top_features:
            return []

        # Get history for those features across all selected dates
        history_q = select(FeatureImportanceRow).where(
            FeatureImportanceRow.model_type == model_type,
            FeatureImportanceRow.date.in_(date_rows),
            FeatureImportanceRow.feature_name.in_(top_features),
        ).order_by(FeatureImportanceRow.date.asc(), FeatureImportanceRow.importance_score.desc())
        rows = session.execute(history_q).scalars().all()

        return [
            {
                "date": r.date,
                "model_type": r.model_type,
                "feature_name": r.feature_name,
                "importance_score": r.importance_score,
            }
            for r in rows
        ]


# ── Section 10 Item 1: Paper Trading Leaderboard Repository CRUD ──────────────

def save_strategy_variant(
    name: str,
    settings_dict: Dict[str, Any],
    starting_capital: float = 10000.0,
    created_date: Optional[str] = None,
) -> int:
    """Save or return existing strategy variant row by name. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    today_str = created_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with Session(get_engine()) as session:
        stmt = select(StrategyVariantRow).where(StrategyVariantRow.name == name)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing:
            return int(existing.id)

        row = StrategyVariantRow(
            name=name,
            settings_json=json.dumps(settings_dict),
            created_date=today_str,
            is_active=True,
            starting_capital=starting_capital,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return int(row.id)


def get_strategy_variants(only_active: bool = True) -> List[Dict[str, Any]]:
    """Return all saved strategy variants. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        stmt = select(StrategyVariantRow)
        if only_active:
            stmt = stmt.where(StrategyVariantRow.is_active == True)
        stmt = stmt.order_by(StrategyVariantRow.id.asc())
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "settings": json.loads(r.settings_json),
                "created_date": r.created_date,
                "is_active": r.is_active,
                "starting_capital": r.starting_capital,
            }
            for r in rows
        ]


def save_strategy_snapshot(
    strategy_id: int,
    date_str: str,
    portfolio_value: float,
    cash: float,
    positions: Dict[str, Any],
    daily_return: float = 0.0,
) -> int:
    """Save a daily snapshot for a strategy variant. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    with Session(get_engine()) as session:
        row = StrategySnapshotRow(
            strategy_id=strategy_id,
            date=date_str,
            portfolio_value=portfolio_value,
            cash=cash,
            positions_json=json.dumps(positions),
            daily_return=daily_return,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return int(row.id)


def get_strategy_snapshots(strategy_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Query portfolio snapshots for a strategy. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        stmt = select(StrategySnapshotRow)
        if strategy_id is not None:
            stmt = stmt.where(StrategySnapshotRow.strategy_id == strategy_id)
        stmt = stmt.order_by(StrategySnapshotRow.date.asc(), StrategySnapshotRow.strategy_id.asc())
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "strategy_id": r.strategy_id,
                "date": r.date,
                "portfolio_value": r.portfolio_value,
                "cash": r.cash,
                "positions": json.loads(r.positions_json or "{}"),
                "daily_return": r.daily_return,
            }
            for r in rows
        ]


def save_strategy_trade(
    strategy_id: int,
    date_str: str,
    ticker: str,
    action: str,
    price: float,
    shares: float,
    pnl: float = 0.0,
) -> int:
    """Record an executed trade for a strategy variant. Returns 0 in stateless mode."""
    if not _db_available:
        return 0
    _assert_safe_write_target()
    with Session(get_engine()) as session:
        row = StrategyTradeRow(
            strategy_id=strategy_id,
            date=date_str,
            ticker=ticker,
            action=action,
            price=price,
            shares=shares,
            pnl=pnl,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return int(row.id)


def get_strategy_trades(strategy_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Query executed trades for a strategy. Returns [] in stateless mode."""
    if not _db_available:
        return []
    with Session(get_engine()) as session:
        stmt = select(StrategyTradeRow)
        if strategy_id is not None:
            stmt = stmt.where(StrategyTradeRow.strategy_id == strategy_id)
        stmt = stmt.order_by(StrategyTradeRow.date.asc(), StrategyTradeRow.id.asc())
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "strategy_id": r.strategy_id,
                "date": r.date,
                "ticker": r.ticker,
                "action": r.action,
                "price": r.price,
                "shares": r.shares,
                "pnl": r.pnl,
            }
            for r in rows
        ]


# ── Walk-Forward Validation Results ───────────────────────────────────────────

def save_walk_forward_results(results: List[Dict[str, Any]]) -> int:
    """
    Save walk-forward fold validation metrics to walk_forward_results table.
    Silently no-ops in stateless mode (DB unavailable).
    """
    if not _db_available or not results:
        return 0
    _assert_safe_write_target()

    saved = 0
    now_dt = datetime.now(timezone.utc)
    with Session(get_engine()) as session:
        for r in results:
            row = WalkForwardResultRow(
                trained_at=now_dt,
                model_type=str(r.get("model_type", "primary")),
                fold_number=int(r.get("fold_number", r.get("fold", 0))),
                train_start=str(r.get("train_start", "")),
                train_end=str(r.get("train_end", "")),
                test_start=str(r.get("test_start", "")),
                test_end=str(r.get("test_end", "")),
                accuracy=float(r.get("accuracy", 0.0)),
                roc_auc=float(r.get("roc_auc", 0.0)),
                brier_score=float(r.get("brier_score", 0.0)),
                n_samples=int(r.get("n_samples", r.get("test_rows", 0))),
            )
            session.add(row)
            saved += 1
        session.commit()

    logger.info("save_walk_forward_results: saved %d fold records.", saved)
    return saved


def save_kill_switch_state(enabled: bool, reason: str = "") -> None:
    """Persists kill switch state to DB so it survives restarts."""
    if not _db_available:
        return
    try:
        with Session(get_engine()) as session:
            existing = session.query(EventLog).filter(
                EventLog.component == "KILL_SWITCH_STATE"
            ).first()
            if existing:
                existing.message = f"{enabled}|{reason}"
                existing.occurred_at = _utcnow()
            else:
                session.add(EventLog(
                    component="KILL_SWITCH_STATE",
                    message=f"{enabled}|{reason}",
                    level="INFO",
                ))
            session.commit()
    except Exception as exc:
        logger.warning("Failed to persist kill switch state: %s", exc)


def load_kill_switch_state() -> tuple[bool, str]:
    """Loads persisted kill switch state from DB."""
    if not _db_available:
        return False, ""
    try:
        with Session(get_engine()) as session:
            row = session.query(EventLog).filter(
                EventLog.component == "KILL_SWITCH_STATE"
            ).first()
            if row and row.message:
                parts = row.message.split("|", 1)
                enabled = parts[0].lower() == "true"
                reason = parts[1] if len(parts) > 1 else ""
                return enabled, reason
    except Exception as exc:
        logger.warning("Failed to load kill switch state: %s", exc)
    return False, ""


def get_walk_forward_history(model_type: Optional[str] = None) -> pd.DataFrame:
    """
    Return walk-forward validation history as a pandas DataFrame.
    Returns empty DataFrame in stateless mode or when no records exist.
    """
    cols = [
        "id", "trained_at", "model_type", "fold_number",
        "train_start", "train_end", "test_start", "test_end",
        "accuracy", "roc_auc", "brier_score", "n_samples",
    ]
    if not _db_available:
        return pd.DataFrame(columns=cols)

    with Session(get_engine()) as session:
        stmt = select(WalkForwardResultRow)
        if model_type:
            stmt = stmt.where(WalkForwardResultRow.model_type == model_type)
        stmt = stmt.order_by(WalkForwardResultRow.trained_at.desc(), WalkForwardResultRow.fold_number.asc())
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame([{
        "id": r.id,
        "trained_at": r.trained_at.strftime("%Y-%m-%d %H:%M:%S") if r.trained_at else "",
        "model_type": r.model_type,
        "fold_number": r.fold_number,
        "train_start": r.train_start,
        "train_end": r.train_end,
        "test_start": r.test_start,
        "test_end": r.test_end,
        "accuracy": r.accuracy,
        "roc_auc": r.roc_auc,
        "brier_score": r.brier_score,
        "n_samples": r.n_samples,
    } for r in rows])


def save_india_portfolio(cash: float, positions: dict) -> None:
    """Saves Indian portfolio state to DB as an event log entry."""
    if not _db_available:
        return
    try:
        import json
        with Session(get_engine()) as session:
            existing = session.query(EventLog).filter(
                EventLog.component == "INDIA_PORTFOLIO"
            ).first()
            data = json.dumps({"cash": cash, "positions": positions})
            if existing:
                existing.message = data
                existing.occurred_at = _utcnow()
            else:
                session.add(EventLog(
                    component="INDIA_PORTFOLIO",
                    message=data,
                    level="INFO",
                ))
            session.commit()
    except Exception as exc:
        logger.warning("Failed to save India portfolio: %s", exc)


def load_india_portfolio() -> tuple:
    """Loads Indian portfolio state from DB."""
    if not _db_available:
        return None, {}
    try:
        import json
        with Session(get_engine()) as session:
            row = session.query(EventLog).filter(
                EventLog.component == "INDIA_PORTFOLIO"
            ).first()
            if row and row.message:
                data = json.loads(row.message)
                return data.get("cash"), data.get("positions", {})
    except Exception as exc:
        logger.warning("Failed to load India portfolio: %s", exc)
    return None, {}

