"""
tests/test_market_isolation_fixes.py — Focused tests for 4 targeted market-isolation cleanup areas.
"""

from datetime import date
import pandas as pd
import pytest

from config.settings import get_ticker_sector, settings
from src.chatbot.gemini_chat import build_live_context_strings
from src.db import repository
from src.reports.weekly_summary import generate_weekly_summary_data


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Patch settings.db_url to isolated SQLite file in tmp_path for each test."""
    db_path = tmp_path / "test_trader_isolation.db"
    db_url = f"sqlite:///{db_path}"
    original_url = settings.db_url
    settings.db_url = db_url
    repository._engine = None
    repository._db_available = True
    repository.create_all_tables()
    yield db_url
    settings.db_url = original_url
    repository._engine = None


def test_orders_and_trades_explicit_market_isolation():
    """Verify that orders and trades with explicit market context do not bleed across markets."""
    repository.create_all_tables()

    # Save orders with explicit market context
    o_us_id = repository.save_order("2026-09-25", "TEST_US", "BUY", 10.0, 100.0, "US test order", market="US")
    o_in_id = repository.save_order("2026-09-25", "TEST_IN", "BUY", 10.0, 100.0, "India test order", market="INDIA")

    # Save trades with explicit market context
    t_us_id = repository.save_trade("2026-09-25", "TEST_US", "BUY", 10.0, 100.0, 0.2, 0.0, market="US")
    t_in_id = repository.save_trade("2026-09-25", "TEST_IN", "BUY", 10.0, 100.0, 0.2, 0.0, market="INDIA")

    us_orders = repository.get_orders(market="US")
    india_orders = repository.get_orders(market="INDIA")

    us_order_ids = [o["id"] for o in us_orders]
    india_order_ids = [o["id"] for o in india_orders]

    assert o_us_id in us_order_ids
    assert o_us_id not in india_order_ids
    assert o_in_id in india_order_ids
    assert o_in_id not in us_order_ids

    us_trades = repository.get_trades(market="US")
    india_trades = repository.get_trades(market="INDIA")

    us_trade_ids = [t["id"] for t in us_trades]
    india_trade_ids = [t["id"] for t in india_trades]

    assert t_us_id in us_trade_ids
    assert t_us_id not in india_trade_ids
    assert t_in_id in india_trade_ids
    assert t_in_id not in us_trade_ids


def test_weekly_summary_dynamic_benchmark():
    """Verify weekly summary fetches benchmark dynamically using settings.get_benchmark(market)."""
    # Seed mock snapshot for US and INDIA so weekly summary runs
    repository.save_portfolio_snapshot("2026-09-25", 10000.0, 10000.0, {}, market="US")
    repository.save_portfolio_snapshot("2026-09-25", 10000.0, 10000.0, {}, market="INDIA")

    us_data = generate_weekly_summary_data(target_date=date(2026, 9, 25), market="US")
    india_data = generate_weekly_summary_data(target_date=date(2026, 9, 25), market="INDIA")

    assert us_data is not None
    assert india_data is not None

    assert us_data["benchmark"] == settings.get_benchmark("US")  # SPY
    assert india_data["benchmark"] == settings.get_benchmark("INDIA")  # ^NSEI


def test_chatbot_context_market_currency_symbols():
    """Verify chatbot live context uses market-appropriate currency symbols ($ vs ₹)."""
    us_ctx = build_live_context_strings(market="US")
    india_ctx = build_live_context_strings(market="INDIA")

    assert "$" in us_ctx["portfolio_data"]
    assert "₹" in india_ctx["portfolio_data"]


def test_indian_sector_mappings():
    """Verify Indian NSE tickers receive specific assigned sectors rather than 'Other'."""
    assert get_ticker_sector("RELIANCE.NS") == "Energy"
    assert get_ticker_sector("TCS.NS") == "Technology"
    assert get_ticker_sector("HDFCBANK.NS") == "Financials"
    assert get_ticker_sector("INFY.NS") == "Technology"
    assert get_ticker_sector("ICICIBANK.NS") == "Financials"
    assert get_ticker_sector("HINDUNILVR.NS") == "Consumer Staples"
    assert get_ticker_sector("SUNPHARMA.NS") == "Healthcare"
    assert get_ticker_sector("LT.NS") == "Industrials"
    assert get_ticker_sector("MARUTI.NS") == "Consumer Cyclical"
    assert get_ticker_sector("BHARTIARTL.NS") == "Communications"


def test_legacy_orders_trades_migration_classification():
    """Verify legacy records (where market column was absent or NULL) are classified by ticker suffix."""
    from sqlalchemy import text
    engine = repository.get_engine()

    # Simulate legacy table creation without 'market' column
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("DROP TABLE IF EXISTS trades"))
        conn.execute(text("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date VARCHAR(10) NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                quantity FLOAT NOT NULL,
                price FLOAT NOT NULL,
                reason TEXT NOT NULL,
                created_at DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date VARCHAR(10) NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                quantity FLOAT NOT NULL,
                fill_price FLOAT NOT NULL,
                cost FLOAT DEFAULT 0.0,
                slippage_cost FLOAT DEFAULT 0.0,
                net_pnl FLOAT DEFAULT 0.0,
                created_at DATETIME
            )
        """))
        # Insert legacy records without explicit market
        conn.execute(text("INSERT INTO orders (run_date, ticker, action, quantity, price, reason) VALUES ('2026-01-01', 'RELIANCE.NS', 'BUY', 10, 2500, 'Legacy India order')"))
        conn.execute(text("INSERT INTO orders (run_date, ticker, action, quantity, price, reason) VALUES ('2026-01-01', 'AAPL', 'BUY', 5, 180, 'Legacy US order')"))
        conn.execute(text("INSERT INTO trades (run_date, ticker, action, quantity, fill_price, slippage_cost, net_pnl) VALUES ('2026-01-01', 'RELIANCE.NS', 'BUY', 10, 2500, 5, 0)"))
        conn.execute(text("INSERT INTO trades (run_date, ticker, action, quantity, fill_price, slippage_cost, net_pnl) VALUES ('2026-01-01', 'AAPL', 'BUY', 5, 180, 1, 0)"))
        conn.commit()

    # Execute schema migration
    repository.create_all_tables()

    # Query with explicit market filters
    india_orders = repository.get_orders(market="INDIA")
    us_orders = repository.get_orders(market="US")
    india_trades = repository.get_trades(market="INDIA")
    us_trades = repository.get_trades(market="US")

    india_order_tickers = [o["ticker"] for o in india_orders]
    us_order_tickers = [o["ticker"] for o in us_orders]
    india_trade_tickers = [t["ticker"] for t in india_trades]
    us_trade_tickers = [t["ticker"] for t in us_trades]

    assert "RELIANCE.NS" in india_order_tickers
    assert "RELIANCE.NS" not in us_order_tickers
    assert "AAPL" in us_order_tickers
    assert "AAPL" not in india_order_tickers

    assert "RELIANCE.NS" in india_trade_tickers
    assert "RELIANCE.NS" not in us_trade_tickers
    assert "AAPL" in us_trade_tickers
    assert "AAPL" not in india_trade_tickers
