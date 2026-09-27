"""
tests/pipeline/test_us_market_only.py — V1 is US-only: market context resolves to US and
any other market is rejected loudly.
"""

from datetime import date
import pytest

from config.settings import settings
from src.chatbot.gemini_chat import build_live_context_strings
from src.db import repository
from src.pipeline.daily_pipeline import _resolve_market
from src.reports.weekly_summary import generate_weekly_summary_data
from src.trading.paper_broker import PaperBroker


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


def test_orders_and_trades_us_market_filter():
    """Orders and trades saved for the US market are returned by the US market filter."""
    o_us_id = repository.save_order("2026-09-25", "AAPL", "BUY", 10.0, 100.0, "US test order", market="US")
    t_us_id = repository.save_trade("2026-09-25", "AAPL", "BUY", 10.0, 100.0, 0.2, 0.0, market="US")

    assert o_us_id in [o["id"] for o in repository.get_orders(market="US")]
    assert t_us_id in [t["id"] for t in repository.get_trades(market="US")]


def test_weekly_summary_uses_spy_benchmark():
    repository.save_portfolio_snapshot("2026-09-25", 10000.0, 10000.0, {}, market="US")
    us_data = generate_weekly_summary_data(target_date=date(2026, 9, 25), market="US")

    assert us_data is not None
    assert us_data["benchmark"] == settings.get_benchmark("US") == "SPY"


def test_chatbot_context_uses_dollar_symbol():
    us_ctx = build_live_context_strings(market="US")
    assert "$" in us_ctx["portfolio_data"]


def test_market_helpers_return_us_values():
    assert settings.get_universe("US") == settings.ticker_list
    assert settings.get_benchmark("US") == "SPY"
    assert settings.get_initial_capital("US") == float(settings.initial_capital)
    assert settings.get_currency_symbol("US") == "$"
    assert settings.get_benchmark(" us ") == "SPY"


@pytest.mark.parametrize("helper", ["get_universe", "get_benchmark", "get_initial_capital", "get_currency_symbol"])
def test_market_helpers_reject_non_us_market(helper):
    with pytest.raises(ValueError, match="US market only"):
        getattr(settings, helper)("INDIA")


def test_pipeline_market_resolution_is_us_only():
    assert _resolve_market(None, None) == "US"
    assert _resolve_market("us", ["AAPL"]) == "US"
    with pytest.raises(ValueError, match="US market only"):
        _resolve_market("INDIA", None)


def test_paper_broker_rejects_non_us_market():
    with pytest.raises(ValueError, match="US market only"):
        PaperBroker(market="INDIA")


def test_legacy_orders_trades_migration_tags_us():
    """Legacy records without a market column are tagged US by the schema migration."""
    from sqlalchemy import text
    engine = repository.get_engine()

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
        conn.execute(text("INSERT INTO orders (run_date, ticker, action, quantity, price, reason) VALUES ('2026-01-01', 'AAPL', 'BUY', 5, 180, 'Legacy US order')"))
        conn.execute(text("INSERT INTO trades (run_date, ticker, action, quantity, fill_price, slippage_cost, net_pnl) VALUES ('2026-01-01', 'AAPL', 'BUY', 5, 180, 1, 0)"))
        conn.commit()

    repository.create_all_tables()

    us_orders = repository.get_orders(market="US")
    us_trades = repository.get_trades(market="US")
    assert [(o["ticker"], o["market"]) for o in us_orders] == [("AAPL", "US")]
    assert [(t["ticker"], t.get("market", "US")) for t in us_trades] == [("AAPL", "US")]
