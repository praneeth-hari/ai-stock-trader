"""
src/db/__init__.py — Exposes the repository's public API at the package level.
"""
from src.db.repository import (
    create_all_tables,
    get_engine,
    get_events,
    get_features,
    get_latest_portfolio_snapshot,
    get_market_data,
    get_orders,
    get_portfolio_snapshot,
    get_predictions,
    get_trades,
    get_walk_forward_history,
    log_event,
    save_features,
    save_market_data,
    save_order,
    save_portfolio_snapshot,
    save_predictions,
    save_trade,
    save_walk_forward_results,
)

__all__ = [
    "create_all_tables",
    "get_engine",
    "get_events",
    "get_features",
    "get_latest_portfolio_snapshot",
    "get_market_data",
    "get_orders",
    "get_portfolio_snapshot",
    "get_predictions",
    "get_trades",
    "get_walk_forward_history",
    "log_event",
    "save_features",
    "save_market_data",
    "save_order",
    "save_portfolio_snapshot",
    "save_predictions",
    "save_trade",
    "save_walk_forward_results",
]

