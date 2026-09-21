"""
src/screening/ — Long-Term (3-5+ Year) Fundamentals-Based Stock Screener.

CRITICAL ARCHITECTURAL ISOLATION GUARANTEE:
This package is completely decoupled from the automated 5-day trading system.
It contains NO dependencies on:
  - src.risk.risk_engine
  - src.portfolio.portfolio
  - src.trading.paper_broker
  - src.pipeline.scheduler

This is a rule-based, informational fundamental analysis checklist only.
"""
