"""
src/db/cloud_db.py — Cloud PostgreSQL (Supabase) Database Connector.

Enables serverless execution on GitHub Actions by persisting state across runs
via Supabase PostgreSQL (free tier).

Features:
  - Connects to Supabase PostgreSQL using the DATABASE_URL environment variable.
  - Automatically normalises postgres:// / postgresql:// dialects to postgresql+psycopg2://.
  - Seamlessly falls back to local SQLite when DATABASE_URL is not set.
  - Retains zero-regression compatibility: repository.py remains the single source of truth.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config.settings import settings

logger = logging.getLogger(__name__)


def normalize_postgres_url(url: str) -> str:
    """
    Ensure SQLAlchemy uses psycopg2 driver for PostgreSQL connections.
    Handles legacy 'postgres://' (common in Heroku / Supabase connection strings)
    and standard 'postgresql://' by prepending the psycopg2 dialect.
    """
    clean = url.strip()
    if clean.startswith("postgres://"):
        return clean.replace("postgres://", "postgresql+psycopg2://", 1)
    if clean.startswith("postgresql://") and not clean.startswith("postgresql+"):
        return clean.replace("postgresql://", "postgresql+psycopg2://", 1)
    return clean


def get_database_url() -> str:
    """
    Resolve the database connection URL:
      1. os.environ['DATABASE_URL'] (cloud override, e.g. Supabase in GitHub Actions)
      2. os.environ['DB_URL']
      3. settings.db_url (defaults to local SQLite: sqlite:///data/processed/trader.db)
    """
    raw_url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL") or settings.db_url
    return normalize_postgres_url(raw_url)


def is_cloud_database() -> bool:
    """Return True if currently configured to run against a PostgreSQL cloud database."""
    url = get_database_url()
    return url.startswith("postgresql") or url.startswith("postgres")


def verify_ci_database_guard(force_check: bool = False) -> None:
    """
    CI Preflight Guard: Ensure GitHub Actions can NEVER silently fall back to SQLite.

    When running in CI / GitHub Actions (or when force_check=True):
      - requires DATABASE_URL to be set in environment
      - requires it to be a valid PostgreSQL URL
      - fails immediately before the trader starts if DATABASE_URL is missing or invalid
      - does NOT modify normal local development behavior
    """
    is_ci = force_check or (
        os.getenv("GITHUB_ACTIONS", "").lower() in ("true", "1")
        or os.getenv("CI", "").lower() in ("true", "1")
    )
    if not is_ci:
        return

    # In CI unit testing under pytest, skip unless explicitly forcing the guard check
    if os.getenv("PYTEST_CURRENT_TEST") and not force_check:
        return

    raw_url = os.environ.get("DATABASE_URL")
    if not raw_url or not raw_url.strip():
        raise RuntimeError(
            "CI DATABASE SAFETY GUARD: DATABASE_URL environment variable is required in CI / GitHub Actions "
            "to prevent silent fallback to SQLite. Please configure the DATABASE_URL repository secret."
        )

    clean = raw_url.strip().lower()
    if not (clean.startswith("postgresql://") or clean.startswith("postgres://") or clean.startswith("postgresql+")):
        raise ValueError(
            f"CI DATABASE SAFETY GUARD: DATABASE_URL must be a PostgreSQL URL (got '{raw_url[:15]}...'). "
            "Cloud runners must never use SQLite."
        )


def get_cloud_engine(url: Optional[str] = None) -> Engine:
    """
    Create a SQLAlchemy engine configured for the cloud or fallback database.
    - PostgreSQL: enables connection pooling and pre-ping for resilient cloud connections.
    - SQLite: disables thread check for multi-threaded runner compatibility.
    """
    target_url = normalize_postgres_url(url) if url else get_database_url()

    if target_url.startswith("sqlite"):
        logger.info("Using local SQLite database: %s", target_url)
        return create_engine(target_url, connect_args={"check_same_thread": False}, echo=False)

    logger.info("Connecting to Cloud PostgreSQL database...")
    return create_engine(
        target_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


def init_cloud_database() -> None:
    """
    Ensure all application tables and indexes are initialized on the cloud database.
    Safe to call repeatedly (idempotent).
    """
    from src.db import repository
    repository.create_all_tables()
    logger.info("Cloud database tables verified and initialized successfully.")


def test_connection() -> bool:
    """Quick diagnostic health-check for cloud database connectivity."""
    try:
        from sqlalchemy import text
        engine = get_cloud_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Cloud database connectivity check failed: %s", exc)
        return False
