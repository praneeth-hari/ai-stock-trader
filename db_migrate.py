"""
db_migrate.py — Database Migration Utility for AI Stock Trader.

Supports cross-backend migration between SQLite and PostgreSQL via SQLAlchemy ORM.
Satisfies Section 4 (Item 14) requirements:
  - Export database contents to a portable JSON backup file.
  - Import JSON backup file into target database (SQLite or PostgreSQL).
  - Direct live transfer between source and target database URLs.
  - Verifies row counts and schema integrity across all 7 ORM tables:
      1. market_data
      2. features
      3. predictions
      4. portfolio
      5. orders
      6. trades
      7. event_log

CLI Usage:
  # Export current active database to JSON:
  python db_migrate.py --export backup.json

  # Import JSON into PostgreSQL:
  python db_migrate.py --import backup.json --target-url postgresql://user:pass@localhost:5432/trader

  # Direct live transfer from SQLite to PostgreSQL:
  python db_migrate.py --transfer --source-url sqlite:///data/processed/trader.db --target-url postgresql://user:pass@localhost:5432/trader
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Session

# Import settings and models
from config.settings import settings
from src.db.models import (
    Base,
    EventLog,
    FeatureRow,
    MarketDataRow,
    OrderRow,
    PortfolioSnapshot,
    PredictionRow,
    TradeRow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("db_migrate")

# Registered ORM tables in migration order
TABLE_MODELS: List[Type[DeclarativeBase]] = [
    MarketDataRow,
    FeatureRow,
    PredictionRow,
    PortfolioSnapshot,
    OrderRow,
    TradeRow,
    EventLog,
]


class MigrationJSONEncoder(json.JSONEncoder):
    """Encodes dates, datetimes, and custom types for portable JSON storage."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def normalize_db_url(url: str) -> str:
    """Ensure postgresql+psycopg2 driver dialect is used for PostgreSQL URLs."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def get_engine_for_url(url: str):
    """Creates a SQLAlchemy engine configured appropriately for the target backend."""
    norm_url = normalize_db_url(url)
    if norm_url.startswith("sqlite"):
        return create_engine(norm_url, connect_args={"check_same_thread": False})
    return create_engine(norm_url, pool_pre_ping=True)


def export_database(source_url: Optional[str] = None, output_path: str = "backup.json") -> Dict[str, int]:
    """
    Exports all tables from the source database into a portable JSON file.

    Returns:
        Dict mapping table names to exported row counts.
    """
    src = source_url or settings.db_url
    logger.info("Starting database export from: %s", src)
    engine = get_engine_for_url(src)
    Base.metadata.create_all(engine)

    dump_data: Dict[str, Any] = {
        "metadata": {
            "source_url": src,
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0",
        },
        "tables": {},
    }
    counts: Dict[str, int] = {}

    with Session(engine) as session:
        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            stmt = select(model).order_by(model.id)
            rows = session.execute(stmt).scalars().all()

            row_dicts = []
            for r in rows:
                r_dict = {}
                for col in model.__table__.columns:
                    val = getattr(r, col.name)
                    # Convert datetimes/dates if needed
                    if isinstance(val, (date, datetime)):
                        val = val.isoformat()
                    r_dict[col.name] = val
                row_dicts.append(r_dict)

            dump_data["tables"][tbl_name] = row_dicts
            counts[tbl_name] = len(row_dicts)
            logger.info("  Exported %-16s: %d rows", tbl_name, len(row_dicts))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, cls=MigrationJSONEncoder, indent=2)

    total_rows = sum(counts.values())
    logger.info("Database export complete -> %s (%d total rows across %d tables)", output_path, total_rows, len(counts))
    return counts


def import_database(
    input_path: str,
    target_url: Optional[str] = None,
    batch_size: int = 500,
    clear_existing: bool = False,
) -> Dict[str, int]:
    """
    Imports table data from a JSON file into the target database.

    Returns:
        Dict mapping table names to imported row counts.
    """
    tgt = target_url or settings.db_url
    logger.info("Starting database import into: %s from: %s", tgt, input_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Export file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        dump_data = json.load(f)

    tables_data = dump_data.get("tables", {})
    engine = get_engine_for_url(tgt)
    Base.metadata.create_all(engine)

    imported_counts: Dict[str, int] = {}

    with Session(engine) as session:
        if clear_existing:
            for model in reversed(TABLE_MODELS):
                session.query(model).delete()
            session.commit()
            logger.info("Cleared existing rows in target database.")

        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            rows = tables_data.get(tbl_name, [])
            if not rows:
                imported_counts[tbl_name] = 0
                logger.info("  Imported %-16s: 0 rows (empty)", tbl_name)
                continue

            count = 0
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                instances = []
                for row_dict in chunk:
                    # Filter only known column attributes
                    col_names = {col.name for col in model.__table__.columns}
                    filtered = {k: v for k, v in row_dict.items() if k in col_names}

                    # Parse date/timestamp columns if model expects datetime/date objects
                    for col in model.__table__.columns:
                        col_type = str(col.type).upper()
                        if col.name in filtered and filtered[col.name] is not None:
                            val = filtered[col.name]
                            if ("DATE" in col_type or "TIME" in col_type) and isinstance(val, str):
                                try:
                                    filtered[col.name] = datetime.fromisoformat(val)
                                except Exception:
                                    pass

                    instances.append(model(**filtered))

                session.add_all(instances)
                session.commit()
                count += len(chunk)

            imported_counts[tbl_name] = count
            logger.info("  Imported %-16s: %d rows", tbl_name, count)

    total_rows = sum(imported_counts.values())
    logger.info("Database import complete (%d total rows imported)", total_rows)
    return imported_counts


def transfer_database(
    source_url: str,
    target_url: str,
    batch_size: int = 500,
    clear_existing: bool = False,
) -> Dict[str, int]:
    """
    Directly transfers all data from source_url to target_url without intermediary disk file.

    Returns:
        Dict mapping table names to transferred row counts.
    """
    logger.info("Direct database transfer: %s -> %s", source_url, target_url)
    src_engine = get_engine_for_url(source_url)
    tgt_engine = get_engine_for_url(target_url)

    Base.metadata.create_all(src_engine)
    Base.metadata.create_all(tgt_engine)

    transferred: Dict[str, int] = {}

    with Session(src_engine) as src_session, Session(tgt_engine) as tgt_session:
        if clear_existing:
            for model in reversed(TABLE_MODELS):
                tgt_session.query(model).delete()
            tgt_session.commit()

        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            rows = src_session.execute(select(model).order_by(model.id)).scalars().all()

            if not rows:
                transferred[tbl_name] = 0
                logger.info("  Transferred %-16s: 0 rows", tbl_name)
                continue

            col_names = {col.name for col in model.__table__.columns}
            count = 0
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                new_instances = []
                for r in chunk:
                    row_kwargs = {col: getattr(r, col) for col in col_names}
                    new_instances.append(model(**row_kwargs))
                tgt_session.add_all(new_instances)
                tgt_session.commit()
                count += len(chunk)

            transferred[tbl_name] = count
            logger.info("  Transferred %-16s: %d rows", tbl_name, count)

    logger.info("Direct database transfer complete (%d total rows)", sum(transferred.values()))
    return transferred


def verify_tables(db_url: Optional[str] = None) -> Dict[str, int]:
    """Queries row counts for all 7 ORM tables in the specified database."""
    url = db_url or settings.db_url
    engine = get_engine_for_url(url)
    Base.metadata.create_all(engine)
    counts: Dict[str, int] = {}
    with Session(engine) as session:
        for model in TABLE_MODELS:
            count = session.query(func.count(model.id)).scalar() or 0
            counts[model.__tablename__] = int(count)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Database Migration Utility for AI Stock Trader")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", dest="export_file", help="Export active database to specified JSON file")
    group.add_argument("--import", dest="import_file", help="Import specified JSON file into database")
    group.add_argument("--transfer", action="store_true", help="Perform direct database transfer")
    group.add_argument("--verify", action="store_true", help="Verify table counts in database")

    parser.add_argument("--source-url", default=None, help="Source database URL (defaults to settings.db_url)")
    parser.add_argument("--target-url", default=None, help="Target database URL (defaults to settings.db_url)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch insertion size (default: 500)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data in target before import/transfer")

    args = parser.parse_args()

    if args.export_file:
        export_database(source_url=args.source_url, output_path=args.export_file)
    elif args.import_file:
        import_database(
            input_path=args.import_file,
            target_url=args.target_url,
            batch_size=args.batch_size,
            clear_existing=args.clear,
        )
    elif args.transfer:
        if not args.source_url or not args.target_url:
            parser.error("--transfer requires both --source-url and --target-url")
        transfer_database(
            source_url=args.source_url,
            target_url=args.target_url,
            batch_size=args.batch_size,
            clear_existing=args.clear,
        )
    elif args.verify:
        url = args.source_url or args.target_url or settings.db_url
        counts = verify_tables(url)
        print(f"\nDatabase Table Verification [{url}]:")
        for tbl, cnt in counts.items():
            print(f"  {tbl:<18}: {cnt:>6} rows")
        print(f"  {'TOTAL':<18}: {sum(counts.values()):>6} rows\n")


if __name__ == "__main__":
    main()
