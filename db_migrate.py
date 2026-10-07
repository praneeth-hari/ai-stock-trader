"""
db_migrate.py — Database Migration Utility for AI Stock Trader.

Supports cross-backend migration between SQLite and PostgreSQL via SQLAlchemy ORM.
Satisfies Section 4 requirements:
  - Export database contents to a portable JSON backup file.
  - Import JSON backup file into target database (SQLite or PostgreSQL).
  - Direct live transfer between source and target database URLs with chunking.
  - Verifies row counts and schema integrity across ALL 19 ORM tables:
      1. strategy_variants (FK parent)
      2. strategy_snapshots (FK child -> strategy_variants.id)
      3. strategy_trades (FK child -> strategy_variants.id)
      4. market_data
      5. features
      6. predictions
      7. portfolio
      8. orders
      9. trades
      10. event_log
      11. decision_log
      12. benchmark_snapshots
      13. sentiment_scores
      14. earnings_calendar
      15. sector_rankings
      16. macro_indicators
      17. correlation_matrix
      18. feature_importance
      19. walk_forward_results

CLI Usage:
  # Preflight check without making any changes:
  python db_migrate.py --preflight --source-url sqlite:///data/processed/trader.db --target-url postgresql://...

  # Direct live transfer from SQLite to PostgreSQL (chunked at 1000 rows):
  python db_migrate.py --transfer --source-url sqlite:///data/processed/trader.db --target-url postgresql://...

  # Deep verification between SQLite source and PostgreSQL target:
  python db_migrate.py --verify-migration --source-url sqlite:///data/processed/trader.db --target-url postgresql://...

  # Export active database to JSON:
  python db_migrate.py --export backup.json

  # Import JSON into database:
  python db_migrate.py --import backup.json --target-url postgresql://...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Session

# Import settings and all 19 models
from config.settings import settings
from src.db.models import (
    Base,
    BenchmarkSnapshotRow,
    CorrelationMatrixRow,
    DecisionLogRow,
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("db_migrate")

# Registered ORM tables in strict foreign-key dependency order:
# 1. Parents before children (strategy_variants before strategy_snapshots/strategy_trades)
# 2. Independent core tables
# 3. Observability and intelligence tables
TABLE_MODELS: List[Type[DeclarativeBase]] = [
    # 1. Strategy parent
    StrategyVariantRow,
    # 2. Strategy children (FK -> strategy_variants.id)
    StrategySnapshotRow,
    StrategyTradeRow,
    # 3. Core market and execution tables
    MarketDataRow,
    FeatureRow,
    PredictionRow,
    PortfolioSnapshot,
    OrderRow,
    TradeRow,
    EventLog,
    # 4. Observability and shadow benchmarks
    DecisionLogRow,
    BenchmarkSnapshotRow,
    # 5. Market intelligence and validation
    SentimentScoreRow,
    EarningsCalendarRow,
    SectorRankingRow,
    MacroIndicatorRow,
    CorrelationMatrixRow,
    FeatureImportanceRow,
    WalkForwardResultRow,
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


def reset_postgres_sequence(session: Session, table_name: str) -> None:
    """Reset PostgreSQL serial/identity sequence to MAX(id) if target is PostgreSQL."""
    try:
        bind = session.get_bind()
        if bind and bind.dialect.name == "postgresql":
            sql = f"""
                SELECT setval(
                    pg_get_serial_sequence('public."{table_name}"', 'id'),
                    COALESCE((SELECT MAX(id) FROM public."{table_name}"), 1),
                    (SELECT COUNT(*) > 0 FROM public."{table_name}")
                );
            """
            session.execute(text(sql))
            session.commit()
    except Exception as e:
        session.rollback()
        logger.warning("Could not reset sequence for table %s: %s", table_name, e)


def sanitize_json(val: Any) -> Any:
    """Sanitizes JSON blobs by converting unquoted NaN, Infinity, -Infinity to null/None."""
    if isinstance(val, str):
        val = re.sub(r"\bNaN\b", "null", val)
        val = re.sub(r"\bInfinity\b", "null", val)
        val = re.sub(r"\b-Infinity\b", "null", val)
        try:
            val = json.loads(val)
        except Exception:
            pass
    if isinstance(val, dict):
        return {k: sanitize_json(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_json(v) for v in val]
    elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val



def export_database(
    source_url: Optional[str] = None,
    output_path: str = "backup.json",
    batch_size: int = 1000,
) -> Dict[str, int]:
    """
    Exports all 19 tables from the source database into a portable JSON file
    using chunked reads to avoid memory exhaustion on large tables.

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
            "version": "2.0",
            "tables_count": len(TABLE_MODELS),
        },
        "tables": {},
    }
    counts: Dict[str, int] = {}

    with Session(engine) as session:
        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            col_names = [col.name for col in model.__table__.columns]
            row_dicts: List[Dict[str, Any]] = []

            last_id = 0
            while True:
                stmt = select(model).where(model.id > last_id).order_by(model.id).limit(batch_size)
                chunk = session.execute(stmt).scalars().all()
                if not chunk:
                    break

                for r in chunk:
                    r_dict = {}
                    for col in col_names:
                        val = getattr(r, col)
                        if isinstance(val, (date, datetime)):
                            val = val.isoformat()
                        r_dict[col] = val
                    row_dicts.append(r_dict)

                last_id = chunk[-1].id

            dump_data["tables"][tbl_name] = row_dicts
            counts[tbl_name] = len(row_dicts)
            logger.info("  Exported %-22s: %d rows", tbl_name, len(row_dicts))

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
    batch_size: int = 1000,
    clear_existing: bool = False,
) -> Dict[str, int]:
    """
    Imports table data from a JSON file into the target database across all 19 tables.

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
                logger.info("  Imported %-22s: 0 rows (empty)", tbl_name)
                continue

            col_names = {col.name for col in model.__table__.columns}
            col_types = {col.name: str(col.type).upper() for col in model.__table__.columns}

            count = 0
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                instances = []
                for row_dict in chunk:
                    filtered = {k: v for k, v in row_dict.items() if k in col_names}

                    for col, val in filtered.items():
                        c_type = col_types.get(col, "")
                        if val is not None:
                            if ("DATE" in c_type or "TIME" in c_type) and isinstance(val, str):
                                try:
                                    filtered[col] = datetime.fromisoformat(val)
                                except Exception:
                                    pass
                            elif "JSON" in c_type:
                                filtered[col] = sanitize_json(val)

                    instances.append(model(**filtered))

                session.add_all(instances)
                session.commit()
                count += len(chunk)

            # Update PostgreSQL sequence to match MAX(id)
            reset_postgres_sequence(session, tbl_name)

            imported_counts[tbl_name] = count
            logger.info("  Imported %-22s: %d rows", tbl_name, count)

    total_rows = sum(imported_counts.values())
    logger.info("Database import complete (%d total rows imported)", total_rows)
    return imported_counts


def transfer_database(
    source_url: str,
    target_url: str,
    batch_size: int = 1000,
    clear_existing: bool = False,
) -> Dict[str, int]:
    """
    Directly transfers all data from source_url to target_url without loading
    large datasets entirely into RAM.
    
    Streams via ID-paging (chunked batches of `batch_size`), commits per chunk,
    and updates PostgreSQL sequences upon completion.

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
            logger.info("Cleared existing rows in target database.")

        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            total_rows = src_session.query(func.count(model.id)).scalar() or 0

            if total_rows == 0:
                transferred[tbl_name] = 0
                logger.info("  Transferred %-22s: 0 rows (empty)", tbl_name)
                continue

            target_rows = tgt_session.query(func.count(model.id)).scalar() or 0
            if target_rows >= total_rows:
                reset_postgres_sequence(tgt_session, tbl_name)
                transferred[tbl_name] = target_rows
                logger.info("  Already migrated %-22s: %d/%d rows (skipping)", tbl_name, target_rows, total_rows)
                continue

            col_names = [col.name for col in model.__table__.columns]
            col_types = {col.name: str(col.type).upper() for col in model.__table__.columns}
            col_lengths = {col.name: getattr(col.type, "length", None) for col in model.__table__.columns}

            last_id = 0
            count = 0
            if target_rows > 0:
                last_id = tgt_session.query(func.max(model.id)).scalar() or 0
                count = target_rows
                logger.info("  Resuming    %-22s from id > %d (%d already in target)", tbl_name, last_id, target_rows)

            while True:
                stmt = select(model).where(model.id > last_id).order_by(model.id).limit(batch_size)
                chunk = src_session.execute(stmt).scalars().all()
                if not chunk:
                    break

                new_instances = []
                for r in chunk:
                    row_kwargs = {}
                    for col in col_names:
                        val = getattr(r, col)
                        if val is not None:
                            c_type = col_types.get(col, "")
                            c_len = col_lengths.get(col)
                            if ("DATE" in c_type or "TIME" in c_type) and isinstance(val, str):
                                try:
                                    val = datetime.fromisoformat(val)
                                except Exception:
                                    pass
                            elif "JSON" in c_type:
                                val = sanitize_json(val)
                            elif isinstance(val, str) and c_len and len(val) > c_len:
                                val = val[:c_len]
                        row_kwargs[col] = val
                    new_instances.append(model(**row_kwargs))

                tgt_session.add_all(new_instances)
                tgt_session.commit()
                count += len(chunk)
                last_id = chunk[-1].id

                if total_rows > batch_size:
                    pct = (count / total_rows) * 100
                    logger.info("  Transferred %-22s: %d/%d rows (%.1f%%)", tbl_name, count, total_rows, pct)

            # Reset PostgreSQL sequence to prevent future PK collision
            reset_postgres_sequence(tgt_session, tbl_name)

            transferred[tbl_name] = count
            logger.info("  Completed   %-22s: %d rows", tbl_name, count)

    total_transferred = sum(transferred.values())
    logger.info("Direct database transfer complete (%d total rows across %d tables)", total_transferred, len(transferred))
    return transferred


def verify_tables(db_url: Optional[str] = None) -> Dict[str, int]:
    """Queries row counts for all 19 ORM tables in the specified database."""
    url = db_url or settings.db_url
    engine = get_engine_for_url(url)
    Base.metadata.create_all(engine)
    counts: Dict[str, int] = {}
    with Session(engine) as session:
        for model in TABLE_MODELS:
            count = session.query(func.count(model.id)).scalar() or 0
            counts[model.__tablename__] = int(count)
    return counts


def preflight_check(source_url: Optional[str] = None, target_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates source and target database connectivity, 19-table availability,
    source row counts, target row counts, and foreign key ordering WITHOUT
    inserting, updating, or deleting any data.

    Returns:
        Dict summarizing preflight findings and pass/fail status.
    """
    src = source_url or settings.db_url
    tgt = target_url or settings.db_url

    logger.info("Starting migration preflight check...")
    src_engine = get_engine_for_url(src)
    tgt_engine = get_engine_for_url(tgt)

    # Validate connections
    try:
        with src_engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        src_conn_ok = True
    except Exception as e:
        logger.error("Source connection failed: %s", e)
        src_conn_ok = False

    try:
        with tgt_engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        tgt_conn_ok = True
    except Exception as e:
        logger.error("Target connection failed: %s", e)
        tgt_conn_ok = False

    if not src_conn_ok or not tgt_conn_ok:
        return {
            "status": False,
            "src_connected": src_conn_ok,
            "tgt_connected": tgt_conn_ok,
            "error": "Database connectivity check failed.",
        }

    # Verify foreign key dependency order
    table_names = [m.__tablename__ for m in TABLE_MODELS]
    sv_idx = table_names.index("strategy_variants")
    ss_idx = table_names.index("strategy_snapshots")
    st_idx = table_names.index("strategy_trades")
    fk_order_valid = (sv_idx < ss_idx) and (sv_idx < st_idx)

    src_counts: Dict[str, int] = {}
    tgt_counts: Dict[str, int] = {}

    with Session(src_engine) as src_session, Session(tgt_engine) as tgt_session:
        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            try:
                s_cnt = src_session.query(func.count(model.id)).scalar() or 0
            except Exception as e:
                s_cnt = -1
                logger.warning("Source table %s query failed: %s", tbl_name, e)
            src_counts[tbl_name] = s_cnt

            try:
                t_cnt = tgt_session.query(func.count(model.id)).scalar() or 0
            except Exception as e:
                t_cnt = -1
                logger.warning("Target table %s query failed: %s", tbl_name, e)
            tgt_counts[tbl_name] = t_cnt

    all_src_ready = all(cnt >= 0 for cnt in src_counts.values())
    all_tgt_ready = all(cnt >= 0 for cnt in tgt_counts.values())

    status = (all_src_ready and all_tgt_ready and fk_order_valid)

    return {
        "status": status,
        "fk_order_valid": fk_order_valid,
        "tables_count": len(TABLE_MODELS),
        "src_counts": src_counts,
        "tgt_counts": tgt_counts,
        "total_source_rows": sum(c for c in src_counts.values() if c > 0),
        "total_target_rows": sum(c for c in tgt_counts.values() if c > 0),
    }


def verify_migration(source_url: str, target_url: str) -> Dict[str, Any]:
    """
    Performs end-to-end verification comparing source and target databases across:
      - All 19 table row counts
      - Min and max dates for date-bearing tables
      - Key trading state (portfolio cash/positions, latest order, latest trade)
    """
    logger.info("Starting migration verification: %s vs %s", source_url, target_url)
    src_engine = get_engine_for_url(source_url)
    tgt_engine = get_engine_for_url(target_url)

    mismatches: List[str] = []
    comparisons: Dict[str, Any] = {}

    with Session(src_engine) as src_session, Session(tgt_engine) as tgt_session:
        for model in TABLE_MODELS:
            tbl_name = model.__tablename__
            src_cnt = src_session.query(func.count(model.id)).scalar() or 0
            tgt_cnt = tgt_session.query(func.count(model.id)).scalar() or 0

            row_match = (src_cnt == tgt_cnt)
            if not row_match:
                mismatches.append(f"{tbl_name}: row count mismatch (src={src_cnt}, tgt={tgt_cnt})")

            date_info: Dict[str, Any] = {}
            col_names = [col.name for col in model.__table__.columns]
            d_col_name = None
            for candidate in ("date", "run_date", "timestamp"):
                if candidate in col_names:
                    d_col_name = candidate
                    break

            if d_col_name and src_cnt > 0:
                col_attr = getattr(model, d_col_name)
                src_min = src_session.query(func.min(col_attr)).scalar()
                src_max = src_session.query(func.max(col_attr)).scalar()
                tgt_min = tgt_session.query(func.min(col_attr)).scalar()
                tgt_max = tgt_session.query(func.max(col_attr)).scalar()

                def _fmt_d(d_val):
                    if isinstance(d_val, datetime):
                        if d_val.tzinfo is not None:
                            d_val = d_val.astimezone(timezone.utc).replace(tzinfo=None)
                        return d_val.isoformat()
                    if isinstance(d_val, date):
                        return d_val.isoformat()
                    return str(d_val)

                src_min_str = _fmt_d(src_min)
                src_max_str = _fmt_d(src_max)
                tgt_min_str = _fmt_d(tgt_min)
                tgt_max_str = _fmt_d(tgt_max)

                date_match = (src_min_str == tgt_min_str and src_max_str == tgt_max_str)
                date_info = {
                    "column": d_col_name,
                    "src_range": (src_min_str, src_max_str),
                    "tgt_range": (tgt_min_str, tgt_max_str),
                    "match": date_match,
                }
                if not date_match:
                    mismatches.append(f"{tbl_name}: date range mismatch")

            comparisons[tbl_name] = {
                "src_count": src_cnt,
                "tgt_count": tgt_cnt,
                "row_match": row_match,
                "date_info": date_info,
            }

        # Trading state checks
        trading_state: Dict[str, Any] = {}
        src_port = src_session.query(PortfolioSnapshot).order_by(PortfolioSnapshot.id.desc()).first()
        tgt_port = tgt_session.query(PortfolioSnapshot).order_by(PortfolioSnapshot.id.desc()).first()
        if src_port and tgt_port:
            port_match = (
                src_port.run_date == tgt_port.run_date
                and abs(src_port.cash - tgt_port.cash) < 1e-4
                and abs(src_port.total_value - tgt_port.total_value) < 1e-4
                and src_port.market == tgt_port.market
            )
            trading_state["portfolio"] = {
                "match": port_match,
                "run_date": src_port.run_date,
                "src_cash": src_port.cash,
                "tgt_cash": tgt_port.cash,
                "src_total_value": src_port.total_value,
                "tgt_total_value": tgt_port.total_value,
            }
            if not port_match:
                mismatches.append("portfolio: latest snapshot mismatch")

        src_ord = src_session.query(OrderRow).order_by(OrderRow.id.desc()).first()
        tgt_ord = tgt_session.query(OrderRow).order_by(OrderRow.id.desc()).first()
        if src_ord and tgt_ord:
            ord_match = (
                src_ord.run_date == tgt_ord.run_date
                and src_ord.ticker == tgt_ord.ticker
                and src_ord.action == tgt_ord.action
            )
            trading_state["orders"] = {
                "match": ord_match,
                "run_date": src_ord.run_date,
                "ticker": src_ord.ticker,
                "action": src_ord.action,
            }
            if not ord_match:
                mismatches.append("orders: latest order mismatch")

        src_trd = src_session.query(TradeRow).order_by(TradeRow.id.desc()).first()
        tgt_trd = tgt_session.query(TradeRow).order_by(TradeRow.id.desc()).first()
        if src_trd and tgt_trd:
            trd_match = (
                src_trd.run_date == tgt_trd.run_date
                and src_trd.ticker == tgt_trd.ticker
                and src_trd.action == tgt_trd.action
            )
            trading_state["trades"] = {
                "match": trd_match,
                "run_date": src_trd.run_date,
                "ticker": src_trd.ticker,
                "action": src_trd.action,
            }
            if not trd_match:
                mismatches.append("trades: latest trade mismatch")

    return {
        "success": (len(mismatches) == 0),
        "mismatches": mismatches,
        "comparisons": comparisons,
        "trading_state": trading_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Database Migration Utility for AI Stock Trader")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", dest="export_file", help="Export active database to specified JSON file")
    group.add_argument("--import", dest="import_file", help="Import specified JSON file into database")
    group.add_argument("--transfer", action="store_true", help="Perform direct database transfer")
    group.add_argument("--verify", action="store_true", help="Verify table counts in database")
    group.add_argument("--preflight", action="store_true", help="Run preflight check between source and target without modifying data")
    group.add_argument("--verify-migration", action="store_true", help="Verify complete migration between source and target")

    parser.add_argument("--source-url", default=None, help="Source database URL (defaults to settings.db_url)")
    parser.add_argument("--target-url", default=None, help="Target database URL (defaults to settings.db_url)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch insertion size (default: 1000)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data in target before import/transfer")

    args = parser.parse_args()

    src_url = args.source_url or settings.db_url
    tgt_url = args.target_url or settings.db_url

    if args.preflight:
        res = preflight_check(source_url=src_url, target_url=tgt_url)
        print("\n=== MIGRATION PREFLIGHT SUMMARY ===")
        print(f"  Foreign Key Ordering: {'PASS' if res['fk_order_valid'] else 'FAIL'}")
        print(f"  Total Tables Checked: {res.get('tables_count', 0)}")
        print(f"  Total Source Rows:    {res.get('total_source_rows', 0):,}")
        print(f"  Total Target Rows:    {res.get('total_target_rows', 0):,}")
        print("\n  Table Details:")
        for tbl, s_cnt in res.get("src_counts", {}).items():
            t_cnt = res.get("tgt_counts", {}).get(tbl, 0)
            print(f"    {tbl:<24}: Source = {s_cnt:>6} rows | Target = {t_cnt:>6} rows")
        verdict = "PASS (Ready for transfer)" if res["status"] else "FAIL"
        print(f"\n  PREFLIGHT VERDICT: {verdict}\n")

    elif args.verify_migration:
        res = verify_migration(source_url=src_url, target_url=tgt_url)
        print("\n=== MIGRATION VERIFICATION SUMMARY ===")
        print(f"  Overall Status: {'PASS' if res['success'] else 'FAIL'}")
        if res["mismatches"]:
            print("  Mismatches:")
            for m in res["mismatches"]:
                print(f"    - {m}")
        else:
            print("  All 19 table row counts and dates match perfectly.")
        print()

    elif args.export_file:
        export_database(source_url=src_url, output_path=args.export_file, batch_size=args.batch_size)

    elif args.import_file:
        import_database(
            input_path=args.import_file,
            target_url=tgt_url,
            batch_size=args.batch_size,
            clear_existing=args.clear,
        )

    elif args.transfer:
        transfer_database(
            source_url=src_url,
            target_url=tgt_url,
            batch_size=args.batch_size,
            clear_existing=args.clear,
        )

    elif args.verify:
        url = args.source_url or args.target_url or settings.db_url
        counts = verify_tables(url)
        print(f"\nDatabase Table Verification [{url}]:")
        for tbl, cnt in counts.items():
            print(f"  {tbl:<24}: {cnt:>6} rows")
        print(f"  {'TOTAL':<24}: {sum(counts.values()):>6} rows\n")


if __name__ == "__main__":
    main()
