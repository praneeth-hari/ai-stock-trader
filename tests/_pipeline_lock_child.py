"""Independent-process helper for tests/test_pipeline_process_lock.py (not collected as a test).

  hold <lock_dir> <market> <ready_file>
      Take the machine-wide lock, write <ready_file>, keep it until <ready_file>.release exists (or killed).
  run <lock_dir> <db_url> <pending_dir> <run_date> <result_file> [<pause_marker>]
      Run the real pipeline on synthetic data against <db_url>, write the result as JSON. With a pause
      marker, stop inside the risk step (after the morning fills) until <pause_marker>.go exists.
All state goes to the paths given; Telegram and live intelligence are stubbed.
"""

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _wait(path: Path, timeout: float = 120.0) -> None:
    end = time.time() + timeout
    while not path.exists() and time.time() < end:
        time.sleep(0.05)


def hold(lock_dir, market, ready_file):
    from config.settings import settings
    from src.pipeline.run_lock import pipeline_run_lock

    settings.pipeline_lock_dir = Path(lock_dir)
    with pipeline_run_lock(market):
        Path(ready_file).write_text(str(os.getpid()))
        _wait(Path(ready_file + ".release"))


def run(lock_dir, db_url, pending_dir, run_date, result_file, pause_marker=None):
    from config.settings import settings
    from src.db import repository
    from src.features.engineer import FEATURE_COLUMNS
    import src.pipeline.daily_pipeline as dp
    import src.trading.paper_broker as pb

    settings.pipeline_lock_dir = Path(lock_dir)
    settings.db_url = db_url
    repository._engine = None
    repository.create_all_tables()
    pb.PENDING_ORDERS_DIR = Path(pending_dir)

    days = [(date(2023, 3, 1) + timedelta(days=i)) for i in range(340)]
    days = [d.strftime("%Y-%m-%d") for d in days if d.weekday() < 5 and d.strftime("%Y-%m-%d") <= run_date]

    def frame(t, c):
        n = len(days)
        return pd.DataFrame({"date": days, "open": [c * 0.995] * n, "high": [c * 1.01] * n, "low": [c * 0.99] * n,
                             "close": [c] * n, "volume": [500_000] * n, "ticker": [t] * n})

    f = lambda X: np.array([[0.5, 0.5]] * (len(X) if hasattr(X, "__len__") else 1))
    model = MagicMock()
    model.predict_proba = MagicMock(side_effect=f)
    model.model = MagicMock()
    model.model.predict_proba = MagicMock(side_effect=f)
    model.model.feature_names_in_ = FEATURE_COLUMNS
    model.metadata = {"model_type": "baseline", "feature_columns": FEATURE_COLUMNS}

    patches = [patch.object(dp, n, MagicMock()) for n in dir(dp) if n.startswith(("send_", "notify_"))]
    patches += [patch.object(dp, n, side_effect=RuntimeError("offline")) for n in
                ("analyze_universe_sentiment", "get_universe_earnings_calendar",
                 "calculate_sector_rotation", "get_macro_environment")]
    patches += [patch.object(dp, "load_active_model", return_value=model)]
    if pause_marker:
        real_risk = dp.evaluate_portfolio_risk

        def paused_risk(**kw):
            Path(pause_marker).write_text(str(os.getpid()))
            _wait(Path(pause_marker + ".go"))
            return real_risk(**kw)

        patches.append(patch.object(dp, "evaluate_portfolio_risk", side_effect=paused_risk))
    for p in patches:
        p.start()
    res = dp.run_daily_pipeline(run_date=run_date, tickers=["AAPL"], _spy_df=frame("SPY", 500.0),
                                _universe_dfs={"AAPL": frame("AAPL", 100.0)})
    Path(result_file).write_text(json.dumps({
        "pid": os.getpid(), "status": res.status, "regime": res.regime, "orders": res.orders_generated,
        "fills": [[x.ticker, x.action] for x in res.fills], "errors": res.errors,
    }))


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    {"hold": hold, "run": run}[mode](*args)
