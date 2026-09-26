"""V1 forward-paper model freeze: the production active model can't be replaced or used unverified."""

import hashlib
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import settings
from src.db import repository
from src.ml.evaluate import (
    ACTIVE_MODEL_FILENAME,
    ModelFreezeError,
    promote_model,
    verify_frozen_active_model,
)
from src.ml.train import rollback_model


@pytest.fixture
def frozen_prod_dir(tmp_path, monkeypatch):
    """A temp directory standing in for the production models dir, with the freeze on."""
    prod = tmp_path / "prod_models"
    prod.mkdir()
    monkeypatch.setattr(settings, "data_models_dir", prod)
    monkeypatch.setattr(settings, "model_freeze_enabled", True)
    return prod


def test_v1_freeze_is_on_by_default_and_pins_the_frozen_model_hash():
    assert settings.model_freeze_enabled is True
    assert settings.frozen_model_sha256 == "09a8f688630438e200519d246008918a947d4208295ad7da54fd9256588963dc"


def test_promotion_into_production_dir_is_blocked_and_active_model_untouched(frozen_prod_dir, tmp_path):
    (frozen_prod_dir / ACTIVE_MODEL_FILENAME).write_bytes(b"frozen-model")
    candidate = tmp_path / "candidate.joblib"
    candidate.write_bytes(b"new-model")

    with pytest.raises(ModelFreezeError, match="MODEL_FREEZE"):
        promote_model(candidate_model_path=candidate, reason="test", author="test")
    assert (frozen_prod_dir / ACTIVE_MODEL_FILENAME).read_bytes() == b"frozen-model"


def test_freeze_does_not_block_non_production_model_dirs(frozen_prod_dir, tmp_path):
    from src.ml.evaluate import assert_model_replacement_allowed

    assert_model_replacement_allowed(tmp_path / "experiment_models", "Promotion")  # no raise


def test_rollback_of_production_model_is_blocked(frozen_prod_dir):
    (frozen_prod_dir / ACTIVE_MODEL_FILENAME).write_bytes(b"frozen-model")
    with pytest.raises(ModelFreezeError):
        rollback_model()
    assert (frozen_prod_dir / ACTIVE_MODEL_FILENAME).read_bytes() == b"frozen-model"


def test_verify_frozen_active_model_checks_presence_and_hash(frozen_prod_dir, monkeypatch):
    with pytest.raises(ModelFreezeError, match="missing"):
        verify_frozen_active_model()

    (frozen_prod_dir / ACTIVE_MODEL_FILENAME).write_bytes(b"frozen-model")
    monkeypatch.setattr(settings, "frozen_model_sha256", hashlib.sha256(b"frozen-model").hexdigest())
    verify_frozen_active_model()  # exact match passes

    (frozen_prod_dir / ACTIVE_MODEL_FILENAME).write_bytes(b"swapped-model")
    with pytest.raises(ModelFreezeError, match="does not match"):
        verify_frozen_active_model()

    monkeypatch.setattr(settings, "model_freeze_enabled", False)
    verify_frozen_active_model()  # freeze off: no-op


def test_real_production_active_model_matches_frozen_hash():
    path = Path(settings.data_models_dir) / ACTIVE_MODEL_FILENAME
    if not path.exists():
        pytest.skip("Production active model not present in this environment.")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == settings.frozen_model_sha256


@pytest.fixture
def isolated_db(tmp_path):
    original_url = settings.db_url
    settings.db_url = f"sqlite:///{tmp_path / 'freeze_test.db'}"
    repository._engine = None
    repository._db_available = True
    repository.create_all_tables()
    yield
    settings.db_url = original_url
    repository._engine = None


def _synthetic_inputs():
    days = [(date(2023, 3, 1) + timedelta(days=i)) for i in range(330)]
    days = [d.strftime("%Y-%m-%d") for d in days if d.weekday() < 5][:223]
    n = len(days)
    spy = pd.DataFrame({"date": days, "open": [500.0] * n, "high": [505.0] * n, "low": [495.0] * n,
                        "close": [500.0] * n, "volume": [1_000_000] * n, "ticker": ["SPY"] * n})
    aapl = pd.DataFrame({"date": days, "open": [99.5] * n, "high": [101.0] * n, "low": [99.0] * n,
                         "close": [100.0] * n, "volume": [500_000] * n, "ticker": ["AAPL"] * n})
    return days[-1], spy, aapl


@pytest.mark.parametrize("model_bytes", [None, b"not-the-frozen-model"])
def test_daily_pipeline_fails_loud_and_never_bootstraps_a_model_while_frozen(
        frozen_prod_dir, isolated_db, model_bytes):
    from src.pipeline.daily_pipeline import run_daily_pipeline
    from src.trading.paper_broker import PaperBroker

    if model_bytes is not None:
        (frozen_prod_dir / ACTIVE_MODEL_FILENAME).write_bytes(model_bytes)
    run_date, spy, aapl = _synthetic_inputs()
    offline = RuntimeError("offline in test")
    broker = PaperBroker()
    broker.load_state()

    with patch("src.pipeline.daily_pipeline.analyze_universe_sentiment", side_effect=offline), \
         patch("src.pipeline.daily_pipeline.get_universe_earnings_calendar", side_effect=offline), \
         patch("src.pipeline.daily_pipeline.calculate_sector_rotation", side_effect=offline), \
         patch("src.pipeline.daily_pipeline.get_macro_environment", side_effect=offline), \
         patch("src.ml.train.train_and_promote") as bootstrap, \
         patch("src.pipeline.daily_pipeline.load_active_model") as load:
        res = run_daily_pipeline(run_date=run_date, tickers=["AAPL"], broker=broker,
                                 _spy_df=spy, _universe_dfs={"AAPL": aapl})

    assert any("MODEL_FREEZE" in e for e in res.errors)
    assert res.orders_generated == 0 and not res.pending_orders
    assert not bootstrap.called and not load.called
    assert not (frozen_prod_dir / ACTIVE_MODEL_FILENAME).exists() or model_bytes is not None
