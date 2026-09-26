"""Suite-wide test isolation: tests must never write production state or tracked reports."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_pending_orders_dir(tmp_path, monkeypatch):
    """Every test writes PaperBroker pending orders to its own temp dir, never to production data/."""
    import src.trading.paper_broker as paper_broker

    monkeypatch.setattr(paper_broker, "PENDING_ORDERS_DIR", tmp_path / "pending_orders")


@pytest.fixture(autouse=True)
def _isolate_tax_report_dir(tmp_path, monkeypatch):
    """Dashboard renders auto-generate the tax report; default its output to a temp dir, not tracked reports/."""
    from src.reports import tax_report

    real_generate = tax_report.generate_tax_report

    def generate_into_tmp(year, country="India", reports_dir=None):
        return real_generate(year=year, country=country, reports_dir=reports_dir or str(tmp_path / "reports"))

    monkeypatch.setattr(tax_report, "generate_tax_report", generate_into_tmp)
