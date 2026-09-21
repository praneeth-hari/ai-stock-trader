"""
tests/test_tax_report.py — Unit tests for SECTION 10 ITEM 3: Tax Report Generator.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.reports.tax_report import (
    DISCLAIMER_TEXT,
    calculate_tax_breakdown,
    export_tax_report_csv,
    export_tax_report_pdf,
    generate_tax_report,
    generate_tax_report_data,
    get_closed_trades_for_year,
)


@pytest.fixture
def mock_db_trades():
    return [
        {
            "id": 1,
            "run_date": "2026-01-10",
            "ticker": "AAPL",
            "action": "BUY",
            "quantity": 10.0,
            "fill_price": 150.0,
            "cost": 3.0,
            "net_pnl": 0.0,
            "slippage_cost": 0.5,
        },
        {
            "id": 2,
            "run_date": "2026-03-15",
            "ticker": "AAPL",
            "action": "SELL",
            "quantity": 10.0,
            "fill_price": 180.0,
            "cost": 3.6,
            "net_pnl": 292.9,
            "slippage_cost": 0.5,
        },
        {
            "id": 3,
            "run_date": "2025-01-05",
            "ticker": "MSFT",
            "action": "BUY",
            "quantity": 5.0,
            "fill_price": 300.0,
            "cost": 3.0,
            "net_pnl": 0.0,
            "slippage_cost": 0.0,
        },
        {
            "id": 4,
            "run_date": "2026-02-20",
            "ticker": "MSFT",
            "action": "SELL",
            "quantity": 5.0,
            "fill_price": 350.0,
            "cost": 3.5,
            "net_pnl": 243.5,
            "slippage_cost": 0.0,
        },
    ]


def test_get_closed_trades_fifo_matching(mock_db_trades):
    with patch("src.db.repository.get_trades", return_value=mock_db_trades):
        closed_2026 = get_closed_trades_for_year(2026)

        assert len(closed_2026) == 2

        aapl_trade = next(t for t in closed_2026 if t["ticker"] == "AAPL")
        assert aapl_trade["entry_date"] == "2026-01-10"
        assert aapl_trade["exit_date"] == "2026-03-15"
        assert aapl_trade["shares"] == 10.0
        assert aapl_trade["buy_price"] == 150.0
        assert aapl_trade["sell_price"] == 180.0
        assert aapl_trade["gross_pnl"] == 300.0
        assert aapl_trade["term"] == "Short"

        msft_trade = next(t for t in closed_2026 if t["ticker"] == "MSFT")
        assert msft_trade["entry_date"] == "2025-01-05"
        assert msft_trade["exit_date"] == "2026-02-20"
        assert msft_trade["holding_days"] > 365
        assert msft_trade["term"] == "Long"


def test_calculate_tax_breakdown_jurisdictions():
    trades = [
        {"net_pnl": 1000.0, "term": "Short"},
        {"net_pnl": 2000.0, "term": "Long"},
    ]

    india_tb = calculate_tax_breakdown(trades, "India")
    assert india_tb["stcg_pnl"] == 1000.0
    assert india_tb["ltcg_pnl"] == 2000.0
    # STCG: 1000 * 0.15 = 150. LTCG: (2000 - 1200) * 0.10 = 80. Total tax = 230
    assert india_tb["estimated_tax"] == 230.0

    usa_tb = calculate_tax_breakdown(trades, "USA")
    # STCG: 1000 * 0.24 = 240. LTCG: 2000 * 0.15 = 300. Total tax = 540
    assert usa_tb["estimated_tax"] == 540.0

    other_tb = calculate_tax_breakdown(trades, "Other")
    # Flat 15% on 3000 = 450
    assert other_tb["estimated_tax"] == 450.0


def test_generate_tax_report_data_structure(mock_db_trades):
    with patch("src.db.repository.get_trades", return_value=mock_db_trades):
        data = generate_tax_report_data(year=2026, country="India")

        assert "header" in data
        assert "annual_summary" in data
        assert "tax_breakdown" in data
        assert "monthly_breakdown" in data
        assert "trades" in data
        assert data["disclaimer"] == DISCLAIMER_TEXT
        assert data["header"]["year"] == 2026
        assert len(data["monthly_breakdown"]) == 12


def test_export_pdf_and_csv(tmp_path, mock_db_trades):
    with patch("src.db.repository.get_trades", return_value=mock_db_trades):
        report_data = generate_tax_report_data(year=2026, country="USA")

        pdf_path = tmp_path / "tax_2026.pdf"
        csv_path = tmp_path / "tax_2026.csv"

        out_pdf = export_tax_report_pdf(report_data, str(pdf_path))
        out_csv = export_tax_report_csv(report_data, str(csv_path))

        assert Path(out_pdf).exists()
        assert Path(out_csv).exists()
        assert Path(out_pdf).stat().st_size > 0
        assert Path(out_csv).stat().st_size > 0


def test_generate_tax_report_end_to_end(tmp_path, mock_db_trades):
    with patch("src.db.repository.get_trades", return_value=mock_db_trades):
        rep = generate_tax_report(year=2026, country="India", reports_dir=str(tmp_path))

        assert "pdf_path" in rep
        assert "csv_path" in rep
        assert Path(rep["pdf_path"]).exists()
        assert Path(rep["csv_path"]).exists()
