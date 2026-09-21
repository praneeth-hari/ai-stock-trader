"""
src/reports/tax_report.py — Section 10 Item 3: Tax Report Generator.

Generates annual trade and tax reports for paper trading activity
in downloadable PDF and CSV formats.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.db import repository

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "This report is generated from a paper trading simulation using fake money. "
    "No real financial transactions occurred. No real taxes are owed. "
    "This report is for educational purposes only. Consult a qualified tax professional for real tax advice."
)


def get_closed_trades_for_year(year: int) -> List[Dict[str, Any]]:
    """
    Retrieves and FIFO-matches closed trades for a given calendar year from DB trade records.
    """
    raw_trades = repository.get_trades(limit=10000)

    open_buys: Dict[str, List[Dict[str, Any]]] = {}
    closed_trades: List[Dict[str, Any]] = []

    sorted_trades = sorted(raw_trades, key=lambda t: (t.get("run_date", ""), t.get("id", 0)))

    for t in sorted_trades:
        ticker = str(t.get("ticker", "")).upper()
        action = str(t.get("action", "")).upper()
        t_date = str(t.get("run_date", ""))
        qty = float(t.get("quantity", 0.0))
        price = float(t.get("fill_price", 0.0))
        fee = float(t.get("cost", 0.0)) + float(t.get("slippage_cost", 0.0))

        if action == "BUY":
            if ticker not in open_buys:
                open_buys[ticker] = []
            open_buys[ticker].append({
                "date": t_date,
                "price": price,
                "qty": qty,
                "fee": fee,
            })
        elif action == "SELL":
            if ticker in open_buys and open_buys[ticker]:
                rem_qty = qty
                while rem_qty > 1e-6 and open_buys[ticker]:
                    buy = open_buys[ticker][0]
                    matched_qty = min(rem_qty, buy["qty"])
                    buy_price = buy["price"]
                    buy_date = buy["date"]
                    buy_fee_portion = buy["fee"] * (matched_qty / buy["qty"]) if buy["qty"] > 0 else 0.0
                    sell_fee_portion = fee * (matched_qty / qty) if qty > 0 else 0.0
                    total_fees = buy_fee_portion + sell_fee_portion

                    gross_pnl = matched_qty * (price - buy_price)
                    net_pnl = gross_pnl - total_fees

                    try:
                        d1 = datetime.datetime.strptime(buy_date, "%Y-%m-%d")
                        d2 = datetime.datetime.strptime(t_date, "%Y-%m-%d")
                        holding_days = max(1, (d2 - d1).days)
                    except Exception:
                        holding_days = 1

                    term = "Long" if holding_days >= 365 else "Short"

                    if t_date.startswith(str(year)):
                        closed_trades.append({
                            "entry_date": buy_date,
                            "exit_date": t_date,
                            "ticker": ticker,
                            "shares": round(matched_qty, 4),
                            "buy_price": round(buy_price, 2),
                            "sell_price": round(price, 2),
                            "gross_pnl": round(gross_pnl, 2),
                            "fees": round(total_fees, 2),
                            "net_pnl": round(net_pnl, 2),
                            "holding_days": holding_days,
                            "term": term,
                        })

                    buy["qty"] -= matched_qty
                    buy["fee"] -= buy_fee_portion
                    rem_qty -= matched_qty

                    if buy["qty"] <= 1e-6:
                        open_buys[ticker].pop(0)

    return closed_trades


def calculate_tax_breakdown(closed_trades: List[Dict[str, Any]], country: str) -> Dict[str, Any]:
    """
    Computes Capital Gains Breakdown for India, USA, or Other jurisdictions.
    """
    stcg_pnl = sum(t["net_pnl"] for t in closed_trades if t["term"] == "Short")
    ltcg_pnl = sum(t["net_pnl"] for t in closed_trades if t["term"] == "Long")
    total_net_pnl = sum(t["net_pnl"] for t in closed_trades)

    country_upper = country.upper()

    if country_upper == "INDIA":
        stcg_tax = max(0.0, stcg_pnl * 0.15) if stcg_pnl > 0 else 0.0
        taxable_ltcg = max(0.0, ltcg_pnl - 1200.0) if ltcg_pnl > 0 else 0.0
        ltcg_tax = taxable_ltcg * 0.10
        est_tax = stcg_tax + ltcg_tax
        rules_desc = [
            f"Short Term Capital Gains (< 1 year): ${stcg_pnl:+.2f} — Taxed at 15% (Est Tax: ${stcg_tax:.2f})",
            f"Long Term Capital Gains (>= 1 year): ${ltcg_pnl:+.2f} — Taxed at 10% above $1,200 exemption (Est Tax: ${ltcg_tax:.2f})",
        ]
    elif country_upper == "USA":
        stcg_tax = max(0.0, stcg_pnl * 0.24) if stcg_pnl > 0 else 0.0
        ltcg_tax = max(0.0, ltcg_pnl * 0.15) if ltcg_pnl > 0 else 0.0
        est_tax = stcg_tax + ltcg_tax
        rules_desc = [
            f"Short Term Capital Gains (< 1 year): ${stcg_pnl:+.2f} — Taxed at ordinary income rate ~24% (Est Tax: ${stcg_tax:.2f})",
            f"Long Term Capital Gains (>= 1 year): ${ltcg_pnl:+.2f} — Taxed at preferential rate 15% (Est Tax: ${ltcg_tax:.2f})",
        ]
    else:  # OTHER
        est_tax = max(0.0, total_net_pnl * 0.15) if total_net_pnl > 0 else 0.0
        rules_desc = [
            f"Standard Capital Gains: ${total_net_pnl:+.2f} — Estimated flat tax rate ~15% (Est Tax: ${est_tax:.2f})"
        ]

    return {
        "country": country,
        "stcg_pnl": round(stcg_pnl, 2),
        "ltcg_pnl": round(ltcg_pnl, 2),
        "estimated_tax": round(est_tax, 2),
        "rules_description": rules_desc,
    }


def generate_tax_report_data(year: int, country: str = "India") -> Dict[str, Any]:
    """
    Generates all structured report data for a given year and country.
    """
    closed_trades = get_closed_trades_for_year(year)

    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t["net_pnl"] > 0]
    losing_trades = [t for t in closed_trades if t["net_pnl"] <= 0]

    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    win_rate_pct = (winning_count / total_trades * 100.0) if total_trades > 0 else 0.0
    loss_rate_pct = (losing_count / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_gains = sum(t["gross_pnl"] for t in winning_trades)
    gross_losses = sum(t["gross_pnl"] for t in losing_trades)
    total_fees = sum(t["fees"] for t in closed_trades)
    net_pnl = sum(t["net_pnl"] for t in closed_trades)

    tax_breakdown = calculate_tax_breakdown(closed_trades, country)

    # Monthly breakdown table
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_rows = []
    for m in range(1, 13):
        m_str = f"{m:02d}"
        m_trades = [t for t in closed_trades if t["exit_date"].startswith(f"{year}-{m_str}")]
        m_pnl = sum(t["net_pnl"] for t in m_trades)
        monthly_rows.append({
            "Month": month_names[m - 1],
            "Net PnL ($)": round(m_pnl, 2),
            "Trades Count": len(m_trades),
        })

    return {
        "header": {
            "title": "AI Stock Trader — Annual Trade Report",
            "year": year,
            "generated_date": datetime.date.today().strftime("%Y-%m-%d"),
            "system": "Paper Trading (Simulated)",
            "note": "This is a paper trading simulation. No real taxes are owed.",
        },
        "annual_summary": {
            "total_trades": total_trades,
            "winning_trades_count": winning_count,
            "winning_trades_pct": round(win_rate_pct, 1),
            "losing_trades_count": losing_count,
            "losing_trades_pct": round(loss_rate_pct, 1),
            "gross_gains": round(gross_gains, 2),
            "gross_losses": round(gross_losses, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
        },
        "tax_breakdown": tax_breakdown,
        "monthly_breakdown": monthly_rows,
        "trades": closed_trades,
        "disclaimer": DISCLAIMER_TEXT,
    }


def export_tax_report_csv(report_data: Dict[str, Any], output_path: str) -> str:
    """
    Exports the report to CSV format at output_path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trades = report_data.get("trades", [])
    if not trades:
        df = pd.DataFrame(columns=[
            "Date Bought", "Date Sold", "Ticker", "Quantity",
            "Buy Price", "Sell Price", "Gross P&L ($)", "Fees/Slippage ($)",
            "Net P&L ($)", "Holding Period (days)", "Term"
        ])
    else:
        df = pd.DataFrame(trades).rename(columns={
            "entry_date": "Date Bought",
            "exit_date": "Date Sold",
            "ticker": "Ticker",
            "shares": "Quantity",
            "buy_price": "Buy Price",
            "sell_price": "Sell Price",
            "gross_pnl": "Gross P&L ($)",
            "fees": "Fees/Slippage ($)",
            "net_pnl": "Net P&L ($)",
            "holding_days": "Holding Period (days)",
            "term": "Term",
        })

    df.to_csv(path, index=False)
    return str(path)


def export_tax_report_pdf(report_data: Dict[str, Any], output_path: str) -> str:
    """
    Exports the tax report to PDF format using ReportLab.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e222d"),
        spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2962ff"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#222222"),
    )
    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#555555"),
    )

    elements = []

    # 1. Header Section
    h = report_data["header"]
    elements.append(Paragraph(h["title"], title_style))
    elements.append(Paragraph(f"<b>Year:</b> {h['year']} | <b>Generated:</b> {h['generated_date']} | <b>System:</b> {h['system']}", body_style))
    elements.append(Paragraph(f"<i>Note: {h['note']}</i>", small_style))
    elements.append(Spacer(1, 10))

    # 2. Annual Summary Section
    elements.append(Paragraph("Annual Performance Summary", section_style))
    s = report_data["annual_summary"]
    summary_table_data = [
        ["Total Trades", str(s["total_trades"]), "Gross Gains", f"+${s['gross_gains']:,.2f}"],
        ["Winning Trades", f"{s['winning_trades_count']} ({s['winning_trades_pct']}%)", "Gross Losses", f"-${abs(s['gross_losses']):,.2f}"],
        ["Losing Trades", f"{s['losing_trades_count']} ({s['losing_trades_pct']}%)", "Total Fees", f"-${abs(s['total_fees']):,.2f}"],
        ["Net P&L", f"{'+' if s['net_pnl'] > 0 else ''}${s['net_pnl']:,.2f}", "", ""],
    ]
    t_summary = Table(summary_table_data, colWidths=[130, 140, 130, 140])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f7fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#111111')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 10))

    # 3. Capital Gains Breakdown Section
    elements.append(Paragraph("Capital Gains & Tax Estimate Breakdown", section_style))
    tb = report_data["tax_breakdown"]
    for r in tb["rules_description"]:
        elements.append(Paragraph(f"• {r}", body_style))
    elements.append(Paragraph(f"<b>Estimated Paper Tax Owed ({tb['country']}):</b> ${tb['estimated_tax']:,.2f}", body_style))
    elements.append(Spacer(1, 10))

    # 4. Monthly Breakdown Section
    elements.append(Paragraph("Monthly P&L Summary", section_style))
    mb_data = [["Month", "Net P&L ($)", "Trades Count"]]
    for m in report_data["monthly_breakdown"]:
        pnl_str = f"{'+' if m['Net PnL ($)'] > 0 else ''}${m['Net PnL ($)']:,.2f}"
        mb_data.append([m["Month"], pnl_str, str(m["Trades Count"])])
    t_mb = Table(mb_data, colWidths=[100, 150, 100])
    t_mb.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2962ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_mb)
    elements.append(Spacer(1, 10))

    # 5. Trade Summary Table Section
    elements.append(Paragraph("Trade Details Summary", section_style))
    trades = report_data["trades"]
    if trades:
        trade_table_data = [["Bought", "Sold", "Ticker", "Qty", "Buy Price", "Sell Price", "Gross PnL", "Fees", "Net PnL", "Days", "Term"]]
        for tr in trades[:100]:  # Cap at top 100 in PDF
            trade_table_data.append([
                tr["entry_date"], tr["exit_date"], tr["ticker"], f"{tr['shares']:.2f}",
                f"${tr['buy_price']:.2f}", f"${tr['sell_price']:.2f}",
                f"${tr['gross_pnl']:+.2f}", f"${tr['fees']:.2f}",
                f"${tr['net_pnl']:+.2f}", str(tr["holding_days"]), tr["term"]
            ])
        t_trades = Table(trade_table_data, colWidths=[48, 48, 40, 35, 45, 45, 48, 40, 48, 32, 36])
        t_trades.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e222d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(t_trades)
    else:
        elements.append(Paragraph("<i>No closed trades recorded for this year.</i>", small_style))

    elements.append(Spacer(1, 12))

    # 6. Disclaimer Section
    elements.append(Paragraph("<b>DISCLAIMER</b>", small_style))
    elements.append(Paragraph(report_data["disclaimer"], small_style))

    doc.build(elements)
    return str(path)


def generate_tax_report(year: int, country: str = "India", reports_dir: str = "reports") -> Dict[str, Any]:
    """
    Main entrypoint: Generates data and exports PDF + CSV report files.
    """
    report_data = generate_tax_report_data(year=year, country=country)

    dir_path = Path(reports_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    pdf_path = str(dir_path / f"tax_{year}.pdf")
    csv_path = str(dir_path / f"tax_{year}.csv")

    export_tax_report_pdf(report_data, pdf_path)
    export_tax_report_csv(report_data, csv_path)

    report_data["pdf_path"] = pdf_path
    report_data["csv_path"] = csv_path

    return report_data
