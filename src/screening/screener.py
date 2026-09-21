"""
src/screening/screener.py — Fundamental Screener Orchestration & Report Generator.

Runs the complete 5-pillar fundamental analysis across the curated universe,
sorts results into quality tiers, and generates human-readable markdown reports.

CLI Usage:
  python -m src.screening.screener --quick
  python -m src.screening.screener --output data/reports/fundamental_screen.md
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.screening.collector import fetch_ticker_fundamentals, fetch_universe_fundamentals
from src.screening.scorer import FundamentalScorecard, score_fundamentals
from src.screening.universe import SCREENER_UNIVERSE, get_screener_tickers

logger = logging.getLogger(__name__)


def run_fundamental_screen(
    tickers: Optional[List[str]] = None,
    fetch_date: Optional[str] = None,
    force_refresh: bool = False,
) -> List[FundamentalScorecard]:
    """
    Executes fundamental screening across the target universe.
    Returns list of scorecards sorted by composite score descending.
    """
    t_list = tickers or get_screener_tickers()
    logger.info("Executing fundamental screen across %d tickers...", len(t_list))

    funds = fetch_universe_fundamentals(
        tickers=t_list,
        fetch_date=fetch_date,
        force_refresh=force_refresh,
    )

    scorecards: List[FundamentalScorecard] = []
    for ticker, fund_data in funds.items():
        card = score_fundamentals(fund_data)
        scorecards.append(card)

    # Sort by score descending; UNRANKED placed at bottom
    scorecards.sort(
        key=lambda c: (-1 if c.status == "INSUFFICIENT_DATA" else c.composite_score),
        reverse=True,
    )
    return scorecards


def scorecards_to_dataframe(scorecards: List[FundamentalScorecard]) -> pd.DataFrame:
    """
    Converts list of scorecards into a structured DataFrame.
    Always includes the 5-pillar breakdown columns by default.
    """
    rows = []
    for sc in scorecards:
        rows.append({
            "ticker": sc.ticker,
            "company": sc.company_name,
            "sector": sc.sector,
            "score": sc.composite_score if sc.status != "INSUFFICIENT_DATA" else "N/A",
            "tier": sc.tier,
            "val_20": sc.pillar_scores["Valuation"].score,
            "prof_20": sc.pillar_scores["Profitability"].score,
            "solv_20": sc.pillar_scores["Solvency"].score,
            "cash_20": sc.pillar_scores["Cash Flow"].score,
            "grow_20": sc.pillar_scores["Growth"].score,
            "pe": sc.key_metrics.get("pe", "N/A"),
            "peg": sc.key_metrics.get("peg", "N/A"),
            "roe": sc.key_metrics.get("roe", "N/A"),
            "de": sc.key_metrics.get("de", "N/A"),
            "fcf": sc.key_metrics.get("fcf_b", "N/A"),
            "div_yield": sc.key_metrics.get("div_yield", "N/A"),
            "status": sc.status,
        })
    return pd.DataFrame(rows)


def generate_markdown_report(
    scorecards: List[FundamentalScorecard],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generates a structured, transparent markdown report for the fundamental screen.
    Includes upfront disclosures, summary tier distribution, and full 5-pillar breakdown.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(scorecards)
    t1 = sum(1 for s in scorecards if "Tier 1" in s.tier)
    t2 = sum(1 for s in scorecards if "Tier 2" in s.tier)
    t3 = sum(1 for s in scorecards if "Tier 3" in s.tier)
    unranked = sum(1 for s in scorecards if s.status == "INSUFFICIENT_DATA")

    lines = [
        "# Long-Term Fundamentals-Based Stock Screener Report",
        f"*Generated on: `{now_str}` | Total Universe Screened: `{total}`*",
        "",
        "> [!IMPORTANT]",
        "> **INFORMATIONAL PURPOSE & DISCLOSURES**:",
        "> 1. **Not Trading Advice**: This is a transparent, rule-based fundamental health checklist, not financial advice or an autonomous trade signal.",
        "> 2. **Survivorship & Selection Bias**: This screen runs across ~45 prominent US large-cap equities active today. It does not account for past bankruptcies or delistings.",
        "> 3. **Backward-Looking Data**: Financial ratios and TTM earnings describe past 12-month accounting data, not guaranteed future performance.",
        "> 4. **No Price Momentum / No Technical Timing**: High-scoring stocks may decline significantly in bear markets. This screen is completely agnostic to technical chart trends.",
        "> 5. **Zero Trading Pipeline Connection**: This tool does NOT interact with `risk_engine.py`, `portfolio.py`, `paper_broker.py`, or the daily scheduler.",
        "",
        "## Summary Tier Distribution",
        f"- **Tier 1 (High Quality / Solid Long-Term Fundamentals, 80–100 pts)**: **{t1}** ({t1/total*100:.1f}%)",
        f"- **Tier 2 (Moderate Quality / Watchlist, 60–79 pts)**: **{t2}** ({t2/total*100:.1f}%)",
        f"- **Tier 3 (Elevated Fundamental Risk, < 60 pts)**: **{t3}** ({t3/total*100:.1f}%)",
        f"- **Unranked (Insufficient Data)**: **{unranked}**",
        "",
        "## Comprehensive 5-Pillar Scorecard Table",
        "*Each pillar is scored out of 20 points (Max 100). Breakdown shown by default for transparent verification.*",
        "",
        "| Ticker | Company | Sector | Score | Tier | Val (20) | Prof (20) | Solv (20) | Cash (20) | Grow (20) | P/E | ROE | D/E | FCF |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for sc in scorecards:
        p = sc.pillar_scores
        score_str = f"**{sc.composite_score}**" if sc.status != "INSUFFICIENT_DATA" else "N/A"
        lines.append(
            f"| `{sc.ticker}` | {sc.company_name} | {sc.sector} | {score_str} | {sc.tier} | "
            f"{p['Valuation'].score} | {p['Profitability'].score} | {p['Solvency'].score} | {p['Cash Flow'].score} | {p['Growth'].score} | "
            f"{sc.key_metrics.get('pe', 'N/A')} | {sc.key_metrics.get('roe', 'N/A')} | {sc.key_metrics.get('de', 'N/A')} | {sc.key_metrics.get('fcf_b', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Pillar Scoring Methodology",
        "1. **Valuation & Pricing Discipline (20 pts)**: Non-financials rewarded for P/E < 25x and PEG < 2.0x; Banks rewarded for P/E < 16x and P/B < 2.0x.",
        "2. **Profitability & Moat (20 pts)**: Non-financials evaluated on ROE >= 15% and Operating Margin >= 15%; Banks evaluated on ROE >= 12% and ROA >= 1.0%.",
        "3. **Solvency & Balance Sheet (20 pts)**: Non-financials evaluated on Debt/Equity <= 100% and Current Ratio >= 1.2x; Banks evaluated on disciplined asset quality (ROA >= 1.1%).",
        "4. **Cash Flow Reality (20 pts)**: Non-financials rewarded for positive Free Cash Flow surplus; Banks evaluated on consistent capital formation.",
        "5. **Growth & Capital Stewardship (20 pts)**: Rewarded for YoY Revenue Growth >= 5% and prudent dividend payout ratio (<= 70%).",
    ])

    report_content = "\n".join(lines)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved fundamental screener report: %s", out_p)

    return report_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-Term Fundamentals-Based Stock Screener")
    parser.add_argument("--quick", action="store_true", help="Screen a quick sample of 5 diverse tickers.")
    parser.add_argument("--force", action="store_true", help="Force refresh fundamental data from yfinance.")
    parser.add_argument("--output", type=str, default=None, help="Output markdown report filepath.")
    parser.add_argument("--ticker", type=str, default=None, help="Screen a single ticker.")
    parser.add_argument("--tickers", type=str, default=None, help="Screen comma-separated list of tickers.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.tickers:
        target_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        target_tickers = [args.ticker.strip().upper()]
    elif args.quick:
        target_tickers = ["AAPL", "JNJ", "JPM", "PG", "XOM"]
    else:
        target_tickers = None

    cards = run_fundamental_screen(tickers=target_tickers, force_refresh=args.force)

    df = scorecards_to_dataframe(cards)
    print("\n" + "=" * 80)
    print("LONG-TERM FUNDAMENTALS-BASED STOCK SCREENER — RESULTS")
    print("=" * 80)
    print(df[["ticker", "sector", "score", "tier", "val_20", "prof_20", "solv_20", "cash_20", "grow_20", "pe", "roe", "fcf"]].to_string(index=False))

    out_file = args.output or f"data/reports/fundamental_screen_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    generate_markdown_report(cards, output_path=Path(out_file))
    print(f"\nMarkdown report written to: {out_file}")


if __name__ == "__main__":
    main()
