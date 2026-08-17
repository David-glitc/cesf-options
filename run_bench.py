#!/usr/bin/env python3
"""Bench CESF vs raw Monte Carlo on realtime (or synthetic) data and paper-trade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cesf_filter import PortfolioGoal
from corpus import fetch_series, synthetic_series
from paper import walk_forward
from viz import write_html

OUT_JSON = HERE / "data" / "bench.json"
OUT_HTML = HERE / "viz" / "index.html"


def load_market(ticker: str, days: int, synthetic: bool) -> tuple[str, list[str], object]:
    if synthetic or ticker == "SYN":
        dates, closes = synthetic_series()
        return "SYN", dates, closes
    try:
        dates, closes = fetch_series(ticker, days=days)
        return ticker, dates, closes
    except Exception as exc:
        print(f"realtime fetch failed ({exc}); using synthetic GBM", file=sys.stderr)
        dates, closes = synthetic_series()
        return "SYN", dates, closes


def main() -> None:
    parser = argparse.ArgumentParser(description="CESF vs RAW paper bench + HTML")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=750)
    parser.add_argument("--paths", type=int, default=48)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--stride", type=int, default=21)
    parser.add_argument("--ev", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--strategy", default="putwrite", choices=("putwrite", "buy"))
    parser.add_argument("--ld", type=float, default=0.35, help="weight on |delta-delta*|")
    parser.add_argument("--lg", type=float, default=1.5, help="weight on |gamma-gamma*|")
    args = parser.parse_args()

    if args.strategy == "putwrite":
        goal = PortfolioGoal(
            ev_target=args.ev if args.ev != 0.02 else 0.01,
            delta_target=0.45 if args.delta == 0.25 else args.delta,
            gamma_target=-0.02 if args.gamma == 0.02 else args.gamma,
            lambda_delta=args.ld,
            lambda_gamma=args.lg,
        )
    else:
        goal = PortfolioGoal(
            ev_target=args.ev,
            delta_target=args.delta,
            gamma_target=args.gamma,
            lambda_delta=args.ld,
            lambda_gamma=args.lg,
        )
    ticker, dates, closes = load_market(args.ticker, args.days, args.synthetic)
    print(
        f"bench {ticker} bars={len(closes)} asof={dates[-1]} "
        f"spot={float(closes[-1]):.2f} paths={args.paths}",
        flush=True,
    )
    report = walk_forward(
        dates,
        closes,
        goal,
        ticker,
        n_paths=args.paths,
        lookback=args.lookback,
        stride=args.stride,
        strategy=args.strategy,
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_html(report, OUT_HTML)
    print(
        f"CESF pnl={report['cesf']['pnl']} wr={report['cesf']['win_rate']:.2f} "
        f"bits={report['mean_cesf_bits']} | "
        f"RAW pnl={report['raw']['pnl']} wr={report['raw']['win_rate']:.2f} | "
        f"agree={report['agreement_rate']:.2f}",
        flush=True,
    )
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"open  {OUT_HTML}", flush=True)


if __name__ == "__main__":
    main()
