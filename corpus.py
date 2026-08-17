"""Build a microgpt-c corpus from historical windows + CESF labels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cesf_filter import PortfolioGoal, calibrate_gbm, run_underlying_cesf, select_contract
from encode import encode_contract_line, encode_transform_line
from features import window_features


def synthetic_closes(
    n: int = 400,
    s0: float = 100.0,
    mu: float = 0.10,
    sigma: float = 0.25,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    shocks = rng.standard_normal(n - 1)
    log_rets = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * shocks
    return np.concatenate([[s0], s0 * np.exp(np.cumsum(log_rets))])


def synthetic_series(
    n: int = 400,
    s0: float = 100.0,
    mu: float = 0.10,
    sigma: float = 0.25,
    seed: int = 7,
) -> tuple[list[str], np.ndarray]:
    from datetime import datetime, timedelta

    closes = synthetic_closes(n=n, s0=s0, mu=mu, sigma=sigma, seed=seed)
    end = datetime.now().date()
    dates = [(end - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]
    return dates, closes


def fetch_series(ticker: str, days: int = 750) -> tuple[list[str], np.ndarray]:
    import yfinance as yf
    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    close_col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if hasattr(close_col, "squeeze"):
        close_col = close_col.squeeze()
    closes = np.asarray(close_col, dtype=float).reshape(-1)
    idx = df.index
    dates = [str(d)[:10] for d in idx]
    if len(dates) != len(closes):
        dates = [str(d)[:10] for d in idx[: len(closes)]]
    return dates, closes


def fetch_closes(ticker: str, days: int = 750) -> np.ndarray:
    _dates, closes = fetch_series(ticker, days=days)
    return closes


def iter_windows(closes: np.ndarray, lookback: int = 120, stride: int = 5):
    for end in range(lookback, len(closes) + 1, stride):
        yield closes[:end][-lookback:]


def build_corpus_lines(
    closes: np.ndarray,
    ticker: str,
    goal: PortfolioGoal,
    n_paths: int = 250,
    lookback: int = 120,
    stride: int = 5,
    include_transforms: bool = True,
) -> list[str]:
    lines: list[str] = []
    for i, window in enumerate(iter_windows(closes, lookback=lookback, stride=stride)):
        feats = window_features(window)
        result = run_underlying_cesf(window, n_paths=n_paths, seed=42 + i)
        system = calibrate_gbm(window, seed=42 + i)
        chosen = select_contract(result, sigma=system.sigma, goal=goal)
        lines.append(
            encode_contract_line(
                ticker,
                feats,
                result.complexity_bits,
                goal.ev_target,
                goal.delta_target,
                goal.gamma_target,
                chosen.contract,
            )
        )
        if include_transforms:
            lines.append(encode_transform_line(window))
    return lines


def write_corpus(path: Path, lines: list[str], contract_repeats: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contracts = [ln for ln in lines if not ln.startswith("W")]
    transforms = [ln for ln in lines if ln.startswith("W")]
    expanded = contracts * max(1, contract_repeats) + transforms
    path.write_text("\n".join(expanded) + "\n", encoding="utf-8")
