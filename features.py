"""Price-window transforms: RSI, SMA50, SMA100, quantized returns."""

from __future__ import annotations

import numpy as np


def rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta = np.diff(closes[-(period + 1) :])
    gains = np.clip(delta, 0.0, None)
    losses = np.clip(-delta, 0.0, None)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def sma(closes: np.ndarray, period: int) -> float:
    if len(closes) < period:
        return float(np.mean(closes))
    return float(np.mean(closes[-period:]))


def ma_digit(price: float, ma: float) -> str:
    """Map price vs moving average to a single digit 0-9 (5 = at the MA)."""
    if ma <= 1e-12:
        return "5"
    pct = (price / ma - 1.0) * 100.0
    return str(int(np.clip(round(pct + 5.0), 0, 9)))


def quantized_returns(closes: np.ndarray, n: int = 14) -> str:
    """Map the last n daily returns to digits 0-9 (5 ≈ 0%)."""
    if len(closes) < n + 1:
        return "5" * n
    rets = np.diff(closes[-(n + 1) :]) / np.maximum(closes[-(n + 1) : -1], 1e-10)
    digits = np.clip(np.round(rets * 100.0 + 5.0), 0, 9).astype(int)
    return "".join(str(int(d)) for d in digits)


def window_features(closes: np.ndarray) -> dict[str, float | str]:
    price = float(closes[-1])
    sma50 = sma(closes, 50)
    sma100 = sma(closes, 100)
    return {
        "price": price,
        "rsi": rsi(closes),
        "sma50": sma50,
        "sma100": sma100,
        "ma50_digit": ma_digit(price, sma50),
        "ma100_digit": ma_digit(price, sma100),
        "qrets": quantized_returns(closes),
    }
