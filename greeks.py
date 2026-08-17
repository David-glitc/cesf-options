"""Black-Scholes price, delta, gamma, and path payoffs."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _d1_d2(spot: float, strike: float, t_years: float, rate: float, sigma: float) -> tuple[float, float]:
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return 0.0, 0.0
    vol_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(kind: str, spot: float, strike: float, t_years: float, rate: float, sigma: float) -> float:
    d1, d2 = _d1_d2(spot, strike, t_years, rate, sigma)
    df = math.exp(-rate * t_years)
    if kind == "C":
        return float(spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
    return float(strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))


def bs_delta(kind: str, spot: float, strike: float, t_years: float, rate: float, sigma: float) -> float:
    d1, _ = _d1_d2(spot, strike, t_years, rate, sigma)
    nd1 = float(norm.cdf(d1))
    return nd1 if kind == "C" else nd1 - 1.0


def bs_gamma(spot: float, strike: float, t_years: float, rate: float, sigma: float) -> float:
    d1, _ = _d1_d2(spot, strike, t_years, rate, sigma)
    denom = spot * sigma * math.sqrt(t_years)
    if denom <= 1e-12:
        return 0.0
    return float(norm.pdf(d1) / denom)


def option_payoff(kind: str, strike: float, terminals: np.ndarray) -> np.ndarray:
    if kind == "C":
        return np.maximum(terminals - strike, 0.0)
    return np.maximum(strike - terminals, 0.0)
