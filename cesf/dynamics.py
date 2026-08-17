"""Stochastic dynamics for generating theoretical possibility space Ω_H."""

from __future__ import annotations

import numpy as np
from scipy import stats


class GBMSystem:
    """Geometric Brownian motion: S = (X, F, μ₀)."""

    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.10,
        sigma: float = 0.25,
        seed: int = 42,
    ):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.seed = seed

    def simulate_paths(
        self,
        horizon: int,
        n_paths: int,
        dt: float = 1.0 / 252,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        paths = np.zeros((n_paths, horizon + 1))
        paths[:, 0] = self.s0
        increments = rng.standard_normal((n_paths, horizon))
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * increments
        paths[:, 1:] = self.s0 * np.exp(np.cumsum(drift + diffusion, axis=1))
        return paths

    def intervene_price_shock(
        self,
        paths: np.ndarray,
        t_int: int,
        factor: float,
        seed: int = 999,
    ) -> np.ndarray:
        """do(X_t ← x') — multiplicative shock at t_int, evolve forward."""
        intervened = paths.copy()
        dt = 1.0 / 252
        rng = np.random.default_rng(seed)
        n_paths, n_steps = paths.shape
        remaining = n_steps - t_int
        increments = rng.standard_normal((n_paths, remaining))
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * increments
        intervened[:, t_int] = paths[:, t_int] * factor
        intervened[:, t_int + 1 :] = intervened[:, t_int].reshape(-1, 1) * np.exp(
            np.cumsum(drift + diffusion, axis=1)
        )
        return intervened


class JumpDiffusionSystem:
    """Merton jump-diffusion calibrated from historical returns."""

    def __init__(self, returns: np.ndarray, name: str = "", seed: int = 42):
        self.name = name
        self.returns = returns
        self.mean_return = float(np.mean(returns))
        self.volatility = float(np.std(returns))
        self.skewness = float(stats.skew(returns))
        self.kurtosis = float(stats.kurtosis(returns, fisher=True))
        self.seed = seed
        self._calibrate_jumps(returns)

    def _calibrate_jumps(self, returns: np.ndarray) -> None:
        excess_kurtosis = max(self.kurtosis, 0.0)
        if excess_kurtosis < 3:
            self.jump_lambda = 0.02
            self.jump_mean = 0.0
            self.jump_std = 0.0
        else:
            self.jump_lambda = min(0.10, excess_kurtosis / 50.0)
            self.jump_mean = self.skewness * self.volatility * 0.3
            self.jump_std = self.volatility * np.sqrt(excess_kurtosis / 6.0)

    def simulate_paths(
        self,
        horizon: int,
        n_paths: int,
        start_price: float = 100.0,
        dt: float = 1.0 / 252,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        paths = np.zeros((n_paths, horizon + 1))
        paths[:, 0] = start_price

        sigma = self.volatility
        mu = self.mean_return
        lam = self.jump_lambda
        z_norm = rng.standard_normal((n_paths, horizon))

        if self.jump_lambda > 0.01:
            k = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1
            drift_adj = (mu - 0.5 * sigma**2 - lam * k) * dt
            diffusion = sigma * np.sqrt(dt) * z_norm
            jump_indicator = rng.poisson(lam, (n_paths, horizon))
            jump_magnitude = rng.normal(self.jump_mean, self.jump_std, (n_paths, horizon))
            log_returns = drift_adj + diffusion + jump_indicator * jump_magnitude
        else:
            drift = (mu - 0.5 * sigma**2) * dt
            log_returns = drift + sigma * np.sqrt(dt) * z_norm

        paths[:, 1:] = start_price * np.exp(np.cumsum(log_returns, axis=1))
        return paths

    @classmethod
    def from_returns(cls, returns: np.ndarray, name: str = "", seed: int = 42) -> JumpDiffusionSystem:
        return cls(returns, name=name, seed=seed)
