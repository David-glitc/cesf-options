"""Two-pass relevance functional R_Q = P × (I + M + η·D)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RelevanceScore:
    R: float
    P: float
    I: float
    M: float
    D: float
    C: float


class RelevanceFunctional:
    """
    R_Q = F_Q(P, I, C, D) specialized for downside-risk assessment.

    P: class probability mass (path count)
    I: maximum drawdown within class
    M: barrier breach severity
    D: terminal displacement persistence
    """

    def __init__(self, barrier: float = 0.80, eta: float = 0.5):
        self.barrier = barrier
        self.eta = eta

    def proxy(self, paths_class: np.ndarray, s0: float) -> float:
        """Pass 1: cheap terminal-price proxy."""
        return max(0.0, (s0 - float(np.mean(paths_class[:, -1]))) / s0)

    def exact(self, paths_class: np.ndarray, s0: float) -> RelevanceScore:
        """Pass 2: full-horizon relevance."""
        p = float(len(paths_class))
        rmax = np.maximum.accumulate(paths_class, axis=1)
        dd = (rmax - paths_class) / np.maximum(rmax, 1e-10)
        i = float(np.max(dd))
        breach = np.maximum(0, self.barrier * s0 - np.min(paths_class, axis=1))
        m = float(np.mean(breach) / s0)
        d = float(np.mean(np.abs(paths_class[:, -1] - s0)) / s0)
        c = float(np.var(paths_class[:, -1]) / (s0**2))
        r = p * (i + m + self.eta * d)
        return RelevanceScore(R=r, P=p, I=i, M=m, D=d, C=c)
