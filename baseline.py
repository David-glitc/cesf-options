"""Controlled non-CESF parallel: same GBM, same C, same grid, same goal.

CESF scores contracts on E_H(Q). This baseline scores the identical
admissible Monte Carlo terminals with uniform path weights (raw Ω_H ∩ C).
No ε-graph, no relevance filter, no operational events.
"""

from __future__ import annotations

import numpy as np

from cesf_filter import (
    PortfolioGoal,
    calibrate_gbm,
    default_query,
    select_from_terminals,
)
from cesf.pipeline import CESFPipeline


def admissible_paths(closes: np.ndarray, n_paths: int, seed: int):
    """Same generator and admissibility C as CESFPipeline; no equivalence step."""
    system = calibrate_gbm(closes, seed=seed)
    query = default_query()
    pipeline = CESFPipeline(system, query, s0=system.s0)
    paths = system.simulate_paths(query.horizon, n_paths)
    mask = pipeline._admissible(paths)
    adm = paths[mask]
    if len(adm) < 10:
        raise ValueError(f"Too few admissible paths: {len(adm)}")
    return system, adm


def select_contract_raw(
    closes: np.ndarray,
    n_paths: int,
    seed: int,
    goal: PortfolioGoal,
):
    system, adm = admissible_paths(closes, n_paths, seed)
    terminals = adm[:, -1]
    weights = np.full(len(terminals), 1.0 / len(terminals))
    scored = select_from_terminals(
        float(adm[0, 0]),
        terminals,
        weights,
        system.sigma,
        goal,
    )
    return scored, system
