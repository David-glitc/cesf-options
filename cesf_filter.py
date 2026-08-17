"""CESF contraction of Monte Carlo futures, then option-contract selection.

Underlying paths are reduced to E_H(Q). Candidate contracts are scored on
that operational event space against portfolio targets (EV, delta, gamma).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _candidate in (HERE, *HERE.parents):
    if (_candidate / "cesf").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from cesf.dynamics import GBMSystem
from cesf.pipeline import CESFPipeline
from cesf.types import AdmissibilityConstraints, OperationalQuery, RelevanceConfig

from contracts import Contract, candidate_grid
from greeks import bs_delta, bs_gamma, bs_price, option_payoff


@dataclass
class PortfolioGoal:
    ev_target: float = 0.02
    delta_target: float = 0.25
    gamma_target: float = 0.02
    lambda_delta: float = 2.0
    lambda_gamma: float = 12.0
    rate: float = 0.04


@dataclass
class ContractScore:
    contract: Contract
    ev_pnl_frac: float
    delta: float
    gamma: float
    premium: float
    score: float


def calibrate_gbm(closes: np.ndarray, seed: int = 42) -> GBMSystem:
    rets = np.diff(np.log(np.maximum(closes, 1e-8)))
    mu = float(np.mean(rets) * 252.0)
    sigma = float(max(np.std(rets) * np.sqrt(252.0), 0.05))
    return GBMSystem(s0=float(closes[-1]), mu=mu, sigma=sigma, seed=seed)


def default_query(horizon: int = 21) -> OperationalQuery:
    return OperationalQuery(
        epsilon=0.088,
        horizon=horizon,
        constraints=AdmissibilityConstraints(max_drawdown=0.60),
        relevance=RelevanceConfig(task="option_ev", barrier=0.80, eta=0.5),
    )


def run_underlying_cesf(
    closes: np.ndarray,
    n_paths: int = 400,
    query: OperationalQuery | None = None,
    seed: int = 42,
):
    system = calibrate_gbm(closes, seed=seed)
    q = query or default_query()
    q.horizon = min(q.horizon, 42)
    return CESFPipeline(system, q, s0=system.s0).run(n_paths=n_paths)


def class_terminals(result) -> tuple[np.ndarray, np.ndarray]:
    """All admissible terminals, probability-weighted.

    Do not collapse a class to its mean price before the payoff. Put/call
    payoffs are convex, so payoff(E[S]) understates E[payoff] and makes
    short premium look better than it is — that is why CESF lagged RAW.
    Equivalence is still used for crash_mass / skip, not for Jensen-biased EV.
    """
    terms = result.adm_paths[:, -1]
    n = max(len(terms), 1)
    w = np.full(n, 1.0 / n)
    return terms, w


def crash_mass(result, barrier: float = 0.80) -> float:
    """Probability mass of classes whose mean terminal is below barrier * S0."""
    if not result.components:
        return 0.0
    s0 = float(result.adm_paths[0, 0])
    n = max(len(result.adm_paths), 1)
    mass = 0.0
    for comp in result.components:
        if float(np.mean(result.adm_paths[comp, -1])) < barrier * s0:
            mass += len(comp) / n
    return float(mass)


def _event_terminals(result) -> tuple[np.ndarray, np.ndarray]:
    """Legacy downside E_H(Q) terminals. Prefer class_terminals for option EV."""
    if not result.events:
        return class_terminals(result)
    terms = []
    weights = []
    n = len(result.adm_paths)
    for comp, _score in result.events:
        pc = result.adm_paths[comp, -1]
        terms.append(pc)
        weights.append(np.full(len(pc), len(comp) / n))
    terminals = np.concatenate(terms)
    w = np.concatenate(weights)
    w = w / max(float(np.sum(w)), 1e-12)
    return terminals, w


def score_contract_from_terminals(
    contract: Contract,
    s0: float,
    terminals: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    goal: PortfolioGoal,
) -> ContractScore:
    strike = contract.strike(s0)
    t_years = contract.t_years()
    premium = bs_price(contract.kind, s0, strike, t_years, goal.rate, sigma)
    delta = bs_delta(contract.kind, s0, strike, t_years, goal.rate, sigma)
    gamma = bs_gamma(s0, strike, t_years, goal.rate, sigma)
    payoffs = option_payoff(contract.kind, strike, terminals)
    expected_payoff = float(np.sum(weights * payoffs))
    signed = contract.qty * (expected_payoff - premium)
    pnl_frac = signed / max(s0, 1e-8)
    pos_delta = contract.qty * delta
    pos_gamma = contract.qty * gamma
    score = (
        -abs(pnl_frac - goal.ev_target)
        - goal.lambda_delta * abs(pos_delta - goal.delta_target)
        - goal.lambda_gamma * abs(pos_gamma - goal.gamma_target)
    )
    return ContractScore(
        contract=contract,
        ev_pnl_frac=pnl_frac,
        delta=pos_delta,
        gamma=pos_gamma,
        premium=premium,
        score=score,
    )


def select_from_terminals(
    s0: float,
    terminals: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    goal: PortfolioGoal,
    grid: list[Contract] | None = None,
) -> ContractScore:
    best: ContractScore | None = None
    for contract in grid or candidate_grid():
        scored = score_contract_from_terminals(
            contract, s0, terminals, weights, sigma, goal
        )
        if best is None or scored.score > best.score:
            best = scored
    assert best is not None
    return best


def score_contract(
    contract: Contract,
    result,
    sigma: float,
    goal: PortfolioGoal,
) -> ContractScore:
    s0 = float(result.adm_paths[0, 0])
    terminals, weights = class_terminals(result)
    return score_contract_from_terminals(
        contract, s0, terminals, weights, sigma, goal
    )


def select_contract(
    result,
    sigma: float,
    goal: PortfolioGoal,
    grid: list[Contract] | None = None,
) -> ContractScore:
    s0 = float(result.adm_paths[0, 0])
    terminals, weights = class_terminals(result)
    return select_from_terminals(s0, terminals, weights, sigma, goal, grid=grid)
