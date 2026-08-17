"""Full CESF reduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cesf.dynamics import GBMSystem, JumpDiffusionSystem
from cesf.equivalence import (
    build_epsilon_graph,
    distance_matrix,
    divergence_analysis,
    find_components,
)
from cesf.metrics import (
    categorize_risks,
    compression_ratio,
    operational_complexity_bits,
)
from cesf.relevance import RelevanceFunctional, RelevanceScore
from cesf.types import OperationalQuery


@dataclass
class PipelineResult:
    omega_size: int
    gamma_size: int
    n_classes: int
    n_events: int
    complexity_bits: float
    compression: float
    persistence_rate: float
    paths: np.ndarray
    adm_paths: np.ndarray
    components: list[list[int]]
    events: list[tuple[list[int], float]]
    scores: list[tuple[list[int], float]]
    categories: dict[str, int]
    class_metrics: list[dict[str, Any]] = field(default_factory=list)
    divergence: dict[str, int | float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "omega_size": self.omega_size,
            "gamma_size": self.gamma_size,
            "n_classes": self.n_classes,
            "n_events": self.n_events,
            "complexity_bits": round(self.complexity_bits, 3),
            "compression": round(self.compression, 1),
            "persistence_rate": round(self.persistence_rate, 4),
            "categories": self.categories,
        }


class CESFPipeline:
    def __init__(
        self,
        system: GBMSystem | JumpDiffusionSystem,
        query: OperationalQuery,
        s0: float = 100.0,
    ):
        self.system = system
        self.query = query
        self.s0 = s0
        self.relevance = RelevanceFunctional(
            barrier=query.relevance.barrier,
            eta=query.relevance.eta,
        )

    def run(self, n_paths: int = 2000, verbose: bool = False) -> PipelineResult:
        q = self.query
        paths = self.system.simulate_paths(q.horizon, n_paths)
        mask = self._admissible(paths)
        adm = paths[mask]
        if len(adm) < 10:
            raise ValueError(f"Too few admissible paths: {len(adm)}")

        dist = distance_matrix(adm)
        graph = build_epsilon_graph(epsilon=q.epsilon, dist_matrix=dist)
        components = find_components(graph, len(adm))
        div = divergence_analysis(components, len(adm))

        proxies = [(c, self.relevance.proxy(adm[c], self.s0)) for c in components]
        theta_tilde = float(np.percentile([s for _, s in proxies], q.relevance.proxy_percentile))
        retained = [(c, s) for c, s in proxies if s >= theta_tilde]

        exacts = [(c, self.relevance.exact(adm[c], self.s0)) for c, _ in retained]
        normed = [(c, e.R / len(adm)) for c, e in exacts]

        score_vals = [s for _, s in normed]
        theta_r = float(np.percentile(score_vals, q.relevance.event_percentile)) if score_vals else 0.0
        events = [(c, s) for c, s in normed if s >= theta_r]
        categories = categorize_risks(normed, len(adm), theta_r)

        n_cls = len(components)
        class_metrics = []
        for i, (c, score) in enumerate(normed):
            pc = adm[c]
            rmax = np.maximum.accumulate(pc, axis=1)
            dd = (rmax - pc) / np.maximum(rmax, 1e-10)
            class_metrics.append({
                "class_id": i,
                "size": len(c),
                "probability": len(c) / len(adm),
                "relevance_score": score,
                "mean_terminal": float(np.mean(pc[:, -1])),
                "max_drawdown": float(np.max(dd)),
            })

        result = PipelineResult(
            omega_size=n_paths,
            gamma_size=len(adm),
            n_classes=n_cls,
            n_events=len(events),
            complexity_bits=operational_complexity_bits(n_cls),
            compression=compression_ratio(n_paths, n_cls),
            persistence_rate=float(div["persistence_rate"]),
            paths=paths,
            adm_paths=adm,
            components=components,
            events=events,
            scores=normed,
            categories=categories,
            class_metrics=class_metrics,
            divergence=div,
        )
        if verbose:
            for k, v in result.summary().items():
                print(f"{k}: {v}")
        return result

    def _admissible(self, paths: np.ndarray) -> np.ndarray:
        c = self.query.constraints
        mask = np.ones(len(paths), dtype=bool)
        if c.no_neg_price:
            mask &= np.all(paths > 0, axis=1)
        if c.max_drawdown is not None:
            rmax = np.maximum.accumulate(paths, axis=1)
            dd = (rmax - paths) / np.maximum(rmax, 1e-10)
            mask &= np.max(dd, axis=1) <= c.max_drawdown
        if c.min_price is not None:
            mask &= np.all(paths >= c.min_price, axis=1)
        return mask
