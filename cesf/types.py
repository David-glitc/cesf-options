"""CESF operational query and configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdmissibilityConstraints:
    """Hard constraints C on admissible trajectories."""

    no_neg_price: bool = True
    max_drawdown: float | None = 0.60
    min_price: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"no_neg_price": self.no_neg_price}
        if self.max_drawdown is not None:
            out["max_drawdown"] = self.max_drawdown
        if self.min_price is not None:
            out["min_price"] = self.min_price
        return out


@dataclass
class RelevanceConfig:
    """Task objective U for the relevance functional."""

    task: str = "downside_risk"
    barrier: float = 0.80
    eta: float = 0.5
    proxy_percentile: float = 40.0
    event_percentile: float = 30.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "barrier": self.barrier,
            "eta": self.eta,
            "proxy_percentile": self.proxy_percentile,
            "event_percentile": self.event_percentile,
        }


@dataclass
class OperationalQuery:
    """
    Operational query Q = (C, ε, H, U).

    C: admissibility constraints
    ε: observational resolution / tolerance
    H: finite prediction horizon (trading days)
    U: task relevance specification
    """

    epsilon: float
    horizon: int
    constraints: AdmissibilityConstraints = field(default_factory=AdmissibilityConstraints)
    relevance: RelevanceConfig = field(default_factory=RelevanceConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperationalQuery:
        c = d.get("constraints", {})
        u = d.get("relevance", {})
        return cls(
            epsilon=float(d["epsilon"]),
            horizon=int(d["horizon"]),
            constraints=AdmissibilityConstraints(
                no_neg_price=c.get("no_neg_price", True),
                max_drawdown=c.get("max_drawdown", 0.60),
                min_price=c.get("min_price"),
            ),
            relevance=RelevanceConfig(
                task=u.get("task", "downside_risk"),
                barrier=u.get("barrier", 0.80),
                eta=u.get("eta", 0.5),
                proxy_percentile=u.get("proxy_percentile", 40.0),
                event_percentile=u.get("event_percentile", 0.30),
            ),
        )
