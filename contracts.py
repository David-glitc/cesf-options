"""Discrete option-contract grid (kind, moneyness, DTE, qty)."""

from __future__ import annotations

from dataclasses import dataclass


MONEYNESS = (-0.10, -0.05, 0.0, 0.05, 0.10)
DTES = (7, 14, 21, 30, 42)
KINDS = ("C", "P")


@dataclass(frozen=True)
class Contract:
    kind: str
    moneyness: float
    dte: int
    qty: int = 1

    def strike(self, spot: float) -> float:
        return spot * (1.0 + self.moneyness)

    def t_years(self) -> float:
        return self.dte / 252.0


def candidate_grid() -> list[Contract]:
    return [
        Contract(kind=kind, moneyness=m, dte=dte)
        for kind in KINDS
        for m in MONEYNESS
        for dte in DTES
    ]


def putwrite_grid() -> list[Contract]:
    """Cboe PUT-style short puts: ATM / 5% OTM, 21–30 DTE."""
    return [
        Contract(kind="P", moneyness=m, dte=dte, qty=-1)
        for m in (0.0, -0.05)
        for dte in (21, 30)
    ]
