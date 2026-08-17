"""Walk-forward paper execution for CESF vs raw-MC baseline.

Each decision uses the same seed, path count, admissibility C, contract
grid, and portfolio goal. Settlement uses the realized underlying path
(hold to DTE or last available bar). The latest bar is an OPEN paper order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cesf_filter import (
    PortfolioGoal,
    calibrate_gbm,
    crash_mass,
    run_underlying_cesf,
    select_contract,
    select_from_terminals,
)
from contracts import Contract, candidate_grid, putwrite_grid
from features import window_features
from greeks import bs_price, option_payoff


MULTIPLIER = 100.0
STARTING_CASH = 100_000.0
CRASH_SKIP = 0.20

# Public Cboe option-overlay benchmarks (total return, disclosed).
PUBLIC_BENCH = {
    "PUT": {
        "name": "Cboe S&P 500 PutWrite (PUT)",
        "ann_10y": 0.084,
        "y2024": 0.1784,
        "y2025": 0.0919,
        "monthly": 0.0073,
    },
    "BXM": {
        "name": "Cboe S&P 500 BuyWrite (BXM)",
        "ann_long": 0.085,
        "y2024": 0.201,
        "y2025": 0.089,
        "sharpe": 0.62,
    },
}


@dataclass
class PaperFill:
    model: str
    date: str
    index: int
    kind: str
    moneyness: float
    dte: int
    qty: int
    strike: float
    premium: float
    delta: float
    gamma: float
    ev_pnl_frac: float
    score: float
    spot_in: float
    spot_out: float | None
    settle_date: str | None
    pnl: float | None
    status: str
    cesf_bits: float | None
    cesf_events: int | None
    cesf_classes: int | None
    rsi: float
    sma50: float
    sma100: float
    crash_mass: float | None = None
    sigma: float | None = None


def _contract_fields(scored, spot: float) -> dict:
    c: Contract = scored.contract
    return {
        "kind": c.kind,
        "moneyness": c.moneyness,
        "dte": c.dte,
        "qty": c.qty,
        "strike": round(c.strike(spot), 4),
        "premium": round(scored.premium, 4),
        "delta": round(scored.delta, 4),
        "gamma": round(scored.gamma, 6),
        "ev_pnl_frac": round(scored.ev_pnl_frac, 6),
        "score": round(scored.score, 6),
        "spot_in": round(spot, 4),
    }


def _settle(contract: Contract, spot: float, premium: float, future: np.ndarray) -> tuple[float, float, str]:
    hold = min(contract.dte, len(future) - 1)
    status = "SETTLED" if hold >= contract.dte else "MARKED"
    if hold <= 0:
        return spot, 0.0, "OPEN"
    spot_out = float(future[hold])
    strike = contract.strike(spot)
    payoff = float(option_payoff(contract.kind, strike, np.array([spot_out]))[0])
    pnl = (payoff - premium) * MULTIPLIER * contract.qty
    return spot_out, pnl, status


def decide_pair(
    window: np.ndarray,
    n_paths: int,
    seed: int,
    goal: PortfolioGoal,
    strategy: str = "putwrite",
):
    """One CESF simulation. CESF and RAW share path-level EV; CESF may skip on crash mass."""
    result = run_underlying_cesf(window, n_paths=n_paths, seed=seed)
    system = calibrate_gbm(window, seed=seed)
    grid = putwrite_grid() if strategy == "putwrite" else candidate_grid()
    scored_c = select_contract(result, sigma=system.sigma, goal=goal, grid=grid)
    s0 = float(result.adm_paths[0, 0])
    raw_terms = result.adm_paths[:, -1]
    raw_w = np.full(len(raw_terms), 1.0 / max(len(raw_terms), 1))
    scored_r = select_from_terminals(
        s0, raw_terms, raw_w, system.sigma, goal, grid=grid
    )
    mass = crash_mass(result, barrier=0.80)
    skip = strategy == "putwrite" and mass >= CRASH_SKIP
    meta = {
        "cesf_bits": round(result.complexity_bits, 4),
        "cesf_events": result.n_events,
        "cesf_classes": result.n_classes,
        "crash_mass": round(mass, 4),
        "sigma": round(system.sigma, 4),
    }
    return (None if skip else scored_c), scored_r, meta


def _fill_from_decision(
    model: str,
    date: str,
    index: int,
    window: np.ndarray,
    future: np.ndarray,
    dates: list[str],
    scored,
    meta: dict,
    live: bool,
) -> PaperFill:
    feats = window_features(window)
    spot = float(window[-1])
    fields = _contract_fields(scored, spot)
    if live or len(future) < 2:
        spot_out, pnl, status = None, None, "OPEN"
        settle_date = None
    else:
        spot_out, pnl, status = _settle(scored.contract, spot, scored.premium, future)
        hold = min(scored.contract.dte, len(future) - 1)
        settle_i = index + hold
        settle_date = dates[settle_i] if settle_i < len(dates) else dates[-1]
        spot_out = round(spot_out, 4)
        pnl = round(pnl, 2)
    return PaperFill(
        model=model,
        date=date,
        index=index,
        settle_date=settle_date,
        spot_out=spot_out,
        pnl=pnl,
        status=status,
        rsi=round(float(feats["rsi"]), 2),
        sma50=round(float(feats["sma50"]), 4),
        sma100=round(float(feats["sma100"]), 4),
        **fields,
        **meta,
    )


def _cash_fill(model: str, date: str, index: int, window: np.ndarray, meta: dict) -> PaperFill:
    feats = window_features(window)
    spot = float(window[-1])
    extra = {
        "cesf_bits": meta.get("cesf_bits") if model == "CESF" else None,
        "cesf_events": meta.get("cesf_events") if model == "CESF" else None,
        "cesf_classes": meta.get("cesf_classes") if model == "CESF" else None,
    }
    return PaperFill(
        model=model,
        date=date,
        index=index,
        kind="CASH",
        moneyness=0.0,
        dte=5,
        qty=0,
        strike=0.0,
        premium=0.0,
        delta=0.0,
        gamma=0.0,
        ev_pnl_frac=0.0,
        score=0.0,
        spot_in=round(spot, 4),
        spot_out=round(spot, 4),
        settle_date=date,
        pnl=0.0,
        status="SKIP",
        rsi=round(float(feats["rsi"]), 2),
        sma50=round(float(feats["sma50"]), 4),
        sma100=round(float(feats["sma100"]), 4),
        crash_mass=meta.get("crash_mass"),
        sigma=meta.get("sigma"),
        **extra,
    )


def annualized_return(pnl: float, cash: float, n_days: int) -> float:
    if cash <= 0 or n_days <= 0:
        return 0.0
    r = pnl / cash
    years = n_days / 252.0
    if years <= 0 or r <= -0.999:
        return 0.0
    return float((1.0 + r) ** (1.0 / years) - 1.0)


def max_drawdown(equity: list[float]) -> float:
    peak = equity[0]
    dd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    return dd


def summarize(fills: list[PaperFill], cash: float, goal: PortfolioGoal) -> dict:
    settled = [f for f in fills if f.pnl is not None]
    pnls = [f.pnl for f in settled]
    wins = sum(1 for p in pnls if p > 0)
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    gross_win = float(sum(win_pnls))
    gross_loss = float(abs(sum(loss_pnls)))
    equity = [cash]
    running = cash
    points = []
    if fills:
        points.append({"time": fills[0].date, "value": round(cash, 2)})
    for fill in fills:
        if fill.pnl is None:
            continue
        running += fill.pnl
        equity.append(running)
        stamp = fill.settle_date or fill.date
        if points and points[-1]["time"] == stamp:
            points[-1]["value"] = round(running, 2)
        else:
            points.append({"time": stamp, "value": round(running, 2)})
    return {
        "trades": len(fills),
        "settled": len(settled),
        "open": sum(1 for f in fills if f.status == "OPEN"),
        "wins": wins,
        "win_rate": (wins / len(settled)) if settled else 0.0,
        "pnl": round(sum(pnls), 2) if pnls else 0.0,
        "mean_pnl": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        "avg_win": round(float(np.mean(win_pnls)), 2) if win_pnls else 0.0,
        "avg_loss": round(float(np.mean(loss_pnls)), 2) if loss_pnls else 0.0,
        "profit_factor": None if gross_loss == 0 else round(gross_win / gross_loss, 3),
        "best": round(max(pnls), 2) if pnls else 0.0,
        "worst": round(min(pnls), 2) if pnls else 0.0,
        "equity": round(equity[-1], 2),
        "max_drawdown": round(max_drawdown(equity), 2),
        "equity_curve": [round(x, 2) for x in equity],
        "equity_points": points,
        "mean_abs_delta": round(float(np.mean([abs(f.delta) for f in fills])), 4) if fills else 0.0,
        "mean_gamma": round(float(np.mean([f.gamma for f in fills])), 6) if fills else 0.0,
        "mean_delta_error": round(float(np.mean([abs(f.delta - goal.delta_target) for f in fills])), 4) if fills else 0.0,
        "mean_gamma_error": round(float(np.mean([abs(f.gamma - goal.gamma_target) for f in fills])), 6) if fills else 0.0,
    }


def reserved_cash(live: dict | None) -> float:
    """Cash-secured short: strike × 100 × |qty|. Long debit: premium × 100 × qty."""
    if not live or live.get("kind") in ("CASH", None):
        return 0.0
    qty = int(live.get("qty") or 0)
    if qty == 0:
        return 0.0
    if qty < 0:
        return abs(qty) * float(live.get("strike") or 0.0) * MULTIPLIER
    return abs(qty) * float(live.get("premium") or 0.0) * MULTIPLIER


def paper_book(
    summary: dict,
    live: dict | None,
    spot: float,
    sigma: float,
    starting: float = STARTING_CASH,
) -> dict:
    realized = float(summary.get("pnl") or 0.0)
    mtm = unrealized_pnl(live or {}, spot, sigma)
    nav = starting + realized + mtm
    reserved = reserved_cash(live)
    free = starting + realized - reserved
    ticket = live or {}
    return {
        "starting": starting,
        "realized": round(realized, 2),
        "mtm": round(mtm, 2),
        "nav": round(nav, 2),
        "reserved": round(reserved, 2),
        "free_cash": round(free, 2),
        "utilization": round(reserved / max(starting + realized, 1e-9), 4),
        "max_drawdown": summary.get("max_drawdown"),
        "win_rate": summary.get("win_rate"),
        "trades": summary.get("trades"),
        "settled": summary.get("settled"),
        "wins": summary.get("wins"),
        "mean_pnl": summary.get("mean_pnl"),
        "avg_win": summary.get("avg_win"),
        "avg_loss": summary.get("avg_loss"),
        "profit_factor": summary.get("profit_factor"),
        "best": summary.get("best"),
        "worst": summary.get("worst"),
        "ann": summary.get("ann"),
        "vs_put": summary.get("vs_put"),
        "delta": ticket.get("delta"),
        "gamma": ticket.get("gamma"),
        "kind": ticket.get("kind"),
        "strike": ticket.get("strike"),
        "dte": ticket.get("dte"),
        "qty": ticket.get("qty"),
        "premium": ticket.get("premium"),
        "moneyness": ticket.get("moneyness"),
        "status": ticket.get("status"),
        "date": ticket.get("date"),
        "rsi": ticket.get("rsi"),
    }


def put_benchmark_points(times: list[str], starting: float, ann: float = 0.084) -> list[dict]:
    if not times:
        return []
    from datetime import date as date_cls

    origin = date_cls.fromisoformat(times[0])
    out = []
    for stamp in times:
        years = (date_cls.fromisoformat(stamp) - origin).days / 365.25
        out.append({"time": stamp, "value": round(starting * ((1.0 + ann) ** years), 2)})
    return out


def _model_meta(model: str, meta: dict) -> dict:
    out = {
        "crash_mass": meta.get("crash_mass"),
        "sigma": meta.get("sigma"),
        "cesf_bits": None,
        "cesf_events": None,
        "cesf_classes": None,
    }
    if model == "CESF":
        out["cesf_bits"] = meta.get("cesf_bits")
        out["cesf_events"] = meta.get("cesf_events")
        out["cesf_classes"] = meta.get("cesf_classes")
    return out


def _moneyness_label(moneyness: float) -> str:
    if abs(float(moneyness or 0.0)) < 1e-9:
        return "ATM"
    return f"{float(moneyness) * 100:+.0f}%"


def contract_phrase(fill: dict) -> str:
    if not fill or fill.get("kind") == "CASH":
        return "CASH"
    side = "short" if int(fill.get("qty") or 0) < 0 else "long"
    return f"{side} {fill['kind']} {_moneyness_label(fill.get('moneyness') or 0)} {fill['dte']}d"


def _reason_open(fill: dict) -> str:
    mass = fill.get("crash_mass")
    mass_txt = "" if mass is None else f" Crash-mass {mass:.1%}."
    if fill.get("kind") == "CASH":
        return f"Veto. Crash-mass {mass:.1%} ≥ 20%. Stay in cash." if mass is not None else "Veto. Stay in cash."
    ev = float(fill.get("ev_pnl_frac") or 0.0)
    core = (
        f"Open {contract_phrase(fill)} @ {float(fill.get('premium') or 0):.2f}. "
        f"EV {ev:.2%} · Δ {float(fill.get('delta') or 0):.2f}."
        f"{mass_txt}"
    )
    if fill.get("model") == "CESF":
        bits = fill.get("cesf_bits")
        classes = fill.get("cesf_classes")
        return core + f" {classes} classes, {bits} bits."
    return core + " RAW same-path EV, no crash veto."


def _reason_close(fill: dict) -> str:
    pnl = fill.get("pnl")
    pnl_txt = "—" if pnl is None else f"{pnl:+.2f}"
    spot_in = fill.get("spot_in")
    spot_out = fill.get("spot_out")
    path = ""
    if spot_in is not None and spot_out is not None:
        path = f" Spot {spot_in:.2f} → {spot_out:.2f}."
    return f"Close {contract_phrase(fill)}. PnL {pnl_txt}.{path}"


def inference_events(fills: list) -> list[dict]:
    """OPEN / SKIP / CLOSE tape from walk-forward fills, oldest first."""
    rows = [asdict(f) if not isinstance(f, dict) else f for f in fills]
    events: list[dict] = []
    for fill in rows:
        action = "SKIP" if fill.get("kind") == "CASH" or fill.get("status") == "SKIP" else "OPEN"
        events.append(
            {
                "id": f"o-{fill['model']}-{fill['index']}",
                "action": action,
                "model": fill["model"],
                "date": fill["date"],
                "contract": contract_phrase(fill),
                "strike": fill.get("strike"),
                "premium": fill.get("premium"),
                "delta": fill.get("delta"),
                "gamma": fill.get("gamma"),
                "ev": fill.get("ev_pnl_frac"),
                "bits": fill.get("cesf_bits"),
                "classes": fill.get("cesf_classes"),
                "crash_mass": fill.get("crash_mass"),
                "sigma": fill.get("sigma"),
                "rsi": fill.get("rsi"),
                "spot": fill.get("spot_in"),
                "pnl": None,
                "reason": _reason_open(fill),
            }
        )
        closed = fill.get("status") in ("SETTLED", "MARKED") and fill.get("kind") != "CASH"
        if closed:
            events.append(
                {
                    "id": f"c-{fill['model']}-{fill['index']}",
                    "action": "CLOSE",
                    "model": fill["model"],
                    "date": fill.get("settle_date") or fill["date"],
                    "contract": contract_phrase(fill),
                    "strike": fill.get("strike"),
                    "premium": fill.get("premium"),
                    "delta": fill.get("delta"),
                    "gamma": fill.get("gamma"),
                    "ev": fill.get("ev_pnl_frac"),
                    "bits": fill.get("cesf_bits"),
                    "classes": fill.get("cesf_classes"),
                    "crash_mass": fill.get("crash_mass"),
                    "sigma": fill.get("sigma"),
                    "rsi": fill.get("rsi"),
                    "spot": fill.get("spot_out"),
                    "pnl": fill.get("pnl"),
                    "reason": _reason_close(fill),
                }
            )
        elif fill.get("status") == "OPEN" and fill.get("kind") != "CASH":
            events.append(
                {
                    "id": f"h-{fill['model']}-{fill['index']}",
                    "action": "HOLD",
                    "model": fill["model"],
                    "date": fill["date"],
                    "contract": contract_phrase(fill),
                    "strike": fill.get("strike"),
                    "premium": fill.get("premium"),
                    "delta": fill.get("delta"),
                    "gamma": fill.get("gamma"),
                    "ev": fill.get("ev_pnl_frac"),
                    "bits": fill.get("cesf_bits"),
                    "classes": fill.get("cesf_classes"),
                    "crash_mass": fill.get("crash_mass"),
                    "sigma": fill.get("sigma"),
                    "rsi": fill.get("rsi"),
                    "spot": fill.get("spot_in"),
                    "pnl": None,
                    "reason": f"Still open. Marking {contract_phrase(fill)} on last price.",
                }
            )
    rank = {"SKIP": 0, "OPEN": 1, "CLOSE": 2, "HOLD": 3}
    events.sort(key=lambda e: (e["date"], rank.get(e["action"], 9), 0 if e["model"] == "CESF" else 1))
    for i, event in enumerate(events, 1):
        event["seq"] = i
    return events


def walk_forward(
    dates: list[str],
    closes: np.ndarray,
    goal: PortfolioGoal,
    ticker: str,
    n_paths: int = 64,
    lookback: int = 120,
    stride: int = 21,
    starting_cash: float = STARTING_CASH,
    strategy: str = "putwrite",
) -> dict:
    cesf_fills: list[PaperFill] = []
    raw_fills: list[PaperFill] = []
    last_i = len(closes) - 1
    sequential = strategy == "putwrite"
    t = lookback
    i = 0
    while t <= len(closes):
        end = t
        window = closes[end - lookback : end]
        future = closes[end - 1 :]
        date = dates[end - 1]
        live = end - 1 == last_i
        scored_c, scored_r, meta = decide_pair(window, n_paths, 42 + i, goal, strategy)
        meta_c = _model_meta("CESF", meta)
        meta_r = _model_meta("RAW", meta)
        if scored_c is None:
            cesf_fills.append(_cash_fill("CESF", date, end - 1, window, meta_c))
            hold = 5
        else:
            cesf_fills.append(
                _fill_from_decision(
                    "CESF", date, end - 1, window, future, dates, scored_c, meta_c, live
                )
            )
            hold = scored_c.contract.dte
        raw_fills.append(
            _fill_from_decision(
                "RAW", date, end - 1, window, future, dates, scored_r, meta_r, live
            )
        )
        if not sequential:
            hold = stride
        i += 1
        t = end + hold
        if live:
            break
        if not sequential and t > len(closes) and end != len(closes):
            t = len(closes)

    agree = sum(
        1
        for a, b in zip(cesf_fills, raw_fills)
        if a.kind == b.kind and a.moneyness == b.moneyness and a.dte == b.dte
    )
    live_c = next((f for f in reversed(cesf_fills) if f.status == "OPEN"), cesf_fills[-1] if cesf_fills else None)
    live_r = next((f for f in reversed(raw_fills) if f.status == "OPEN"), raw_fills[-1] if raw_fills else None)
    bits = [f.cesf_bits for f in cesf_fills if f.cesf_bits is not None]
    n_days = max(len(closes) - lookback, 1)
    cesf_sum = summarize(cesf_fills, starting_cash, goal)
    raw_sum = summarize(raw_fills, starting_cash, goal)
    cesf_sum["ann"] = round(annualized_return(cesf_sum["pnl"], starting_cash, n_days), 4)
    raw_sum["ann"] = round(annualized_return(raw_sum["pnl"], starting_cash, n_days), 4)
    put_ann = PUBLIC_BENCH["PUT"]["ann_10y"]
    cesf_sum["vs_put"] = round(cesf_sum["ann"] - put_ann, 4)
    raw_sum["vs_put"] = round(raw_sum["ann"] - put_ann, 4)
    return {
        "ticker": ticker,
        "strategy": strategy,
        "thesis": (
            "CESF compresses the underlying futures into operational events, then picks the "
            "put that hits EV 1% / delta 0.45 / gamma −0.02. RAW scores the same paths without "
            "the filter. CESF only vetoes a short put when crash-mass is high."
        ),
        "why_cesf_lagged": (
            "CESF previously priced payoff(E[S]) on class means (Jensen-biased) and crash events. "
            "EV now uses all class member terminals; CESF only vetoes short puts when crash-mass is high."
        ),
        "goal": {
            "ev_target": goal.ev_target,
            "delta_target": goal.delta_target,
            "gamma_target": goal.gamma_target,
        },
        "public": PUBLIC_BENCH,
        "n_paths": n_paths,
        "lookback": lookback,
        "stride": stride,
        "starting_cash": starting_cash,
        "multiplier": MULTIPLIER,
        "spot": float(closes[-1]),
        "asof": dates[-1],
        "bars": len(closes),
        "agreement_rate": (agree / len(cesf_fills)) if cesf_fills else 0.0,
        "mean_cesf_bits": round(float(np.mean(bits)), 3) if bits else 0.0,
        "cesf": cesf_sum,
        "raw": raw_sum,
        "fills": [asdict(f) for f in cesf_fills + raw_fills],
        "tape": inference_events(cesf_fills + raw_fills),
        "live": {
            "CESF": asdict(live_c) if live_c else None,
            "RAW": asdict(live_r) if live_r else None,
        },
        "price": [{"date": d, "close": round(float(p), 4)} for d, p in zip(dates, closes)],
    }


def unrealized_pnl(fill: dict, spot: float, sigma: float, rate: float = 0.04) -> float:
    if not fill or fill.get("kind") == "CASH" or int(fill.get("qty") or 0) == 0:
        return 0.0
    t_years = max(int(fill["dte"]), 1) / 252.0
    now = bs_price(fill["kind"], float(spot), float(fill["strike"]), t_years, rate, float(sigma))
    return (now - float(fill["premium"])) * MULTIPLIER * int(fill["qty"])
