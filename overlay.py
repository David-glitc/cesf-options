"""Underlying + option overlay: marks, strikes, open span, projected expiry."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from greeks import bs_price, option_payoff
from paper import contract_phrase

RATE = 0.04
ASSET_BARS = 180


def trading_days_ahead(start: date, count: int) -> list[str]:
    out: list[str] = []
    cursor = start
    while len(out) < max(int(count), 0):
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            out.append(cursor.isoformat())
    return out


def _as_map(price: list[dict]) -> list[dict]:
    return [{"time": p["date"], "value": float(p["close"])} for p in price if p.get("date")]


def _held_bars(open_date: str, bar_date: str, dates: list[str]) -> int:
    held = 0
    for stamp in dates:
        if stamp < open_date:
            continue
        if stamp > bar_date:
            break
        held += 1
    return max(held - 1, 0)


def _mark(kind: str, spot: float, strike: float, left: int, sigma: float) -> float:
    if left <= 0:
        return float(option_payoff(kind, strike, np.array([spot]))[0])
    return float(bs_price(kind, spot, strike, left / 252.0, RATE, sigma))


def _ticket_path(
    ticket: dict | None,
    price: list[dict],
    sigma: float,
    live_spot: float,
    today: str,
) -> list[dict]:
    if not ticket or ticket.get("kind") in ("CASH", None) or int(ticket.get("qty") or 0) == 0:
        return []
    kind = str(ticket["kind"])
    strike = float(ticket["strike"])
    dte = int(ticket["dte"])
    open_date = str(ticket["date"])
    settle = ticket.get("settle_date") if ticket.get("status") == "SETTLED" else None
    dates = [p["date"] for p in price]
    points: list[dict] = []
    for bar in price:
        stamp = bar["date"]
        if stamp < open_date:
            continue
        if settle and stamp > settle:
            break
        left = max(dte - _held_bars(open_date, stamp, dates), 0)
        px = float(bar["close"])
        points.append({"time": stamp, "value": round(_mark(kind, px, strike, left, sigma), 4)})
    if ticket.get("status") in ("OPEN", "FILLED") and (not points or points[-1]["time"] != today):
        left = max(dte - _held_bars(open_date, today, dates + [today]), 0)
        points.append({"time": today, "value": round(_mark(kind, live_spot, strike, left, sigma), 4)})
    elif points and points[-1]["time"] == today:
        left = max(dte - _held_bars(open_date, today, dates + [today]), 0)
        points[-1]["value"] = round(_mark(kind, live_spot, strike, left, sigma), 4)
    return points


def _project(
    ticket: dict | None,
    live_spot: float,
    sigma: float,
    today: str,
    last_mark: float | None,
    dates: list[str],
) -> tuple[list[dict], list[dict], int]:
    if not ticket or ticket.get("kind") in ("CASH", None) or int(ticket.get("qty") or 0) == 0:
        return [], [], 0
    if ticket.get("status") in ("SETTLED", "SKIP"):
        return [], [], 0
    kind = str(ticket["kind"])
    strike = float(ticket["strike"])
    dte = int(ticket["dte"])
    open_date = str(ticket["date"])
    held = _held_bars(open_date, today, dates)
    left = max(dte - held, 0)
    if left <= 0:
        return [], [], 0
    future = trading_days_ahead(date.fromisoformat(today), left)
    asset_proj = [{"time": stamp, "value": round(live_spot, 4)} for stamp in future]
    option_proj = []
    for i, stamp in enumerate(future, 1):
        remain = max(left - i, 0)
        option_proj.append(
            {"time": stamp, "value": round(_mark(kind, live_spot, strike, remain, sigma), 4)}
        )
    if last_mark is not None and option_proj:
        option_proj[0]["value"] = round(float(last_mark), 4)
    return asset_proj, option_proj, left


def asset_overlay(report: dict, spot: float, sigma: float, today: str) -> dict:
    price = list(report.get("price") or [])[-ASSET_BARS:]
    if not price:
        return {
            "asset": [],
            "asset_proj": [],
            "option": {"CESF": [], "RAW": []},
            "option_proj": {"CESF": [], "RAW": []},
            "strikes": [],
            "markers": [],
            "open_from": None,
            "expiry": None,
        }
    asset = _as_map(price)
    if asset[-1]["time"] == today:
        asset[-1]["value"] = round(float(spot), 4)
    else:
        asset.append({"time": today, "value": round(float(spot), 4)})

    dates = [p["date"] for p in price]
    if today not in dates:
        dates = dates + [today]
    live = report.get("live") or {}
    option = {}
    option_proj = {}
    asset_proj: list[dict] = []
    strikes = []
    markers = []
    open_from = None
    expiry = None
    left = 0
    for model, color in (("CESF", "#b6d95c"), ("RAW", "#7ba3d9")):
        ticket = live.get(model)
        path = _ticket_path(ticket, price, sigma, spot, today)
        option[model] = path
        last_mark = path[-1]["value"] if path else None
        a_proj, o_proj, left = _project(ticket, spot, sigma, today, last_mark, dates)
        option_proj[model] = o_proj
        if a_proj and not asset_proj:
            asset_proj = a_proj
        if ticket and ticket.get("kind") not in ("CASH", None) and int(ticket.get("qty") or 0) != 0:
            strike = float(ticket["strike"])
            label = f"{model} {int(round(strike))}{ticket['kind']}"
            strikes.append(
                {
                    "model": model,
                    "strike": round(strike, 4),
                    "label": label,
                    "premium": ticket.get("premium"),
                    "color": color,
                    "status": ticket.get("status"),
                    "contract": contract_phrase(ticket),
                }
            )
            open_from = ticket.get("date")
            future = trading_days_ahead(date.fromisoformat(str(ticket["date"])), int(ticket["dte"]))
            expiry = future[-1] if future else None
            markers.append(
                {
                    "time": ticket["date"],
                    "position": "belowBar",
                    "color": color,
                    "shape": "arrowUp",
                    "text": f"OPEN {label}",
                }
            )
            settled = ticket.get("status") == "SETTLED" and ticket.get("settle_date")
            if settled:
                markers.append(
                    {
                        "time": ticket["settle_date"],
                        "position": "aboveBar",
                        "color": color,
                        "shape": "arrowDown",
                        "text": f"CLOSE {label}",
                    }
                )
            elif expiry:
                markers.append(
                    {
                        "time": expiry,
                        "position": "aboveBar",
                        "color": color,
                        "shape": "circle",
                        "text": f"EXP {label}" + (f"  {left}d" if left else ""),
                    }
                )
    # Deduplicate identical strike labels for the axis.
    uniq = []
    seen = set()
    for row in strikes:
        key = (row["strike"], row["model"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    if asset_proj and asset and asset_proj[0]["time"] != asset[-1]["time"]:
        asset_proj = [dict(asset[-1])] + asset_proj
    for model in ("CESF", "RAW"):
        path = option.get(model) or []
        proj = option_proj.get(model) or []
        if path and proj and proj[0]["time"] != path[-1]["time"]:
            option_proj[model] = [dict(path[-1])] + proj
    return {
        "asset": asset,
        "asset_proj": asset_proj,
        "option": option,
        "option_proj": option_proj,
        "strikes": uniq,
        "markers": markers,
        "open_from": open_from,
        "expiry": expiry,
        "spot": round(float(spot), 4),
    }
