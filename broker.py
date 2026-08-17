"""Paper broker: match CESF/RAW tickets to live option quotes and fill."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import time

import pandas as pd


@dataclass
class PaperExecution:
    t: str
    model: str
    action: str
    symbol: str
    expiry: str
    strike: float
    kind: str
    qty: int
    bid: float
    ask: float
    last: float
    fill_px: float
    mid: float
    slippage: float
    iv: float | None
    venue: str
    status: str
    note: str


def _calendar_days(dte: int) -> int:
    return max(int(round(dte * 365.0 / 252.0)), 1)


def pick_expiry(expiries: list[str], dte: int, today: date | None = None) -> str:
    today = today or date.today()
    target = today + timedelta(days=_calendar_days(dte))
    return min(expiries, key=lambda e: abs((date.fromisoformat(e) - target).days))


def pick_put_row(puts: pd.DataFrame, strike: float) -> pd.Series:
    if puts is None or puts.empty:
        raise RuntimeError("empty put chain")
    idx = (puts["strike"] - strike).abs().idxmin()
    return puts.loc[idx]


def fill_price(qty: int, bid: float, ask: float, last: float) -> float:
    """Sell at bid, buy at ask. Fall back to last then mid."""
    bid = float(bid or 0.0)
    ask = float(ask or 0.0)
    last = float(last or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
    if qty < 0:
        px = bid if bid > 0 else (last if last > 0 else mid)
    else:
        px = ask if ask > 0 else (last if last > 0 else mid)
    if px <= 0 and mid > 0:
        px = mid
    return float(px)


_PUTS: dict = {}


def fetch_puts(ticker: str, expiry: str) -> pd.DataFrame:
    import yfinance as yf

    key = (ticker, expiry, int(time.time() // 30))
    cached = _PUTS.get(key)
    if cached is not None:
        return cached
    chain = yf.Ticker(ticker).option_chain(expiry)
    _PUTS.clear()
    _PUTS[key] = chain.puts
    return chain.puts


def quote_and_fill(ticker: str, ticket: dict, model: str) -> PaperExecution:
    if not ticket or ticket.get("kind") in ("CASH", None) or int(ticket.get("qty") or 0) == 0:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return PaperExecution(
            t=now, model=model, action="SKIP", symbol="", expiry="", strike=0.0,
            kind="CASH", qty=0, bid=0.0, ask=0.0, last=0.0, fill_px=0.0, mid=0.0,
            slippage=0.0, iv=None, venue="paper", status="SKIP",
            note="no live option — CESF cash skip",
        )
    import yfinance as yf

    tk = yf.Ticker(ticker)
    expiries = list(tk.options or [])
    if not expiries:
        raise RuntimeError(f"no option expiries for {ticker}")
    expiry = pick_expiry(expiries, int(ticket["dte"]))
    puts = fetch_puts(ticker, expiry)
    row = pick_put_row(puts, float(ticket["strike"]))
    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    qty = int(ticket["qty"])
    px = fill_price(qty, bid, ask, last)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (last if last > 0 else px)
    action = "SELL" if qty < 0 else "BUY"
    symbol = str(row.get("contractSymbol") or "")
    iv = row.get("impliedVolatility")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return PaperExecution(
        t=now,
        model=model,
        action=action,
        symbol=symbol,
        expiry=expiry,
        strike=float(row["strike"]),
        kind=str(ticket.get("kind") or "P"),
        qty=qty,
        bid=bid,
        ask=ask,
        last=last,
        fill_px=round(px, 4),
        mid=round(float(mid), 4),
        slippage=round(px - float(mid), 4),
        iv=None if iv is None or (isinstance(iv, float) and pd.isna(iv)) else float(iv),
        venue="yahoo-paper",
        status="FILLED" if px > 0 else "REJECTED",
        note=f"paper {action} {symbol} @ {px:.2f} (bid {bid:.2f} / ask {ask:.2f})",
    )


def mark_from_quote(execution: dict, ticker: str) -> float:
    """Unrealized vs fill: short (fill - mid)*100, long (mid - fill)*100."""
    if not execution or execution.get("status") != "FILLED":
        return 0.0
    puts = fetch_puts(ticker, execution["expiry"])
    row = pick_put_row(puts, float(execution["strike"]))
    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
    qty = int(execution.get("qty") or -1)
    return (mid - float(execution["fill_px"])) * 100.0 * qty
