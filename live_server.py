#!/usr/bin/env python3
"""Paper put-write dashboard: CESF vs RAW vs Cboe PUT.

Polls last price, marks open lots with Black-Scholes, serves /api/state.
CESF is run once per boot (cached), not every tick. No broker, no option-chain fill.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cesf_filter import PortfolioGoal, calibrate_gbm
from corpus import fetch_series, synthetic_series
from overlay import asset_overlay
from paper import STARTING_CASH, paper_book, put_benchmark_points, walk_forward, unrealized_pnl

STATE: dict = {
    "ready": False,
    "error": None,
    "ticks": [],
    "report": None,
    "spot": None,
    "updated": None,
    "sigma": 0.25,
}
LOCK = threading.Lock()
HTML = HERE / "viz" / "live.html"


def putwrite_goal(ev: float, ld: float, lg: float) -> PortfolioGoal:
    return PortfolioGoal(
        ev_target=ev,
        delta_target=0.45,
        gamma_target=-0.02,
        lambda_delta=ld,
        lambda_gamma=lg,
    )


def fetch_last_price(ticker: str) -> float:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?interval=1m&range=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "cesf-live/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    meta = payload["chart"]["result"][0]["meta"]
    px = meta.get("regularMarketPrice") or meta.get("previousClose")
    if px is None:
        raise RuntimeError("no last price")
    return float(px)


def _with_live_nav(points: list[dict], nav: float, today: str) -> list[dict]:
    out = [dict(p) for p in points]
    if not out:
        return [{"time": today, "value": round(nav, 2)}]
    if out[-1]["time"] == today:
        out[-1]["value"] = round(nav, 2)
    else:
        out.append({"time": today, "value": round(nav, 2)})
    return out


def books_payload(report: dict, spot: float, sigma: float) -> dict:
    live = report.get("live") or {}
    return {
        "CESF": paper_book(report["cesf"], live.get("CESF"), spot, sigma),
        "RAW": paper_book(report["raw"], live.get("RAW"), spot, sigma),
    }


def equity_payload(report: dict, books: dict, today: str) -> dict:
    cesf_pts = _with_live_nav(report["cesf"].get("equity_points") or [], books["CESF"]["nav"], today)
    raw_pts = _with_live_nav(report["raw"].get("equity_points") or [], books["RAW"]["nav"], today)
    times = sorted({p["time"] for p in cesf_pts + raw_pts})
    put_ann = report["public"]["PUT"]["ann_10y"]
    return {
        "CESF": cesf_pts,
        "RAW": raw_pts,
        "PUT": put_benchmark_points(times, STARTING_CASH, put_ann),
    }


def snapshot_tick(report: dict, spot: float, sigma: float) -> dict:
    live = report["live"]
    cesf_u = unrealized_pnl(live.get("CESF") or {}, spot, sigma)
    raw_u = unrealized_pnl(live.get("RAW") or {}, spot, sigma)
    cesf_eq = report["cesf"]["equity"] + cesf_u
    raw_eq = report["raw"]["equity"] + raw_u
    put_ann = report["public"]["PUT"]["ann_10y"]
    n_days = max(report["bars"] - report["lookback"], 1)

    def ann(eq: float) -> float:
        r = (eq - STARTING_CASH) / STARTING_CASH
        years = n_days / 252.0
        return float((1.0 + r) ** (1.0 / max(years, 1e-6)) - 1.0)

    today = datetime.now(timezone.utc).date().isoformat()
    books = books_payload(report, spot, sigma)
    return {
        "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today": today,
        "spot": round(spot, 4),
        "cesf_mtm": round(cesf_u, 2),
        "raw_mtm": round(raw_u, 2),
        "cesf_equity": round(cesf_eq, 2),
        "raw_equity": round(raw_eq, 2),
        "cesf_ann": round(ann(cesf_eq), 4),
        "raw_ann": round(ann(raw_eq), 4),
        "put_ann": put_ann,
        "bxm_ann": report["public"]["BXM"]["ann_long"],
        "books": books,
        "equity": equity_payload(report, books, today),
        "overlay": asset_overlay(report, spot, sigma, today),
    }


def boot(ticker: str, synthetic: bool, paths: int, days: int) -> None:
    try:
        if synthetic or ticker == "SYN":
            name, dates, closes = "SYN", *synthetic_series()
        else:
            dates, closes = fetch_series(ticker, days=days)
            name = ticker
        goal = putwrite_goal(0.01, 0.35, 1.5)
        report = walk_forward(
            dates, closes, goal, name, n_paths=paths, lookback=120, stride=21, strategy="putwrite"
        )
        sigma = float(calibrate_gbm(closes[-120:]).sigma)
        spot = float(closes[-1])
        try:
            if not synthetic and ticker != "SYN":
                spot = fetch_last_price(ticker)
        except Exception:
            pass
        tick = snapshot_tick(report, spot, sigma)
        with LOCK:
            STATE.update(
                ready=True,
                error=None,
                report=report,
                spot=spot,
                sigma=sigma,
                ticker=name,
                ticks=[tick],
                updated=tick["t"],
                synthetic=synthetic,
            )
        print(
            f"paper boot {name} spot={spot:.2f} CESF nav={tick['books']['CESF']['nav']:.0f} "
            f"RAW nav={tick['books']['RAW']['nav']:.0f} PUT={tick['put_ann']:.1%}",
            flush=True,
        )
    except Exception as exc:
        with LOCK:
            STATE.update(ready=False, error=str(exc))
        raise


def poll_loop(ticker: str, interval: float, synthetic: bool) -> None:
    while True:
        time.sleep(interval)
        with LOCK:
            if not STATE.get("ready"):
                continue
            report = STATE["report"]
            sigma = STATE["sigma"]
            last = STATE["spot"]
        try:
            spot = last if synthetic or ticker == "SYN" else fetch_last_price(ticker)
        except Exception:
            spot = last
        tick = snapshot_tick(report, spot, sigma)
        with LOCK:
            STATE["spot"] = spot
            STATE["updated"] = tick["t"]
            STATE["ticks"] = (STATE["ticks"] + [tick])[-400:]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str, head: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        head = self.command == "HEAD"
        if self.path in ("/", "/live", "/live.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8", head=head)
            return
        if self.path.startswith("/api/state"):
            with LOCK:
                ticks = STATE.get("ticks") or []
                report = STATE.get("report")
                tick = ticks[-1] if ticks else None
                fills = [] if not report else (report.get("fills") or [])[-80:]
                payload = {
                    "ready": STATE.get("ready"),
                    "error": STATE.get("error"),
                    "ticker": STATE.get("ticker"),
                    "spot": STATE.get("spot"),
                    "updated": STATE.get("updated"),
                    "tick": tick,
                    "ticks": ticks[-240:],
                    "public": None if not report else report.get("public"),
                    "thesis": None if not report else report.get("thesis"),
                    "why": None if not report else report.get("why_cesf_lagged"),
                    "cesf": None if not report else report.get("cesf"),
                    "raw": None if not report else report.get("raw"),
                    "live": None if not report else report.get("live"),
                    "books": None if not tick else tick.get("books"),
                    "equity": None if not tick else tick.get("equity"),
                    "overlay": None if not tick else tick.get("overlay"),
                    "fills": fills,
                    "tape": [] if not report else report.get("tape") or [],
                    "goal": None if not report else report.get("goal"),
                    "mean_bits": None if not report else report.get("mean_cesf_bits"),
                    "strategy": None if not report else report.get("strategy"),
                    "asof": None if not report else report.get("asof"),
                    "starting_cash": STARTING_CASH,
                    "venue": "paper",
                }
            self._send(200, json.dumps(payload).encode(), "application/json", head=head)
            return
        if self.path.startswith("/api/stream"):
            if head:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                return
            self._stream()
            return
        self._send(404, b"not found", "text/plain", head=head)

    def _stream(self) -> None:
        parsed = urlparse(self.path)
        pace_raw = (parse_qs(parsed.query).get("pace") or ["140"])[0]
        try:
            pace = max(0.0, min(float(pace_raw), 2000.0)) / 1000.0
        except ValueError:
            pace = 0.14
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with LOCK:
            tape = list(((STATE.get("report") or {}).get("tape")) or [])
        try:
            for event in tape:
                self.wfile.write(
                    ("data: " + json.dumps({"type": "event", **event}) + "\n\n").encode()
                )
                self.wfile.flush()
                if pace:
                    time.sleep(pace)
            self.wfile.write(b'data: {"type":"live"}\n\n')
            self.wfile.flush()
            while True:
                time.sleep(8)
                with LOCK:
                    tick = (STATE.get("ticks") or [None])[-1]
                    live = ((STATE.get("report") or {}).get("live")) or {}
                payload = {"type": "mark", "tick": tick, "live": live}
                self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="CESF paper put-write dashboard")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--paths", type=int, default=48)
    parser.add_argument("--days", type=int, default=520)
    parser.add_argument("--interval", type=float, default=8.0)
    args = parser.parse_args()
    boot(args.ticker, args.synthetic, args.paths, args.days)
    threading.Thread(
        target=poll_loop,
        args=(args.ticker, args.interval, args.synthetic),
        daemon=True,
    ).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard http://127.0.0.1:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
