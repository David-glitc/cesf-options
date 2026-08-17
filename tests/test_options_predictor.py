#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np

from cesf_filter import PortfolioGoal, calibrate_gbm, run_underlying_cesf, select_contract
from contracts import Contract
from corpus import build_corpus_lines, synthetic_closes, synthetic_series
from encode import context_prefix, decode_contract, encode_contract_line
from features import rsi, sma, window_features
from greeks import bs_delta, option_payoff
from baseline import select_contract_raw
from broker import fill_price, pick_expiry
from overlay import asset_overlay, trading_days_ahead, _mark
from paper import walk_forward, reserved_cash, paper_book, STARTING_CASH, inference_events
from viz import render_html


class FeatureTests(unittest.TestCase):
    def test_rsi_flat_is_near_fifty(self):
        closes = np.full(40, 100.0)
        self.assertAlmostEqual(rsi(closes), 50.0, places=5)

    def test_sma_matches_mean(self):
        closes = np.arange(1, 51, dtype=float)
        self.assertAlmostEqual(sma(closes, 50), float(np.mean(closes)))


class EncodeTests(unittest.TestCase):
    def test_roundtrip_contract(self):
        closes = synthetic_closes(n=150, seed=1)
        feats = window_features(closes)
        contract = Contract(kind="C", moneyness=0.05, dte=21)
        line = encode_contract_line("AAPL", feats, 3.0, 0.02, 0.25, 0.02, contract)
        self.assertLessEqual(len(line), 48)
        decoded = decode_contract(line)
        self.assertEqual(decoded.kind, "C")
        self.assertAlmostEqual(decoded.moneyness, 0.05)
        self.assertEqual(decoded.dte, 21)
        self.assertTrue(context_prefix(line).endswith(">"))


class GreekTests(unittest.TestCase):
    def test_call_delta_between_zero_and_one(self):
        delta = bs_delta("C", 100.0, 100.0, 21 / 252, 0.04, 0.25)
        self.assertGreater(delta, 0.4)
        self.assertLess(delta, 0.7)

    def test_call_payoff(self):
        pay = option_payoff("C", 100.0, np.array([90.0, 110.0]))
        np.testing.assert_allclose(pay, [0.0, 10.0])


class CesfSelectTests(unittest.TestCase):
    def test_select_contract_on_synthetic(self):
        closes = synthetic_closes(n=180, seed=2)
        result = run_underlying_cesf(closes[-120:], n_paths=80, seed=3)
        system = calibrate_gbm(closes[-120:], seed=3)
        goal = PortfolioGoal(ev_target=0.02, delta_target=0.25, gamma_target=0.02)
        scored = select_contract(result, sigma=system.sigma, goal=goal)
        self.assertIn(scored.contract.kind, ("C", "P"))
        self.assertGreater(result.gamma_size, 10)
        self.assertGreaterEqual(result.n_classes, 1)

    def test_positive_delta_prefers_calls(self):
        closes = synthetic_closes(n=180, seed=2)
        result = run_underlying_cesf(closes[-120:], n_paths=80, seed=3)
        system = calibrate_gbm(closes[-120:], seed=3)
        goal = PortfolioGoal(
            ev_target=0.0,
            delta_target=0.45,
            gamma_target=0.0,
            lambda_delta=4.0,
            lambda_gamma=0.0,
        )
        scored = select_contract(result, sigma=system.sigma, goal=goal)
        self.assertEqual(scored.contract.kind, "C")

    def test_corpus_lines_parse(self):
        closes = synthetic_closes(n=200, seed=4)
        goal = PortfolioGoal()
        lines = build_corpus_lines(
            closes,
            "SYN",
            goal,
            n_paths=60,
            lookback=120,
            stride=40,
            include_transforms=True,
        )
        self.assertGreaterEqual(len(lines), 4)
        contracts = [ln for ln in lines if not ln.startswith("W")]
        decode_contract(contracts[0])


class ParallelBenchTests(unittest.TestCase):
    def test_raw_and_cesf_same_trade_count(self):
        dates, closes = synthetic_series(n=200, seed=5)
        goal = PortfolioGoal(ev_target=0.02, delta_target=0.25, gamma_target=0.02)
        report = walk_forward(
            dates, closes, goal, "SYN", n_paths=50, lookback=120, stride=40, strategy="buy"
        )
        self.assertEqual(report["cesf"]["trades"], report["raw"]["trades"])
        self.assertGreaterEqual(report["cesf"]["trades"], 2)
        self.assertIn(report["live"]["CESF"]["kind"], ("C", "P"))
        self.assertIn(report["live"]["RAW"]["kind"], ("C", "P"))
        html = render_html(report)
        self.assertIn("CESF", html)
        self.assertIn("RAW", html)
        self.assertIn("const DATA", html)

    def test_putwrite_has_public_benchmarks(self):
        dates, closes = synthetic_series(n=220, seed=8)
        goal = PortfolioGoal(ev_target=0.01, delta_target=0.45, gamma_target=-0.02)
        report = walk_forward(
            dates, closes, goal, "SYN", n_paths=40, lookback=120, stride=21, strategy="putwrite"
        )
        self.assertEqual(report["strategy"], "putwrite")
        self.assertIn("ann", report["cesf"])
        self.assertIn("PUT", report["public"])
        self.assertIn(report["live"]["RAW"]["kind"], ("P", "CASH"))
        self.assertIn("equity_points", report["cesf"])
        self.assertIn("thesis", report)
        tape = report["tape"]
        self.assertTrue(any(e["action"] in ("OPEN", "SKIP") for e in tape))
        self.assertTrue(any(e["action"] in ("CLOSE", "HOLD") for e in tape))
        self.assertTrue(all("reason" in e for e in tape))


    def test_baseline_returns_grid_contract(self):
        closes = synthetic_closes(n=150, seed=6)
        goal = PortfolioGoal()
        scored, system = select_contract_raw(closes[-120:], n_paths=60, seed=3, goal=goal)
        self.assertIn(scored.contract.kind, ("C", "P"))
        self.assertGreater(system.sigma, 0.0)


class PortfolioTests(unittest.TestCase):
    def test_short_put_reserves_strike_times_multiplier(self):
        self.assertEqual(reserved_cash({"kind": "P", "qty": -1, "strike": 743.0, "premium": 5.12}), 74300.0)

    def test_cash_skip_reserves_nothing(self):
        self.assertEqual(reserved_cash({"kind": "CASH", "qty": 0, "strike": 0}), 0.0)

    def test_paper_book_nav_is_cash_plus_realized_plus_mtm(self):
        summary = {
            "pnl": 200.0,
            "max_drawdown": -50.0,
            "win_rate": 1.0,
            "trades": 2,
            "settled": 1,
            "wins": 1,
            "mean_pnl": 200.0,
            "avg_win": 200.0,
            "avg_loss": 0.0,
            "profit_factor": None,
            "best": 200.0,
            "worst": 200.0,
            "ann": 0.1,
            "vs_put": 0.016,
        }
        live = {"kind": "CASH", "qty": 0}
        book = paper_book(summary, live, spot=100.0, sigma=0.2)
        self.assertEqual(book["nav"], STARTING_CASH + 200.0)
        self.assertEqual(book["reserved"], 0.0)
        self.assertEqual(book["free_cash"], STARTING_CASH + 200.0)

    def test_inference_events_open_then_close(self):
        fills = [
            {
                "model": "CESF", "date": "2026-01-02", "index": 10, "kind": "P",
                "moneyness": 0.0, "dte": 21, "qty": -1, "strike": 100.0, "premium": 1.5,
                "delta": 0.45, "gamma": -0.02, "ev_pnl_frac": 0.01, "score": -0.01,
                "spot_in": 100.0, "spot_out": 102.0, "settle_date": "2026-02-02",
                "pnl": 150.0, "status": "SETTLED", "cesf_bits": 0.0, "cesf_events": 1,
                "cesf_classes": 1, "crash_mass": 0.04, "sigma": 0.2, "rsi": 55,
            }
        ]
        tape = inference_events(fills)
        self.assertEqual([e["action"] for e in tape], ["OPEN", "CLOSE"])
        self.assertIn("Crash-mass", tape[0]["reason"])
        self.assertIn("PnL", tape[1]["reason"])


class OverlayTests(unittest.TestCase):
    def test_trading_days_skip_weekend(self):
        from datetime import date

        days = trading_days_ahead(date(2026, 8, 14), 5)
        self.assertEqual(len(days), 5)
        self.assertTrue(all(date.fromisoformat(d).weekday() < 5 for d in days))

    def test_shorter_tenor_atm_put_is_cheaper(self):
        self.assertGreater(_mark("P", 100.0, 100.0, 21, 0.2), _mark("P", 100.0, 100.0, 5, 0.2))

    def test_overlay_has_spot_and_strike(self):
        dates, closes = synthetic_series(n=220, seed=8)
        goal = PortfolioGoal(ev_target=0.01, delta_target=0.45, gamma_target=-0.02)
        report = walk_forward(
            dates, closes, goal, "SYN", n_paths=40, lookback=120, stride=21, strategy="putwrite"
        )
        overlay = asset_overlay(report, float(closes[-1]), 0.2, dates[-1])
        self.assertGreaterEqual(len(overlay["asset"]), 2)
        live = report["live"]["RAW"]
        if live and live.get("kind") != "CASH":
            self.assertTrue(overlay["strikes"])
            self.assertEqual(overlay["strikes"][0]["strike"], live["strike"])
            self.assertTrue(overlay["option"]["RAW"] or overlay["option"]["CESF"])


class BrokerTests(unittest.TestCase):
    def test_sell_fills_at_bid(self):
        self.assertEqual(fill_price(-1, 1.50, 1.70, 1.60), 1.50)

    def test_buy_fills_at_ask(self):
        self.assertEqual(fill_price(1, 1.50, 1.70, 1.60), 1.70)

    def test_pick_expiry_near_21dte(self):
        from datetime import date

        expiry = pick_expiry(
            ["2026-08-21", "2026-09-18", "2026-12-18"],
            21,
            today=date(2026, 8, 17),
        )
        self.assertEqual(expiry, "2026-09-18")


if __name__ == "__main__":
    unittest.main()
