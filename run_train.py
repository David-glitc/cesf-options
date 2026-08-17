#!/usr/bin/env python3
"""Build a CESF-labeled options corpus and train microgpt-c on it."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cesf_filter import PortfolioGoal, calibrate_gbm, run_underlying_cesf, select_contract
from corpus import build_corpus_lines, fetch_closes, synthetic_closes, write_corpus
from encode import context_prefix, encode_contract_line
from features import window_features

MICROGPT_DIR = HERE / "microgpt-c"
BIN = MICROGPT_DIR / "microgpt"
SRC = MICROGPT_DIR / "src" / "microgpt.c"
CORPUS = HERE / "data" / "options_corpus.txt"


def compile_microgpt() -> None:
    import platform

    cmd = [
        "cc",
        "-O3",
        "-march=native",
        "-ffast-math",
        "-Wall",
        "-DBLOCK_SIZE=48",
        "-o",
        str(BIN),
        str(SRC),
        "-lm",
    ]
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        cmd[3:3] = ["-mavx2", "-mfma"]
    subprocess.check_call(cmd, cwd=MICROGPT_DIR)


def train_and_sample(prefix: str | None, steps: int, samples: int) -> None:
    cmd = [
        str(BIN),
        str(CORPUS),
        f"--steps={steps}",
        f"--samples={samples}",
        "--no-bench",
    ]
    if prefix:
        cmd.append(f"--prefix={prefix}")
    cmd.append("--temp=0.2")
    cmd.append("--greedy")
    subprocess.check_call(cmd, cwd=MICROGPT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="CESF + microgpt-c options trainer")
    parser.add_argument("--ticker", default="SYN")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--paths", type=int, default=250)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--ev", type=float, default=0.02, help="target expected P&L / spot")
    parser.add_argument("--delta", type=float, default=0.25, help="target option delta")
    parser.add_argument("--gamma", type=float, default=0.02, help="target option gamma")
    parser.add_argument("--repeats", type=int, default=80, help="repeat contract lines for the tiny GPT")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    goal = PortfolioGoal(
        ev_target=args.ev,
        delta_target=args.delta,
        gamma_target=args.gamma,
    )
    if args.synthetic or args.ticker == "SYN":
        ticker = "SYN"
        closes = synthetic_closes()
    else:
        ticker = args.ticker
        try:
            closes = fetch_closes(ticker)
        except Exception as exc:
            print(f"yfinance failed ({exc}); falling back to synthetic GBM", file=sys.stderr)
            ticker = "SYN"
            closes = synthetic_closes()

    print(f"building corpus ticker={ticker} bars={len(closes)} paths={args.paths}", flush=True)
    lines = build_corpus_lines(
        closes,
        ticker,
        goal,
        n_paths=args.paths,
        lookback=args.lookback,
        stride=args.stride,
    )
    write_corpus(CORPUS, lines, contract_repeats=args.repeats)
    n_written = len(CORPUS.read_text().splitlines())
    print(f"wrote {n_written} lines ({len(lines)} unique) -> {CORPUS}", flush=True)

    window = closes[-args.lookback :]
    feats = window_features(window)
    result = run_underlying_cesf(window, n_paths=args.paths, seed=1)
    system = calibrate_gbm(window, seed=1)
    chosen = select_contract(result, sigma=system.sigma, goal=goal)
    teacher = encode_contract_line(
        ticker,
        feats,
        result.complexity_bits,
        goal.ev_target,
        goal.delta_target,
        goal.gamma_target,
        chosen.contract,
    )
    prefix = context_prefix(teacher)
    print("CESF teacher contract:", teacher, flush=True)
    print(
        f"  bits={result.complexity_bits:.2f} events={result.n_events} "
        f"classes={result.n_classes} ev={chosen.ev_pnl_frac:.4f} "
        f"delta={chosen.delta:.3f} gamma={chosen.gamma:.4f}",
        flush=True,
    )
    print("prefix:", prefix, flush=True)

    if args.skip_train:
        return
    compile_microgpt()
    train_and_sample(prefix, steps=args.steps, samples=args.samples)


if __name__ == "__main__":
    main()
