"""Compact character encoding for microgpt-c (BLOCK_SIZE=48).

Contract line:
  TICK rsi m h bits E±ev D±delta G±gamma > KIND M±mn Tdd

Transform line (raw window → RSI / MAs):
  W + 14 return digits + / rsi m h .
"""

from __future__ import annotations

import re

from contracts import Contract
from features import window_features

_CONTRACT_RE = re.compile(
    r"^(?P<ticker>.{4})"
    r"(?P<rsi>\d{2})"
    r"(?P<m>\d)(?P<h>\d)"
    r"(?P<bits>\d{2})"
    r"E(?P<es>[+-])(?P<ev>\d{2})"
    r"D(?P<ds>[+-])(?P<delta>\d{2})"
    r"G(?P<gs>[+-])(?P<gamma>\d{2})"
    r">(?P<kind>[CP])"
    r"M(?P<ms>[+-])(?P<mn>\d{2})"
    r"T(?P<dte>\d{2})$"
)


def _pad_ticker(ticker: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", ticker.upper())[:4]
    return t.ljust(4, "-")


def _signed2(prefix: str, value: float, scale: float) -> str:
    n = int(round(value * scale))
    n = max(-99, min(99, n))
    sign = "+" if n >= 0 else "-"
    return f"{prefix}{sign}{abs(n):02d}"


def _rsi2(rsi_value: float) -> str:
    return f"{int(max(0, min(99, round(rsi_value)))):02d}"


def encode_contract_line(
    ticker: str,
    feats: dict,
    complexity_bits: float,
    ev_target: float,
    delta_target: float,
    gamma_target: float,
    contract: Contract,
) -> str:
    body = (
        f"{_pad_ticker(ticker)}"
        f"{_rsi2(float(feats['rsi']))}"
        f"{feats['ma50_digit']}{feats['ma100_digit']}"
        f"{int(max(0, min(99, round(complexity_bits * 10)))):02d}"
        f"{_signed2('E', ev_target, 100.0)}"
        f"{_signed2('D', delta_target, 100.0)}"
        f"{_signed2('G', gamma_target, 1000.0)}"
        f">{contract.kind}"
        f"{_signed2('M', contract.moneyness, 100.0)}"
        f"T{contract.dte:02d}."
    )
    return body


def encode_transform_line(closes) -> str:
    feats = window_features(closes)
    return f"W{feats['qrets']}/{_rsi2(float(feats['rsi']))}{feats['ma50_digit']}{feats['ma100_digit']}."


def context_prefix(line: str) -> str:
    if ">" not in line:
        return line
    return line.split(">", 1)[0] + ">"


def decode_contract(line: str) -> Contract:
    match = _CONTRACT_RE.match(line.strip().rstrip("."))
    if not match:
        raise ValueError(f"unreadable contract line: {line!r}")
    sign = 1.0 if match.group("ms") == "+" else -1.0
    return Contract(
        kind=match.group("kind"),
        moneyness=sign * int(match.group("mn")) / 100.0,
        dte=int(match.group("dte")),
    )
