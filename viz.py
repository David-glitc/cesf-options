"""Render the CESF vs raw-MC paper-trade story as a single HTML page."""

from __future__ import annotations

import json
from pathlib import Path


def _contract_label(fill: dict | None) -> str:
    if not fill:
        return "—"
    sign = "+" if fill["moneyness"] >= 0 else ""
    return f"{fill['kind']} {sign}{fill['moneyness']*100:.0f}%  {fill['dte']}d"


def render_html(report: dict) -> str:
    payload = json.dumps(report, default=str)
    ticker = report["ticker"]
    asof = report["asof"]
    spot = report["spot"]
    goal = report["goal"]
    cesf = report["cesf"]
    raw = report["raw"]
    live_c = report["live"]["CESF"]
    live_r = report["live"]["RAW"]
    winner = "CESF" if cesf["pnl"] >= raw["pnl"] else "RAW"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CESF vs raw Monte Carlo — {ticker}</title>
<style>
  :root {{
    --bg: #0f1210;
    --ink: #e8eee6;
    --muted: #8b9586;
    --line: #2a3228;
    --cesf: #c4f542;
    --raw: #7ab8ff;
    --card: #171c18;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font: 15px/1.45 "Source Serif 4", "Iowan Old Style", Georgia, serif;
    background: var(--bg);
    color: var(--ink);
  }}
  header, main {{ max-width: 1100px; margin: 0 auto; padding: 0 24px; }}
  header {{ padding-top: 36px; padding-bottom: 8px; }}
  h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 8px; }}
  .lede {{ color: var(--muted); max-width: 46em; }}
  .meta {{ font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); margin-top: 12px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 24px 0; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 18px;
  }}
  .card h2 {{ margin: 0 0 10px; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; }}
  .ticket {{ font-family: ui-monospace, monospace; font-size: 22px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .kpi .v {{ font-family: ui-monospace, monospace; font-size: 20px; }}
  .kpi .l {{ color: var(--muted); font-size: 12px; font-family: ui-sans-serif, system-ui, sans-serif; }}
  .cesf {{ color: var(--cesf); }}
  .raw {{ color: var(--raw); }}
  svg {{ width: 100%; height: 280px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 12px; }}
  th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 8px 6px; border-bottom: 1px solid var(--line); }}
  td {{ padding: 7px 6px; border-bottom: 1px solid var(--line); }}
  .pos {{ color: var(--cesf); }}
  .neg {{ color: #ff8a7a; }}
  footer {{ color: var(--muted); font-size: 12px; padding: 28px 0 48px; }}
  @media (max-width: 800px) {{
    .grid, .kpis {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Same market. Same goal. Two futures.</h1>
  <p class="lede">
    Both models see identical GBM Monte Carlo paths, the same admissibility filter C,
    the same option grid, and the same portfolio targets
    (EV* {goal['ev_target']}, Δ* {goal['delta_target']}, Γ* {goal['gamma_target']}).
    The only difference: <span class="cesf">CESF</span> scores contracts on the
    operational event space E_H(Q). <span class="raw">RAW</span> scores them on every
    admissible path with equal weight — no ε-graph, no relevance filter.
  </p>
  <p class="meta">{ticker} · last {spot:.2f} · {asof} · {report['bars']} bars ·
    {report['n_paths']} paths · paper {report['starting_cash']:.0f} USD · multiplier {report['multiplier']:.0f}
    · agreement {report['agreement_rate']*100:.0f}% · mean CESF complexity {report['mean_cesf_bits']:.2f} bits</p>
</header>
<main>
  <div class="grid">
    <div class="card">
      <h2>Live paper ticket — CESF</h2>
      <div class="ticket cesf">{_contract_label(live_c)}</div>
      <p class="meta">premium {live_c['premium'] if live_c else '—'} · Δ {live_c['delta'] if live_c else '—'} · Γ {live_c['gamma'] if live_c else '—'} · {live_c['status'] if live_c else ''}</p>
    </div>
    <div class="card">
      <h2>Live paper ticket — RAW MC</h2>
      <div class="ticket raw">{_contract_label(live_r)}</div>
      <p class="meta">premium {live_r['premium'] if live_r else '—'} · Δ {live_r['delta'] if live_r else '—'} · Γ {live_r['gamma'] if live_r else '—'} · {live_r['status'] if live_r else ''}</p>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="l">CESF paper P&amp;L</div><div class="v cesf" id="cesf-pnl"></div></div>
    <div class="kpi"><div class="l">RAW paper P&amp;L</div><div class="v raw" id="raw-pnl"></div></div>
    <div class="kpi"><div class="l">CESF win rate</div><div class="v" id="cesf-wr"></div></div>
    <div class="kpi"><div class="l">Leader</div><div class="v">{winner}</div></div>
  </div>

  <h2 style="margin:28px 0 8px;font-size:16px;">Paper equity (settled contracts)</h2>
  <svg id="equity" viewBox="0 0 1000 280" preserveAspectRatio="none"></svg>
  <p class="meta">Source: walk-forward paper blotter · 1 lot × 100 multiplier · hold to DTE or last bar</p>

  <div class="grid">
    <div class="card">
      <h2>CESF book</h2>
      <p>P&amp;L {cesf['pnl']} · equity {cesf['equity']} · DD {cesf['max_drawdown']} · Δ err {cesf['mean_delta_error']} · Γ err {cesf['mean_gamma_error']} · settled {cesf['settled']}/{cesf['trades']}</p>
    </div>
    <div class="card">
      <h2>RAW book</h2>
      <p>P&amp;L {raw['pnl']} · equity {raw['equity']} · DD {raw['max_drawdown']} · Δ err {raw['mean_delta_error']} · Γ err {raw['mean_gamma_error']} · settled {raw['settled']}/{raw['trades']}</p>
    </div>
  </div>

  <h2 style="margin:28px 0 8px;font-size:16px;">Blotter</h2>
  <table>
    <thead>
      <tr>
        <th>date</th><th>model</th><th>contract</th><th>spot in</th><th>spot out</th>
        <th>premium</th><th>Δ</th><th>bits</th><th>P&amp;L</th><th>status</th>
      </tr>
    </thead>
    <tbody id="blotter"></tbody>
  </table>
</main>
<footer>
  Controlled parallel: seed, n_paths, C, grid, and (EV*, Δ*, Γ*) are shared.
  CESF contracts possibility with ε=0.088, H=42. Not live brokerage — paper marks only.
</footer>
<script>
const DATA = {payload};
function money(x) {{
  const n = Number(x);
  const s = (n < 0 ? "-" : "") + "$" + Math.abs(n).toFixed(2);
  return s;
}}
function label(f) {{
  const sign = f.moneyness >= 0 ? "+" : "";
  return f.kind + " " + sign + (f.moneyness * 100).toFixed(0) + "% " + f.dte + "d";
}}
document.getElementById("cesf-pnl").textContent = money(DATA.cesf.pnl);
document.getElementById("raw-pnl").textContent = money(DATA.raw.pnl);
document.getElementById("cesf-wr").textContent = (DATA.cesf.win_rate * 100).toFixed(0) + "% vs " + (DATA.raw.win_rate * 100).toFixed(0) + "% RAW";

const svg = document.getElementById("equity");
function seriesPath(arr, color) {{
  if (!arr.length) return;
  const w = 1000, h = 280, pad = 24;
  const both = DATA.cesf.equity_curve.concat(DATA.raw.equity_curve);
  const mn = Math.min.apply(null, both), mx = Math.max.apply(null, both);
  const xs = (i) => pad + (w - 2 * pad) * i / Math.max(arr.length - 1, 1);
  const ys = (v) => h - pad - (h - 2 * pad) * ((v - mn) / Math.max(mx - mn, 1e-9));
  let d = "";
  arr.forEach((v, i) => {{ d += (i ? "L" : "M") + xs(i) + " " + ys(v) + " "; }});
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", d);
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", color);
  p.setAttribute("stroke-width", "2.2");
  svg.appendChild(p);
}}
seriesPath(DATA.cesf.equity_curve, "#c4f542");
seriesPath(DATA.raw.equity_curve, "#7ab8ff");

const tb = document.getElementById("blotter");
DATA.fills.sort((a, b) => a.index - b.index || a.model.localeCompare(b.model));
for (const f of DATA.fills) {{
  const tr = document.createElement("tr");
  const pnl = f.pnl == null ? "—" : money(f.pnl);
  const cls = f.pnl == null ? "" : (f.pnl >= 0 ? "pos" : "neg");
  tr.innerHTML = `<td>${{f.date}}</td><td class="${{f.model === "CESF" ? "cesf" : "raw"}}">${{f.model}}</td>
    <td>${{label(f)}}</td><td>${{f.spot_in}}</td><td>${{f.spot_out ?? "—"}}</td>
    <td>${{f.premium}}</td><td>${{f.delta}}</td><td>${{f.cesf_bits ?? "—"}}</td>
    <td class="${{cls}}">${{pnl}}</td><td>${{f.status}}</td>`;
  tb.appendChild(tr);
}}
</script>
</body>
</html>
"""


def write_html(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
