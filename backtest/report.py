"""
backtest/report.py

Generates visual performance reports from backtest results.
Creates equity curves, drawdown charts, trade distributions,
and a comprehensive HTML report.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from typing import List
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


DARK_BG   = "#0f1318"
PANEL_BG  = "#161c24"
ACCENT    = "#00e5a0"
ACCENT2   = "#0099ff"
DANGER    = "#ff4466"
WARN      = "#ffaa00"
MUTED     = "#5a6a7a"
TEXT      = "#d4dde8"
GRID      = "#1e2730"


def plot_results(results: dict, strategy_name: str = "Strategy", output_dir: str = "backtest_results"):
    """
    Generate a full visual performance report.
    Saves PNG chart and HTML report to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    equity   = results["equity_curve"]
    trades   = results["trades"]
    metrics  = results["metrics"]
    benchmark = results.get("benchmark")

    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    fig.suptitle(
        f"SET AI Trader · {strategy_name} · Backtest Report",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Equity Curve ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PANEL_BG)
    ax1.plot(equity.index, equity["total"], color=ACCENT, linewidth=2, label="Strategy")
    if benchmark is not None and len(benchmark) > 0:
        bench_aligned = benchmark.reindex(equity.index, method="ffill")
        ax1.plot(bench_aligned.index, bench_aligned, color=MUTED,
                 linewidth=1.2, linestyle="--", label="Buy & Hold", alpha=0.8)
    ax1.fill_between(equity.index, equity["total"],
                     equity["total"].min(), alpha=0.08, color=ACCENT)
    ax1.set_title("Equity Curve", color=TEXT, fontsize=10, pad=8)
    ax1.set_ylabel("Portfolio Value (THB)", color=MUTED, fontsize=8)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"฿{x:,.0f}"))
    ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT, edgecolor=GRID)
    _style_ax(ax1)

    # ── Panel 2: Drawdown ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(PANEL_BG)
    rolling_max = equity["total"].cummax()
    drawdown    = (equity["total"] - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown, 0, color=DANGER, alpha=0.4)
    ax2.plot(drawdown.index, drawdown, color=DANGER, linewidth=1)
    ax2.set_title("Drawdown", color=TEXT, fontsize=10, pad=8)
    ax2.set_ylabel("Drawdown %", color=MUTED, fontsize=8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    _style_ax(ax2)

    # ── Panel 3: Trade P&L Distribution ──────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.set_facecolor(PANEL_BG)
    closed = [t for t in trades if not t.is_open and t.realized_pct is not None]
    if closed:
        pnls = [t.realized_pct for t in closed]
        colors = [ACCENT if p > 0 else DANGER for p in pnls]
        dates  = [t.exit_date for t in closed]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.8, width=0.6)
        ax3.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax3.set_title("Trade P&L (per trade %)", color=TEXT, fontsize=10, pad=8)
        ax3.set_ylabel("Return %", color=MUTED, fontsize=8)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax3.set_xlabel("Trade #", color=MUTED, fontsize=8)
    _style_ax(ax3)

    # ── Panel 4: Key Metrics ──────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_facecolor(PANEL_BG)
    ax4.axis("off")
    m = metrics
    grade, _ = _grade(m)
    grade_color = ACCENT if grade in ("A+", "A") else WARN if grade == "B" else DANGER

    kv_pairs = [
        ("TOTAL RETURN",    f"{m['total_return_pct']:+.1f}%"),
        ("CAGR",            f"{m['cagr_pct']:+.1f}%"),
        ("SHARPE RATIO",    f"{m['sharpe_ratio']:.2f}"),
        ("MAX DRAWDOWN",    f"{m['max_drawdown_pct']:.1f}%"),
        ("WIN RATE",        f"{m['win_rate_pct']:.1f}%"),
        ("PROFIT FACTOR",   f"{m['profit_factor']:.2f}"),
        ("TOTAL TRADES",    str(m['total_trades'])),
        ("AVG HOLD",        f"{m['avg_holding_days']:.0f} days"),
    ]
    if m.get("benchmark_return"):
        kv_pairs.append(("vs BENCHMARK", f"{m['alpha']:+.1f}%"))

    ax4.text(0.5, 0.97, "PERFORMANCE", ha="center", va="top", color=MUTED,
             fontsize=7, fontweight="bold", transform=ax4.transAxes, fontfamily="monospace")
    ax4.text(0.5, 0.88, grade, ha="center", va="top", color=grade_color,
             fontsize=32, fontweight="bold", transform=ax4.transAxes)

    for i, (k, v) in enumerate(kv_pairs):
        y = 0.72 - i * 0.085
        ax4.text(0.05, y, k, ha="left", va="top", color=MUTED,
                 fontsize=7, fontfamily="monospace", transform=ax4.transAxes)
        col = ACCENT if (("RETURN" in k or "CAGR" in k or "vs" in k) and "+" in v) else \
              DANGER if (("RETURN" in k or "CAGR" in k or "vs" in k) and "-" in v) else TEXT
        ax4.text(0.95, y, v, ha="right", va="top", color=col,
                 fontsize=8, fontweight="bold", fontfamily="monospace", transform=ax4.transAxes)

    # ── Panel 5: Monthly Returns Heatmap ─────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PANEL_BG)
    monthly = equity["total"].resample("ME").last().pct_change().dropna() * 100
    if len(monthly) > 0:
        years  = sorted(set(monthly.index.year))
        months = list(range(1, 13))
        matrix = pd.DataFrame(index=years, columns=months, dtype=float)
        for dt, val in monthly.items():
            matrix.loc[dt.year, dt.month] = val

        im = ax5.imshow(matrix.values.astype(float),
                        cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto")
        ax5.set_xticks(range(12))
        ax5.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                            color=MUTED, fontsize=6)
        ax5.set_yticks(range(len(years)))
        ax5.set_yticklabels(years, color=MUTED, fontsize=6)
        ax5.set_title("Monthly Returns", color=TEXT, fontsize=10, pad=8)

        # Add text values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax5.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=5, color="white" if abs(val) > 5 else TEXT)

    _style_ax(ax5)

    # ── Panel 6: P&L Histogram ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL_BG)
    if closed:
        pnls = [t.realized_pct for t in closed]
        ax6.hist(pnls, bins=20, color=ACCENT, alpha=0.7, edgecolor=DARK_BG)
        ax6.axvline(0, color=DANGER, linewidth=1.5, linestyle="--")
        ax6.axvline(np.mean(pnls), color=WARN, linewidth=1.5, linestyle=":")
        ax6.set_title("Return Distribution", color=TEXT, fontsize=10, pad=8)
        ax6.set_xlabel("Return %", color=MUTED, fontsize=8)
        ax6.set_ylabel("# Trades", color=MUTED, fontsize=8)
    _style_ax(ax6)

    # Save
    chart_path = os.path.join(output_dir, "backtest_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"  Chart saved: {chart_path}")

    # Generate HTML report
    html_path = generate_html_report(results, strategy_name, output_dir, chart_path)
    return chart_path, html_path


def generate_html_report(results, strategy_name, output_dir, chart_path):
    """Generate a standalone HTML report with embedded chart."""
    m      = results["metrics"]
    trades = results["trades"]
    closed = [t for t in trades if not t.is_open]
    grade, grade_comment = _grade(m)
    grade_color = "#00e5a0" if grade in ("A+","A") else "#ffaa00" if grade == "B" else "#ff4466"

    # Embed chart as base64
    import base64
    with open(chart_path, "rb") as f:
        chart_b64 = base64.b64encode(f.read()).decode()

    trade_rows = ""
    for t in sorted(closed, key=lambda x: x.exit_date or datetime.min, reverse=True)[:50]:
        color = "#00e5a0" if t.realized_pnl >= 0 else "#ff4466"
        exit_r = t.exit_reason or ""
        trade_rows += f"""
        <tr>
          <td>{t.symbol}</td>
          <td>{t.entry_date.strftime('%Y-%m-%d')}</td>
          <td>{t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '-'}</td>
          <td>{t.entry_price:.2f}</td>
          <td>{f"{t.exit_price:.2f}" if t.exit_price else '-'}</td>
          <td>{t.shares}</td>
          <td style="color:{color}">{t.realized_pnl:+,.0f}</td>
          <td style="color:{color}">{t.realized_pct:+.1f}%</td>
          <td>{t.holding_days}d</td>
          <td><span class="tag tag-{exit_r.lower().replace('_','-')}">{exit_r}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report: {strategy_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #090c10; color: #d4dde8; font-family: 'Segoe UI', sans-serif; padding: 32px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #5a6a7a; font-size: 13px; margin-bottom: 32px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: #161c24; border: 1px solid #1e2730; padding: 18px 20px; }}
  .kpi-label {{ font-size: 10px; letter-spacing: 0.15em; color: #5a6a7a; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; font-family: monospace; }}
  .grade {{ font-size: 48px; font-weight: 900; color: {grade_color}; margin: 0 24px; }}
  .grade-row {{ display: flex; align-items: center; background: #161c24; border: 1px solid #1e2730; padding: 20px 24px; margin-bottom: 32px; }}
  .grade-text {{ color: #5a6a7a; font-size: 13px; }}
  img {{ width: 100%; border: 1px solid #1e2730; margin-bottom: 32px; }}
  table {{ width: 100%; border-collapse: collapse; background: #161c24; font-size: 12px; font-family: monospace; }}
  th {{ background: #0f1318; color: #5a6a7a; padding: 10px 14px; text-align: left; font-size: 10px; letter-spacing: 0.1em; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1e2730; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .tag {{ font-size: 9px; padding: 2px 7px; border-radius: 2px; font-weight: 700; }}
  .tag-stop-loss {{ background: rgba(255,68,102,0.15); color: #ff4466; }}
  .tag-take-profit {{ background: rgba(0,229,160,0.15); color: #00e5a0; }}
  .tag-end-of-backtest {{ background: rgba(90,106,122,0.15); color: #5a6a7a; }}
  .section-title {{ font-size: 11px; letter-spacing: 0.15em; color: #5a6a7a; margin-bottom: 12px; font-weight: 700; }}
  .up {{ color: #00e5a0; }} .down {{ color: #ff4466; }}
</style>
</head>
<body>
<h1>SET AI Trader · Backtest Report</h1>
<div class="sub">{strategy_name} · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="grade-row">
  <div class="grade">{grade}</div>
  <div>
    <div style="font-size:16px; font-weight:700; margin-bottom:6px">Strategy Grade</div>
    <div class="grade-text">{grade_comment}</div>
  </div>
</div>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-label">TOTAL RETURN</div><div class="kpi-value {'up' if m['total_return_pct']>0 else 'down'}">{m['total_return_pct']:+.1f}%</div></div>
  <div class="kpi"><div class="kpi-label">CAGR</div><div class="kpi-value {'up' if m['cagr_pct']>0 else 'down'}">{m['cagr_pct']:+.1f}%</div></div>
  <div class="kpi"><div class="kpi-label">SHARPE RATIO</div><div class="kpi-value">{m['sharpe_ratio']:.2f}</div></div>
  <div class="kpi"><div class="kpi-label">MAX DRAWDOWN</div><div class="kpi-value down">{m['max_drawdown_pct']:.1f}%</div></div>
  <div class="kpi"><div class="kpi-label">WIN RATE</div><div class="kpi-value">{m['win_rate_pct']:.1f}%</div></div>
  <div class="kpi"><div class="kpi-label">PROFIT FACTOR</div><div class="kpi-value">{m['profit_factor']:.2f}</div></div>
  <div class="kpi"><div class="kpi-label">TOTAL TRADES</div><div class="kpi-value">{m['total_trades']}</div></div>
  <div class="kpi"><div class="kpi-label">AVG HOLD</div><div class="kpi-value">{m['avg_holding_days']:.0f}d</div></div>
  <div class="kpi"><div class="kpi-label">STARTING CAPITAL</div><div class="kpi-value">฿{m['starting_capital']:,.0f}</div></div>
  <div class="kpi"><div class="kpi-label">ENDING CAPITAL</div><div class="kpi-value {'up' if m['ending_capital']>m['starting_capital'] else 'down'}">฿{m['ending_capital']:,.0f}</div></div>
</div>

<img src="data:image/png;base64,{chart_b64}" alt="Backtest Chart">

<div class="section-title">TRADE LOG (Last 50 Trades)</div>
<table>
<thead><tr>
  <th>SYMBOL</th><th>ENTRY</th><th>EXIT</th><th>BUY ฿</th><th>SELL ฿</th>
  <th>SHARES</th><th>P&L (฿)</th><th>RETURN</th><th>DAYS</th><th>EXIT REASON</th>
</tr></thead>
<tbody>{trade_rows}</tbody>
</table>
</body>
</html>"""

    html_path = os.path.join(output_dir, "backtest_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Report saved: {html_path}")
    return html_path


def _style_ax(ax):
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _grade(m):
    sharpe = m["sharpe_ratio"]
    cagr   = m["cagr_pct"]
    dd     = abs(m["max_drawdown_pct"])
    if sharpe >= 1.5 and cagr >= 20 and dd <= 20: return "A+", "Exceptional. Institutional-grade performance."
    if sharpe >= 1.0 and cagr >= 15 and dd <= 25: return "A",  "Strong. Consistent alpha with controlled risk."
    if sharpe >= 0.7 and cagr >= 10:               return "B",  "Good. Solid returns, acceptable risk."
    if sharpe >= 0.5 and cagr >= 5:                return "C",  "Mediocre. Needs more work before live trading."
    if cagr > 0:                                   return "D",  "Weak. Barely positive. Significant work needed."
    return "F", "Strategy loses money. Do not trade live."
