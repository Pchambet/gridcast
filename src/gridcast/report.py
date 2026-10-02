"""Build the static report page (``site/index.html``) and refresh README result blocks.

The page depends only on committed result tables and, when present, the live forecast
log, so the Pages workflow can rebuild it without downloading any raw data.
"""

from __future__ import annotations

import html
import json
import re

import pandas as pd
import plotly.graph_objects as go

from gridcast import config, live
from gridcast.conformal import METHODS
from gridcast.evaluate import POINT_MODELS

REPO = "https://github.com/Pchambet/gridcast"
TEAL, AMBER, SLATE, INK = "#0d9488", "#d97706", "#64748b", "#0f172a"
METHOD_COLORS = {
    "qr_raw": "#a855f7",
    "split_static": AMBER,
    "split_static_aci": "#0369a1",
    "split_rolling": "#94a3b8",
    "cqr_rolling": "#475569",
    "cqr_rolling_aci": TEAL,
}


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _gw(x: float) -> str:
    return f"{x / 1000:.2f} GW"


def _load() -> dict:
    r = config.RESULTS
    return {
        "summary": json.loads((r / "summary.json").read_text()),
        "intervals": pd.read_csv(r / "interval_metrics.csv"),
        "points": pd.read_csv(r / "point_metrics.csv"),
        "monthly": pd.read_csv(r / "monthly_mape.csv", parse_dates=["month"]),
        "daily": pd.read_csv(r / "coverage_daily.csv", parse_dates=["day"]),
        "fan": pd.read_csv(r / "fan_week.csv"),
        "gamma": pd.read_csv(r / "aci_gamma_sensitivity.csv"),
        "rte_bias": pd.read_csv(r / "rte_bias.csv", dtype={"segment": str}),
    }


# --- Charts (plotly figure dicts; theming happens in the browser) --------------------


def _layout(**kw) -> dict:
    base = {
        "margin": {"l": 56, "r": 16, "t": 64, "b": 36},
        "height": 340,
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        "xaxis": {"showgrid": False},
        "yaxis": {"zeroline": False},
    }
    base.update(kw)
    return base


def _crisis_shape() -> dict:
    lo, hi = config.CRISIS
    return {
        "type": "rect", "xref": "x", "yref": "paper", "x0": str(lo.date()),
        "x1": str(hi.date()), "y0": 0, "y1": 1, "fillcolor": "rgba(100,116,139,0.10)",
        "line": {"width": 0}, "layer": "below",
    }  # fmt: skip


def chart_coverage(daily: pd.DataFrame, level: float) -> go.Figure:
    fig = go.Figure()
    shown = {"split_static", "cqr_rolling_aci"}
    for method in METHODS:
        g = daily[(daily["method"] == method.name) & (daily["level"] == level)]
        g = g.sort_values("day").set_index("day")
        roll = g["hits"].rolling("30D").sum() / g["hours"].rolling("30D").sum()
        roll = roll[roll.index >= g.index[0] + pd.Timedelta(days=29)]
        fig.add_trace(
            go.Scatter(
                x=roll.index, y=roll.round(4), name=method.label, mode="lines",
                line={"width": 2 if method.name in shown else 1.4,
                      "color": METHOD_COLORS[method.name]},
                visible=True if method.name in shown else "legendonly",
                hovertemplate="%{y:.1%}",
            )
        )  # fmt: skip
    band = {"type": "rect", "xref": "paper", "x0": 0, "x1": 1, "y0": level - 0.05,
            "y1": min(level + 0.05, 1), "fillcolor": "rgba(13,148,136,0.10)",
            "line": {"width": 0}, "layer": "below"}  # fmt: skip
    nominal = {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": level, "y1": level,
               "line": {"dash": "dash", "width": 1, "color": SLATE}}  # fmt: skip
    fig.update_layout(
        **_layout(
            shapes=[_crisis_shape(), band, nominal],
            height=360,
            yaxis={"tickformat": ".0%", "title": "30-day coverage", "range": [0.45, 1.0]},
        )
    )
    return fig


def chart_mape(monthly: pd.DataFrame) -> go.Figure:
    monthly = monthly[monthly["month"] >= config.EVAL_START]
    fig = go.Figure()
    styles = {
        "point": (TEAL, "solid"), "point_raw": (TEAL, "dash"), "rte_j1_adj": (INK, "solid"),
        "rte_j1": (SLATE, "dash"), "point_oracle": (TEAL, "dot"), "naive": (SLATE, "dot"),
    }  # fmt: skip
    hidden = {"naive", "point_raw", "point_oracle"}
    for col, label in POINT_MODELS.items():
        color, dash = styles[col]
        fig.add_trace(
            go.Scatter(
                x=monthly["month"], y=monthly[col].round(5), name=label, mode="lines",
                line={"color": color, "dash": dash, "width": 2},
                visible="legendonly" if col in hidden else True, hovertemplate="%{y:.2%}",
            )
        )  # fmt: skip
    fig.update_layout(
        **_layout(
            shapes=[_crisis_shape()],
            yaxis={"tickformat": ".1%", "title": "monthly MAPE", "rangemode": "tozero"},
        )
    )
    return fig


def chart_fan(
    frame: pd.DataFrame, rte_col: str = "rte_j1", rte_label: str = "RTE J-1"
) -> go.Figure:
    t = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(config.PARIS)
    t = t.dt.tz_localize(None)
    fig = go.Figure()
    for lo, hi, alpha, name in (("lo95", "hi95", 0.14, "95% interval"),
                                ("lo80", "hi80", 0.30, "80% interval")):  # fmt: skip
        fig.add_trace(go.Scatter(x=t, y=frame[hi] / 1e3, mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip"))  # fmt: skip
        fig.add_trace(
            go.Scatter(x=t, y=frame[lo] / 1e3, mode="lines", line={"width": 0}, fill="tonexty",
                       fillcolor=f"rgba(13,148,136,{alpha})", name=name, hoverinfo="skip")
        )  # fmt: skip
    fig.add_trace(go.Scatter(x=t, y=(frame["point"] / 1e3).round(2), name="gridcast",
                             line={"color": TEAL, "width": 2},
                             hovertemplate="%{y:.1f} GW"))  # fmt: skip
    if frame[rte_col].notna().any():
        fig.add_trace(go.Scatter(x=t, y=(frame[rte_col] / 1e3).round(2), name=rte_label,
                                 line={"color": SLATE, "width": 1.5, "dash": "dash"},
                                 hovertemplate="%{y:.1f} GW"))  # fmt: skip
    if frame["load"].notna().any():
        fig.add_trace(go.Scatter(x=t, y=(frame["load"] / 1e3).round(2), name="actual",
                                 line={"color": INK, "width": 2},
                                 hovertemplate="%{y:.1f} GW"))  # fmt: skip
    fig.update_layout(**_layout(yaxis={"title": "GW"}))
    return fig


# --- HTML -----------------------------------------------------------------------------


def _interval_rows(table: pd.DataFrame, period: str) -> str:
    sub = table[table["period"] == period]
    rows = []
    for m in METHODS:
        cells = [f"<td>{html.escape(m.label)}</td>"]
        for level in config.LEVELS:
            r = sub[(sub["method"] == m.name) & (sub["level"] == level)].iloc[0]
            off = abs(r["coverage"] - level) > 0.02
            cls = ' class="off"' if off else ""
            cells.append(
                f"<td{cls}>{_pct(r['coverage'])}<span class=ci> [{_pct(r['coverage_lo'])}, "
                f"{_pct(r['coverage_hi'])}]</span></td>"
                f"<td>{r['width_mw'] / 1000:.2f}</td><td>{r['winkler_mw'] / 1000:.2f}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def _period_rows(table: pd.DataFrame, points: pd.DataFrame) -> str:
    rows = []
    for period in ("pre-crisis", "crisis", "post-crisis"):
        p = points[points["period"] == period].set_index("model")
        i = table[(table["period"] == period) & (table["level"] == 0.8)].set_index("method")
        rows.append(
            f"<tr><td>{period}</td><td>{_pct(p.loc['point', 'mape'], 2)}</td>"
            f"<td>{_pct(p.loc['rte_j1_adj', 'mape'], 2)}</td>"
            f"<td>{_pct(i.loc['split_static', 'coverage'])}</td>"
            f"<td>{_pct(i.loc['split_static_aci', 'coverage'])}</td>"
            f"<td>{_pct(i.loc['cqr_rolling_aci', 'coverage'])}</td></tr>"
        )
    return "\n".join(rows)


def _live_section(log: pd.DataFrame) -> tuple[str, dict | None]:
    if log.empty:
        return (
            (
                "<p class=muted>The daily job has not published yet. From its first run, "
                "tomorrow's forecast and the running track record appear here.</p>"
            ),
            None,
        )
    latest_day = log["day"].max()
    latest = log[log["day"] == latest_day]
    rec = live.track_record(log)
    peak = latest.loc[latest["point"].idxmax()]
    peak_time = pd.Timestamp(peak["time"]).tz_convert(config.PARIS)
    lead = (
        f"<p>Forecast for <strong>{latest_day:%A %d %B %Y}</strong>, issued "
        f"{pd.Timestamp(latest['issued_at'].iloc[0]).tz_convert(config.PARIS):%d %b %H:%M} "
        f"Paris time. Peak {_gw(peak['point'])} at {peak_time:%H:%M} "
        f"(80%: {_gw(peak['lo80'])} to {_gw(peak['hi80'])}).</p>"
    )
    if rec["days"] == 0:
        track = "<p class=muted>No forecast has been scored yet.</p>"
    else:
        track = (
            "<div class=scroll><table><thead><tr><th>Scored days</th><th>80% coverage</th><th>95% coverage</th>"
            "<th>MAPE gridcast</th><th>MAPE RTE J-1</th></tr></thead><tbody><tr>"
            f"<td>{rec['days']} ({rec['first_day']} to {rec['last_day']})</td>"
            f"<td>{_pct(rec['coverage80'])}</td><td>{_pct(rec['coverage95'])}</td>"
            f"<td>{_pct(rec['mape'], 2)}</td><td>{_pct(rec['mape_rte'], 2)}</td>"
            "</tr></tbody></table></div>"
        )
    recent = log[log["day"] >= latest_day - pd.Timedelta(days=6)]
    return lead + '<div class=chart id="c-live"></div>' + track, chart_fan(recent).to_json()


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gridcast</title>
<meta name="description" content="Day-ahead forecast of French electricity demand with
calibrated conformal intervals, benchmarked against RTE and re-run daily.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet"
 href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {{ --bg:#ffffff; --surface:#f8fafc; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --accent:#0d9488; --warn:#b45309; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0b1120;
  --surface:#111827; --ink:#e2e8f0; --muted:#94a3b8; --line:#1f2937; --accent:#2dd4bf;
  --warn:#fbbf24; }} }}
:root[data-theme="dark"] {{ --bg:#0b1120; --surface:#111827; --ink:#e2e8f0; --muted:#94a3b8;
  --line:#1f2937; --accent:#2dd4bf; --warn:#fbbf24; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:48px 16px 64px; }}
h1 {{ font-size:2.1rem; margin:0 0 4px; letter-spacing:-0.02em; }}
h2 {{ font-size:1.3rem; margin:48px 0 8px; letter-spacing:-0.01em; }}
p {{ margin:8px 0 12px; }}
a {{ color:var(--accent); }}
.lede {{ font-size:1.12rem; color:var(--muted); margin-top:0; }}
.muted, .note {{ color:var(--muted); font-size:0.9rem; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px;
  margin:28px 0 8px; }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; }}
.kpi b {{ display:block; font-size:1.6rem; font-variant-numeric:tabular-nums; }}
.kpi span {{ color:var(--muted); font-size:0.85rem; }}
.chart {{ width:100%; min-height:340px; margin:8px 0 4px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:0.88rem; margin:12px 0;
  font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:600; }}
td.off {{ color:var(--warn); font-weight:600; }}
.ci {{ color:var(--muted); font-size:0.8em; }}
footer {{ margin-top:56px; padding-top:16px; border-top:1px solid var(--line);
  color:var(--muted); font-size:0.88rem; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
</style>
</head>
<body>
<main>
<h1>gridcast</h1>
<p class="lede">Day-ahead forecast of French electricity demand with intervals that stay
honest, benchmarked against the grid operator and re-run every day in public.</p>
<p class="muted">Backtest {eval_start} to {eval_end} &middot; {eval_hours} hourly day-ahead
forecasts &middot; issued daily at 12:00 Paris time &middot;
<a href="{repo}">source code</a></p>

<div class="kpis">
 <div class="kpi"><b>{k_cov80}</b><span>coverage of the published 80% interval
 (target 80%)</span></div>
 <div class="kpi"><b>{k_cov95}</b><span>coverage of the published 95% interval
 (target 95%)</span></div>
 <div class="kpi"><b>{k_mape}</b><span>gridcast MAPE vs {k_rte} for RTE's own J-1
 forecast (both level-corrected)</span></div>
 <div class="kpi"><b>{k_win}</b><span>of 30-day windows within &plusmn;5 pp of 80%, vs
 {k_win_static} with a frozen calibration</span></div>
</div>

<h2>Tomorrow</h2>
{live_html}

<h2>Calibration through the energy crisis</h2>
<p>Conformal prediction guarantees coverage on average, for exchangeable data. Demand
forecast errors are not exchangeable: they jump when heating starts in November, shrink
in spring, and in autumn 2022 the whole level of French demand dropped. A calibration
frozen on April 2021 to March 2022 still covers {k_cov_static} of hours overall, yet its
30-day coverage swings between roughly 50% and 100% and is within &plusmn;5 pp of target
only {k_win_static} of the time. The published method (rolling CQR with adaptive
conformal inference) is on target {k_win} of the time at 80% and {k_win95} at 95%. No
method fully absorbs the crisis winter: over {crisis} the published 80% interval covered
{k_cov_crisis}. Click legend entries to compare the other methods; shaded: energy
crisis.</p>
<div class="chart" id="c-cov80"></div>
<p class="note">The 95% version tells the same story:</p>
<div class="chart" id="c-cov95"></div>

<h2>All interval methods, {eval_start} to {eval_end}</h2>
<div class="scroll"><table>
<thead><tr><th>Method</th><th>80% coverage [95% CI]</th><th>width GW</th><th>Winkler GW</th>
<th>95% coverage [95% CI]</th><th>width GW</th><th>Winkler GW</th></tr></thead>
<tbody>{interval_rows}</tbody></table></div>
<p class="note">Coverage CIs from a week-block bootstrap ({reps} resamples). Winkler
(interval) score: width plus 2/&alpha; times the miss distance; lower is better.
Highlighted: coverage more than 2 pp from nominal.</p>

<h2>By period</h2>
<div class="scroll"><table>
<thead><tr><th>Period</th><th>MAPE gridcast</th><th>MAPE RTE J-1 (corr.)</th>
<th>80% cov. frozen</th><th>80% cov. frozen + ACI</th><th>80% cov. CQR + ACI</th></tr>
</thead><tbody>{period_rows}</tbody></table></div>

<h2>Point accuracy: RTE wins, once compared fairly</h2>
<p>Scored naively against the consolidated demand series, RTE's published J-1 forecast
looks worse than gridcast ({k_rte_raw} vs {k_mape} MAPE). That comparison is wrong. From
2023 on, RTE's forecast sits {k_rte_bias} below the consolidated values on average (it was
within 1% in 2015-2021), most at night and in the early afternoon, while on the
real-time feed that replaces consolidated data from July 2026 its bias is {k_rt_bias}
and its MAPE {k_rt_mape}. Part of the gap is likely definitional (consolidated versus real-time data);
either way, a stable offset says nothing about skill. Both forecasts are therefore
corrected the same way, by their own mean error over the trailing 28 days at the same
local hour, using only errors at least two days old. On that footing RTE is clearly
better: {k_rte} vs {k_mape}. RTE uses far richer inputs (dozens of weather stations,
cloud cover, embedded solar, operational knowledge); gridcast sees one public
temperature forecast for ten cities. With observed instead of forecast weather the same
model reaches {k_oracle}. The contribution here is not a better point forecast but a
transparent, calibrated uncertainty layer anyone can audit.</p>
<div class="chart" id="c-mape"></div>

<h2>A week of the December 2022 cold snap</h2>
<div class="chart" id="c-fan"></div>

<h2>How it works</h2>
<ul>
<li><strong>Information set.</strong> The forecast for day D is issued at 12:00 Paris on
D-1. It may use demand up to 11:00 (one hour publication lag), calendar features and
temperature forecasts whose model run had finished by then (24 h-ahead values for the
first hours of D, 48 h-ahead after that). A test poisons every later value and checks
the features do not change.</li>
<li><strong>Model.</strong> LightGBM on calendar, similar-day demand lags, the latest
week-over-week level drift and a population-weighted national temperature (ten largest
metro areas) with exponential smoothing for thermal inertia. Refit monthly on all past
days. Temperature forecasts are bias-corrected per hour on a trailing 60-day window, and
the demand forecast by its own mean error over the trailing 28 days.</li>
<li><strong>Intervals.</strong> Split conformal on absolute residuals, conformalized
quantile regression (CQR) on LightGBM quantiles, and adaptive conformal inference
(Gibbs &amp; Cand&egrave;s, 2021) on top. Because the forecast for D+1 is issued before
D ends, every method learns from its errors with a two-day delay.</li>
<li><strong>Live.</strong> A scheduled job retrains, forecasts tomorrow, appends to a
public log on the <code>live-data</code> branch, scores past forecasts and redeploys
this page.</li>
</ul>

<h2>Limitations</h2>
<ul>
<li>Only temperature has archived day-ahead forecasts before 2024 in the open archive;
cloud cover and wind, which RTE uses, are not in the model.</li>
<li>Training uses reanalysis (ERA5) temperature, prediction uses forecasts; the
conformal layer absorbs the mismatch, the point forecast pays for it.</li>
<li>Twenty days (30 Dec 2023 to 19 Jan 2024) use JMA instead of GFS forecasts, the
only archived model covering them.</li>
<li>RTE's J-1 forecast may be published later than 12:00 on D-1, so the comparison
slightly favours RTE.</li>
<li>The model is trained on consolidated demand; the last two backtest months and the
live track record are scored on RTE's real-time series, which differs slightly.</li>
<li>Coverage is marginal over hours, not conditional on weather regime or hour; see the
repository for coverage by hour.</li>
</ul>

<footer>Built by <a href="https://github.com/Pchambet">Pierre Chambet</a> &mdash;
decision science for operations under uncertainty. Data: RTE eco2mix via ODR&Eacute;
(Licence Ouverte 2.0), Open-Meteo (CC BY 4.0). Page generated {generated}.</footer>
</main>
<script>
const FIGS = {figs};
function theme() {{
  const forced = document.documentElement.dataset.theme;
  const dark = forced ? forced === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  return dark
    ? {{ink: "#e2e8f0", grid: "#1f2937", muted: "#94a3b8"}}
    : {{ink: "#0f172a", grid: "#e2e8f0", muted: "#64748b"}};
}}
function draw() {{
  const t = theme();
  for (const [id, fig] of Object.entries(FIGS)) {{
    const el = document.getElementById(id);
    if (!el || !fig) continue;
    const layout = Object.assign({{}}, fig.layout, {{
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {{family: "Inter, system-ui, sans-serif", color: t.ink, size: 12}},
    }});
    for (const ax of ["xaxis", "yaxis"]) {{
      layout[ax] = Object.assign({{}}, fig.layout[ax] || {{}},
        {{gridcolor: t.grid, linecolor: t.grid, tickfont: {{color: t.muted}}}});
    }}
    // Ink-coloured series (actuals, RTE) follow the theme so they stay visible in dark mode.
    const data = fig.data.map((tr) => (tr.line && tr.line.color === "{ink}")
      ? Object.assign({{}}, tr, {{line: Object.assign({{}}, tr.line, {{color: t.ink}})}})
      : tr);
    Plotly.react(el, data, layout, {{displayModeBar: false, responsive: true}});
  }}
}}
draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
</script>
</body>
</html>
"""


def build() -> str:
    d = _load()
    s = d["summary"]
    ev_iv = s["intervals"]["evaluation"]
    ev_pt = s["point"]["evaluation"]
    log = live.read_log()
    live_html, live_fig = _live_section(log)
    rt = d["rte_bias"].set_index("segment").loc["real-time feed"]
    figs = {
        "c-live": live_fig,
        "c-cov80": chart_coverage(d["daily"], 0.8).to_json(),
        "c-cov95": chart_coverage(d["daily"], 0.95).to_json(),
        "c-mape": chart_mape(d["monthly"]).to_json(),
        "c-fan": chart_fan(d["fan"], "rte_j1_adj", "RTE J-1, level-corrected").to_json(),
    }
    page = TEMPLATE.format(
        repo=REPO,
        ink=INK,
        eval_start=s["eval_start"],
        eval_end=s["eval_end"],
        eval_hours=f"{s['eval_hours']:,}",
        crisis=f"{s['crisis'][0]} to {s['crisis'][1]}",
        k_cov80=_pct(ev_iv["cqr_rolling_aci@80"]["coverage"]),
        k_cov95=_pct(ev_iv["cqr_rolling_aci@95"]["coverage"]),
        k_mape=_pct(ev_pt["point"]["mape"], 2),
        k_rte=_pct(ev_pt["rte_j1_adj"]["mape"], 2),
        k_rte_bias=_pct(
            -d["rte_bias"].query("segment in ['2023', '2024', '2025']")["bias_pct"].mean()
        ),
        k_rt_bias=f"{rt['bias_pct'] * 100:+.1f}%",
        k_rt_mape=_pct(rt["mape"], 2),
        k_cov_static=_pct(ev_iv["split_static@80"]["coverage"]),
        k_win95=_pct(ev_iv["cqr_rolling_aci@95"]["rolling30_within_5pp"], 0),
        k_cov_crisis=_pct(s["intervals"]["crisis"]["cqr_rolling_aci@80"]["coverage"]),
        k_rte_raw=_pct(ev_pt["rte_j1"]["mape"], 2),
        k_oracle=_pct(ev_pt["point_oracle"]["mape"], 2),
        k_win=_pct(ev_iv["cqr_rolling_aci@80"]["rolling30_within_5pp"], 0),
        k_win_static=_pct(ev_iv["split_static@80"]["rolling30_within_5pp"], 0),
        live_html=live_html,
        interval_rows=_interval_rows(d["intervals"], "evaluation"),
        period_rows=_period_rows(d["intervals"], d["points"]),
        reps=config.BOOTSTRAP_REPS,
        figs="{" + ",".join(f'"{k}": {v or "null"}' for k, v in figs.items()) + "}",
        generated=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
    )
    config.SITE.mkdir(parents=True, exist_ok=True)
    out = config.SITE / "index.html"
    out.write_text(page)
    update_readme(s, d["intervals"])
    return str(out)


# --- README blocks -------------------------------------------------------------------


def readme_tables(summary: dict, intervals: pd.DataFrame) -> dict[str, str]:
    ev = intervals[intervals["period"] == "evaluation"]
    lines = [
        (
            "| Method | 80% coverage [95% CI] | width (GW) | Winkler (GW) "
            "| 95% coverage [95% CI] | width (GW) | Winkler (GW) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        cells = [m.label]
        for level in config.LEVELS:
            r = ev[(ev["method"] == m.name) & (ev["level"] == level)].iloc[0]
            cells += [
                f"{_pct(r['coverage'])} [{_pct(r['coverage_lo'])}, {_pct(r['coverage_hi'])}]",
                f"{r['width_mw'] / 1000:.2f}",
                f"{r['winkler_mw'] / 1000:.2f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    pt = summary["point"]
    plines = [
        "| Forecast | MAPE [95% CI] | RMSE (GW) | MAPE, crisis | vs RTE corrected [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for col, label in POINT_MODELS.items():
        e, c = pt["evaluation"][col], pt["crisis"][col]
        diff = (
            "—"
            if col == "rte_j1_adj"
            else f"{e['mape_vs_rte'] * 100:+.2f} pp [{e['mape_vs_rte_lo'] * 100:+.2f}, "
            f"{e['mape_vs_rte_hi'] * 100:+.2f}]"
        )
        plines.append(
            f"| {label} | {_pct(e['mape'], 2)} [{_pct(e['mape_lo'], 2)}, {_pct(e['mape_hi'], 2)}]"
            f" | {e['rmse_mw'] / 1000:.2f} | {_pct(c['mape'], 2)} | {diff} |"
        )
    return {"interval-table": "\n".join(lines), "point-table": "\n".join(plines)}


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace the body between ``<!-- BEGIN:key -->`` and ``<!-- END:key -->`` markers."""
    for key, body in blocks.items():
        pattern = re.compile(rf"(<!-- BEGIN:{key} -->\n)(?:.*?\n)?(<!-- END:{key} -->)", re.DOTALL)
        text = pattern.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
    return text


def update_readme(summary: dict, intervals: pd.DataFrame) -> None:
    """Rewrite the generated README tables so they can never drift from the results."""
    path = config.ROOT / "README.md"
    if path.exists():
        path.write_text(replace_blocks(path.read_text(), readme_tables(summary, intervals)))
