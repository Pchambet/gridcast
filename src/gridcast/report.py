"""Build the static report page (``site/index.html``) and refresh README result blocks.

The page depends only on committed result tables and, when present, the live forecast
log, so the Pages workflow can rebuild it without downloading any raw data.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go

from gridcast import config, live
from gridcast.conformal import METHODS
from gridcast.evaluate import POINT_MODELS
from gridcast.figures import month_span

REPO = "https://github.com/Pchambet/gridcast"
TEAL, AMBER, SLATE, INK = "#0d9488", "#d97706", "#64748b", "#0f172a"
# (colour, dash) per interval method, within the shared teal / amber / slate palette.
METHOD_STYLES = {
    "qr_raw": ("#fbbf24", "solid"),
    "split_static": (AMBER, "solid"),
    "split_static_aci": (AMBER, "dot"),
    "split_rolling": ("#94a3b8", "solid"),
    "cqr_rolling": (SLATE, "solid"),
    "cqr_rolling_aci": (TEAL, "solid"),
}
SHOWN_METHODS = {"split_static", "cqr_rolling", "cqr_rolling_aci"}
STALE_AFTER = pd.Timedelta(days=2)


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _pp(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f} pp"


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
        "rte_hour": pd.read_csv(r / "rte_bias_by_hour.csv"),
    }


# --- Charts (plotly figure dicts; theming happens in the browser) --------------------
# Dates are sent as short strings and values rounded: the page stays small.


def _layout(**kw: object) -> dict:
    base: dict = {
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
    for method in METHODS:
        g = daily[(daily["method"] == method.name) & (daily["level"] == level)]
        g = g.sort_values("day").set_index("day")
        roll = g["hits"].rolling("30D").sum() / g["hours"].rolling("30D").sum()
        roll = roll[roll.index >= g.index[0] + pd.Timedelta(days=29)]
        color, dash = METHOD_STYLES[method.name]
        shown = method.name in SHOWN_METHODS
        fig.add_trace(
            go.Scatter(
                x=list(roll.index.strftime("%Y-%m-%d")), y=roll.round(4).tolist(),
                name=method.label, mode="lines",
                line={"width": 2 if method.name == "cqr_rolling_aci" else 1.4,
                      "color": color, "dash": dash},
                visible=True if shown else "legendonly", hovertemplate="%{y:.1%}",
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
    months = list(monthly["month"].dt.strftime("%Y-%m-%d"))
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
                x=months, y=monthly[col].round(5).tolist(), name=label, mode="lines",
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
    times = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(config.PARIS)
    t = list(times.dt.strftime("%Y-%m-%d %H:%M"))

    def gw(col: str) -> list:
        return (frame[col] / 1e3).round(2).tolist()

    fig = go.Figure()
    for lo, hi, alpha, name in (("lo95", "hi95", 0.14, "95% interval"),
                                ("lo80", "hi80", 0.30, "80% interval")):  # fmt: skip
        fig.add_trace(go.Scatter(x=t, y=gw(hi), mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip"))  # fmt: skip
        fig.add_trace(
            go.Scatter(x=t, y=gw(lo), mode="lines", line={"width": 0}, fill="tonexty",
                       fillcolor=f"rgba(13,148,136,{alpha})", name=name, hoverinfo="skip")
        )  # fmt: skip
    fig.add_trace(go.Scatter(x=t, y=gw("point"), name="gridcast",
                             line={"color": TEAL, "width": 2},
                             hovertemplate="%{y:.1f} GW"))  # fmt: skip
    if frame[rte_col].notna().any():
        fig.add_trace(go.Scatter(x=t, y=gw(rte_col), name=rte_label,
                                 line={"color": SLATE, "width": 1.5, "dash": "dash"},
                                 hovertemplate="%{y:.1f} GW"))  # fmt: skip
    if frame["load"].notna().any():
        fig.add_trace(go.Scatter(x=t, y=gw("load"), name="actual",
                                 line={"color": INK, "width": 2},
                                 hovertemplate="%{y:.1f} GW"))  # fmt: skip
    fig.update_layout(**_layout(yaxis={"title": "GW"}))
    return fig


# --- HTML -----------------------------------------------------------------------------


def _interval_rows(table: pd.DataFrame, period: str, level: float) -> str:
    sub = table[(table["period"] == period) & (table["level"] == level)]
    rows = []
    for m in METHODS:
        r = sub[sub["method"] == m.name].iloc[0]
        cls = ' class="off"' if abs(r["coverage"] - level) > 0.02 else ""
        rows.append(
            f"<tr><td>{html.escape(m.label)}</td>"
            f"<td{cls}>{_pct(r['coverage'])}<span class=ci>[{_pct(r['coverage_lo'])}, "
            f"{_pct(r['coverage_hi'])}]</span></td>"
            f"<td>{r['width_mw'] / 1000:.2f}</td><td>{r['winkler_mw'] / 1000:.2f}</td>"
            f"<td>{_pct(r['rolling30_within_5pp'], 0)}</td></tr>"
        )
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


def _live_section(log: pd.DataFrame, now: pd.Timestamp) -> tuple[str, str | None, bool]:
    """Live HTML, its chart, and whether a scored track record exists yet."""
    issued = log[log["kind"] == "live"] if not log.empty else log
    if issued.empty:
        return (
            (
                "<p class=muted>The daily job has not published yet. From its first run, "
                "tomorrow's forecast and the running track record appear here.</p>"
            ),
            None,
            False,
        )
    latest_day = issued["day"].max()
    latest = issued[issued["day"] == latest_day]
    issued_at = pd.Timestamp(latest["issued_at"].iloc[0]).tz_convert(config.PARIS)
    rec = live.track_record(log)
    peak = latest.loc[latest["point"].idxmax()]
    peak_time = pd.Timestamp(peak["time"]).tz_convert(config.PARIS)
    stale = ""
    if now - issued_at > STALE_AFTER:
        stale = (
            f"<p class=warn>The latest forecast was issued on {issued_at:%d %B %Y}: the "
            "daily job has not run since. The figures below are not current.</p>"
        )
    lead = (
        f"{stale}<p>Forecast for <strong>{latest_day:%A %d %B %Y}</strong>, issued "
        f"{issued_at:%d %b %Y %H:%M} Paris time. Peak {_gw(peak['point'])} at "
        f"{peak_time:%H:%M} (80%: {_gw(peak['lo80'])} to {_gw(peak['hi80'])}).</p>"
    )
    if rec["days"] == 0:
        track = "<p class=muted>No live forecast has been scored yet.</p>"
    else:
        missing = rec["hours_without_bands"]
        note = f" {missing} scored hours had no interval and are left out." if missing else ""
        track = (
            "<div class=scroll><table><thead><tr><th>Scored days</th><th>80% coverage</th>"
            "<th>95% coverage</th><th>MAPE gridcast</th><th>MAPE RTE J-1</th></tr></thead>"
            f"<tbody><tr><td>{rec['days']} ({rec['first_day']} to {rec['last_day']})</td>"
            f"<td>{_pct(rec['coverage80'])}</td><td>{_pct(rec['coverage95'])}</td>"
            f"<td>{_pct(rec['mape'], 2)}</td><td>{_pct(rec['mape_rte'], 2)}</td>"
            "</tr></tbody></table></div>"
            f"<p class=note>Live forecasts only, scored on RTE's real-time series; RTE's "
            f"J-1 as published.{note}</p>"
        )
    recent = issued[issued["day"] >= latest_day - pd.Timedelta(days=6)]
    html_out = lead + '<div class=chart id="c-live"></div>' + track
    return html_out, chart_fan(recent).to_json(), rec["days"] > 0


def _hour_profile(rte_hour: pd.DataFrame) -> str:
    """One sentence on how RTE's bias against the consolidated series varies by hour."""
    since = rte_hour.set_index("local_hour")["bias_since_2023"]
    worst = since.nsmallest(3).sort_index()
    best = since.idxmax()
    hours = ", ".join(f"{h:02d}:00" for h in worst.index)
    return (
        f"The gap depends strongly on the hour: largest at {hours} ({_pct(worst.min())} to "
        f"{_pct(worst.max())}), smallest at {best:02d}:00 ({since[best] * 100:+.1f}%)"
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gridcast</title>
<meta name="description" content="Day-ahead forecast of French electricity demand with
conformal intervals, benchmarked against RTE and re-issued every day.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3E%3Cpath d=%27M1 12 L5 6 L9 9 L15 2%27 stroke=%27%230d9488%27 stroke-width=%272.5%27 fill=%27none%27/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet"
 href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {{ --bg:#ffffff; --surface:#f8fafc; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --accent:#0d9488; --warn:#b45309; --shade:rgba(15,23,42,0.12); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0b1120;
  --surface:#111827; --ink:#e2e8f0; --muted:#94a3b8; --line:#1f2937; --accent:#2dd4bf;
  --warn:#fbbf24; --shade:rgba(0,0,0,0.45); }} }}
:root[data-theme="dark"] {{ --bg:#0b1120; --surface:#111827; --ink:#e2e8f0; --muted:#94a3b8;
  --line:#1f2937; --accent:#2dd4bf; --warn:#fbbf24; --shade:rgba(0,0,0,0.45); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:48px 16px 64px; }}
h1 {{ font-size:2.1rem; margin:0 0 4px; letter-spacing:-0.02em; }}
h2 {{ font-size:1.3rem; margin:48px 0 8px; letter-spacing:-0.01em; }}
h3 {{ font-size:1rem; margin:20px 0 0; }}
p {{ margin:8px 0 12px; }}
a {{ color:var(--accent); }}
.lede {{ font-size:1.12rem; color:var(--muted); margin-top:0; }}
.muted, .note {{ color:var(--muted); font-size:0.9rem; }}
.warn {{ color:var(--warn); font-weight:600; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px;
  margin:28px 0 8px; }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; }}
.kpi b {{ display:block; font-size:1.6rem; font-variant-numeric:tabular-nums; }}
.kpi span {{ color:var(--muted); font-size:0.85rem; }}
.chart {{ width:100%; min-height:340px; margin:8px 0 4px; }}
/* Horizontal scroll with edge shadows that only show when there is more to see. */
.scroll {{ overflow-x:auto;
  background: linear-gradient(to right, var(--bg) 30%, transparent) left / 24px 100% no-repeat local,
    linear-gradient(to left, var(--bg) 30%, transparent) right / 24px 100% no-repeat local,
    radial-gradient(farthest-side at 0 50%, var(--shade), transparent) left / 10px 100% no-repeat scroll,
    radial-gradient(farthest-side at 100% 50%, var(--shade), transparent) right / 10px 100% no-repeat scroll; }}
table {{ border-collapse:collapse; width:100%; font-size:0.88rem; margin:12px 0;
  font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--line);
  white-space:nowrap; vertical-align:top; }}
th:first-child, td:first-child {{ text-align:left; white-space:normal; min-width:150px; }}
th {{ color:var(--muted); font-weight:600; white-space:normal; }}
td.off {{ color:var(--warn); font-weight:600; }}
@media (max-width:520px) {{ table {{ font-size:0.8rem; }} th, td {{ padding:6px 5px; }}
  th:first-child, td:first-child {{ min-width:118px; }} }}
.ci {{ display:block; color:var(--muted); font-size:0.78em; font-weight:400; }}
footer {{ margin-top:56px; padding-top:16px; border-top:1px solid var(--line);
  color:var(--muted); font-size:0.88rem; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
</style>
</head>
<body>
<main>
<h1>gridcast</h1>
<p class="lede">Can a day-ahead forecast of French electricity demand publish uncertainty
bands that stay honest through an energy crisis? On average yes, month to month only
partly: a measured answer, plus a forecast re-issued every day in public.</p>
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
{live_top}
<h2>Calibration through the energy crisis</h2>
<p>Conformal prediction guarantees coverage on average, for exchangeable data. Demand
forecast errors are not exchangeable: they jump when heating starts in November, shrink
in spring, and in autumn 2022 the whole level of French demand dropped. A calibration
frozen on April 2021 to March 2022 still covers {k_cov_static} of hours overall, yet its
30-day coverage swings between roughly 50% and 100% and is within &plusmn;5 pp of target
only {k_win_static} of the time. Recalibrating on a rolling 90-day window with
conformalized quantile regression (CQR) does most of the repair: {k_win_rolling} of
windows on target. Adaptive conformal inference (ACI) at the pre-set step adds a little
at 80% ({k_win}) and nothing at 95%, where it widens the bands ({k_w95_rolling} to
{k_w95} GW) and worsens the interval score ({k_wk95_rolling} to {k_wk95} GW). No method
fully absorbs the crisis winter: over {crisis} the published 80% interval covered
{k_cov_crisis}. Click legend entries to compare the other methods; shaded: energy
crisis.</p>
<div class="chart" id="c-cov80"></div>
<p class="note">The 95% version. Coverage cannot exceed 100%, so here the &plusmn;5 pp
band is [90%, 100%] and only flags under-coverage: the on-target shares at 80% and 95%
are not comparable.</p>
<div class="chart" id="c-cov95"></div>

<h2>All interval methods, {eval_start} to {eval_end}</h2>
<h3>80% intervals</h3>
<div class="scroll"><table>
<thead><tr><th>Method</th><th>coverage [95% CI]</th><th>width GW</th><th>Winkler GW</th>
<th>30-day on target</th></tr></thead>
<tbody>{interval_rows80}</tbody></table></div>
<h3>95% intervals</h3>
<div class="scroll"><table>
<thead><tr><th>Method</th><th>coverage [95% CI]</th><th>width GW</th><th>Winkler GW</th>
<th>30-day &ge; 90%</th></tr></thead>
<tbody>{interval_rows95}</tbody></table></div>
<p class="note">Coverage CIs from a week-block bootstrap ({reps} resamples). Winkler
(interval) score: width plus 2/&alpha; times the miss distance; lower is better.
Highlighted: coverage more than 2 pp from nominal. ACI clips its level when it would ask
for an infinite interval (Gibbs &amp; Cand&egrave;s's guarantee assumes it does not); the
published 95% interval was capped at the widest calibration score on {k_capped95} of
the scored days.</p>

<h2>By period</h2>
<div class="scroll"><table>
<thead><tr><th>Period</th><th>MAPE gridcast</th><th>MAPE RTE (corrected)</th>
<th>80% coverage, frozen</th><th>frozen + ACI</th><th>CQR + ACI</th></tr>
</thead><tbody>{period_rows}</tbody></table></div>
<p class="note">Pre-crisis is only five months (April to August 2022), all in the summer,
when every method over-covers. Frozen + ACI starts from its burn-in state and, at
&gamma; = 0.02 per day, narrows too slowly to catch up, hence its higher coverage there.
Too short a window to judge.</p>

<h2>Point accuracy: RTE wins, once compared fairly</h2>
<p>Scored naively, both forecasts as produced, gridcast beats RTE's published J-1 forecast
({k_raw} vs {k_rte_raw} MAPE against the consolidated demand series). That comparison is
confounded by a level gap that public data cannot attribute to definitions or to forecast
error. From 2023 on, RTE's forecast runs {k_rte_bias} below the consolidated values on
average (within 1% in 2015-2021). {k_hour_profile}. On the only real-time data in the
backtest ({k_rt_hours} hours, July and August 2026) its bias is {k_rt_bias} and its MAPE
{k_rt_mape}, but two summer months settle nothing. Both forecasts are therefore corrected
the same way, by their own mean error over the trailing 28 days at the same local hour,
using only errors at least two days old. On that footing RTE is better overall, {k_rte}
vs {k_mape}, and in each period, though not significantly in the five pre-crisis months
(difference {k_pre_diff}, 95% CI {k_pre_ci}), where the level correction itself costs
gridcast {k_lc_pre}. gridcast is ahead in {k_ahead} of {k_months} months, {k_ahead_span},
when demand is flat and temperature-driven. RTE likely benefits from richer inputs (many
more weather stations, cloud cover, embedded solar, operational knowledge) and may issue
later than 12:00; gridcast sees one public temperature forecast for ten cities. With
observed instead of forecast weather the same model reaches {k_oracle}. The contribution
here is not a better point forecast but a transparent, calibrated uncertainty layer
anyone can audit.</p>
<div class="chart" id="c-mape"></div>

<h2>A week of the December 2022 cold snap</h2>
<div class="chart" id="c-fan"></div>
{live_bottom}
<h2>How it works</h2>
<ul>
<li><strong>Information set.</strong> The forecast for day D is issued at 12:00 Paris on
D-1. It may use demand up to 11:00 (one hour publication lag), calendar features and
temperature forecasts whose model run had finished by then (24 h-ahead values for the
first hours of D, 48 h-ahead after that). A test poisons every later value and checks
the features, the backtest refit and the live forecast do not change.</li>
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
this page. Days without a live forecast (before the first run, or after an outage) are
hindcast with the backtest protocol so calibration never runs dry; they are kept out of
the track record.</li>
</ul>

<h2>Limitations</h2>
<ul>
<li>The backtest information set is better than the live one. Demand lags and all
feedback use RTE's consolidated series, published months later; at issue time only the
real-time vintage existed. Observed temperature before the issue time is ERA5
reanalysis, which arrives about 5.5 days late. Masking those days as the live job must
moves MAPE from {k_mape_pub} to {k_mape_lag} and the published 80% coverage from
{k_cov_pub} to {k_cov_lag}; the vintage effect cannot be measured with public data.</li>
<li>Only temperature has archived day-ahead forecasts before 2024 in the open archive;
cloud cover and wind, which RTE uses, are not in the model.</li>
<li>Training uses reanalysis (ERA5) temperature, prediction uses forecasts; the
conformal layer absorbs the mismatch, the point forecast pays for it. The uncalibrated
quantiles ignore weather-forecast error by construction: fed observed weather, they
cover {k_raw_oracle} at nominal 80% instead of {k_raw_pub}, so the mismatch explains
little of their under-coverage.</li>
<li>{k_filled} of forecast temperature ({k_filled_span}) come from JMA instead of GFS,
the only archived model covering them.</li>
<li>Features were designed knowing that French demand dropped in 2022 (the week-ratio
feature exists to absorb such level shifts); hyperparameters were set a priori, not
tuned.</li>
<li>RTE's J-1 forecast may be published later than 12:00 on D-1, so the comparison
slightly favours RTE.</li>
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


def _filled_text(info: dict) -> tuple[str, str]:
    hours = info["hours"]
    if not hours:
        return "No hours", "none in the backtest window"
    first, last = (pd.Timestamp(info[k]) for k in ("first", "last"))
    return (
        f"{hours} hours ({hours / 24:.1f} days)",
        f"{first:%d %b %Y %H:%M} to {last:%d %b %Y %H:%M} UTC",
    )


def build() -> str:
    d = _load()
    s = d["summary"]
    ev_iv = s["intervals"]["evaluation"]
    ev_pt = s["point"]["evaluation"]
    pre = s["point"]["pre-crisis"]
    info = s["information_sets"]
    now = pd.Timestamp.now(tz="UTC")
    live_html, live_fig, has_record = _live_section(live.read_log(), now)
    live_block = f"\n<h2>Tomorrow</h2>\n{live_html}\n"
    rt = d["rte_bias"].set_index("segment").loc["real-time feed"]
    filled, filled_span = _filled_text(s["gap_filled_forecast_hours"])
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
        live_top=live_block if has_record else "",
        live_bottom="" if has_record else live_block,
        k_cov80=_pct(ev_iv["cqr_rolling_aci@80"]["coverage"]),
        k_cov95=_pct(ev_iv["cqr_rolling_aci@95"]["coverage"]),
        k_mape=_pct(ev_pt["point"]["mape"], 2),
        k_rte=_pct(ev_pt["rte_j1_adj"]["mape"], 2),
        k_raw=_pct(ev_pt["point_raw"]["mape"], 2),
        k_rte_raw=_pct(ev_pt["rte_j1"]["mape"], 2),
        k_rte_bias=_pct(
            -d["rte_bias"].query("segment in ['2023', '2024', '2025']")["bias_pct"].mean()
        ),
        k_hour_profile=_hour_profile(d["rte_hour"]),
        k_rt_hours=f"{int(rt['hours']):,}",
        k_rt_bias=f"{rt['bias_pct'] * 100:+.1f}%",
        k_rt_mape=_pct(rt["mape"], 2),
        k_pre_diff=_pp(pre["point"]["mape_vs_rte"]),
        k_pre_ci=f"[{_pp(pre['point']['mape_vs_rte_lo'])}, {_pp(pre['point']['mape_vs_rte_hi'])}]",
        k_lc_pre=_pp(pre["point"]["mape"] - pre["point_raw"]["mape"]),
        k_ahead=s["months_ahead_of_rte"]["ahead"],
        k_months=s["months_ahead_of_rte"]["months"],
        k_ahead_span=month_span(s["months_ahead_of_rte"]["calendar_months"]),
        k_oracle=_pct(ev_pt["point_oracle"]["mape"], 2),
        k_cov_static=_pct(ev_iv["split_static@80"]["coverage"]),
        k_cov_crisis=_pct(s["intervals"]["crisis"]["cqr_rolling_aci@80"]["coverage"]),
        k_win=_pct(ev_iv["cqr_rolling_aci@80"]["rolling30_within_5pp"], 0),
        k_win_static=_pct(ev_iv["split_static@80"]["rolling30_within_5pp"], 0),
        k_win_rolling=_pct(ev_iv["cqr_rolling@80"]["rolling30_within_5pp"], 0),
        k_w95_rolling=f"{ev_iv['cqr_rolling@95']['width_mw'] / 1000:.2f}",
        k_w95=f"{ev_iv['cqr_rolling_aci@95']['width_mw'] / 1000:.2f}",
        k_wk95_rolling=f"{ev_iv['cqr_rolling@95']['winkler_mw'] / 1000:.2f}",
        k_wk95=f"{ev_iv['cqr_rolling_aci@95']['winkler_mw'] / 1000:.2f}",
        k_capped95=s["aci_capped_days"]["cqr_rolling_aci@95"],
        k_mape_pub=_pct(info["published"]["mape"], 3),
        k_mape_lag=_pct(info["ERA5 delayed as live"]["mape"], 3),
        k_cov_pub=_pct(info["published"]["cqr_rolling_aci_coverage80"], 2),
        k_cov_lag=_pct(info["ERA5 delayed as live"]["cqr_rolling_aci_coverage80"], 2),
        k_raw_pub=_pct(info["published"]["qr_raw_coverage80"]),
        k_raw_oracle=_pct(info["oracle weather"]["qr_raw_coverage80"]),
        k_filled=filled,
        k_filled_span=filled_span,
        interval_rows80=_interval_rows(d["intervals"], "evaluation", 0.8),
        interval_rows95=_interval_rows(d["intervals"], "evaluation", 0.95),
        period_rows=_period_rows(d["intervals"], d["points"]),
        reps=config.BOOTSTRAP_REPS,
        figs="{" + ",".join(f'"{k}": {v or "null"}' for k, v in figs.items()) + "}",
        generated=now.strftime("%Y-%m-%d %H:%M UTC"),
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


def _block(body: str) -> Callable[[re.Match[str]], str]:
    return lambda m: m.group(1) + body + "\n" + m.group(2)


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace the body between ``<!-- BEGIN:key -->`` and ``<!-- END:key -->`` markers."""
    for key, body in blocks.items():
        pattern = re.compile(rf"(<!-- BEGIN:{key} -->\n)(?:.*?\n)?(<!-- END:{key} -->)", re.DOTALL)
        text = pattern.sub(_block(body), text)
    return text


def update_readme(summary: dict, intervals: pd.DataFrame) -> None:
    """Rewrite the generated README tables so they can never drift from the results."""
    path = config.ROOT / "README.md"
    if path.exists():
        path.write_text(replace_blocks(path.read_text(), readme_tables(summary, intervals)))
