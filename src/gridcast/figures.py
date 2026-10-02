"""Static README figures (matplotlib), built only from the committed result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from gridcast import config
from gridcast.conformal import METHOD_BY_NAME, METHODS

matplotlib.use("Agg")

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
CRISIS_FILL = "#f1f5f9"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _results(name: str) -> pd.DataFrame:
    return pd.read_csv(config.RESULTS / name)


def _summary() -> dict:
    return json.loads((config.RESULTS / "summary.json").read_text())


def _shade_crisis(ax, label: bool = True) -> None:
    lo, hi = config.CRISIS
    ax.axvspan(lo, hi, color=CRISIS_FILL, zorder=0, lw=0)
    if label:
        ax.text(
            lo + (hi - lo) / 2, 1.0, "energy crisis", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color=SLATE, fontsize=8.5,
        )  # fmt: skip


def _save(fig, name: str) -> Path:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _rolling(daily: pd.DataFrame, method: str, level: float, window: int = 30) -> pd.Series:
    g = daily[(daily["method"] == method) & (daily["level"] == level)].copy()
    g["day"] = pd.to_datetime(g["day"])
    g = g.sort_values("day").set_index("day")
    roll = g["hits"].rolling(f"{window}D").sum() / g["hours"].rolling(f"{window}D").sum()
    return roll[roll.index >= g.index[0] + pd.Timedelta(days=window - 1)]


def hero() -> Path:
    """Rolling 30-day coverage: frozen split conformal vs the published CQR + ACI."""
    daily = _results("coverage_daily.csv")
    s = _summary()["intervals"]["evaluation"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    for ax, level in zip(axes, config.LEVELS, strict=True):
        tag = round(level * 100)
        _shade_crisis(ax, label=False)
        ax.axhspan(level - 0.05, min(level + 0.05, 1.0), color=TEAL, alpha=0.08, lw=0)
        ax.axhline(level, color=INK, lw=0.8, ls=(0, (4, 3)))
        for method, color, name in (
            ("split_static", AMBER, "frozen split conformal"),
            ("cqr_rolling_aci", TEAL, "CQR + ACI (published)"),
        ):
            r = _rolling(daily, method, level)
            share = s[f"{method}@{tag}"]["rolling30_within_5pp"]
            ax.plot(r.index, r.to_numpy(), color=color, lw=1.6,
                    label=f"{name}: on target {share:.0%} of the time")  # fmt: skip
        ax.legend(loc="lower left", bbox_to_anchor=(0, 0.98), ncol=2, frameon=False,
                  fontsize=8.5, handlelength=1.6, borderaxespad=0)  # fmt: skip
        ax.set_ylabel(f"{tag}% interval:\n30-day coverage")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        lo = min(0.5 if level < 0.9 else 0.75, ax.get_ylim()[0])
        ax.set_ylim(lo, 1.005)
    mid = config.CRISIS[0] + (config.CRISIS[1] - config.CRISIS[0]) / 2
    axes[1].text(mid, 0.02, "energy\ncrisis", transform=axes[1].get_xaxis_transform(),
                 ha="center", va="bottom", color=SLATE, fontsize=8)  # fmt: skip
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    s80, a80 = (s[f"{m}@80"]["rolling30_within_5pp"] for m in ("split_static", "cqr_rolling_aci"))
    c_s = s["split_static@80"]["coverage"]
    fig.suptitle(
        f"Right on average, wrong every season: a frozen 80% interval covers {c_s:.1%} overall\n"
        f"but sits within ±5 pp of target only {s80:.0%} of the time; online CQR + ACI {a80:.0%}",
        x=0.01, y=1.0, ha="left", fontsize=12, fontweight="bold", color=INK,
    )  # fmt: skip
    fig.text(
        0.01, -0.02,
        "Trailing 30-day share of hourly actuals inside the day-ahead interval; teal band: "
        "nominal ±5 pp. Frozen: |residual| quantile from Apr 2021 - Mar 2022, never updated.\n"
        "CQR + ACI: 90-day rolling conformalized quantile regression, adaptive conformal "
        "inference (step 0.02/day), errors fed back with a 2-day delay. Apr 2022 - Aug 2026.",
        fontsize=8, color=SLATE,
    )  # fmt: skip
    fig.subplots_adjust(hspace=0.22, top=0.86)
    return _save(fig, "hero_coverage.png")


def point_accuracy() -> Path:
    monthly = _results("monthly_mape.csv")
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly = monthly[monthly["month"] >= config.EVAL_START]
    ev = _summary()["point"]["evaluation"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _shade_crisis(ax, label=False)
    series = (
        ("point", TEAL, "gridcast", "-"),
        ("rte_j1_adj", INK, "RTE J-1, level-corrected", "-"),
        ("rte_j1", SLATE, "RTE J-1, as published", (0, (3, 2))),
    )
    for col, color, label, ls in series:
        ax.plot(monthly["month"], monthly[col], color=color, lw=1.6, ls=ls,
                label=f"{label} ({ev[col]['mape']:.2%})")  # fmt: skip
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.99), ncol=3, frameon=False, fontsize=8.5,
              borderaxespad=0)  # fmt: skip
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_ylabel("monthly MAPE")
    ax.set_ylim(0, None)
    ax.set_xlim(monthly["month"].min() - pd.Timedelta(days=15), None)
    ax.set_title(
        f"Against a fair benchmark, RTE's day-ahead forecast is more accurate: "
        f"{ev['rte_j1_adj']['mape']:.2%} vs {ev['point']['mape']:.2%} MAPE",
        pad=26,
    )
    fig.text(
        0.0, -0.03,
        "Both forecasts are level-corrected with their own errors from the trailing 28 days "
        "(same local hour, errors at least 2 days old); shaded: energy crisis. As published, RTE's J-1 sits ~2% "
        f"below the consolidated series since 2023 ({ev['rte_j1']['mape']:.2%} MAPE). "
        f"Seasonal naive: {ev['naive']['mape']:.1%}.",
        fontsize=8, color=SLATE,
    )  # fmt: skip
    return _save(fig, "point_accuracy.png")


def interval_scores() -> Path:
    table = _results("interval_metrics.csv")
    table = table[table["period"] == "evaluation"]
    order = [m.name for m in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, level in zip(axes, config.LEVELS, strict=True):
        sub = table[table["level"] == level].set_index("method").loc[order]
        ypos = range(len(order))[::-1]
        for y, (name, row) in zip(ypos, sub.iterrows(), strict=True):
            color = TEAL if name.endswith("aci") else (AMBER if name == "qr_raw" else SLATE)
            ax.plot([row["coverage_lo"], row["coverage_hi"]], [y, y], color=color, lw=2.5)
            ax.plot(row["coverage"], y, "o", color=color, ms=7, mec="white", mew=1.5)
            ax.annotate(
                f"{row['coverage']:.1%} · width {row['width_mw'] / 1000:.1f} GW",
                xy=(row["coverage_hi"], y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK,
            )  # fmt: skip
        ax.axvline(level, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.set_yticks(list(ypos), [METHOD_BY_NAME[m].label for m in order])
        ax.set_xlabel(f"empirical coverage of {round(level * 100)}% intervals (95% CI)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.06)
        ax.grid(axis="y", visible=False)
    raw = table[(table["method"] == "qr_raw") & (table["level"] == 0.8)]["coverage"].iloc[0]
    fig.suptitle(
        f"Uncalibrated quantile regression covers {raw:.0%} at nominal 80%; "
        "every conformal method restores the level",
        x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK,
    )  # fmt: skip
    fig.subplots_adjust(top=0.88, wspace=0.08)
    return _save(fig, "interval_coverage.png")


def fan_week() -> Path:
    week = _results("fan_week.csv")
    week["time"] = pd.to_datetime(week["time"], utc=True).dt.tz_convert(config.PARIS)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    t = week["time"]
    ax.fill_between(t, week["lo95"] / 1e3, week["hi95"] / 1e3, color=TEAL, alpha=0.15, lw=0)
    ax.fill_between(t, week["lo80"] / 1e3, week["hi80"] / 1e3, color=TEAL, alpha=0.3, lw=0)
    ax.plot(t, week["load"] / 1e3, color=INK, lw=1.6, label="actual")
    ax.plot(t, week["point"] / 1e3, color=TEAL, lw=1.4, label="gridcast")
    ax.plot(t, week["rte_j1_adj"] / 1e3, color=SLATE, lw=1.0, ls=(0, (3, 2)),
            label="RTE J-1, level-corrected")  # fmt: skip
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.99), ncol=3, frameon=False, fontsize=8.5,
              borderaxespad=0)  # fmt: skip
    inside = ((week["load"] >= week["lo80"]) & (week["load"] <= week["hi80"])).mean()
    ax.set_ylabel("French demand (GW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=config.PARIS))
    peak = week.loc[week["load"].idxmax()]
    ax.set_title(
        f"Cold snap of December 2022: demand peaks at {peak['load'] / 1e3:.1f} GW; "
        f"{inside:.0%} of hours fall inside the 80% band",
        pad=24,
    )
    fig.text(
        0.0, -0.02,
        "Day-ahead forecasts issued at 12:00 the day before. Bands: CQR rolling + ACI, "
        "80% (dark) and 95% (light).", fontsize=8, color=SLATE,
    )  # fmt: skip
    return _save(fig, "fan_week.png")


def demand_context() -> Path:
    monthly = _results("monthly_demand.csv")
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly["m"] = monthly["month"].dt.month
    ref = monthly[(monthly["month"] >= "2015-01-01") & (monthly["month"] < "2020-01-01")]
    normal = ref.groupby("m")["load_gw"].mean()
    monthly["gap"] = monthly["load_gw"] / monthly["m"].map(normal) - 1
    show = monthly[monthly["month"] >= "2019-01-01"]
    winter = show[(show["month"] >= "2022-11-01") & (show["month"] <= "2023-02-28")]["gap"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _shade_crisis(ax)
    colors = [TEAL if g >= 0 else AMBER for g in show["gap"]]
    ax.bar(show["month"], show["gap"], width=25, color=colors, lw=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("vs 2015-2019 same month")
    ax.set_title(
        f"French demand fell {-winter:.0%} below its pre-2020 level in winter 2022-23 "
        "and has not come back",
        pad=18,
    )
    fig.text(
        0.0, -0.03,
        "Monthly mean consumption relative to the 2015-2019 average of the same calendar "
        "month (not weather-adjusted). Source: RTE eco2mix.", fontsize=8, color=SLATE,
    )  # fmt: skip
    return _save(fig, "demand_context.png")


def coverage_by_hour() -> Path:
    table = _results("coverage_by_hour.csv")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    level = 0.8
    ax.axhspan(level - 0.05, level + 0.05, color=TEAL, alpha=0.08, lw=0)
    ax.axhline(level, color=INK, lw=0.8, ls=(0, (4, 3)))
    spread = {}
    for method, color in (("split_rolling", SLATE), ("cqr_rolling_aci", TEAL)):
        sub = table[(table["method"] == method) & (table["level"] == level)]
        spread[method] = sub["coverage"].max() - sub["coverage"].min()
        ax.plot(sub["hour"], sub["coverage"], color=color, lw=1.8, marker="o", ms=4,
                label=METHOD_BY_NAME[method].label)  # fmt: skip
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.99), ncol=2, frameon=False, fontsize=8.5,
              borderaxespad=0)  # fmt: skip
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("local hour of the target day")
    ax.set_ylabel("coverage of 80% intervals")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(
        "Coverage is marginal, not per hour: nights are over-covered, afternoons under-covered;"
        f"\nCQR narrows the hour-to-hour spread from {spread['split_rolling'] * 100:.0f} to "
        f"{spread['cqr_rolling_aci'] * 100:.0f} pp",
        pad=24,
    )
    return _save(fig, "coverage_by_hour.png")


def gamma_frontier() -> Path:
    table = _results("aci_gamma_sensitivity.csv")
    sub = table[table["method"] == "cqr_rolling_aci"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for level, color in zip(config.LEVELS, (TEAL, AMBER), strict=True):
        g = sub[sub["level"] == level].sort_values("gamma")
        ax.plot(g["gamma"], g["rolling30_within_5pp"], color=color, lw=1.8, marker="o", ms=6)
        for _, row in g.iterrows():
            ax.annotate(
                f"{row['width_mw'] / 1000:.1f} GW", xy=(row["gamma"], row["rolling30_within_5pp"]),
                xytext=(0, -14 if level < 0.9 else 8), textcoords="offset points",
                ha="center", fontsize=7.5, color=SLATE,
            )  # fmt: skip
        ax.annotate(
            f"{round(level * 100)}% intervals", xy=(g["gamma"].iloc[-1], g["rolling30_within_5pp"].iloc[-1]),
            xytext=(10, 0), textcoords="offset points", color=color, fontsize=9, va="center",
            fontweight="bold",
        )  # fmt: skip
    ax.axvline(config.ACI_GAMMA, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.text(config.ACI_GAMMA, 0.02, " pre-set γ", color=INK, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(list(config.ACI_GAMMA_GRID), [f"{g:g}" for g in config.ACI_GAMMA_GRID])
    ax.minorticks_off()
    ax.set_xlabel("ACI step size γ (per day)")
    ax.set_ylabel("30-day windows within ±5 pp")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.05)
    lo = sub[(sub["level"] == 0.8)].set_index("gamma")
    gmax = max(config.ACI_GAMMA_GRID)
    ax.set_title(
        f"Faster adaptation buys local calibration: γ={gmax:g} keeps 80% coverage on target "
        f"{lo.loc[gmax, 'rolling30_within_5pp']:.0%} of the time for "
        f"{lo.loc[gmax, 'width_mw'] / lo.loc[config.ACI_GAMMA, 'width_mw'] - 1:+.0%} width",
        pad=12,
    )
    fig.text(0.0, -0.04, "Labels: mean interval width. CQR rolling + ACI, 2022-04 to 2026-08.",
             fontsize=8, color=SLATE)  # fmt: skip
    fig.subplots_adjust(right=0.85)
    return _save(fig, "gamma_frontier.png")


def make_all() -> list[Path]:
    return [hero(), point_accuracy(), interval_scores(), fan_week(), demand_context(),
            coverage_by_hour(), gamma_frontier()]  # fmt: skip
