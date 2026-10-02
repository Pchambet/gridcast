"""Turn backtest predictions into the result tables behind the README and report.

Every number quoted in the README comes from ``data/results/summary.json`` or one of
the CSV tables written here. Evaluation rows are the hours where the actual, our
forecast, RTE's forecast and the naive forecast all exist, so every model is scored on
exactly the same hours.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from gridcast import config, metrics
from gridcast.conformal import METHOD_BY_NAME, METHODS, all_intervals, online_intervals
from gridcast.features import mos_correct
from gridcast.models import bias_adjusted, level_correct, qcol

RESULTS = config.RESULTS
INTERVALS = config.INTERIM / "intervals.parquet"

POINT_MODELS = {
    "point": "gridcast",
    "point_raw": "gridcast, no level correction",
    "rte_j1_adj": "RTE J-1, level-corrected",
    "rte_j1": "RTE J-1, as published",
    "naive": "Seasonal naive (D-7)",
    "point_oracle": "gridcast with observed weather (oracle)",
}
PRIMARY = "cqr_rolling_aci"
# Information sets fed to the same monthly models (column suffixes in the backtest output).
VARIANTS = {"published": "", "oracle weather": "_oracle", "ERA5 delayed as live": "_era5lag"}
TOLERANCE = 0.05  # a 30-day window is "on target" within +-5 pp of nominal coverage
FAN_WEEK = "2022-12-10"  # the December 2022 cold snap, at the height of the crisis


def periods() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    one = pd.Timedelta(days=1)
    return {
        "evaluation": (config.EVAL_START, config.BACKTEST_END),
        "pre-crisis": (config.EVAL_START, config.CRISIS[0] - one),
        "crisis": config.CRISIS,
        "post-crisis": (config.CRISIS[1] + one, config.BACKTEST_END),
    }


def scored_rows(pred: pd.DataFrame) -> pd.DataFrame:
    """Evaluation-window hours where every compared forecast exists."""
    cols = ["load", *POINT_MODELS]
    keep = (pred["day"] >= config.EVAL_START) & (pred["day"] <= config.BACKTEST_END)
    return pred[keep].dropna(subset=cols)


def _in(frame: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    return frame[(frame["day"] >= period[0]) & (frame["day"] <= period[1])]


def _mean_of(values: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda i: float(values[i].mean())


def _mean_gap(a: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda i: float(a[i].mean() - b[i].mean())


def point_table(pred: pd.DataFrame, reps: int = config.BOOTSTRAP_REPS) -> pd.DataFrame:
    rows = []
    for pname, span in periods().items():
        sub = _in(pred, span)
        y = sub["load"].to_numpy()
        blocks = metrics.week_blocks(sub.index)
        # Differences are taken against the level-corrected RTE forecast: the fair benchmark.
        rte_ape = metrics.ape(y, sub["rte_j1_adj"].to_numpy())
        for col, label in POINT_MODELS.items():
            yhat = sub[col].to_numpy()
            ape = metrics.ape(y, yhat)
            est, lo, hi = metrics.block_bootstrap(_mean_of(ape), blocks, reps)
            d_est, d_lo, d_hi = metrics.block_bootstrap(_mean_gap(ape, rte_ape), blocks, reps)
            rows.append(
                {
                    "period": pname,
                    "model": col,
                    "label": label,
                    "hours": len(sub),
                    "mape": est,
                    "mape_lo": lo,
                    "mape_hi": hi,
                    "rmse_mw": metrics.rmse(y, yhat),
                    "mae_mw": metrics.mae(y, yhat),
                    "bias_mw": float(np.mean(yhat - y)),
                    "mape_vs_rte": d_est,
                    "mape_vs_rte_lo": d_lo,
                    "mape_vs_rte_hi": d_hi,
                }
            )
    return pd.DataFrame(rows)


def _attach(pred: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Long interval table joined with outcomes, restricted to scored hours."""
    joined = intervals.merge(
        pred[["day", "hour", "load"]].reset_index(), on="time", how="inner"
    ).dropna(subset=["lo", "hi"])
    joined["hit"] = metrics.hits(joined["load"], joined["lo"], joined["hi"])
    joined["width"] = joined["hi"] - joined["lo"]
    alpha = 1 - joined["level"]
    joined["winkler"] = metrics.interval_score(
        joined["load"].to_numpy(),
        joined["lo"].to_numpy(),
        joined["hi"].to_numpy(),
        alpha.to_numpy(),
    )
    joined["pinball"] = 0.5 * (
        metrics.pinball(joined["load"].to_numpy(), joined["lo"].to_numpy(), alpha / 2)
        + metrics.pinball(joined["load"].to_numpy(), joined["hi"].to_numpy(), 1 - alpha / 2)
    )
    return joined


def daily_coverage(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["method", "level", "day"])
        .agg(hits=("hit", "sum"), hours=("hit", "size"), width=("width", "mean"))
        .reset_index()
    )


def rolling_coverage(daily: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Trailing ``window``-day empirical coverage, per method and level."""
    out = []
    for (method, level), grp in daily.groupby(["method", "level"]):
        g = grp.sort_values("day").set_index("day")
        hits = g["hits"].rolling(f"{window}D").sum()
        hours = g["hours"].rolling(f"{window}D").sum()
        span_ok = (g.index - g.index[0]) >= pd.Timedelta(days=window - 1)
        frame = pd.DataFrame({"coverage": (hits / hours).where(span_ok)}, index=g.index)
        frame["method"], frame["level"] = method, level
        out.append(frame.reset_index())
    return pd.concat(out, ignore_index=True).dropna()


def on_target(rolling: pd.Series, level: float) -> float:
    """Share of 30-day windows whose coverage is within 5 pp of nominal.

    Coverage cannot exceed 100%, so at the 95% level this band is [90%, 100%] and only
    flags under-coverage; the 80% and 95% shares are not comparable with each other.
    """
    return float(np.mean(np.abs(rolling - level) <= TOLERANCE))


def interval_table(scored: pd.DataFrame, reps: int = config.BOOTSTRAP_REPS) -> pd.DataFrame:
    roll = rolling_coverage(daily_coverage(scored))
    rows = []
    for pname, span in periods().items():
        sub = _in(scored, span)
        rsub = _in(roll, span)
        for (method, level), grp in sub.groupby(["method", "level"], sort=False):
            grp = grp.set_index("time")
            hit = grp["hit"].to_numpy(dtype=float)
            est, lo, hi = metrics.block_bootstrap(
                _mean_of(hit), metrics.week_blocks(grp.index), reps
            )
            r = rsub[(rsub["method"] == method) & (rsub["level"] == level)]["coverage"]
            rows.append(
                {
                    "period": pname,
                    "method": method,
                    "label": METHOD_BY_NAME[method].label,
                    "level": level,
                    "coverage": est,
                    "coverage_lo": lo,
                    "coverage_hi": hi,
                    "width_mw": grp["width"].mean(),
                    "winkler_mw": grp["winkler"].mean(),
                    "pinball_mw": grp["pinball"].mean(),
                    "rolling30_within_5pp": on_target(r, level),
                    "rolling30_min": float(r.min()),
                }
            )
    return pd.DataFrame(rows)


def gamma_sensitivity(pred: pd.DataFrame) -> pd.DataFrame:
    """In-sample sensitivity to the ACI step size, over the whole window and the crisis."""
    rows = []
    for gamma in config.ACI_GAMMA_GRID:
        for name in ("split_static_aci", "cqr_rolling_aci"):
            parts = []
            for level in config.LEVELS:
                iv = online_intervals(pred, METHOD_BY_NAME[name], level, gamma=gamma)
                iv["method"], iv["level"] = name, level
                parts.append(iv.reset_index())
            scored = _attach(scored_rows(pred), pd.concat(parts, ignore_index=True))
            roll = rolling_coverage(daily_coverage(_in(scored, periods()["evaluation"])))
            for pname in ("evaluation", "crisis"):
                sub, rsub = _in(scored, periods()[pname]), _in(roll, periods()[pname])
                for level, grp in sub.groupby("level"):
                    r = rsub[rsub["level"] == level]["coverage"]
                    rows.append(
                        {
                            "period": pname,
                            "gamma": gamma,
                            "method": name,
                            "level": level,
                            "coverage": grp["hit"].mean(),
                            "width_mw": grp["width"].mean(),
                            "winkler_mw": grp["winkler"].mean(),
                            "rolling30_within_5pp": on_target(r, level),
                        }
                    )
    return pd.DataFrame(rows)


def coverage_by_hour(scored: pd.DataFrame) -> pd.DataFrame:
    sub = _in(scored, periods()["evaluation"])
    return sub.groupby(["method", "level", "hour"])["hit"].mean().rename("coverage").reset_index()


def monthly_mape(pred: pd.DataFrame) -> pd.DataFrame:
    month = pred["day"].dt.to_period("M").dt.to_timestamp()
    y = pred["load"]
    out = {
        col: pd.Series(metrics.ape(y, pred[col]), index=pred.index).groupby(month).mean()
        for col in POINT_MODELS
    }
    frame = pd.DataFrame(out)
    frame.index.name = "month"
    return frame.reset_index()


def monthly_demand(table: pd.DataFrame) -> pd.DataFrame:
    local = table.copy()
    local["month"] = local.index.tz_convert(config.PARIS).tz_localize(None).to_period("M")
    out = local.groupby("month").agg(load_gw=("load", "mean"), temp_c=("temp_obs", "mean"))
    out["load_gw"] /= 1000
    out = out[out.index <= config.BACKTEST_END.to_period("M")]
    out.index = out.index.to_timestamp()
    return out.reset_index()


def rte_bias(table: pd.DataFrame) -> pd.DataFrame:
    """RTE's published J-1 error by year (consolidated series) and on the real-time feed."""
    t = table[(table.index >= pd.Timestamp("2015-01-01", tz="UTC"))].dropna(
        subset=["load", "rte_j1"]
    )
    t = t[t.index < (config.BACKTEST_END + pd.Timedelta(days=1)).tz_localize(config.PARIS)]
    err = (t["rte_j1"] - t["load"]) / t["load"]
    is_rt = t["realtime"].fillna(0).to_numpy() == 1
    year = t.index.tz_convert(config.PARIS).year.astype(str)
    segment = np.where(is_rt, "real-time feed", year)
    frame = pd.DataFrame({"segment": segment, "bias": err, "ape": err.abs()})
    out = frame.groupby("segment", sort=False).agg(
        hours=("bias", "size"), bias_pct=("bias", "mean"), mape=("ape", "mean")
    )
    return out.reset_index()


def rte_bias_by_hour(table: pd.DataFrame) -> pd.DataFrame:
    """RTE's J-1 bias against the consolidated series by local hour, before and since 2023.

    The gap is far from a constant offset: it depends strongly on the hour of day.
    """
    t = table[table["realtime"].fillna(0) == 0].dropna(subset=["load", "rte_j1"])
    local = t.index.tz_convert(config.PARIS)
    err = (t["rte_j1"] - t["load"]) / t["load"]
    out = {}
    for name, keep in (
        ("bias_2015_2021", (local.year >= 2015) & (local.year <= 2021)),
        ("bias_since_2023", local.year >= 2023),
    ):
        out[name] = err[keep].groupby(local.hour[keep]).mean()
    frame = pd.DataFrame(out)
    frame.index.name = "local_hour"
    return frame.reset_index()


def information_set_ablation(raw: pd.DataFrame, scored: pd.Index) -> pd.DataFrame:
    """Re-run level correction and the published intervals under other information sets.

    ``raw`` holds the backtest output before level correction, with the published
    columns and the ``*_oracle`` (observed target-day weather) and ``*_era5lag``
    (reanalysis masked over the last days before issue, as live) variants. Scores are
    taken on the same evaluation hours as everything else.
    """
    cols = ["point", *[qcol(q) for q in config.QUANTILES]]
    rows = []
    for variant, suffix in VARIANTS.items():
        frame = raw[["day", "load"]].copy()
        for col in cols:
            frame[col] = raw[col + suffix]
        frame = level_correct(frame)
        sub = frame.loc[scored]
        row = {"variant": variant, "mape": metrics.mape(sub["load"], sub["point"])}
        for level in config.LEVELS:
            tag = round(level * 100)
            for name in ("qr_raw", PRIMARY):
                iv = online_intervals(frame, METHOD_BY_NAME[name], level).loc[scored]
                ok = iv["lo"].notna() & iv["hi"].notna()
                y = sub.loc[ok, "load"].to_numpy()
                lo, hi = iv.loc[ok, "lo"].to_numpy(), iv.loc[ok, "hi"].to_numpy()
                row[f"{name}_coverage{tag}"] = metrics.coverage(y, lo, hi)
                row[f"{name}_width{tag}_mw"] = float(np.mean(hi - lo))
                if name == PRIMARY:
                    row[f"{name}_winkler{tag}_mw"] = float(
                        metrics.interval_score(y, lo, hi, 1 - level).mean()
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def temperature_bias(table: pd.DataFrame) -> pd.DataFrame:
    """Mean error of the 24 h-ahead temperature forecast vs ERA5, raw and MOS-corrected."""
    lo = config.BACKTEST_START.tz_localize(config.PARIS)
    hi = (config.BACKTEST_END + pd.Timedelta(days=1)).tz_localize(config.PARIS)
    corrected = mos_correct(table)["temp_fc_d1"]
    window = (table.index >= lo) & (table.index < hi)
    t = table[window]
    hour = t.index.tz_convert(config.PARIS).hour
    out = pd.DataFrame(
        {
            "raw_bias_c": (t["temp_fc_d1"] - t["temp_obs"]).groupby(hour).mean(),
            "mos_bias_c": (corrected[window] - t["temp_obs"]).groupby(hour).mean(),
            "raw_mae_c": (t["temp_fc_d1"] - t["temp_obs"]).abs().groupby(hour).mean(),
            "mos_mae_c": (corrected[window] - t["temp_obs"]).abs().groupby(hour).mean(),
        }
    )
    out.index.name = "local_hour"
    return out.reset_index()


def fan_week(
    pred: pd.DataFrame, intervals: pd.DataFrame, start: str, days: int = 7
) -> pd.DataFrame:
    lo_t = pd.Timestamp(start)
    sub = pred[(pred["day"] >= lo_t) & (pred["day"] < lo_t + pd.Timedelta(days=days))]
    out = sub[["day", "hour", "load", "point", "rte_j1", "rte_j1_adj"]].copy()
    for level in config.LEVELS:
        iv = intervals[(intervals["method"] == PRIMARY) & (intervals["level"] == level)]
        iv = iv.set_index("time").reindex(out.index)
        tag = round(level * 100)
        out[f"lo{tag}"], out[f"hi{tag}"] = iv["lo"], iv["hi"]
    return out.reset_index()


def summary(
    point: pd.DataFrame,
    interval: pd.DataFrame,
    *,
    monthly: pd.DataFrame,
    intervals: pd.DataFrame,
    ablation: pd.DataFrame,
    table: pd.DataFrame,
) -> dict:
    def pt(model: str, period: str = "evaluation") -> pd.Series:
        return point[(point["model"] == model) & (point["period"] == period)].iloc[0]

    def iv(method: str, level: float, period: str = "evaluation") -> pd.Series:
        sel = (
            (interval["method"] == method)
            & (interval["level"] == level)
            & (interval["period"] == period)
        )
        return interval[sel].iloc[0]

    out: dict = {
        "eval_start": str(config.EVAL_START.date()),
        "eval_end": str(config.BACKTEST_END.date()),
        "eval_hours": int(pt("point")["hours"]),
        "crisis": [str(config.CRISIS[0].date()), str(config.CRISIS[1].date())],
        "aci_gamma": config.ACI_GAMMA,
        "point": {},
        "intervals": {},
    }
    for period in periods():
        out["point"][period] = {
            m: {
                k: round(float(pt(m, period)[k]), 5)
                for k in (
                    "mape",
                    "mape_lo",
                    "mape_hi",
                    "rmse_mw",
                    "bias_mw",
                    "mape_vs_rte",
                    "mape_vs_rte_lo",
                    "mape_vs_rte_hi",
                )
            }
            for m in POINT_MODELS
        }
        out["intervals"][period] = {
            f"{m.name}@{round(lv * 100)}": {
                k: round(float(iv(m.name, lv, period)[k]), 5)
                for k in (
                    "coverage",
                    "coverage_lo",
                    "coverage_hi",
                    "width_mw",
                    "winkler_mw",
                    "rolling30_within_5pp",
                    "rolling30_min",
                )
            }
            for m in METHODS
            for lv in config.LEVELS
        }

    months = monthly[monthly["month"] >= config.EVAL_START]
    ahead = months[months["point"] < months["rte_j1_adj"]]["month"]
    out["months_ahead_of_rte"] = {
        "months": len(months),
        "ahead": len(ahead),
        "list": [f"{m:%Y-%m}" for m in ahead],
        "calendar_months": sorted({int(m.month) for m in ahead}),
    }
    local = intervals["time"].dt.tz_convert(config.PARIS).dt.tz_localize(None).dt.normalize()
    ev = _in(intervals.assign(day=local), periods()["evaluation"])
    out["aci_capped_days"] = {
        f"{m.name}@{round(lv * 100)}": int(
            ev[(ev["method"] == m.name) & (ev["level"] == lv) & ev["capped"]]["day"].nunique()
        )
        for m in METHODS
        if m.aci
        for lv in config.LEVELS
    }
    span = (table.index >= config.BACKTEST_START.tz_localize(config.PARIS)) & (
        table.index < (config.BACKTEST_END + pd.Timedelta(days=1)).tz_localize(config.PARIS)
    )
    filled = table.index[span & (table.get("temp_fc_filled", 0) == 1)]
    out["gap_filled_forecast_hours"] = {
        "hours": len(filled),
        "first": str(filled.min()) if len(filled) else None,
        "last": str(filled.max()) if len(filled) else None,
    }
    out["information_sets"] = {
        r["variant"]: {k: round(float(v), 5) for k, v in r.items() if k != "variant"}
        for r in ablation.to_dict("records")
    }
    return out


def run(table: pd.DataFrame, pred: pd.DataFrame, reps: int = config.BOOTSTRAP_REPS) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pred.copy()
    pred = level_correct(pred)
    oracle = bias_adjusted(pred["point_oracle"], pred["load"])
    pred["point_oracle"] = oracle.fillna(pred["point_oracle"])
    pred["rte_j1_adj"] = bias_adjusted(table["rte_j1"], table["load"]).reindex(pred.index)
    intervals = all_intervals(pred)
    intervals.to_parquet(INTERVALS)

    rows = scored_rows(pred)
    point = point_table(rows, reps)
    scored = _attach(rows, intervals)
    interval = interval_table(scored, reps)
    gamma = gamma_sensitivity(pred)
    daily = daily_coverage(scored)
    monthly = monthly_mape(rows)
    ablation = information_set_ablation(raw, rows.index)

    point.to_csv(RESULTS / "point_metrics.csv", index=False, float_format="%.6g")
    interval.to_csv(RESULTS / "interval_metrics.csv", index=False, float_format="%.6g")
    gamma.to_csv(RESULTS / "aci_gamma_sensitivity.csv", index=False, float_format="%.6g")
    daily.to_csv(RESULTS / "coverage_daily.csv", index=False, float_format="%.6g")
    coverage_by_hour(scored).to_csv(
        RESULTS / "coverage_by_hour.csv", index=False, float_format="%.6g"
    )
    monthly.to_csv(RESULTS / "monthly_mape.csv", index=False, float_format="%.6g")
    ablation.to_csv(RESULTS / "information_sets.csv", index=False, float_format="%.6g")
    monthly_demand(table).to_csv(RESULTS / "monthly_demand.csv", index=False, float_format="%.6g")
    temperature_bias(table).to_csv(
        RESULTS / "temperature_bias.csv", index=False, float_format="%.4g"
    )
    rte_bias(table).to_csv(RESULTS / "rte_bias.csv", index=False, float_format="%.6g")
    rte_bias_by_hour(table).to_csv(
        RESULTS / "rte_bias_by_hour.csv", index=False, float_format="%.6g"
    )
    fan_week(pred, intervals, FAN_WEEK).to_csv(
        RESULTS / "fan_week.csv", index=False, float_format="%.6g"
    )
    seed = pred[
        pred["day"] > config.BACKTEST_END - pd.Timedelta(days=config.ROLLING_WINDOW_DAYS + 5)
    ]
    variant_cols = [c for c in seed if c.endswith(("_oracle", "_era5lag"))]
    seed.drop(columns=[*variant_cols, "refit", "rte_j1_adj"]).to_parquet(
        RESULTS / "calibration_seed.parquet"
    )

    result = summary(
        point, interval, monthly=monthly, intervals=intervals, ablation=ablation, table=table
    )
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
