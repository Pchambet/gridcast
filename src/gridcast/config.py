"""Project-wide constants: paths, study periods, data sources and design choices.

Everything that shapes a reported number lives here, so a reader can audit the
experimental design in one place instead of hunting through modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(os.environ.get("GRIDCAST_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = DATA / "results"
LIVE = DATA / "live"
FIGURES = ROOT / "docs" / "figures"
SITE = ROOT / "site"

PARIS = ZoneInfo("Europe/Paris")

# --- Forecasting protocol -------------------------------------------------------------
# The forecast for local day D is issued at 12:00 Europe/Paris on D-1 (the timing of the
# day-ahead electricity auction). Hourly demand for [h, h+1) is assumed published one
# hour after it ends, so at issue time the last known hour is [10:00, 11:00) on D-1.
ISSUE_HOUR_LOCAL = 12
PUBLICATION_LAG = pd.Timedelta(hours=1)

# Weather forecasts come from an archive storing, for each valid time t, the value
# forecast 24 h earlier (d1) and 48 h earlier (d2). A d1 value comes from a run started
# at most t - 24 h; it is usable only if that run was finished by the issue time. With
# a conservative production delay of 4 h, d1 is used iff t - 24 h <= issue - 4 h, and
# d2 (always available) otherwise. This never uses a run finished after issue time.
RUN_AVAILABILITY_DELAY = pd.Timedelta(hours=4)

# --- Study periods -------------------------------------------------------------------
HISTORY_START = pd.Timestamp("2012-01-01")
BACKTEST_START = pd.Timestamp("2021-04-01")  # first month with archived day-ahead forecasts
BACKTEST_END = pd.Timestamp("2026-08-31")  # frozen so that reruns reproduce the README
CALIBRATION_END = pd.Timestamp("2022-03-31")  # burn-in year: calibration only, not scored
EVAL_START = CALIBRATION_END + pd.Timedelta(days=1)
CRISIS = (pd.Timestamp("2022-09-01"), pd.Timestamp("2023-03-31"))

# --- Probabilistic layer -------------------------------------------------------------
LEVELS = (0.80, 0.95)
QUANTILES = (0.025, 0.10, 0.90, 0.975)
ROLLING_WINDOW_DAYS = 90
# Errors of day D are fully observed only after the issue time of D+2 (the issue for
# D+1 happens at noon on D, before D is over), so conformal feedback is lagged 2 days.
FEEDBACK_DELAY_DAYS = 2
ACI_GAMMA = 0.02  # per-day step size, fixed before looking at evaluation results
ACI_GAMMA_GRID = (0.005, 0.01, 0.02, 0.05, 0.1)

# --- Model --------------------------------------------------------------------------
# Fixed a priori (no tuning on the evaluation window); native LightGBM names.
LGBM_PARAMS: dict = {
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "lambda_l2": 1.0,
    "num_threads": 3,
    "verbose": -1,
    "seed": 7,
}
LGBM_ROUNDS = 300
LGBM_MAX_BIN = 127

BOOTSTRAP_REPS = 1000
SEED = 7


# --- Data sources -------------------------------------------------------------------
ODRE_EXPORT = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/parquet"
)
ECO2MIX_DEF = "eco2mix-national-cons-def"
ECO2MIX_TR = "eco2mix-national-tr"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
FORECAST_MODELS = ("gfs_seamless", "jma_seamless")  # primary, gap-filler


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    population: int  # INSEE "aire d'attraction des villes" 2020, population 2017


# Ten largest French metropolitan areas; together about 40% of the population.
CITIES: tuple[City, ...] = (
    City("Paris", 48.8566, 2.3522, 13_025_000),
    City("Lyon", 45.7640, 4.8357, 2_281_000),
    City("Marseille", 43.2965, 5.3698, 1_873_000),
    City("Lille", 50.6292, 3.0573, 1_510_000),
    City("Toulouse", 43.6047, 1.4442, 1_454_000),
    City("Bordeaux", 44.8378, -0.5792, 1_364_000),
    City("Nantes", 47.2184, -1.5536, 1_011_000),
    City("Nice", 43.7102, 7.2620, 1_006_000),
    City("Strasbourg", 48.5734, 7.7521, 853_000),
    City("Rennes", 48.1173, -1.6778, 733_000),
)


def city_weights() -> pd.Series:
    """Population weights summing to one, indexed by city name."""
    pop = pd.Series({c.name: c.population for c in CITIES}, dtype=float)
    return pop / pop.sum()
