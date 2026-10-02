"""Point and quantile gradient-boosting models, and the online level correction."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast import config
from gridcast.features import FEATURES


@dataclass
class DemandModel:
    """One L2 booster for the point forecast and one quantile booster per level."""

    quantiles: tuple[float, ...] = config.QUANTILES
    params: dict = field(default_factory=lambda: dict(config.LGBM_PARAMS))
    rounds: int = field(default_factory=lambda: config.LGBM_ROUNDS)
    boosters: dict[str, lgb.Booster] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> DemandModel:
        rows = frame.dropna(subset=["load", "temp"])
        data = lgb.Dataset(
            rows[FEATURES],
            rows["load"],
            params={"max_bin": config.LGBM_MAX_BIN},
            free_raw_data=False,
        )
        self.boosters = {
            "point": lgb.train({**self.params, "objective": "regression"}, data, self.rounds)
        }
        for q in self.quantiles:
            params = {**self.params, "objective": "quantile", "alpha": q}
            self.boosters[qcol(q)] = lgb.train(params, data, self.rounds)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[FEATURES]
        out = pd.DataFrame(
            {name: b.predict(x) for name, b in self.boosters.items()}, index=frame.index
        )
        qs = [qcol(q) for q in self.quantiles]
        # Independently fitted quantiles can cross; sorting restores monotonicity and
        # never increases pinball loss (Chernozhukov et al., 2010).
        out[qs] = np.sort(out[qs].to_numpy(), axis=1)
        return out


def qcol(q: float) -> str:
    return f"q{q:.3f}"


def bias_adjusted(
    forecast: pd.Series, actual: pd.Series, window_days: int = 28, delay_days: int = 2
) -> pd.Series:
    """Subtract a forecast's own trailing mean error at the same local hour.

    Only errors at least ``delay_days`` old enter the correction, so it is computable at
    issue time. Since 2023 RTE's published J-1 runs about 2% below the consolidated
    demand series it is scored against (``data/results/rte_bias.csv``). Public data
    cannot tell whether that gap is definitional (the consolidated and real-time files
    never overlap) or forecast error, so both forecasters get this same correction.
    """
    err = (forecast - actual).dropna()
    hours = forecast.index.tz_convert("Europe/Paris").hour
    err_hours = err.index.tz_convert("Europe/Paris").hour
    bias = pd.Series(np.nan, index=forecast.index)
    for h in range(24):
        trailing = err[err_hours == h].rolling(f"{window_days}D", min_periods=7).mean()
        trailing.index = trailing.index + pd.Timedelta(days=delay_days)
        target = forecast.index[hours == h]
        bias[hours == h] = trailing.reindex(target, method="ffill").to_numpy()
    return forecast - bias


def level_correct(pred: pd.DataFrame) -> pd.DataFrame:
    """Shift the point forecast and its quantiles by the model's own recent bias.

    ``pred`` is indexed by UTC hour with the raw ``point``, quantile columns and the
    realised ``load``. The raw point is kept as ``point_raw``. Exactly the same
    correction is applied to RTE's forecast in the evaluation, so the comparison is
    symmetric: both forecasters may learn from their own errors once these are known.
    """
    out = pred.sort_index().copy()
    out["point_raw"] = out["point"]
    offset = (out["point"] - bias_adjusted(out["point"], out["load"])).fillna(0.0)
    cols = ["point", *[qcol(q) for q in config.QUANTILES if qcol(q) in out]]
    out[cols] = out[cols].sub(offset, axis=0)
    return out
