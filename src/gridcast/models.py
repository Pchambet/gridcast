"""Point and quantile gradient-boosting models plus the two reference forecasts."""

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
    rounds: int = config.LGBM_ROUNDS
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


def seasonal_naive(frame: pd.DataFrame) -> pd.Series:
    """Same local hour one week earlier: the floor any model must beat."""
    return frame["load_d7"]
