"""Shared fixtures: a synthetic hourly table spanning both DST changes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from gridcast import config, infoset


def make_table(start: str = "2023-09-01", end: str = "2024-04-30", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="UTC", name="time")
    hours = idx.tz_convert("Europe/Paris").hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    temp = 12 + 8 * np.cos(2 * np.pi * (doy - 200) / 365) + 4 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp = temp + rng.normal(0, 1, len(idx))
    load = 52_000 + 6_000 * np.sin(2 * np.pi * (hours - 6) / 24) - 1_500 * (temp - 15)
    load = load + rng.normal(0, 500, len(idx))
    return pd.DataFrame(
        {
            "load": load,
            "rte_j1": load + rng.normal(0, 600, len(idx)),
            "rte_j": load + rng.normal(0, 400, len(idx)),
            "temp_obs": temp,
            "temp_fc_d1": temp + 0.8 + rng.normal(0, 1, len(idx)),
            "temp_fc_d2": temp + 0.8 + rng.normal(0, 1.5, len(idx)),
        },
        index=idx,
    )


@pytest.fixture(scope="session")
def table() -> pd.DataFrame:
    return make_table()


def poison_future(table: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Replace everything unknowable at the issue time of ``day`` by garbage."""
    poisoned = table.copy()
    rng = np.random.default_rng(1)
    after_cutoff = poisoned.index >= infoset.demand_cutoff(day)
    after_issue = poisoned.index >= infoset.issue_time(day)
    poisoned.loc[after_cutoff, "load"] = rng.normal(1e6, 1e5, after_cutoff.sum())
    poisoned.loc[after_issue, "temp_obs"] = rng.normal(80, 10, after_issue.sum())
    # Forecasts are legitimate inputs only if issued before the issue time; corrupt the
    # 24 h-ahead values that were not yet available (valid more than 20 h after issue).
    late = poisoned.index > infoset.issue_time(day) + pd.Timedelta(hours=20)
    poisoned.loc[late, "temp_fc_d1"] = rng.normal(80, 10, late.sum())
    return poisoned


@pytest.fixture(scope="session")
def poison() -> Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]:
    return poison_future


@pytest.fixture
def fast_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Few boosting rounds: tests check plumbing and information sets, not accuracy."""
    monkeypatch.setattr(config, "LGBM_ROUNDS", 20)
