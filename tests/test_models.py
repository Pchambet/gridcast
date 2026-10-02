import numpy as np
import pandas as pd
import pytest

from gridcast.features import FEATURES
from gridcast.models import DemandModel, bias_adjusted, level_correct, qcol


def _hourly(days: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")


def test_bias_adjustment_removes_a_constant_offset_after_warmup():
    idx = _hourly(60)
    actual = pd.Series(50_000 + 1_000 * np.sin(np.arange(len(idx)) / 5), index=idx)
    forecast = actual - 1_200.0  # always 1.2 GW too low
    adjusted = bias_adjusted(forecast, actual)
    late = idx >= idx[0] + pd.Timedelta(days=10)
    np.testing.assert_allclose(adjusted[late], actual[late], atol=1e-6)


def test_bias_adjustment_never_uses_errors_younger_than_two_days():
    idx = _hourly(40)
    actual = pd.Series(50_000.0, index=idx)
    forecast = pd.Series(49_000.0, index=idx)
    base = bias_adjusted(forecast, actual)
    shocked_actual = actual.copy()
    cut = idx[-1] - pd.Timedelta(hours=47)  # the last two days' outcomes
    shocked_actual[idx >= cut] += 5_000.0
    after = bias_adjusted(forecast, shocked_actual)
    pd.testing.assert_series_equal(base, after)
    assert base.iloc[-1] == pytest.approx(50_000.0)


def test_level_correction_shifts_point_and_quantiles_together():
    idx = _hourly(40)
    load = pd.Series(50_000.0, index=idx)
    pred = pd.DataFrame({"load": load, "point": 50_800.0}, index=idx)
    for q, off in ((0.025, -2000), (0.1, -1000), (0.9, 1000), (0.975, 2000)):
        pred[qcol(q)] = 50_800.0 + off
    out = level_correct(pred)
    late = idx >= idx[0] + pd.Timedelta(days=10)
    np.testing.assert_allclose(out.loc[late, "point"], 50_000.0)
    np.testing.assert_allclose(out.loc[late, qcol(0.9)] - out.loc[late, "point"], 1000.0)
    assert (out["point_raw"] == 50_800.0).all()
    assert (out.loc[~late, "point"].iloc[:24] == 50_800.0).all()  # no history yet: untouched


def test_demand_model_recovers_a_linear_temperature_response():
    # Ground truth: load falls 1.5 GW per degree, other features are pure noise.
    rng = np.random.default_rng(0)
    n = 4000

    def sample(size: int) -> pd.DataFrame:
        frame = pd.DataFrame(rng.normal(0, 1, (size, len(FEATURES))), columns=FEATURES)
        frame["temp"] = rng.uniform(-5, 30, size)
        frame["load"] = 60_000 - 1_500 * frame["temp"] + rng.normal(0, 800, size)
        return frame

    model = DemandModel(quantiles=(0.1, 0.9)).fit(sample(n))  # production settings otherwise
    test = sample(2000)
    pred = model.predict(test)
    truth = 60_000 - 1_500 * test["temp"]
    assert np.sqrt(np.mean((pred["point"] - truth) ** 2)) < 400  # well under the noise
    # Slope over the bulk of the range, read off the fitted point model.
    grid = test.iloc[:1].loc[np.repeat(test.index[:1], 2)].reset_index(drop=True)
    grid["temp"] = [5.0, 20.0]
    slope = np.diff(model.predict(grid)["point"].to_numpy())[0] / 15.0
    assert slope == pytest.approx(-1_500, rel=0.1)
    # Independently fitted quantiles are returned in order (no crossing).
    qs = pred[[qcol(0.1), qcol(0.9)]].to_numpy()
    assert (np.diff(qs, axis=1) >= 0).all()
