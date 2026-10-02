import pandas as pd

from gridcast import backtest, config, live


def test_backtest_refit_ignores_everything_after_the_first_issue_of_the_month(
    table, poison, fast_model, tmp_path, monkeypatch
):
    # The model for a month is trained before that month's first issue time, so poisoning
    # everything unknowable then cannot change the forecast for the month's first day.
    monkeypatch.setattr(config, "INTERIM", tmp_path)
    monkeypatch.setattr(backtest, "CHECKPOINTS", tmp_path / "months")
    monkeypatch.setattr(backtest, "PREDICTIONS", tmp_path / "predictions.parquet")
    day = pd.Timestamp("2024-02-01")
    clean = backtest.run_backtest(table, start=day, end=day)
    dirty = backtest.run_backtest(poison(table, day), start=day, end=day)
    cols = ["point", *live.QCOLS] + [f"{c}_era5lag" for c in ["point", *live.QCOLS]]
    pd.testing.assert_frame_equal(clean[cols], dirty[cols])
    # The oracle columns use the target day's observed weather, so they must react.
    assert not clean["point_oracle"].equals(dirty["point_oracle"])
