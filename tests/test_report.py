import pandas as pd

from gridcast import live
from gridcast.report import _live_section, replace_blocks


def test_generated_readme_blocks_are_replaced_and_the_rest_kept():
    text = "intro\n<!-- BEGIN:t -->\nold\nrows\n<!-- END:t -->\noutro\n<!-- BEGIN:u -->\nx\n<!-- END:u -->"
    out = replace_blocks(text, {"t": "| new |", "missing": "ignored"})
    assert (
        out
        == "intro\n<!-- BEGIN:t -->\n| new |\n<!-- END:t -->\noutro\n<!-- BEGIN:u -->\nx\n<!-- END:u -->"
    )
    assert replace_blocks(out, {"t": "| new |"}) == out  # idempotent


def test_empty_block_is_filled():
    text = "<!-- BEGIN:t -->\n<!-- END:t -->"
    assert replace_blocks(text, {"t": "a\nb"}) == "<!-- BEGIN:t -->\na\nb\n<!-- END:t -->"


def test_live_section_flags_a_stale_forecast_and_ignores_hindcasts():
    times = pd.date_range("2026-10-02 22:00", periods=24, freq="h", tz="UTC")
    log = pd.DataFrame({"time": times, "day": pd.Timestamp("2026-10-03"), "hour": range(24)})
    log["issued_at"] = pd.Timestamp("2026-10-02 10:00", tz="UTC")
    log["kind"] = "live"
    for col in ["point_raw", "point", *live.QCOLS, *live.BANDS, "rte_j1", "load"]:
        log[col] = 50_000.0
    log[["load", "rte_j1"]] = float("nan")
    hind = log.assign(day=pd.Timestamp("2026-10-04"), kind="hindcast")
    fresh, _, _ = _live_section(pd.concat([log, hind]), pd.Timestamp("2026-10-02 12:00", tz="UTC"))
    stale, _, has_record = _live_section(log, pd.Timestamp("2026-10-06 12:00", tz="UTC"))
    assert "class=warn" not in fresh and "03 October 2026" in fresh
    assert "class=warn" in stale and not has_record


def test_latest_forecast_json_follows_the_portfolio_contract():
    from gridcast.report import latest_forecast

    times = pd.date_range("2026-10-02 22:00", periods=24, freq="h", tz="UTC")
    log = pd.DataFrame({"time": times, "day": pd.Timestamp("2026-10-03"), "hour": range(24)})
    log["issued_at"] = pd.Timestamp("2026-10-02 10:00", tz="UTC")
    log["kind"] = "live"
    for col in ["point_raw", "point", *live.QCOLS, "rte_j1", "load"]:
        log[col] = 50_000.0
    log[["lo80", "hi80", "lo95", "hi95"]] = [48_000.0, 52_000.0, 47_000.0, 53_000.0]
    log[["load"]] = float("nan")
    log.loc[3, "rte_j1"] = float("nan")
    hind = log.assign(day=pd.Timestamp("2026-10-04"), kind="hindcast")

    out = latest_forecast(pd.concat([log, hind]))
    assert out["target_date"] == "2026-10-03"  # hindcasts never count as the latest forecast
    assert out["issued_at"] == "2026-10-02T10:00:00Z"
    assert len(out["points"]) == 24
    first = out["points"][0]
    assert first == {
        "t": "2026-10-02T22:00:00Z", "p50": 50000.0, "lo80": 48000.0, "hi80": 52000.0,
        "lo95": 47000.0, "hi95": 53000.0, "bench": 50000.0,
    }  # fmt: skip
    assert out["points"][3]["bench"] is None  # missing benchmark is null, never NaN
    assert out["track_record"] == {"days": 0}


def test_latest_forecast_is_none_before_the_first_live_run():
    from gridcast.report import latest_forecast

    assert latest_forecast(pd.DataFrame(columns=live.LOG_COLUMNS)) is None
