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
