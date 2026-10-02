"""Command-line entry point: ``uv run gridcast <step>``."""

from __future__ import annotations

import argparse
import time


def _data(args: argparse.Namespace) -> None:
    from gridcast import data

    data.fetch_demand(refresh=args.refresh)
    data.fetch_weather(refresh=args.refresh)
    table = data.build_hourly()
    print(f"hourly table: {len(table):,} rows, {table.index.min()} -> {table.index.max()}")


def _backtest(_: argparse.Namespace) -> None:
    from gridcast import backtest, data

    pred = backtest.run_backtest(data.load_hourly())
    print(f"predictions: {len(pred):,} hours")


def _evaluate(_: argparse.Namespace) -> None:
    from gridcast import backtest, data, evaluate

    result = evaluate.run(data.load_hourly(), backtest.load_predictions())
    ev = result["point"]["evaluation"]
    print(
        f"MAPE gridcast {ev['point']['mape']:.2%} | RTE {ev['rte_j1']['mape']:.2%} | "
        f"naive {ev['naive']['mape']:.2%}"
    )


def _figures(_: argparse.Namespace) -> None:
    from gridcast import figures

    for path in figures.make_all():
        print(f"wrote {path}")


def _report(_: argparse.Namespace) -> None:
    from gridcast import report

    print(f"wrote {report.build()}")


def _live(args: argparse.Namespace) -> None:
    import pandas as pd

    from gridcast import data, live

    now = pd.Timestamp(args.now, tz="UTC") if args.now else None
    live.run(data.load_hourly(), now=now)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gridcast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="download / refresh raw data and build the hourly table")
    p.add_argument("--refresh", action="store_true", help="re-fetch the growing sources")
    p.set_defaults(func=_data)
    sub.add_parser("backtest", help="monthly rolling-origin backtest").set_defaults(func=_backtest)
    sub.add_parser("evaluate", help="metrics and result tables").set_defaults(func=_evaluate)
    sub.add_parser("figures", help="static README figures").set_defaults(func=_figures)
    sub.add_parser("report", help="build site/index.html").set_defaults(func=_report)
    p = sub.add_parser("live", help="forecast tomorrow and update the forecast log")
    p.add_argument("--now", help="pretend the job runs at this UTC time (testing)")
    p.set_defaults(func=_live)

    args = parser.parse_args(argv)
    tic = time.perf_counter()
    args.func(args)
    print(f"[{args.command}] done in {time.perf_counter() - tic:.0f} s")


if __name__ == "__main__":
    main()
