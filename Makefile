.PHONY: setup data run report live test lint all clean

setup:  ## install the locked environment
	uv sync --locked

data:  ## download (cached) demand and weather, build the hourly table
	uv run gridcast data

run:  ## backtest, evaluate, figures (~30 min on 3 cores)
	uv run gridcast backtest
	uv run gridcast evaluate
	uv run gridcast figures

report:  ## build site/index.html and refresh README tables
	uv run gridcast report

live:  ## simulate the daily job: refresh data, forecast tomorrow, rebuild the page
	uv run gridcast data --refresh
	uv run gridcast live
	uv run gridcast report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

all: setup data run report

clean:  ## remove generated, non-committed artefacts
	rm -rf data/interim site
