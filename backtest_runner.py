"""Walk-forward February 2026 forecasts using only weather known at issue time."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.pipeline import ForecastAgent

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["timestamp", "turbine_id", "predicted_normalized_power"]


def run_backtest(
    start_date: str | date,
    end_date: str | date,
    horizon_hours: int = 48,
    *,
    agent: ForecastAgent | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Forecast every target day from the previous day's fixed 12:00 UTC issue."""
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    if end < start:
        raise ValueError("end_date must not be before start_date")

    active_agent = agent or ForecastAgent()
    rows: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    for target_day in pd.date_range(start, end, freq="1D"):
        result = active_agent.run(
            issue_date=(target_day - pd.Timedelta(days=1)).date(),
            horizon_hours=horizon_hours,
        )
        if result.status == "error" or result.forecast is None:
            detail = result.steps[-1]["detail"] if result.steps else "unknown error"
            raise RuntimeError(f"Backtest failed for {target_day.date()}: {detail}")

        forecast = result.forecast.copy()
        target_time = pd.to_datetime(forecast["target_time"], utc=True)
        selected = forecast.loc[
            target_time.between(target_day, target_day + pd.Timedelta(hours=23)),
            ["target_time", "turbine_id", "predicted_power"],
        ].copy()
        expected_rows = 24 * len(result.turbine_ids)
        if len(selected) != expected_rows:
            raise ValueError(
                f"Backtest got {len(selected)}/{expected_rows} rows for {target_day.date()}"
            )
        selected = selected.rename(
            columns={
                "target_time": "timestamp",
                "predicted_power": "predicted_normalized_power",
            }
        )
        rows.append(selected)
        runs.append(
            {
                "target_date": target_day.date().isoformat(),
                "issue_date": result.issue_date.isoformat(),
                "status": result.status,
                "steps": [f"{step['stage']}:{step['status']}" for step in result.steps],
            }
        )
        LOGGER.info("Backtest target day %s complete", target_day.date())

    submission = pd.concat(rows, ignore_index=True).sort_values(
        ["timestamp", "turbine_id"],
        kind="stable",
    )
    return submission.loc[:, OUTPUT_COLUMNS].reset_index(drop=True), runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-02-01")
    parser.add_argument("--end-date", default="2026-02-28")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    submission, runs = run_backtest(args.start_date, args.end_date, args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "forecast_february_2026.csv"
    output = submission.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output.to_csv(csv_path, index=False)

    report_path = args.output_dir / "backtest_report.json"
    report_path.write_text(
        json.dumps({"csv": str(csv_path), "rows": len(output), "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(output)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
