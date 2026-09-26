"""Build a clean hourly SCADA training set for both wind turbines."""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_TURBINE_1_PATH: Final = PROJECT_ROOT / "data" / (
    "Dataset_HackAlemAI_для_участников_11_03_2023_28_02_2026_turbine.xlsx"
)
DEFAULT_TURBINE_2_PATH: Final = PROJECT_ROOT / "data" / (
    "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 2.csv"
)
DEFAULT_OUTPUT_PATH: Final = PROJECT_ROOT / "data" / "processed" / (
    "scada_hourly.parquet"
)

TIMESTAMP_COLUMN: Final = "timestamp"
TARGET_COLUMN: Final = "power"
SIGNAL_COLUMNS: Final = ["wind_speed", "temperature", TARGET_COLUMN]
REQUIRED_COLUMNS: Final = {TIMESTAMP_COLUMN, *SIGNAL_COLUMNS}

COLUMN_ALIASES: Final = {
    "timestamp": TIMESTAMP_COLUMN,
    "статистическоевремя": TIMESTAMP_COLUMN,
    "windspeed": "wind_speed",
    "windspeedms": "wind_speed",
    "средняяскоростьветраms": "wind_speed",
    "temperature": "temperature",
    "temperaturec": "temperature",
    "средняятемператураокружающейсредыc": "temperature",
    "power": TARGET_COLUMN,
    "normalizedactivepower": TARGET_COLUMN,
    "нормализованнаяактивнаямощность": TARGET_COLUMN,
}


def _column_key(value: object) -> str:
    """Normalize Russian and English column names for reliable matching."""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    normalized = normalized.replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", normalized)


def _read_source(path: Path) -> pd.DataFrame:
    """Read an XLSX/XLS or UTF-8 CSV SCADA source."""
    if not path.is_file():
        raise FileNotFoundError(f"SCADA source not found: {path}")

    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Unsupported SCADA format {suffix!r}: {path}")


def load_turbine_data(path: Path, turbine_id: int) -> pd.DataFrame:
    """Load one turbine, standardize columns, and validate source values."""
    if turbine_id not in {1, 2}:
        raise ValueError(f"Unsupported turbine_id={turbine_id}; expected 1 or 2")

    raw = _read_source(path)
    rename_map: dict[object, str] = {}
    for column in raw.columns:
        standard_name = COLUMN_ALIASES.get(_column_key(column))
        if standard_name is not None:
            rename_map[column] = standard_name

    standardized = raw.rename(columns=rename_map)
    duplicated = standardized.columns[standardized.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Ambiguous mapped columns in {path}: {duplicated}")

    missing = REQUIRED_COLUMNS.difference(standardized.columns)
    if missing:
        raise ValueError(
            f"SCADA source {path} is missing {sorted(missing)}; "
            f"available columns: {list(raw.columns)}"
        )

    data = standardized.loc[:, [TIMESTAMP_COLUMN, *SIGNAL_COLUMNS]].copy()
    data[TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TIMESTAMP_COLUMN],
        errors="coerce",
        utc=True,
    )
    for column in SIGNAL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_timestamps = int(data[TIMESTAMP_COLUMN].isna().sum())
    if invalid_timestamps:
        LOGGER.warning(
            "Dropping %d rows with invalid timestamps from turbine %d",
            invalid_timestamps,
            turbine_id,
        )
        data = data.dropna(subset=[TIMESTAMP_COLUMN])

    if data.empty:
        raise ValueError(f"SCADA source has no valid timestamped rows: {path}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].clip(lower=0.0, upper=1.0)
    data["turbine_id"] = turbine_id
    data = data.sort_values(TIMESTAMP_COLUMN, kind="stable")

    LOGGER.info(
        "Loaded turbine %d: %d rows, %s through %s",
        turbine_id,
        len(data),
        data[TIMESTAMP_COLUMN].min(),
        data[TIMESTAMP_COLUMN].max(),
    )
    return data


def _interpolate_short_gaps(series: pd.Series, max_gap_hours: int = 2) -> pd.Series:
    """Interpolate complete interior gaps no longer than ``max_gap_hours``."""
    missing = series.isna()
    if not missing.any():
        return series

    run_ids = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_lengths = missing.groupby(run_ids).transform("sum")
    eligible = missing & run_lengths.le(max_gap_hours)
    interpolated = series.interpolate(method="time", limit_area="inside")

    result = series.copy()
    result.loc[eligible] = interpolated.loc[eligible]
    return result


def resample_hourly(scada: pd.DataFrame) -> pd.DataFrame:
    """Resample each turbine independently to left-labelled UTC hours."""
    hourly = (
        scada.set_index(TIMESTAMP_COLUMN)
        .groupby("turbine_id")
        .resample("1h", label="left", closed="left")[SIGNAL_COLUMNS]
        .mean()
        .reset_index()
        .sort_values(["turbine_id", TIMESTAMP_COLUMN], kind="stable")
    )

    interpolated_groups: list[pd.DataFrame] = []
    for turbine_id, group in hourly.groupby("turbine_id", sort=True):
        turbine = group.set_index(TIMESTAMP_COLUMN).copy()
        for column in SIGNAL_COLUMNS:
            turbine[column] = _interpolate_short_gaps(turbine[column])
        turbine["turbine_id"] = int(turbine_id)
        interpolated_groups.append(turbine.reset_index())

    if not interpolated_groups:
        raise ValueError("Hourly SCADA set is empty")

    result = pd.concat(interpolated_groups, ignore_index=True)
    result[TARGET_COLUMN] = result[TARGET_COLUMN].clip(0.0, 1.0)
    return result.sort_values(
        [TIMESTAMP_COLUMN, "turbine_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_train_set(
    turbine_1_path: Path = DEFAULT_TURBINE_1_PATH,
    turbine_2_path: Path = DEFAULT_TURBINE_2_PATH,
) -> pd.DataFrame:
    """Load and combine both turbines into one clean hourly UTC data set."""
    turbine_1 = load_turbine_data(turbine_1_path, turbine_id=1)
    turbine_2 = load_turbine_data(turbine_2_path, turbine_id=2)
    combined = pd.concat([turbine_1, turbine_2], ignore_index=True)
    return resample_hourly(combined)


def save_parquet(data: pd.DataFrame, output_path: Path) -> None:
    """Atomically persist a prepared data set as parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    data.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    LOGGER.info("Saved %d rows to %s", len(data), output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbine-1", type=Path, default=DEFAULT_TURBINE_1_PATH)
    parser.add_argument("--turbine-2", type=Path, default=DEFAULT_TURBINE_2_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and save the hourly SCADA data set."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    try:
        data = build_train_set(args.turbine_1, args.turbine_2)
        save_parquet(data, args.output)
    except Exception:
        LOGGER.exception("Failed to build the SCADA training set")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
