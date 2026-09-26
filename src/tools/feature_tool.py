"""Physical and calendar feature engineering for wind forecasts."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN: Final = "timestamp"
REQUIRED_WEATHER_COLUMNS: Final = {
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
}
RAW_WEATHER_COLUMNS: Final = [
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "temperature_2m",
]
ENGINEERED_FEATURE_COLUMNS: Final = [
    "delta_v",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_u",
    "wind_v",
    "v3",
    "hour_sin",
    "hour_cos",
    "day_of_year",
]
MODEL_FEATURE_COLUMNS: Final = [
    *RAW_WEATHER_COLUMNS,
    *ENGINEERED_FEATURE_COLUMNS,
    "turbine_id",
]


def _utc_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Return validated UTC timestamps from a column or DatetimeIndex."""
    if TIMESTAMP_COLUMN in frame.columns:
        timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.Series(
            pd.to_datetime(frame.index, errors="coerce", utc=True),
            index=frame.index,
            name=TIMESTAMP_COLUMN,
        )
    else:
        raise ValueError("Weather data requires a timestamp column or DatetimeIndex")

    if timestamps.isna().any():
        count = int(timestamps.isna().sum())
        raise ValueError(f"Weather data contains {count} invalid timestamps")
    return timestamps


def add_physical_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic wind physics and UTC calendar features."""
    missing = REQUIRED_WEATHER_COLUMNS.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather data is missing columns: {sorted(missing)}")

    featured = weather.copy()
    timestamps = _utc_timestamps(featured)
    featured[TIMESTAMP_COLUMN] = timestamps.to_numpy()

    numeric_columns = set(RAW_WEATHER_COLUMNS).intersection(featured.columns)
    for column in numeric_columns:
        featured[column] = pd.to_numeric(featured[column], errors="coerce")

    direction_radians = np.deg2rad(featured["wind_direction_100m"] % 360.0)
    wind_speed_100m = featured["wind_speed_100m"]

    featured["delta_v"] = wind_speed_100m - featured["wind_speed_10m"]
    featured["wind_direction_sin"] = np.sin(direction_radians)
    featured["wind_direction_cos"] = np.cos(direction_radians)
    featured["wind_u"] = -wind_speed_100m * featured["wind_direction_sin"]
    featured["wind_v"] = -wind_speed_100m * featured["wind_direction_cos"]
    featured["v3"] = wind_speed_100m.pow(3)

    utc_index = pd.DatetimeIndex(featured[TIMESTAMP_COLUMN])
    hour = utc_index.hour.astype(float)
    featured["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    featured["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    featured["day_of_year"] = utc_index.dayofyear.astype("int16")
    return featured


def select_model_features(featured: pd.DataFrame) -> pd.DataFrame:
    """Return model features in a stable, validated order."""
    missing = set(MODEL_FEATURE_COLUMNS).difference(featured.columns)
    if missing:
        raise ValueError(f"Feature frame is missing columns: {sorted(missing)}")
    return featured.loc[:, MODEL_FEATURE_COLUMNS].copy()
