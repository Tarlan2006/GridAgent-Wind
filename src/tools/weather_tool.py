"""Open-Meteo historical forecast client with a local parquet cache."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
API_URL: Final = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL_NAME: Final = "ecmwf_ifs"
DEFAULT_CACHE_DIR: Final = PROJECT_ROOT / "data" / "weather_cache"
TURBINE_CONFIG_PATH: Final = PROJECT_ROOT / "data" / "turbines.json"
WEATHER_VARIABLES: Final = (
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "temperature_2m",
)


def _load_turbine_coordinates() -> dict[int, tuple[float, float]]:
    """Load the two weather coordinates from the project data configuration."""
    try:
        config = json.loads(TURBINE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read turbine configuration: {TURBINE_CONFIG_PATH}"
        ) from exc

    coordinates: dict[int, tuple[float, float]] = {}
    for turbine_id in (1, 2):
        turbine = config.get(f"turbine_{turbine_id}")
        if not isinstance(turbine, dict):
            raise TypeError(f"Configuration is missing turbine_{turbine_id}")
        latitude = turbine.get("lat")
        longitude = turbine.get("lon")
        if not isinstance(latitude, (int, float)) or not isinstance(
            longitude,
            (int, float),
        ):
            raise TypeError(
                f"Configuration has invalid coordinates for turbine {turbine_id}"
            )
        coordinates[turbine_id] = (float(latitude), float(longitude))
    return coordinates


TURBINE_COORDINATES: Final = _load_turbine_coordinates()


class HistoricalWeatherService:
    """Fetch and cache hourly ECMWF IFS historical forecasts."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        timeout_seconds: float = 30.0,
        offline: bool = False,
        session: requests.Session | None = None,
        api_url: str = API_URL,
        model: str = MODEL_NAME,
        variables: tuple[str, ...] = WEATHER_VARIABLES,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.api_url = api_url
        self.model = model
        self.variables = variables
        self.timeout_seconds = timeout_seconds
        self.offline = offline
        self._session = session or requests.Session()

    def _cache_path(self, turbine_id: int) -> Path:
        suffix = "" if self.api_url == API_URL else "_previous_runs"
        return self.cache_dir / f"turbine_{turbine_id}_{self.model}{suffix}.parquet"

    @staticmethod
    def _parse_date(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)

    def _load_cache(self, turbine_id: int) -> pd.DataFrame:
        path = self._cache_path(turbine_id)
        if not path.is_file():
            return pd.DataFrame()
        try:
            cached = pd.read_parquet(path)
        except Exception:
            LOGGER.exception("Unable to read weather cache %s", path)
            raise

        required = {"timestamp", "turbine_id", *self.variables}
        missing = required.difference(cached.columns)
        if missing:
            raise ValueError(
                f"Weather cache {path} is missing columns: {sorted(missing)}"
            )
        cached["timestamp"] = pd.to_datetime(
            cached["timestamp"],
            errors="raise",
            utc=True,
        )
        return cached.sort_values("timestamp", kind="stable")

    @staticmethod
    def _requested_slice(
        data: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        if data.empty:
            return data
        start = pd.Timestamp(start_date, tz="UTC")
        end_exclusive = pd.Timestamp(end_date + timedelta(days=1), tz="UTC")
        mask = data["timestamp"].between(start, end_exclusive, inclusive="left")
        return data.loc[mask].copy()

    @staticmethod
    def _has_complete_coverage(
        data: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> bool:
        if data.empty:
            return False
        expected = pd.date_range(
            start=pd.Timestamp(start_date, tz="UTC"),
            end=pd.Timestamp(end_date + timedelta(days=1), tz="UTC"),
            freq="1h",
            inclusive="left",
        )
        available = pd.DatetimeIndex(data["timestamp"].drop_duplicates())
        return expected.isin(available).all()

    def _request_chunk(
        self,
        turbine_id: int,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        latitude, longitude = TURBINE_COORDINATES[turbine_id]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(self.variables),
            "models": self.model,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
        LOGGER.info(
            "Requesting Open-Meteo turbine=%d dates=%s..%s",
            turbine_id,
            start_date,
            end_date,
        )
        response = self._session.get(
            self.api_url,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Open-Meteo response must be a JSON object")
        return self._payload_to_frame(payload, turbine_id, self.variables)

    @staticmethod
    def _payload_to_frame(
        payload: dict[str, Any],
        turbine_id: int,
        variables: tuple[str, ...] = WEATHER_VARIABLES,
    ) -> pd.DataFrame:
        if payload.get("error"):
            raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")

        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise TypeError("Open-Meteo response has no hourly object")

        missing = {"time", *variables}.difference(hourly)
        if missing:
            raise ValueError(
                f"Open-Meteo response is missing fields: {sorted(missing)}"
            )

        lengths = {len(hourly[name]) for name in ("time", *variables)}
        if len(lengths) != 1:
            raise ValueError("Open-Meteo hourly arrays have different lengths")

        frame = pd.DataFrame(
            {"timestamp": pd.to_datetime(hourly["time"], errors="raise", utc=True)}
        )
        for variable in variables:
            frame[variable] = pd.to_numeric(hourly[variable], errors="coerce")
        frame["turbine_id"] = turbine_id
        return frame

    def _fetch_range(
        self,
        turbine_id: int,
        start_date: date,
        end_date: date,
        *,
        chunk_days: int = 31,
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=chunk_days - 1),
                end_date,
            )
            chunks.append(
                self._request_chunk(turbine_id, chunk_start, chunk_end)
            )
            chunk_start = chunk_end + timedelta(days=1)
        return pd.concat(chunks, ignore_index=True)

    def _save_cache(self, turbine_id: int, data: pd.DataFrame) -> None:
        path = self._cache_path(turbine_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp.parquet")
        data.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
        LOGGER.info("Saved %d cached weather rows to %s", len(data), path)

    def get_weather(
        self,
        turbine_id: int,
        start_date: str | date,
        end_date: str | date,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return a complete hourly UTC interval, using cache when possible."""
        if turbine_id not in TURBINE_COORDINATES:
            raise ValueError(f"Unknown turbine_id={turbine_id}")

        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if start > end:
            raise ValueError("start_date must not be after end_date")

        cached = self._load_cache(turbine_id)
        cached_slice = self._requested_slice(cached, start, end)
        if not force_refresh and self._has_complete_coverage(
            cached_slice,
            start,
            end,
        ):
            LOGGER.info("Weather cache hit for turbine %d", turbine_id)
            return cached_slice.reset_index(drop=True)

        if self.offline:
            raise RuntimeError(
                f"Offline cache does not cover turbine {turbine_id} "
                f"from {start} through {end}"
            )

        fetched = self._fetch_range(turbine_id, start, end)
        combined = pd.concat([cached, fetched], ignore_index=True)
        combined = (
            combined.sort_values("timestamp", kind="stable")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
        self._save_cache(turbine_id, combined)
        return self._requested_slice(combined, start, end).reset_index(drop=True)


OpenMeteoHistoricalWeatherClient = HistoricalWeatherService


# --- Forecast "as of" an issue time: only NWP runs published before it -------

PREVIOUS_RUNS_API_URL: Final = "https://previous-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_MODEL: Final = "ecmwf_ifs025"
AS_OF_BASE_VARIABLES: Final = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "temperature_2m",
)
MAX_LEAD_DAYS: Final = 3
RUN_PUBLICATION_DELAY: Final = pd.Timedelta(hours=7)


def _lagged_name(variable: str, lead_days: int) -> str:
    return variable if lead_days == 0 else f"{variable}_previous_day{lead_days}"


def previous_runs_service(*, offline: bool = False) -> HistoricalWeatherService:
    """Weather service over forecasts that were published before issue time."""
    return HistoricalWeatherService(
        offline=offline,
        api_url=PREVIOUS_RUNS_API_URL,
        model=PREVIOUS_RUNS_MODEL,
        variables=tuple(
            _lagged_name(variable, lead_days)
            for lead_days in range(MAX_LEAD_DAYS + 1)
            for variable in AS_OF_BASE_VARIABLES
        ),
    )


def get_forecast_as_of(
    turbine_id: int,
    issue_time: pd.Timestamp | str,
    horizon_hours: int = 48,
    *,
    offline: bool = False,
    service: HistoricalWeatherService | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return weather from NWP runs available at ``issue_time`` only."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")

    issued = pd.Timestamp(issue_time)
    issued = issued.tz_localize("UTC") if issued.tzinfo is None else issued.tz_convert("UTC")
    hours = pd.date_range(issued + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h")
    weather_service = service or previous_runs_service(offline=offline)
    raw = weather_service.get_weather(
        turbine_id,
        hours[0].date(),
        hours[-1].date(),
        force_refresh=force_refresh,
    )
    raw = raw.set_index("timestamp").reindex(hours)

    latest_run_day = (issued - RUN_PUBLICATION_DELAY).normalize()
    lead_days = ((hours.normalize() - latest_run_day).days).to_numpy()
    if lead_days.max() > MAX_LEAD_DAYS:
        raise ValueError(f"Horizon needs forecasts older than {MAX_LEAD_DAYS} days")

    frame = pd.DataFrame({"timestamp": hours, "turbine_id": turbine_id})
    for variable in AS_OF_BASE_VARIABLES:
        values = [
            raw[_lagged_name(variable, int(lead_day))].iloc[index]
            for index, lead_day in enumerate(lead_days)
        ]
        frame[variable] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()

    shear = np.log(80 / 10) / np.log(100 / 10)
    frame["wind_speed_80m"] = frame["wind_speed_10m"] + shear * (
        frame["wind_speed_100m"] - frame["wind_speed_10m"]
    )
    frame["lead_days"] = lead_days
    if frame[list(AS_OF_BASE_VARIABLES)].isna().any().any():
        raise RuntimeError(
            f"Incomplete as-of forecast for turbine {turbine_id} issued {issued}"
        )
    return frame
