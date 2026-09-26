"""Single source of truth for a forecast: context -> weather -> features -> model -> validation.

A forecast issued on day D covers target hours starting at D+1 00:00 UTC,
e.g. issue 2026-01-31 -> 2026-02-01 00:00 .. 2026-02-02 23:00 for 48h.
All numbers stay in Python; the LLM only receives ``ForecastResult.summary()``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

try:
    from src.tools.feature_tool import (
        MODEL_FEATURE_COLUMNS,
        RAW_WEATHER_COLUMNS,
        add_physical_features,
    )
    from src.tools.inference_tool import (
        PROJECT_ROOT,
        SHUTDOWN_WIND_SPEED_MPS,
        WindPowerPredictor,
    )
    from src.tools.weather_tool import (
        PREVIOUS_RUNS_MODEL,
        HistoricalWeatherService,
        get_forecast_as_of,
    )
except ModuleNotFoundError:
    from tools.feature_tool import (
        MODEL_FEATURE_COLUMNS,
        RAW_WEATHER_COLUMNS,
        add_physical_features,
    )
    from tools.inference_tool import (
        PROJECT_ROOT,
        SHUTDOWN_WIND_SPEED_MPS,
        WindPowerPredictor,
    )
    from tools.weather_tool import (
        PREVIOUS_RUNS_MODEL,
        HistoricalWeatherService,
        get_forecast_as_of,
    )

LOGGER = logging.getLogger(__name__)

TURBINE_IDS: Final = (1, 2)
ALLOWED_HORIZONS: Final = (24, 48)
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "outputs" / "forecasts"
ISSUE_HOUR_UTC: Final = 12

# Validation thresholds (assumptions: no turbine passport available).
LOW_WIND_MPS: Final = 3.0
LOW_WIND_MAX_POWER: Final = 0.2
FLAT_FORECAST_STD: Final = 0.01
ICING_TEMPERATURE_C: Final = 0.0
ICING_HUMIDITY_PCT: Final = 90.0
DIVERGENCE_MAE_THRESHOLD: Final = 0.15


@dataclass
class ForecastResult:
    """Everything one cycle produced; ``steps`` is the visible agent activity."""

    issue_date: date
    horizon_hours: int
    turbine_ids: tuple[int, ...]
    status: str = "ok"  # ok | warn | error
    steps: list[dict[str, Any]] = field(default_factory=list)
    forecast: pd.DataFrame | None = None
    weather_hash: str | None = None
    model_info: dict[str, Any] | None = None
    turbines: dict[str, dict[str, Any]] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    changes: dict[str, Any] | None = None

    @property
    def key(self) -> tuple[date, int, tuple[int, ...]]:
        return self.issue_date, self.horizon_hours, self.turbine_ids

    def summary(self) -> dict[str, Any]:
        """Compact JSON-safe digest for the LLM: no hourly rows."""
        window = target_window(self.issue_date, self.horizon_hours)
        return {
            "issue_date": self.issue_date.isoformat(),
            "forecast_window_utc": f"{window[0].isoformat()} .. {window[-1].isoformat()}",
            "horizon_hours": self.horizon_hours,
            "turbine_ids": list(self.turbine_ids),
            "status": self.status,
            "power_unit": "нормализованная мощность 0..1 от номинала, по каждой турбине отдельно",
            "turbines": self.turbines,
            "checks": [c for c in self.checks if c["status"] != "ok"],
            "changes": self.changes,
            "model": self.model_info,
            "steps": [{k: s[k] for k in ("stage", "status", "detail")} for s in self.steps],
        }


def target_window(issue_date: date, horizon_hours: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(issue_date + timedelta(days=1), tz="UTC")
    return pd.date_range(start, periods=horizon_hours, freq="1h")


def weather_hash(weather: pd.DataFrame) -> str:
    """Stable fingerprint of the model inputs, used to detect weather updates."""
    ordered = weather.sort_values(["turbine_id", "timestamp"], kind="stable")
    columns = ["timestamp", "turbine_id", *RAW_WEATHER_COLUMNS]
    hashed = pd.util.hash_pandas_object(ordered.loc[:, columns], index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()[:12]


def compare_forecasts(new: pd.DataFrame, old: pd.DataFrame) -> dict[str, Any] | None:
    """Difference on overlapping (target_time, turbine_id) hours."""
    merged = new.merge(old, on=["target_time", "turbine_id"], suffixes=("", "_old"))
    if merged.empty:
        return None
    diff = (merged["predicted_power"] - merged["predicted_power_old"]).abs()
    worst = merged.loc[diff.idxmax()]
    return {
        "hours_compared": len(merged),
        "mean_abs_diff": round(float(diff.mean()), 3),
        "max_abs_diff": round(float(diff.max()), 3),
        "max_diff_time_utc": worst["target_time"].isoformat(),
        "max_diff_turbine": int(worst["turbine_id"]),
    }


def _hours(frame: pd.DataFrame, mask: pd.Series, limit: int = 5) -> list[str]:
    return [t.strftime("%m-%d %H:%M") for t in frame.loc[mask, "target_time"].head(limit)]


def summarize_turbines(forecast: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-turbine statistics; turbines are never summed together."""
    turbines = {}
    for turbine_id, group in forecast.groupby("turbine_id"):
        peak = group.loc[group["predicted_power"].idxmax()]
        turbines[f"turbine_{turbine_id}"] = {
            "mean_power": round(float(group["predicted_power"].mean()), 3),
            "peak_power": round(float(peak["predicted_power"]), 3),
            "peak_time_utc": peak["target_time"].isoformat(),
            "mean_wind_100m_ms": round(float(group["wind_speed_100m"].mean()), 2),
            "min_temperature_c": round(float(group["temperature_2m"].min()), 1),
        }
    return turbines


def validate_forecast(forecast: pd.DataFrame) -> list[dict[str, Any]]:
    """Rule-based sanity checks; the LLM only explains them."""
    checks = []

    shutdown = forecast["wind_speed_100m"] > SHUTDOWN_WIND_SPEED_MPS
    checks.append({
        "name": "shutdown_wind",
        "status": "warn" if shutdown.any() else "ok",
        "detail": (
            f"часов с ветром > {SHUTDOWN_WIND_SPEED_MPS} м/с: {int(shutdown.sum())} "
            f"{_hours(forecast, shutdown)} — мощность обнулена правилом отключения турбины"
            if shutdown.any() else f"ветер не превышает {SHUTDOWN_WIND_SPEED_MPS} м/с"
        ),
    })

    low_wind = (forecast["wind_speed_100m"] < LOW_WIND_MPS) & (forecast["predicted_power"] > LOW_WIND_MAX_POWER)
    checks.append({
        "name": "low_wind_high_power",
        "status": "warn" if low_wind.any() else "ok",
        "detail": f"часов с ветром < {LOW_WIND_MPS} м/с и мощностью > {LOW_WIND_MAX_POWER}: "
                  f"{int(low_wind.sum())} {_hours(forecast, low_wind)}",
    })

    flat = [int(t) for t, g in forecast.groupby("turbine_id") if g["predicted_power"].std() < FLAT_FORECAST_STD]
    checks.append({
        "name": "flat_forecast",
        "status": "warn" if flat else "ok",
        "detail": f"почти постоянный прогноз у турбин {flat}" if flat else "прогноз меняется во времени",
    })

    cold = forecast["temperature_2m"] <= ICING_TEMPERATURE_C
    if "relative_humidity_2m" in forecast.columns:
        icing = cold & (forecast["relative_humidity_2m"] >= ICING_HUMIDITY_PCT)
        checks.append({
            "name": "icing_risk",
            "status": "warn" if icing.any() else "ok",
            "detail": f"часов с t ≤ 0 °C и влажностью ≥ {ICING_HUMIDITY_PCT}%: "
                      f"{int(icing.sum())} {_hours(forecast, icing)}",
        })
    else:
        checks.append({
            "name": "icing_risk",
            "status": "info" if cold.any() else "ok",
            "detail": f"часов с t ≤ 0 °C: {int(cold.sum())}; влажность не запрашивается, "
                      "риск обледенения оценён только по температуре",
        })
    return checks


class ForecastAgent:
    """Runs the forecast cycle; shared by the UI button and the LLM tools."""

    def __init__(
        self,
        weather_service: HistoricalWeatherService | None = None,
        predictor: WindPowerPredictor | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.weather_service = weather_service
        self.predictor = predictor or WindPowerPredictor()
        self.output_dir = output_dir
        self.results: dict[tuple[date, int, tuple[int, ...]], ForecastResult] = {}
        self.last_result: ForecastResult | None = None

    @staticmethod
    def _step(result: ForecastResult, stage: str, action: Callable[[], tuple[Any, str]]) -> Any:
        """Run one stage and record it; an error stops the cycle loudly."""
        started = time.perf_counter()
        try:
            value, detail = action()
        except Exception as error:
            LOGGER.exception("forecast stage failed stage=%s", stage)
            result.steps.append({
                "stage": stage, "status": "error", "detail": f"{type(error).__name__}: {error}",
                "ms": round((time.perf_counter() - started) * 1000),
            })
            raise
        result.steps.append({
            "stage": stage, "status": "ok", "detail": detail,
            "ms": round((time.perf_counter() - started) * 1000),
        })
        LOGGER.info("forecast stage=%s detail=%s", stage, detail)
        return value

    def _context(self, result: ForecastResult) -> tuple[dict[str, Any], str]:
        window = target_window(result.issue_date, result.horizon_hours)
        metadata = self.predictor.metadata()  # fails fast if the model artifact is missing
        info = {"objective": metadata["objective"], "validation_metrics": metadata["validation_metrics"]}
        return info, (
            f"выпуск {result.issue_date}, окно {window[0]:%m-%d %H:%M}..{window[-1]:%m-%d %H:%M} UTC, "
            f"турбины {list(result.turbine_ids)}, модель {Path(metadata['model_path']).name}"
        )

    def _fetch_weather(
        self, result: ForecastResult, *, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, str]:
        window = target_window(result.issue_date, result.horizon_hours)
        issue_time = pd.Timestamp(result.issue_date, tz="UTC") + pd.Timedelta(hours=ISSUE_HOUR_UTC)
        fetch_hours = int((window[-1] - issue_time) / pd.Timedelta(hours=1))
        frames = []
        for turbine_id in result.turbine_ids:
            try:
                weather = get_forecast_as_of(
                    turbine_id,
                    issue_time,
                    fetch_hours,
                    offline=False,
                    service=self.weather_service,
                    force_refresh=force_refresh,
                )
            except (requests.RequestException, OSError) as error:
                started = time.perf_counter()
                weather = get_forecast_as_of(
                    turbine_id,
                    issue_time,
                    fetch_hours,
                    offline=True,
                    service=self.weather_service,
                )
                result.steps.append({
                    "stage": "weather_fallback_cache",
                    "status": "ok",
                    "detail": f"{type(error).__name__}: использован локальный кэш",
                    "ms": round((time.perf_counter() - started) * 1000),
                })
            weather = weather.loc[weather["timestamp"].isin(window)]
            complete = weather.dropna(subset=list(RAW_WEATHER_COLUMNS))
            if len(complete) != len(window):
                raise ValueError(
                    f"неполная погода для турбины {turbine_id}: {len(complete)}/{len(window)} часов"
                )
            frames.append(weather)
        weather = pd.concat(frames, ignore_index=True).sort_values(["turbine_id", "timestamp"], kind="stable")
        weather = weather.reset_index(drop=True)
        return weather, (
            f"{PREVIOUS_RUNS_MODEL}: {len(weather)} часовых строк без пропусков, "
            f"хэш {weather_hash(weather)}"
        )

    @staticmethod
    def _features(weather: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        featured = add_physical_features(weather)
        numeric = [c for c in MODEL_FEATURE_COLUMNS if c != "turbine_id"]
        missing = featured[numeric].isna().any()
        if missing.any():
            raise ValueError(f"признаки с пропусками: {missing[missing].index.tolist()}")
        return featured, f"{len(MODEL_FEATURE_COLUMNS)} признаков × {len(featured)} строк"

    def _model(self, result: ForecastResult, featured: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        predictions = self.predictor.predict(featured.loc[:, ["timestamp", "turbine_id", *RAW_WEATHER_COLUMNS]])
        predictions["turbine_id"] = predictions["turbine_id"].astype(int)
        joined = featured.merge(predictions, on=["timestamp", "turbine_id"], how="inner", validate="one_to_one")
        if len(joined) != len(featured):
            raise ValueError(f"предсказаний {len(joined)} на {len(featured)} строк погоды")

        first_target = target_window(result.issue_date, result.horizon_hours)[0]
        forecast = joined.rename(columns={"timestamp": "target_time"})
        forecast.insert(0, "issue_date", result.issue_date.isoformat())
        forecast["lead_hours"] = ((forecast["target_time"] - first_target) / pd.Timedelta(hours=1)).astype(int) + 1
        columns = ["issue_date", "target_time", "lead_hours", "turbine_id", "predicted_power",
                   "wind_speed_100m", "temperature_2m"]
        if "relative_humidity_2m" in forecast.columns:
            columns.append("relative_humidity_2m")
        forecast = forecast.loc[:, columns]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        turbines = "".join(str(t) for t in result.turbine_ids)
        path = self.output_dir / f"forecast_{result.issue_date.isoformat()}_t{turbines}_{result.horizon_hours}h.csv"
        forecast.to_csv(path, index=False)
        return forecast, f"{len(forecast)} почасовых прогнозов → {path.name}"

    def _validation(self, result: ForecastResult) -> tuple[None, str]:
        result.turbines = summarize_turbines(result.forecast)
        result.checks = validate_forecast(result.forecast)
        result.changes = self._compare_previous_issue(result)
        flagged = [c["name"] for c in result.checks if c["status"] == "warn"]
        return None, f"проверок {len(result.checks)}, предупреждений {len(flagged)} {flagged or ''}".strip()

    def _previous_issue_forecast(self, issue_date: date) -> pd.DataFrame | None:
        """All forecasts issued the day before (any turbines/horizon), memory first, then CSV."""
        previous_date = issue_date - timedelta(days=1)
        frames = [
            previous.forecast for key, previous in self.results.items()
            if key[0] == previous_date and previous.forecast is not None
        ]
        if not frames:
            for path in sorted(self.output_dir.glob(f"forecast_{previous_date.isoformat()}_*.csv")):
                frame = pd.read_csv(path)
                frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
                frames.append(frame)
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        return combined.drop_duplicates(["target_time", "turbine_id"], keep="last")

    def _compare_previous_issue(self, result: ForecastResult) -> dict[str, Any] | None:
        previous = self._previous_issue_forecast(result.issue_date)
        changes = compare_forecasts(result.forecast, previous) if previous is not None else None
        if changes is None:
            return None
        changes["against"] = f"прогноз от {result.issue_date - timedelta(days=1)}"
        diverged = changes["mean_abs_diff"] > DIVERGENCE_MAE_THRESHOLD
        result.checks.append({
            "name": "divergence_from_previous_issue",
            "status": "warn" if diverged else "ok",
            "detail": f"средн. |Δ| {changes['mean_abs_diff']} на {changes['hours_compared']} общих часах "
                      f"(порог {DIVERGENCE_MAE_THRESHOLD})",
        })
        return changes

    def run(
        self,
        issue_date: date,
        horizon_hours: int = 48,
        turbine_ids: tuple[int, ...] = TURBINE_IDS,
        *,
        weather: pd.DataFrame | None = None,
    ) -> ForecastResult:
        """Full cycle; never raises on stage failures, they land in ``steps``."""
        if horizon_hours not in ALLOWED_HORIZONS:
            raise ValueError(f"horizon_hours must be one of {ALLOWED_HORIZONS}")
        turbine_ids = tuple(sorted(set(turbine_ids)))
        if not turbine_ids or not set(turbine_ids) <= set(TURBINE_IDS):
            raise ValueError(f"turbine_ids must be a subset of {TURBINE_IDS}")

        result = ForecastResult(issue_date=issue_date, horizon_hours=horizon_hours, turbine_ids=turbine_ids)
        try:
            result.model_info = self._step(result, "context", lambda: self._context(result))
            if weather is None:
                weather = self._step(result, "weather", lambda: self._fetch_weather(result))
            result.weather_hash = weather_hash(weather)
            featured = self._step(result, "features", lambda: self._features(weather))
            result.forecast = self._step(result, "model", lambda: self._model(result, featured))
            self._step(result, "validation", lambda: self._validation(result))
        except Exception:  # noqa: BLE001 - failures are recorded by _step for the UI
            result.status = "error"
        else:
            if any(check["status"] == "warn" for check in result.checks):
                result.status = "warn"

        self.results[result.key] = result
        self.last_result = result
        return result

    def check_update(
        self,
        issue_date: date,
        horizon_hours: int = 48,
        turbine_ids: tuple[int, ...] = TURBINE_IDS,
    ) -> ForecastResult:
        """Re-download weather; recompute only if the model inputs changed."""
        turbine_ids = tuple(sorted(set(turbine_ids)))
        previous = self.results.get((issue_date, horizon_hours, turbine_ids))
        if previous is None or previous.status == "error":
            return self.run(issue_date, horizon_hours, turbine_ids)

        probe = ForecastResult(issue_date=issue_date, horizon_hours=horizon_hours, turbine_ids=turbine_ids)
        try:
            fresh = self._step(probe, "weather_update", lambda: self._fetch_weather(probe, force_refresh=True))
        except Exception:  # noqa: BLE001 - retain the last valid forecast on update failure
            previous.steps.extend(probe.steps)
            self.last_result = previous
            return previous

        fresh_hash = weather_hash(fresh)
        if fresh_hash == previous.weather_hash:
            probe.steps[-1]["detail"] = f"хэш {fresh_hash} не изменился — пересчёт не нужен"
            previous.steps.extend(probe.steps)
            self.last_result = previous
            return previous

        probe.steps[-1]["detail"] = f"погода обновилась: {previous.weather_hash} → {fresh_hash}, пересчёт"
        rerun = self.run(issue_date, horizon_hours, turbine_ids, weather=fresh)
        rerun.steps = probe.steps + rerun.steps
        if rerun.forecast is not None and previous.forecast is not None:
            rerun.changes = {
                "vs_previous_run": compare_forecasts(rerun.forecast, previous.forecast),
                "vs_previous_issue": rerun.changes,
            }
        return rerun


STATUS_ICONS: Final = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "ℹ️"}


def activity_markdown(result: ForecastResult) -> str:
    """Render real pipeline stages and validation checks for the UI."""
    lines = [
        (
            f"**Выпуск {result.issue_date} · {result.horizon_hours} ч · турбины {list(result.turbine_ids)} · "
            f"{STATUS_ICONS[result.status]} {result.status}**"
        ),
        "",
        "| # | Этап | | Детали | мс |",
        "|---|---|---|---|---|",
    ]
    for index, step in enumerate(result.steps, start=1):
        detail = str(step["detail"]).replace("|", "/")
        lines.append(f"| {index} | {step['stage']} | {STATUS_ICONS[step['status']]} | {detail} | {step['ms']} |")
    if result.checks:
        lines += ["", "**Проверки**", ""]
        lines += [f"- {STATUS_ICONS[c['status']]} `{c['name']}` — {c['detail']}" for c in result.checks]
    if result.changes:
        lines += ["", f"**Изменения:** `{result.changes}`"]
    return "\n".join(lines)


def plot_frame(result: ForecastResult | None) -> pd.DataFrame:
    """Long-format frame for gr.LinePlot, one line per turbine."""
    if result is None or result.forecast is None:
        return pd.DataFrame({"target_time": pd.Series(dtype="datetime64[ns]"),
                             "predicted_power": pd.Series(dtype=float), "turbine": pd.Series(dtype=str)})
    frame = result.forecast.loc[:, ["target_time", "turbine_id", "predicted_power"]].copy()
    frame["target_time"] = frame["target_time"].dt.tz_localize(None)
    frame["turbine"] = "Турбина " + frame["turbine_id"].astype(str)
    return frame.drop(columns="turbine_id")
