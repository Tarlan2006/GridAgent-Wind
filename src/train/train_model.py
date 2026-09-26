"""Train the deterministic LightGBM wind-power model."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools.feature_tool import (
    MODEL_FEATURE_COLUMNS,
    add_physical_features,
    select_model_features,
)
from src.tools.weather_tool import (
    DEFAULT_CACHE_DIR,
    HistoricalWeatherService,
)
from src.train.build_train_set import (
    DEFAULT_TURBINE_1_PATH,
    DEFAULT_TURBINE_2_PATH,
    build_train_set,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final = _PROJECT_ROOT
DEFAULT_MODEL_PATH: Final = PROJECT_ROOT / "models" / "wind_power_lgb.pkl"
TRAIN_END_EXCLUSIVE: Final = pd.Timestamp("2026-01-01", tz="UTC")
VALIDATION_END_EXCLUSIVE: Final = pd.Timestamp("2026-02-01", tz="UTC")
TARGET_COLUMN: Final = "power"
RANDOM_SEED: Final = 42


def _load_lightgbm() -> type[Any]:
    """Import LightGBM with a clear dependency error."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required; install dependencies from requirements.txt"
        ) from exc
    return LGBMRegressor


def _make_estimator(objective: str) -> Any:
    """Create a deterministic LightGBM regressor."""
    estimator_type = _load_lightgbm()
    return estimator_type(
        objective=objective,
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )


def load_weather_for_scada(
    scada: pd.DataFrame,
    service: HistoricalWeatherService,
) -> pd.DataFrame:
    """Load cached/API weather for the exact SCADA date range."""
    start_date = scada["timestamp"].min().date()
    end_date = scada["timestamp"].max().date()
    frames = [
        service.get_weather(turbine_id, start_date, end_date)
        for turbine_id in (1, 2)
    ]
    return pd.concat(frames, ignore_index=True)


def assemble_training_frame(
    scada: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge hourly SCADA targets with weather and create model features."""
    merged = scada.merge(
        weather,
        on=["timestamp", "turbine_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("SCADA and weather data have no matching UTC hours")

    featured = add_physical_features(merged)
    required = [*MODEL_FEATURE_COLUMNS, TARGET_COLUMN, "timestamp"]
    before_drop = len(featured)
    featured = featured.dropna(subset=required).copy()
    dropped = before_drop - len(featured)
    if dropped:
        LOGGER.warning("Dropped %d rows with incomplete training features", dropped)

    featured[TARGET_COLUMN] = featured[TARGET_COLUMN].clip(0.0, 1.0)
    featured = featured.loc[
        featured["timestamp"] < VALIDATION_END_EXCLUSIVE
    ].copy()
    return featured.sort_values(
        ["timestamp", "turbine_id"],
        kind="stable",
    ).reset_index(drop=True)


def split_fixed_holdout(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pre-2026 training data from the January 2026 holdout."""
    train = frame.loc[frame["timestamp"] < TRAIN_END_EXCLUSIVE].copy()
    validation = frame.loc[
        (frame["timestamp"] >= TRAIN_END_EXCLUSIVE)
        & (frame["timestamp"] < VALIDATION_END_EXCLUSIVE)
    ].copy()
    if train.empty:
        raise ValueError("Training split before 2026-01-01 is empty")
    if validation.empty:
        raise ValueError("January 2026 validation split is empty")
    return train, validation


def _model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the stable LightGBM matrix with a categorical turbine ID."""
    matrix = select_model_features(frame)
    matrix["turbine_id"] = matrix["turbine_id"].astype("category")
    return matrix


def _metrics(target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(target, clipped))),
        "mae": float(mean_absolute_error(target, clipped)),
    }


def cross_validate(
    train: pd.DataFrame,
    *,
    objective: str,
    n_splits: int = 5,
) -> list[dict[str, float]]:
    """Run leakage-safe CV over unique timestamps using TimeSeriesSplit."""
    unique_timestamps = pd.DatetimeIndex(train["timestamp"].unique()).sort_values()
    if len(unique_timestamps) <= n_splits:
        raise ValueError("Not enough unique timestamps for TimeSeriesSplit")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(unique_timestamps),
        start=1,
    ):
        fold_train_times = unique_timestamps[train_indices]
        fold_valid_times = unique_timestamps[valid_indices]
        fold_train = train.loc[train["timestamp"].isin(fold_train_times)]
        fold_valid = train.loc[train["timestamp"].isin(fold_valid_times)]

        model = _make_estimator(objective)
        model.fit(_model_matrix(fold_train), fold_train[TARGET_COLUMN])
        metrics = _metrics(
            fold_valid[TARGET_COLUMN],
            model.predict(_model_matrix(fold_valid)),
        )
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)
        LOGGER.info(
            "TimeSeriesSplit fold=%d RMSE=%.6f MAE=%.6f",
            fold,
            metrics["rmse"],
            metrics["mae"],
        )
    return fold_metrics


def train_and_evaluate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    objective: str,
) -> tuple[Any, dict[str, float], list[dict[str, float]]]:
    """Cross-validate, fit through December 2025, and score January 2026."""
    cv_metrics = cross_validate(train, objective=objective)
    model = _make_estimator(objective)
    model.fit(_model_matrix(train), train[TARGET_COLUMN])
    validation_metrics = _metrics(
        validation[TARGET_COLUMN],
        model.predict(_model_matrix(validation)),
    )
    LOGGER.info(
        "January 2026 holdout RMSE=%.6f MAE=%.6f",
        validation_metrics["rmse"],
        validation_metrics["mae"],
    )
    return model, validation_metrics, cv_metrics


def save_model_artifact(
    model: Any,
    output_path: Path,
    *,
    objective: str,
    validation_metrics: dict[str, float],
    cv_metrics: list[dict[str, float]],
) -> None:
    """Atomically save the model together with its feature contract."""
    artifact = {
        "model": model,
        "feature_names": MODEL_FEATURE_COLUMNS.copy(),
        "categorical_features": ["turbine_id"],
        "target_name": TARGET_COLUMN,
        "objective": objective,
        "train_end_exclusive": TRAIN_END_EXCLUSIVE.isoformat(),
        "validation_end_exclusive": VALIDATION_END_EXCLUSIVE.isoformat(),
        "validation_metrics": validation_metrics,
        "cv_metrics": cv_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.pkl")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(output_path)
    LOGGER.info("Saved LightGBM artifact to %s", output_path)


def run_training(
    *,
    turbine_1_path: Path,
    turbine_2_path: Path,
    cache_dir: Path,
    output_path: Path,
    objective: str,
    offline: bool,
) -> dict[str, float]:
    """Execute the complete deterministic training workflow."""
    scada = build_train_set(turbine_1_path, turbine_2_path)
    weather_service = HistoricalWeatherService(cache_dir, offline=offline)
    weather = load_weather_for_scada(scada, weather_service)
    frame = assemble_training_frame(scada, weather)
    train, validation = split_fixed_holdout(frame)
    LOGGER.info(
        "Training rows=%d validation rows=%d features=%d",
        len(train),
        len(validation),
        len(MODEL_FEATURE_COLUMNS),
    )
    model, validation_metrics, cv_metrics = train_and_evaluate(
        train,
        validation,
        objective=objective,
    )
    save_model_artifact(
        model,
        output_path,
        objective=objective,
        validation_metrics=validation_metrics,
        cv_metrics=cv_metrics,
    )
    return validation_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbine-1", type=Path, default=DEFAULT_TURBINE_1_PATH)
    parser.add_argument("--turbine-2", type=Path, default=DEFAULT_TURBINE_2_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--objective",
        choices=("regression_l1", "huber"),
        default="huber",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require weather data to be available in the parquet cache",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    try:
        metrics = run_training(
            turbine_1_path=args.turbine_1,
            turbine_2_path=args.turbine_2,
            cache_dir=args.cache_dir,
            output_path=args.output,
            objective=args.objective,
            offline=args.offline,
        )
    except Exception:
        LOGGER.exception("Wind-power training failed")
        return 1
    LOGGER.info("Training complete: %s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
