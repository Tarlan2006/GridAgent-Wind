"""Train and benchmark local wind-power regression models.

The script combines measurements from two turbines, aggregates the original
10-minute observations to hourly values, creates leakage-safe weather lags,
benchmarks LightGBM, Random Forest, and CatBoost, then serializes the best
pipeline.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypeAlias

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_TURBINE_1_PATH: Final = PROJECT_ROOT / "data" / "turbine_1.csv"
DEFAULT_TURBINE_2_PATH: Final = PROJECT_ROOT / "data" / "turbine_22.csv"
DEFAULT_MODEL_PATH: Final = Path(__file__).with_name("wind_model.pkl")

TRAIN_END_EXCLUSIVE: Final = pd.Timestamp("2026-01-01")
VALIDATION_END_EXCLUSIVE: Final = pd.Timestamp("2026-02-01")
TARGET_COLUMN: Final = "power"
SIGNAL_COLUMNS: Final = ["wind_speed", "temperature", TARGET_COLUMN]
FEATURE_COLUMNS: Final = [
    "wind_speed",
    "temperature",
    "hour_sin",
    "hour_cos",
    "day_of_year",
    "wind_speed_lag_1h",
    "temperature_lag_1h",
    "wind_speed_lag_2h",
    "temperature_lag_2h",
    "turbine_id",
]
ModelMetrics: TypeAlias = dict[str, dict[str, float]]

# Keys are normalized by ``_column_key`` so harmless differences in spaces,
# letter case, degree symbols, and punctuation do not break ingestion.
COLUMN_ALIASES: Final = {
    "id": "source_id",
    "timestamp": "timestamp",
    "статистическоевремя": "timestamp",
    "windspeed": "wind_speed",
    "средняяскоростьветраms": "wind_speed",
    "power": TARGET_COLUMN,
    "normalizedactivepower": TARGET_COLUMN,
    "нормализованнаяактивнаямощность": TARGET_COLUMN,
    "temperature": "temperature",
    "средняятемператураокружающейсредыc": "temperature",
}


def _column_key(value: object) -> str:
    """Return a stable key for Russian or English source column names."""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    normalized = normalized.replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", normalized)


def _read_source(path: Path) -> pd.DataFrame:
    """Read a supported tabular source without altering its contents."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Unsupported dataset format {suffix!r}: {path}")


def load_turbine_data(path: Path, turbine_id: int) -> pd.DataFrame:
    """Load, standardize, and validate one turbine dataset."""
    raw = _read_source(path)
    rename_map: dict[object, str] = {}

    for column in raw.columns:
        alias = COLUMN_ALIASES.get(_column_key(column))
        if alias is not None:
            rename_map[column] = alias

    standardized = raw.rename(columns=rename_map)
    required = {"timestamp", "wind_speed", "temperature", TARGET_COLUMN}
    missing = required.difference(standardized.columns)
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(raw.columns)}"
        )

    duplicated = standardized.columns[standardized.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"Dataset {path} contains ambiguous mapped columns: {duplicated}"
        )

    data = standardized.loc[:, ["timestamp", *SIGNAL_COLUMNS]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for column in SIGNAL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_mask = data[["timestamp", *SIGNAL_COLUMNS]].isna().any(axis=1)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        LOGGER.warning(
            "Dropping %d invalid rows from turbine %d (%s)",
            invalid_count,
            turbine_id,
            path,
        )
        data = data.loc[~invalid_mask].copy()

    if data.empty:
        raise ValueError(f"Dataset contains no valid rows: {path}")

    invalid_target = ~data[TARGET_COLUMN].between(0.0, 1.0, inclusive="both")
    if invalid_target.any():
        count = int(invalid_target.sum())
        minimum = float(data.loc[invalid_target, TARGET_COLUMN].min())
        maximum = float(data.loc[invalid_target, TARGET_COLUMN].max())
        raise ValueError(
            f"Dataset {path} has {count} power values outside [0, 1] "
            f"(min={minimum:.4f}, max={maximum:.4f})"
        )

    data["turbine_id"] = turbine_id
    data = data.sort_values("timestamp", kind="stable")

    duplicate_count = int(data["timestamp"].duplicated().sum())
    if duplicate_count:
        LOGGER.warning(
            "Turbine %d contains %d duplicate timestamps; hourly resampling "
            "will average them",
            turbine_id,
            duplicate_count,
        )

    LOGGER.info(
        "Loaded turbine %d: %d rows from %s to %s",
        turbine_id,
        len(data),
        data["timestamp"].min(),
        data["timestamp"].max(),
    )
    return data


def resample_hourly(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signals hourly, keeping turbine time series isolated."""
    hourly = (
        data.set_index("timestamp")
        .groupby("turbine_id")
        .resample("1h")[SIGNAL_COLUMNS]
        .mean()
        .reset_index()
    )
    return hourly.sort_values(["turbine_id", "timestamp"], kind="stable")


def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features and per-turbine weather lags."""
    featured = hourly.copy()
    hour = featured["timestamp"].dt.hour.astype(float)
    featured["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    featured["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    featured["day_of_year"] = featured["timestamp"].dt.dayofyear.astype(
        "int16"
    )

    grouped = featured.groupby("turbine_id", sort=False)
    for lag_hours in (1, 2):
        featured[f"wind_speed_lag_{lag_hours}h"] = grouped[
            "wind_speed"
        ].shift(lag_hours)
        featured[f"temperature_lag_{lag_hours}h"] = grouped[
            "temperature"
        ].shift(lag_hours)

    return featured.set_index("timestamp")


def split_train_validation(
    featured: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the fixed temporal split and exclude February 2026 onward."""
    complete = featured.dropna()
    train = complete.loc[complete.index < TRAIN_END_EXCLUSIVE].copy()
    validation = complete.loc[
        (complete.index >= TRAIN_END_EXCLUSIVE)
        & (complete.index < VALIDATION_END_EXCLUSIVE)
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty before 2026-01-01")
    if validation.empty:
        raise ValueError("Validation split is empty for January 2026")

    LOGGER.info(
        "Prepared %d train rows and %d validation rows",
        len(train),
        len(validation),
    )
    return train, validation


def _make_preprocessor() -> ColumnTransformer:
    """Encode turbine identity categorically and pass numeric features through."""
    return ColumnTransformer(
        transformers=[
            (
                "turbine_id",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["turbine_id"],
            )
        ],
        remainder="passthrough",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_candidates() -> dict[str, Pipeline]:
    """Build deterministic benchmark pipelines."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        LOGGER.error("Unable to import LightGBM: %s", exc)
        raise RuntimeError(
            "LightGBM is required. Install project dependencies before training."
        ) from exc

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        LOGGER.error("Unable to import CatBoost: %s", exc)
        raise RuntimeError(
            "CatBoost is required. Install project dependencies before training."
        ) from exc

    return {
        "lightgbm": Pipeline(
            steps=[
                ("preprocess", _make_preprocessor()),
                (
                    "regressor",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", _make_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            steps=[
                ("preprocess", _make_preprocessor()),
                (
                    "regressor",
                    CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.05,
                        depth=8,
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                        thread_count=-1,
                    ),
                ),
            ]
        ),
    }


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model inputs in the stable training/inference column order."""
    return frame.loc[:, FEATURE_COLUMNS].copy()


def benchmark_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[str, Pipeline, ModelMetrics]:
    """Fit candidates, clip predictions, and select the lowest RMSE."""
    x_train = _feature_matrix(train)
    y_train = train[TARGET_COLUMN]
    x_validation = _feature_matrix(validation)
    y_validation = validation[TARGET_COLUMN]

    fitted: dict[str, Pipeline] = {}
    scores: ModelMetrics = {}
    for name, model in build_candidates().items():
        LOGGER.info("Training %s", name)
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_validation), dtype=float)
        predictions = np.clip(predictions, 0.0, 1.0)
        rmse = float(np.sqrt(mean_squared_error(y_validation, predictions)))
        mae = float(mean_absolute_error(y_validation, predictions))
        fitted[name] = model
        scores[name] = {"rmse": rmse, "mae": mae}
        LOGGER.info(
            "Validation metrics [%s]: RMSE=%.6f MAE=%.6f",
            name,
            rmse,
            mae,
        )

    best_name = min(scores, key=lambda name: scores[name]["rmse"])
    LOGGER.info(
        "Selected %s with validation RMSE %.6f",
        best_name,
        scores[best_name]["rmse"],
    )
    return best_name, fitted[best_name], scores


def fit_and_save(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_path: Path,
) -> tuple[str, ModelMetrics]:
    """Select the best model, refit it through January, and save atomically."""
    best_name, selected_model, scores = benchmark_models(train, validation)
    final_frame = pd.concat([train, validation], axis=0)
    final_model = clone(selected_model)
    final_model.fit(
        _feature_matrix(final_frame),
        final_frame[TARGET_COLUMN],
    )

    # These attributes keep the artifact self-describing without wrapping the
    # estimator in a custom class that would complicate joblib deserialization.
    final_model.model_name_ = best_name
    final_model.validation_rmse_ = scores[best_name]["rmse"]
    final_model.validation_mae_ = scores[best_name]["mae"]
    final_model.validation_scores_ = scores
    final_model.feature_columns_ = FEATURE_COLUMNS.copy()
    final_model.training_cutoff_ = str(VALIDATION_END_EXCLUSIVE.date())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    joblib.dump(final_model, temporary_path)
    temporary_path.replace(model_path)
    LOGGER.info("Saved final model to %s", model_path)
    return best_name, scores


def run_training(
    turbine_1_path: Path,
    turbine_2_path: Path,
    model_path: Path,
) -> tuple[str, ModelMetrics]:
    """Execute the complete deterministic training pipeline."""
    turbine_1 = load_turbine_data(turbine_1_path, turbine_id=1)
    turbine_2 = load_turbine_data(turbine_2_path, turbine_id=2)
    combined = pd.concat([turbine_1, turbine_2], ignore_index=True)
    hourly = resample_hourly(combined)
    featured = add_features(hourly)
    train, validation = split_train_validation(featured)
    return fit_and_save(train, validation, model_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a local wind-power model for two turbines."
    )
    parser.add_argument(
        "--turbine-1",
        type=Path,
        default=DEFAULT_TURBINE_1_PATH,
        help=f"Path to turbine 1 XLSX/CSV (default: {DEFAULT_TURBINE_1_PATH})",
    )
    parser.add_argument(
        "--turbine-2",
        type=Path,
        default=DEFAULT_TURBINE_2_PATH,
        help=f"Path to turbine 2 CSV (default: {DEFAULT_TURBINE_2_PATH})",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path for the best model (default: {DEFAULT_MODEL_PATH})",
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
        best_name, scores = run_training(
            turbine_1_path=args.turbine_1,
            turbine_2_path=args.turbine_2,
            model_path=args.model_output,
        )
    except Exception:
        LOGGER.exception("Training pipeline failed")
        return 1
    LOGGER.info("Benchmark complete: best=%s scores=%s", best_name, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
