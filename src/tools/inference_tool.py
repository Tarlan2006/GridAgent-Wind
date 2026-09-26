"""Local inference for the deterministic wind-power model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

try:
    from src.tools.feature_tool import (
        MODEL_FEATURE_COLUMNS,
        RAW_WEATHER_COLUMNS,
        add_physical_features,
        select_model_features,
    )
except ModuleNotFoundError:
    from tools.feature_tool import (
        MODEL_FEATURE_COLUMNS,
        RAW_WEATHER_COLUMNS,
        add_physical_features,
        select_model_features,
    )

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH: Final = PROJECT_ROOT / "models" / "wind_power_lgb.pkl"
SHUTDOWN_WIND_SPEED_MPS: Final = 25.0


class ModelArtifactError(RuntimeError):
    """Raised when the local model artifact cannot be used safely."""


class WindPowerPredictor:
    """Load the local LightGBM artifact and apply physical safety limits."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._artifact: dict[str, Any] | None = None

    def _load_artifact(self) -> dict[str, Any]:
        if self._artifact is not None:
            return self._artifact
        if not self.model_path.is_file():
            raise ModelArtifactError(
                f"Model artifact is not available: {self.model_path}. "
                "Run python -m src.train.train_model first."
            )

        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ModelArtifactError(
                f"Model artifact {self.model_path} has an invalid format"
            )

        feature_names = artifact.get("feature_names")
        if feature_names != MODEL_FEATURE_COLUMNS:
            raise ModelArtifactError(
                "Model feature contract does not match the current feature tool"
            )
        self._artifact = artifact
        return artifact

    @staticmethod
    def _validate_input(weather_data: pd.DataFrame) -> pd.DataFrame:
        if weather_data.empty:
            raise ValueError("Weather data must contain at least one row")
        required = {"timestamp", "turbine_id", *RAW_WEATHER_COLUMNS}
        missing = required.difference(weather_data.columns)
        if missing:
            raise ValueError(
                f"Weather data is missing columns: {sorted(missing)}"
            )

        validated = weather_data.copy()
        validated["timestamp"] = pd.to_datetime(
            validated["timestamp"],
            errors="coerce",
            utc=True,
        )
        if validated["timestamp"].isna().any():
            raise ValueError("Weather data contains invalid timestamps")

        turbine_ids = pd.to_numeric(
            validated["turbine_id"],
            errors="coerce",
        )
        if turbine_ids.isna().any() or not turbine_ids.isin((1, 2)).all():
            raise ValueError("turbine_id must contain only 1 or 2")
        validated["turbine_id"] = turbine_ids.astype("int8")
        return validated

    def predict(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Predict normalized power and enforce the turbine safety envelope."""
        artifact = self._load_artifact()
        validated = self._validate_input(weather_data)
        featured = add_physical_features(validated)
        features = select_model_features(featured)
        features["turbine_id"] = features["turbine_id"].astype("category")

        if features.isna().any().any():
            missing_columns = features.columns[features.isna().any()].tolist()
            raise ValueError(
                f"Weather data contains missing model features: {missing_columns}"
            )

        raw_predictions = artifact["model"].predict(features)
        predictions = np.clip(np.asarray(raw_predictions, dtype=float), 0.0, 1.0)
        shutdown_mask = validated["wind_speed_100m"].gt(
            SHUTDOWN_WIND_SPEED_MPS
        ).to_numpy()
        predictions[shutdown_mask] = 0.0

        return pd.DataFrame(
            {
                "timestamp": validated["timestamp"].to_numpy(),
                "turbine_id": validated["turbine_id"].to_numpy(),
                "predicted_power": predictions,
            }
        )

    def metadata(self) -> dict[str, Any]:
        """Return safe model metadata for logs and user-facing status."""
        artifact = self._load_artifact()
        return {
            "model_path": str(self.model_path),
            "objective": artifact.get("objective"),
            "validation_metrics": artifact.get("validation_metrics"),
            "train_end_exclusive": artifact.get("train_end_exclusive"),
            "validation_end_exclusive": artifact.get("validation_end_exclusive"),
        }
