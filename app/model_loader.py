"""
app/model_loader.py

Owns the currently-loaded model + its metadata. Separated from main.py so
route handlers stay thin, and so this can be unit-tested independently of
the FastAPI app object.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib

from app.config import settings

logger = logging.getLogger("diabetes-api")


class ModelState:
    def __init__(self):
        self.model = None
        self.threshold: float = 0.5
        self.model_type: str = "unknown"
        self.feature_order: list[str] = []
        self.model_version: str = "unknown"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_local(self):
        """
        Loads the ModelBundle dataclass saved by ModelTrainer. This is a
        dataclass, not a dict — access via .model/.threshold/.model_type/
        .feature_columns/.mlflow_run_id, never bundle["key"] or bundle.get().
        """
        bundle = joblib.load(Path(settings.model_path))
        self.model = bundle.model
        self.threshold = bundle.threshold
        self.model_type = bundle.model_type
        self.feature_order = bundle.feature_columns
        self.model_version = f"local:{bundle.model_type}:{bundle.mlflow_run_id[:8]}"

    def load_mlflow(self):
        """
        Pulls the current Production-stage model from the MLflow Registry,
        plus its metadata (threshold, feature order) from the
        model_metadata.json artifact ModelTrainer logs alongside it — there
        is no model_card.json in this pipeline; don't reintroduce a
        dependency on one.
        """
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        # Note: get_latest_versions(stages=...) is the deprecated stages API
        # (MLflow recommends aliases as of 2.9+). Kept for now since
        # model_pusher.py still writes stages, not aliases — migrate both
        # together if you switch.
        versions = client.get_latest_versions(settings.mlflow_model_name, stages=[settings.mlflow_model_stage])
        if not versions:
            raise RuntimeError(
                f"No model version found in stage '{settings.mlflow_model_stage}' "
                f"for '{settings.mlflow_model_name}'"
            )
        version_info = versions[0]

        metadata = mlflow.artifacts.load_dict(f"runs:/{version_info.run_id}/model_metadata.json")

        self.model = mlflow.sklearn.load_model(f"models:/{settings.mlflow_model_name}/{settings.mlflow_model_stage}")
        self.threshold = float(metadata["threshold"])
        self.model_type = metadata["model_type"]
        self.feature_order = metadata["feature_columns"]
        self.model_version = f"mlflow:v{version_info.version}"

    def load(self):
        try:
            if settings.model_source == "mlflow":
                logger.info("Loading model from MLflow registry: %s/%s", settings.mlflow_model_name, settings.mlflow_model_stage)
                self.load_mlflow()
            else:
                logger.info("Loading model from local artifact: %s", settings.model_path)
                self.load_local()
            logger.info(
                "Model loaded — version=%s type=%s threshold=%.3f features=%d",
                self.model_version, self.model_type, self.threshold, len(self.feature_order),
            )
        except Exception as e:
            # Fail loudly in logs, don't crash the process — /health reports
            # unhealthy and the K8s readiness probe holds traffic back
            # instead of the pod crash-looping.
            logger.error("Failed to load model: %s", e)
            self.model = None


model_state = ModelState()