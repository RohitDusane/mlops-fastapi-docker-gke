"""
app/model_loader.py

Owns the currently-loaded model + its metadata. Separated from main.py so
route handlers stay thin, and so this can be unit-tested independently of
the FastAPI app object.
"""

import logging
from pathlib import Path
# from typing import Optional
# import json, os
# import joblib

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

    # def load_local(self):
    #     """
    #     Loads the ModelBundle dataclass saved by ModelTrainer. This is a
    #     dataclass, not a dict — access via .model/.threshold/.model_type/
    #     .feature_columns/.mlflow_run_id, never bundle["key"] or bundle.get().
    #     """
    #     model_path = Path(settings.model_path)
    #     metadata_path = Path(settings.metadata_path)

    #     if not model_path.exists():
    #         raise FileNotFoundError(f"Model not found: {model_path}")

    #     if not metadata_path.exists():
    #         raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    #     logger.info("Loading model from %s", model_path)
    #     # Load pipeline
    #     self.model = joblib.load(model_path)

    #     # Load metadata
    #     with metadata_path.open("r", encoding="utf-8") as f:
    #         metadata = json.load(f)

    #     self.threshold = float(metadata["threshold"])
    #     self.model_type = metadata["model_type"]
    #     self.feature_order = metadata["feature_columns"]

    #     run_id = metadata.get("mlflow_run_id", "local")
    #     self.model_version = f"local:{self.model_type}:{run_id[:8]}"

        
    def load_mlflow(self):
        """
        Pulls the current Production-stage model from the MLflow Registry,
        plus its metadata (threshold, feature order) from the
        model_metadata.json artifact ModelTrainer logs alongside it — there
        is no model_card.json in this pipeline; don't reintroduce a
        dependency on one.
        """
        import mlflow
        import mlflow.pyfunc
        from mlflow.artifacts import download_artifacts
        import json
        import mlflow.lightgbm        

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        model_uri = (
            f"models:/{settings.mlflow_model_name}"
            f"@{settings.mlflow_model_alias}"
        )
        logger.info(f"Loading Model {model_uri}")

        self.model = mlflow.pyfunc.load_model(model_uri)
        # pyfunc_model = mlflow.pyfunc.load_model(model_uri)


        # print("MODEL IMPL:", type(self.model._model_impl))

        # if hasattr(self.model._model_impl, "sklearn_model"):
        #     print("Native sklearn model:",
        #         type(self.model._model_impl.sklearn_model))
            
        # model = mlflow.lightgbm.load_model(model_uri)
        self.model_version = model_uri

        local_path = download_artifacts(model_uri)

        # logger.info("Downloaded model artifact path: %s", local_path)

        # for root, dirs, files in os.walk(local_path):
        #     logger.info("DIR=%s FILES=%s", root, files)

        # metadata = mlflow.artifacts.load_dict(f"{model_uri}/model_metadata.json")
        
        # Download complete model artifact directory
        # local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

        metadata_file = Path(local_path) / "model_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(metadata_file)

        with metadata_file.open() as f:
            metadata = json.load(f)

        self.threshold = float(metadata["threshold"])
        self.model_type = metadata["model_type"]
        self.feature_order = metadata["feature_columns"]

        # self.model_version = (
        #     f"{settings.mlflow_model_name}"
        #     f"@{settings.mlflow_model_alias}"
        # )

    def load(self):
        try:
            # if settings.model_source == "mlflow":
            logger.info("Loading model from MLflow registry: %s/%s", settings.mlflow_model_name, settings.mlflow_model_alias)
            self.load_mlflow()
            # else:
            #     logger.info("Loading model from local artifact: %s", settings.model_path)
            #     self.load_local()
            logger.info(
                "Model loaded successfully | "
                "version=%s | "
                "type=%s | "
                "threshold=%.3f | "
                "features=%d",
                self.model_version,
                self.model_type,
                self.threshold,
                len(self.feature_order),
            )
            logger.debug("Feature order: %s", self.feature_order)
        except Exception:
            # Fail loudly in logs, don't crash the process — /health reports
            # unhealthy and the K8s readiness probe holds traffic back
            # instead of the pod crash-looping.
            logger.exception("Failed to load model")
            raise

model_state = ModelState()