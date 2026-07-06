import shutil
import sys

import mlflow
from mlflow.tracking import MlflowClient

from diabetes_risk_prediction.entity.config_entity import ModelPusherConfig
from diabetes_risk_prediction.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
            self.config = model_pusher_config
        except Exception as e:
            raise CustomException(f"Error initializing ModelPusher: {e}", sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model was not accepted by evaluation — skipping push.")
                return ModelPusherArtifact(
                    is_model_pushed=False,
                    mlflow_model_version=None,
                    saved_model_path="",
                )

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = MlflowClient()

            # Find the model version tied to this training run and promote
            # it to Production; archive whatever was Production before.
            all_versions = client.search_model_versions(
                f"name='{self.config.mlflow_registered_model_name}'"
            )
            matching = [v for v in all_versions if v.run_id == self.model_trainer_artifact.mlflow_run_id]

            if not matching:
                raise CustomException(
                    f"No MLflow model version found for run_id {self.model_trainer_artifact.mlflow_run_id}",
                    sys,
                )

            version = matching[0].version
            client.transition_model_version_stage(
                name=self.config.mlflow_registered_model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logging.info(f"Promoted model version {version} to Production in MLflow Registry.")

            # Also copy to a local "saved_models" dir — this is what the
            # FastAPI service loads when MODEL_SOURCE=local instead of
            # pulling from MLflow over HTTP at startup.
            import os

            os.makedirs(self.config.saved_model_dir, exist_ok=True)
            saved_path = os.path.join(self.config.saved_model_dir, "diabetes_model.pkl")
            shutil.copy(self.model_trainer_artifact.trained_model_file_path, saved_path)

            return ModelPusherArtifact(
                is_model_pushed=True,
                mlflow_model_version=str(version),
                saved_model_path=saved_path,
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model pushing: {e}", sys)