import sys

import mlflow
from mlflow.tracking import MlflowClient

from diabetes_risk_prediction.entity.config_entity import ModelEvaluationConfig
from diabetes_risk_prediction.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging


class ModelEvaluation:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.config = model_evaluation_config
        except Exception as e:
            raise CustomException(f"Error initializing ModelEvaluation: {e}", sys)

    def _get_current_production_metrics(self) -> dict | None:
        """
        Looks up the metrics logged against whichever model version is
        currently tagged "Production" in the MLflow Registry. Returns None
        if there is no Production model yet (first-ever training run).
        """
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = MlflowClient()

            versions = client.get_latest_versions(
                self.config.mlflow_registered_model_name, stages=["Production"]
            )
            if not versions:
                logging.info("No existing Production model found — first model will be accepted by default.")
                return None

            production_run_id = versions[0].run_id
            run = client.get_run(production_run_id)
            return {
                k.replace("test_", ""): v
                for k, v in run.data.metrics.items()
                if k.startswith("test_")
            }
        except Exception as e:
            logging.warning(f"Could not fetch current Production model metrics: {e}")
            return None

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation.")
            new_metrics = self.model_trainer_artifact.test_metrics
            current_metrics = self._get_current_production_metrics()

            target_metric = "roc_auc"

            if current_metrics is None:
                is_accepted = True
                improved_score = new_metrics[target_metric]
                logging.info("No baseline to compare against — accepting new model.")
            else:
                improved_score = new_metrics[target_metric] - current_metrics[target_metric]
                is_accepted = improved_score >= self.config.changed_threshold_score
                logging.info(
                    f"New {target_metric}: {new_metrics[target_metric]} | "
                    f"Current Production {target_metric}: {current_metrics[target_metric]} | "
                    f"Delta: {improved_score:.4f} | Accepted: {is_accepted}"
                )

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                improved_score=round(improved_score, 4),
                trained_model_metrics=new_metrics,
                best_model_metrics=current_metrics,
                trained_model_file_path=self.model_trainer_artifact.trained_model_file_path,
            )
        except Exception as e:
            raise CustomException(f"Error during model evaluation: {e}", sys)