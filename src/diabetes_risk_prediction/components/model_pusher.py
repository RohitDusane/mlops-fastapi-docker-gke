
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
                logging.info("Model rejected by evaluation — skipping promotion.")
                return ModelPusherArtifact(
                    is_model_pushed=False,
                    mlflow_model_version=None,
                )

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = MlflowClient()

            # Find registered model version created by this training run
            # all_versions = client.search_model_versions(f"name='{self.config.mlflow_registered_model_name}'")
            # matching = [v for v in all_versions if v.run_id == self.model_trainer_artifact.mlflow_run_id]

            versions = client.search_model_versions(f"name='{self.config.mlflow_registered_model_name}'")
            logging.info(
                "Searching model versions for run_id=%s",
                self.model_trainer_artifact.mlflow_run_id,
            )

            for v in versions:
                logging.info(
                    "version=%s run_id=%s source=%s",
                    v.version,
                    v.run_id,
                    v.source,
                )
            matching = [
                v for v in versions
                if v.run_id == self.model_trainer_artifact.mlflow_run_id
            ]
            
            if not matching:
                raise CustomException(f"No MLflow model version found for run_id {self.model_trainer_artifact.mlflow_run_id}", sys,)

            version = matching[0].version
            model_name = self.config.mlflow_registered_model_name
            alias = self.config.mlflow_model_alias

            # GOVERNENCE TAGS
            # Add audit metadata
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="validation_status",
                value="passed"
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="deployment_target",
                value="fastapi",
            )

            # PROMOTE MODEL
            # Assign champion alias
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=str(version)
            )
            
            logging.info(
                "Assigned MLflow alias | "
                "model=%s | alias=%s | version=%s",
                model_name, alias, version,)

            return ModelPusherArtifact(
                is_model_pushed=True,
                mlflow_model_version=str(version),
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model pushing: {e}", sys)