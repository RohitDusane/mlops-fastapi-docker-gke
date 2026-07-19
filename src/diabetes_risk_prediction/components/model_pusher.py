import subprocess
import sys
from datetime import datetime, timezone
 
# import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
 
from diabetes_risk_prediction.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from diabetes_risk_prediction.entity.config_entity import ModelPusherConfig
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

    @staticmethod
    def _current_git_sha() -> str:
        # CHANGE: new — same traceability addition as model_evaluation.py
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"
 
    def _find_version_for_run(self, client: MlflowClient, model_name: str, run_id: str) -> ModelVersion:
        # CHANGE: previously fetched ALL versions for the model
        # (search_model_versions(f"name='{model_name}'")) and filtered
        # client-side, logging every single version at INFO level on every
        # push. That scales linearly with registry size and gets noisy fast.
        # MLflow's search filter supports run_id directly — ask the server
        # for exactly the one version that matters.
        matches = list(client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'"))
        if not matches:
            raise CustomException(f"No MLflow model version found for run_id {run_id}", sys)
        return matches[0]
 
    def _previous_champion_version(self, client: MlflowClient, model_name: str, alias: str) -> str | None:
        # CHANGE: new — before overwriting the alias, record what it
        # previously pointed at. Without this, there's no way to know what
        # to roll back to if the new champion misbehaves in production —
        # you'd be reconstructing it from MLflow's UI history by hand.
        try:
            previous = client.get_model_version_by_alias(model_name, alias)
            return previous.version
        except MlflowException as e:
            if ("RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower()):
                return None  # no previous champion — first promotion ever
            raise CustomException(f"Failed to fetch current champion: {e}", sys,)  
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model rejected by evaluation — skipping promotion.")
                return ModelPusherArtifact(is_model_pushed=False, mlflow_model_version=None,)

            # mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)

            model_name = self.config.mlflow_registered_model_name
            alias = self.config.mlflow_model_alias
            run_id = self.model_trainer_artifact.mlflow_run_id
 
            version_info = self._find_version_for_run(client, model_name, run_id)
            version = version_info.version
 
            previous_version = self._previous_champion_version(client, model_name, alias)  # CHANGE: new
            if previous_version:
                logging.info(
                    "Current '%s' is version %s — will be superseded by version %s.",
                    alias,
                    previous_version,
                    version,
                )

                client.set_model_version_tag(
                    name=model_name,
                    version=previous_version,
                    key="superseded_by",
                    value=str(version),
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=previous_version,
                    key="promotion_status",
                    value="retired",
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="previous_champion",
                    value=str(previous_version),
                )

            promoted_at = datetime.now(timezone.utc).isoformat()
 
            # ---- Governance tags — audit metadata ----
            client.set_model_version_tag(name=model_name, version=version, key="validation_status", value="passed")
            client.set_model_version_tag(name=model_name, version=version, key="deployment_target", value="fastapi")
            client.set_model_version_tag(name=model_name, version=version, key="promoted_at", value=promoted_at)  # CHANGE: new
            client.set_model_version_tag(name=model_name, version=version, key="git_commit", value=self._current_git_sha())  # CHANGE: new
            client.set_model_version_tag(
                name=model_name, version=version, key="evaluation_score",
                value=str(self.model_evaluation_artifact.improved_score),)
            client.set_model_version_tag(
                name=model_name, version=version, key="mlflow_run_id",
                value=self.model_trainer_artifact.mlflow_run_id,
            )
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_by",
                value="training_pipeline",
            )

            metrics = self.model_evaluation_artifact.trained_model_metrics
            for key, value in metrics.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key=f"metric_{key}",
                    value=str(value),
                )
            # ---- Promote ----
            client.set_registered_model_alias(name=model_name, alias=alias, version=str(version))
 
            # CHANGE: new — verify the alias actually points where expected
            # after the call, rather than assuming success just because no
            # exception was raised. Catches silent no-ops or eventual-
            # consistency delays on the server side.
            confirmed = client.get_model_version_by_alias(model_name, alias)
            if confirmed.version != str(version):
                raise CustomException(
                    f"Alias verification failed: expected version {version}, "
                    f"but '{alias}' points to version {confirmed.version} after promotion",
                    sys,
                )
            
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promotion_status",
                value="champion",
            )
 
            logging.info(
                "Promoted and verified | model=%s | alias=%s | version=%s | previous=%s",
                model_name, alias, version, previous_version,)
 
            return ModelPusherArtifact(
                is_model_pushed=True, 
                mlflow_model_version=str(version),
                alias=alias,
                registered_model_name=model_name,)
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model pushing: {e}", sys)
