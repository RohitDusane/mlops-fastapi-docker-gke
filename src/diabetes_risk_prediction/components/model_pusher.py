import subprocess
import sys
from datetime import datetime, timezone
 
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
 
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
 
    def _find_version_for_run(self, client: MlflowClient, model_name: str, run_id: str):
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
        except MlflowException:
            return None  # no previous champion — first promotion ever
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model rejected by evaluation — skipping promotion.")
                return ModelPusherArtifact(is_model_pushed=False, mlflow_model_version=None,)

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = MlflowClient()

            model_name = self.config.mlflow_registered_model_name
            alias = self.config.mlflow_model_alias
            run_id = self.model_trainer_artifact.mlflow_run_id
 
            version_info = self._find_version_for_run(client, model_name, run_id)
            version = version_info.version
 
            previous_version = self._previous_champion_version(client, model_name, alias)  # CHANGE: new
            if previous_version:
                logging.info(f"Current '{alias}' is version {previous_version} — will be superseded by version {version}.")
                # CHANGE: tag the outgoing champion so its demotion is
                # visible in MLflow's UI, not just inferred from the alias
                # having moved.
                client.set_model_version_tag(
                    name=model_name, version=previous_version,
                    key="superseded_by", value=str(version),)
 
            promoted_at = datetime.now(timezone.utc).isoformat()
 
            # ---- Governance tags — audit metadata ----
            client.set_model_version_tag(name=model_name, version=version, key="validation_status", value="passed")
            client.set_model_version_tag(name=model_name, version=version, key="deployment_target", value="fastapi")
            client.set_model_version_tag(name=model_name, version=version, key="promoted_at", value=promoted_at)  # CHANGE: new
            client.set_model_version_tag(name=model_name, version=version, key="git_commit", value=self._current_git_sha())  # CHANGE: new
            client.set_model_version_tag(
                name=model_name, version=version, key="evaluation_score",
                value=str(self.model_evaluation_artifact.improved_score),)
 
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
 
            logging.info(
                "Promoted and verified | model=%s | alias=%s | version=%s | previous=%s",
                model_name, alias, version, previous_version,)
 
            return ModelPusherArtifact(is_model_pushed=True, mlflow_model_version=str(version),)
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model pushing: {e}", sys)
