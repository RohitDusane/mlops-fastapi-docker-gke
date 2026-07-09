import sys
import os
import json
import mlflow
from mlflow.tracking import MlflowClient

from diabetes_risk_prediction.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from diabetes_risk_prediction.entity.config_entity import ModelEvaluationConfig
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging

# Composite score weights — recall weighted highest since this is a
# screening tool (missing an at-risk patient is costlier than an
# unnecessary follow-up), but not exclusively, since optimizing purely for
# recall is trivially gamed by a near-always-positive model. The precision
# guard below is the other half of that same concern.
SCORE_WEIGHTS = {"roc_auc": 0.4, "recall": 0.35, "f1_score": 0.25}


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

    @staticmethod
    def _safe_get(metrics: dict, key: str, default: float = 0.0) -> float:
        return metrics.get(key, default)

    def _get_current_production_metrics(self) -> dict | None:
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = MlflowClient()

            # Note: get_latest_versions(stages=...) is deprecated as of
            # MLflow 2.9 in favor of aliases (e.g. "@champion"). Left as-is
            # since model_pusher.py still uses transition_model_version_stage
            # — migrate both together if you move to the alias API, not
            # just this side.
            versions = client.get_latest_versions(
                self.config.mlflow_registered_model_name, stages=["Production"]
            )
            if not versions:
                logging.info("No Production model found — first model will be accepted by default.")
                return None

            run = client.get_run(versions[0].run_id)
            metrics = run.data.metrics
            return {
                "roc_auc": self._safe_get(metrics, "test_roc_auc"),
                "f1_score": self._safe_get(metrics, "test_f1_score"),
                "recall": self._safe_get(metrics, "test_recall"),
                "precision": self._safe_get(metrics, "test_precision"),
            }
        except Exception as e:
            logging.warning(f"Failed to fetch current Production metrics: {e}")
            return None

    @staticmethod
    def _compute_score(metrics: dict) -> float:
        return sum(SCORE_WEIGHTS[k] * metrics.get(k, 0.0) for k in SCORE_WEIGHTS)

    def _passes_precision_guard(self, new_metrics: dict, current_metrics: dict | None) -> tuple[bool, str]:
        if current_metrics is None:
            return True, ""
        current_precision = current_metrics.get("precision", 0.0)
        if current_precision == 0:
            return True, ""  # nothing to regress against
        regression = (current_precision - new_metrics.get("precision", 0.0)) / current_precision
        if regression > self.config.max_precision_regression:
            return False, (
                f"precision regressed {regression:.1%} "
                f"({current_precision:.4f} -> {new_metrics.get('precision', 0.0):.4f}), "
                f"exceeding the {self.config.max_precision_regression:.0%} guard"
            )
        return True, ""

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation.")

            raw_new_metrics = self.model_trainer_artifact.test_metrics
            new_metrics = {
                "roc_auc": self._safe_get(raw_new_metrics, "roc_auc"),
                "f1_score": self._safe_get(raw_new_metrics, "f1_score"),
                "recall": self._safe_get(raw_new_metrics, "recall"),
                "precision": self._safe_get(raw_new_metrics, "precision"),
            }
            current_metrics = self._get_current_production_metrics()

            if current_metrics is None:
                is_accepted = True
                improved_score = self._compute_score(new_metrics)
                logging.info("No baseline model — accepting first model.")
            else:
                score_delta = self._compute_score(new_metrics) - self._compute_score(current_metrics)
                score_ok = score_delta > self.config.changed_threshold_score

                precision_ok, precision_reason = self._passes_precision_guard(new_metrics, current_metrics)
                is_accepted = score_ok and precision_ok
                improved_score = score_delta

                logging.info(
                    f"New composite score delta: {score_delta:.4f} (threshold: {self.config.changed_threshold_score}) | "
                    f"Precision guard passed: {precision_ok}"
                    + (f" — {precision_reason}" if not precision_ok else "")
                    + f" | Accepted: {is_accepted}"
                )
                logging.info(f"New metrics: {new_metrics} | Current Production metrics: {current_metrics}")
            
            evaluation_dir = self.config.evaluation_dir
            os.makedirs(evaluation_dir, exist_ok=True)

            metrics_path = os.path.join(evaluation_dir, "metrics.json")

            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "model_accepted": is_accepted,
                        "improved_score": improved_score,
                        "trained_metrics": new_metrics,
                        "production_metrics": current_metrics
                    },
                    f, indent=4)

            logging.info(f"Evaluation metrics saved: {metrics_path}")

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                improved_score=round(improved_score, 4),
                trained_model_metrics=new_metrics,
                best_model_metrics=current_metrics,
                trained_model_file_path=self.model_trainer_artifact.trained_model_file_path,
            )
        except Exception as e:
            raise CustomException(f"Error during model evaluation: {e}", sys)