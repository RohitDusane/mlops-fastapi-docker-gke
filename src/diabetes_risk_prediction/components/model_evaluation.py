import json
import os
import subprocess
import sys
import numpy as np
 
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone
 
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
            if not 0 <= self.config.max_precision_regression <= 1:
                raise CustomException("max_precision_regression must be between 0 and 1", sys,)

            if self.config.changed_threshold_score < 0:
                raise CustomException("changed_threshold_score must be non-negative", sys,)
        except Exception as e:
            raise CustomException(f"Error initializing ModelEvaluation: {e}", sys)

    @staticmethod
    def _safe_get(metrics: dict, key: str, default: float = 0.0) -> float:
        return metrics.get(key, default)
    
    def _validate_metrics(self, metrics: dict) -> dict:
        required = [
            "roc_auc",
            "recall",
            "f1_score",
            "precision",
        ]

        missing = [key for key in required if key not in metrics]
        if missing:
            raise CustomException(f"Missing evaluation metrics: {missing}", sys,)

        invalid = {key: value for key, value in metrics.items() if value < 0 or value > 1}
        if invalid:
            raise CustomException(f"Invalid metric values: {invalid}", sys,)
        return metrics

    def _get_current_production_metrics(self) -> dict | None:
        """
        Returns the current @champion's test metrics, or None if there is
        genuinely no champion registered yet (expected on the very first
        training run — this is the only case that should auto-accept).
        """
        client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
        try:
            # CHANGE: get_model_version_by_alias returns a single ModelVersion
            # object, not a list. The previous code did `versions[0].run_id`,
            # which raised TypeError on every call where a champion actually
            # existed — silently caught by the broad except below, always
            # returning None, which meant every challenger was auto-accepted
            # regardless of quality. This was the highest-severity bug here:
            # the quality gate was never actually being enforced in practice.
            version_info = client.get_model_version_by_alias(
                self.config.mlflow_registered_model_name,
                self.config.mlflow_model_alias,  # CHANGE: use config value, not hardcoded "champion"
            )
            run = client.get_run(version_info.run_id)  # CHANGE: no indexing — this is one object
            metrics = run.data.metrics
            return {
                "roc_auc": self._safe_get(metrics, "test_roc_auc"),
                "f1_score": self._safe_get(metrics, "test_f1_score"),
                "recall": self._safe_get(metrics, "test_recall"),
                "precision": self._safe_get(metrics, "test_precision"),
            }
        except MlflowException as e:
            # CHANGE: narrowed from a bare `except Exception` to specifically
            # MlflowException here. An alias that genuinely doesn't exist yet
            # raises this — that's the real "first model ever" case, and the
            # only case that should return None. A connection error, auth
            # failure, or malformed response is a different, more serious
            # problem and should NOT be silently treated as "no baseline."
            if "not found" in str(e).lower() or "RESOURCE_DOES_NOT_EXIST" in str(e):
                logging.info(
                    f"No '{self.config.mlflow_model_alias}' alias found for "
                    f"'{self.config.mlflow_registered_model_name}' — treating as first model."
                )
                return None
            # CHANGE: any other MLflow-side error (auth, malformed response,
            # etc.) is now re-raised rather than silently swallowed — a
            # broken comparison should fail the pipeline loudly, not quietly
            # let an unvalidated model through.
            raise CustomException(f"MLflow error while fetching current champion metrics: {e}", sys)
        except Exception as e:
            # CHANGE: genuine infrastructure failures (network unreachable,
            # DNS failure, timeout) are also re-raised now instead of being
            # treated as "no champion" — see rationale above.
            raise CustomException(f"Failed to reach MLflow while evaluating challenger: {e}", sys)

    @staticmethod
    def _compute_score(metrics: dict) -> float:
        return sum(SCORE_WEIGHTS[k] * metrics[k] for k in SCORE_WEIGHTS)

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
    @staticmethod
    def _current_git_sha() -> str:
        # CHANGE: new — every evaluation record now carries the exact code
        # version that produced it. Cheap, and closes one of the traceability
        # gaps flagged earlier (git SHA in artifact metadata).
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    
    @staticmethod
    def _make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: ModelEvaluation._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelEvaluation._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation.")
            # mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            raw_new_metrics = self.model_trainer_artifact.test_metrics

            logging.info(
                "Evaluating model=%s run_id=%s",
                self.model_trainer_artifact.model_type,
                self.model_trainer_artifact.mlflow_run_id,
            )
                        
            self._validate_metrics(raw_new_metrics)

            new_metrics = {
                "roc_auc": raw_new_metrics["roc_auc"],
                "f1_score": raw_new_metrics["f1_score"],
                "recall": raw_new_metrics["recall"],
                "precision": raw_new_metrics["precision"],
            }
            current_metrics = self._get_current_production_metrics()
            if current_metrics is not None:
                self._validate_metrics(current_metrics)
            rejection_reasons = []  # captured and persisted for audit, not just logged
 
            if current_metrics is None:
                logging.info("No baseline model — accepting validated first model."    )
                is_accepted = True
                improved_score = self._compute_score(new_metrics)
            else:
                score_delta = self._compute_score(new_metrics) - self._compute_score(current_metrics)
                score_ok = score_delta > self.config.changed_threshold_score
                if not score_ok:
                    rejection_reasons.append(
                        f"composite score delta {score_delta:.4f} did not exceed "
                        f"threshold {self.config.changed_threshold_score}"
                    )

                precision_ok, precision_reason = self._passes_precision_guard(new_metrics, current_metrics)
                if not precision_ok:
                    rejection_reasons.append(precision_reason)

                is_accepted = score_ok and precision_ok
                improved_score = score_delta
                
                logging.info(
                    f"New composite score delta: {score_delta:.4f} (threshold: {self.config.changed_threshold_score}) | "
                    f"Precision guard passed: {precision_ok}"
                    + (f" — {precision_reason}" if not precision_ok else "")
                    + f" | Accepted: {is_accepted}"
                )
                # logging.info(f"New metrics: {new_metrics} | Current champion metrics: {current_metrics}")
                logging.info(
                    "New metrics=%s | Champion metrics=%s",
                    new_metrics,
                    current_metrics,
                )
            evaluation_dir = self.config.evaluation_dir
            os.makedirs(evaluation_dir, exist_ok=True)
            metrics_path = os.path.join(evaluation_dir, "metrics.json")

            with open(metrics_path, "w") as f:
                json.dump(
                    self._make_json_serializable(
                        {
                            "model_accepted": is_accepted,
                            "improved_score": improved_score,
                            "trained_metrics": new_metrics,
                            "champion_metrics": current_metrics,
                            "rejection_reasons": rejection_reasons,
                            "mlflow_run_id": self.model_trainer_artifact.mlflow_run_id,
                            "registered_model_name": self.config.mlflow_registered_model_name,
                            "alias": self.config.mlflow_model_alias,
                            "git_commit": self._current_git_sha(),
                            "evaluated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ),
                    f,
                    indent=4
                )

            logging.info(f"Evaluation metrics saved: {metrics_path}")

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                improved_score=round(improved_score, 4),
                trained_model_metrics=new_metrics,
                best_model_metrics=current_metrics,
                # trained_model_file_path=self.model_trainer_artifact.trained_model_file_path,
                mlflow_run_id=self.model_trainer_artifact.mlflow_run_id,
                registered_model_name= self.config.mlflow_registered_model_name,
                alias= self.config.mlflow_model_alias,
            )
        except Exception as e:
            raise CustomException(f"Error during model evaluation: {e}", sys)