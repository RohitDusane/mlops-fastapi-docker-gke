import sys

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from diabetes_risk_prediction.entity.config_entity import ModelTrainerConfig
from diabetes_risk_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging
from diabetes_risk_prediction.utils.common import load_numpy_array_data, save_object


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.config = model_trainer_config
        except Exception as e:
            raise CustomException(f"Error initializing ModelTrainer: {e}", sys)

    def _compute_metrics(self, model, X, y) -> dict:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred), 4),
            "recall": round(recall_score(y, y_pred), 4),
            "f1_score": round(f1_score(y, y_pred), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
        }

    def _objective(self, trial, X_train, y_train):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "bootstrap": True,
            "max_samples": 0.7,
            "random_state": 24,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=self.config.target_metric, n_jobs=-1)
        mean_score = scores.mean()

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric(f"cv_{self.config.target_metric}_mean", mean_score)
            mlflow.log_metric(f"cv_{self.config.target_metric}_std", scores.std())

        return mean_score

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training.")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            with mlflow.start_run(run_name="hyperparameter-search") as parent_run:
                logging.info(f"Running Optuna study: {self.config.n_trials} trials")
                study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(),)
                study.optimize(
                    lambda trial: self._objective(trial, X_train, y_train),
                    n_trials=self.config.n_trials,
                )

                best_params = study.best_params
                logging.info(f"Best CV {self.config.target_metric}: {study.best_value:.4f}")
                logging.info(f"Best hyperparameters: {best_params}")

                final_model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42, n_jobs=-1)
                final_model.fit(X_train, y_train)

                train_metrics = self._compute_metrics(final_model, X_train, y_train)
                test_metrics = self._compute_metrics(final_model, X_test, y_test)

                # ---- Quality gates ----
                # 1. Minimum acceptable performance — fail loudly rather than
                #    silently shipping a model that barely beats a coin flip.
                if test_metrics[self.config.target_metric] < self.config.expected_score:
                    raise CustomException(
                        f"Model did not meet minimum expected {self.config.target_metric} "
                        f"({test_metrics[self.config.target_metric]} < {self.config.expected_score})",
                        sys,
                    )

                # 2. Overfitting check — large train/test gap signals the
                #    model memorized training data rather than generalizing.
                score_gap = train_metrics[self.config.target_metric] - test_metrics[self.config.target_metric]
                if score_gap > self.config.overfitting_threshold:
                    raise CustomException(
                        f"Overfitting detected: train {self.config.target_metric} "
                        f"({train_metrics[self.config.target_metric]}) exceeds test "
                        f"({test_metrics[self.config.target_metric]}) by {score_gap:.4f}, "
                        f"over the {self.config.overfitting_threshold} threshold",
                        sys,
                    )

                mlflow.log_params(best_params)
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow.sklearn.log_model(
                    final_model,
                    artifact_path="model",
                    registered_model_name=self.config.mlflow_registered_model_name,
                )
                run_id = parent_run.info.run_id

            model_path = f"{self.config.trainer_dir}/{self.config.trained_model_file_name}"
            save_object(model_path, final_model)

            logging.info(f"Model training completed. Test metrics: {test_metrics}")

            return ModelTrainerArtifact(
                trained_model_file_path=model_path,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                mlflow_run_id=run_id,
                best_hyperparameters=best_params,
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model training: {e}", sys)