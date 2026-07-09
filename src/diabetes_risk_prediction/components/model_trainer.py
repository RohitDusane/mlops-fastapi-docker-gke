import sys
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# try:
#     from xgboost import XGBClassifier
#     _XGBOOST_AVAILABLE = True
# except ImportError:
#     _XGBOOST_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from diabetes_risk_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from diabetes_risk_prediction.entity.config_entity import ModelTrainerConfig
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging
from diabetes_risk_prediction.utils.common import load_numpy_array_data, load_object
import json
import os

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

    # ---------------- Metrics ---------------- #

    def _compute_metrics(self, model, X, y, threshold: float) -> dict:
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred), 4),
            "f1_score": round(f1_score(y, y_pred), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
            "pr_auc": round(average_precision_score(y, y_proba), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        }

    # ---------------- Threshold tuning ---------------- #

    def optimize_threshold(self, model, X_val, y_val) -> float:
        """
        Picks the highest decision threshold that still keeps recall at or
        above `target_recall_floor` — maximizes precision subject to a
        recall floor, rather than optimizing raw F1, since in a screening
        context a false negative (missed at-risk patient) is worse than a
        false positive (unnecessary follow-up test).

        Caveat: this is tuned on a held-out split of the training data
        (X_val), then the final model is refit on the FULL training set
        afterward — meaning the refit model's probability surface shifts
        slightly from the one the threshold was tuned against. This is a
        reasonable approximation, not a perfect calibration; a more rigorous
        version would tune the threshold via out-of-fold predictions across
        the full training set (cross_val_predict) instead of a single
        train/val split. Worth upgrading if threshold precision matters more
        than the extra CV cost.
        """
        y_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

        valid = np.where(recall[:-1] >= self.config.target_recall_floor)[0]
        if len(valid):
            return float(thresholds[valid[-1]])

        logging.warning(
            f"No threshold achieves the target recall floor "
            f"({self.config.target_recall_floor}) — falling back to 0.5"
        )
        return 0.5

    # ---------------- Multi-algorithm search space ---------------- #

    def _suggest_hyperparams(self, trial, model_type: str, scale_pos_weight: float) -> dict:
        if model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, step=25),
                "max_depth": trial.suggest_int("rf_max_depth", 4, 16),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 8),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
                "class_weight": "balanced",
                "bootstrap": True,
                "max_samples": 0.7,
                "random_state": self.config.random_state,
                "n_jobs": 1,   # parallelism lives at the CV-fold level, not here — avoids oversubscription
            }
        if model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 600, step=50),
                "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("lgbm_num_leaves", 31, 200),
                "max_depth": trial.suggest_int("lgbm_max_depth", -1, 12),
                "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 10, 100),
                "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 10.0, log=True),
                # Computed from the actual training data's class ratio, not
                # a hardcoded literal — stays correct if class balance shifts
                # on retraining, unlike a fixed magic number.
                "scale_pos_weight": scale_pos_weight,
                "random_state": self.config.random_state,
                "n_jobs": 1,
                "verbosity": -1,
            }
        if model_type == "logistic_regression":
            # Fast, interpretable baseline — useful as a sanity floor: if a
            # tuned RF/LGBM can't beat simple logistic regression by much,
            # that's worth knowing, not just worth beating.
            return {
                "C": trial.suggest_float("logreg_C", 1e-3, 10.0, log=True),
                "class_weight": "balanced",
                "max_iter": 2000,
                "random_state": self.config.random_state,
                "n_jobs": 1,
            }
        # if model_type == "xgboost":
        #     if not _XGBOOST_AVAILABLE:
        #         raise ValueError("xgboost is not installed — pip install xgboost, or remove it from candidate_algorithms")
        #     return {
        #         "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 600, step=50),
        #         "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
        #         "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        #         "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        #         "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        #         "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        #         "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
        #         "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
        #         "scale_pos_weight": scale_pos_weight,
        #         "random_state": self.config.random_state,
        #         "n_jobs": 1,
        #         "eval_metric": "aucpr",
        #     }
        raise ValueError(f"Unknown model_type: {model_type}")

    def _build_model(self, model_type: str, params: dict):
        if model_type == "random_forest":
            return RandomForestClassifier(**params)
        if model_type == "lightgbm":
            return LGBMClassifier(**params)
        if model_type == "logistic_regression":
            return LogisticRegression(**params)
        # if model_type == "xgboost":
        #     return XGBClassifier(**params)
        raise ValueError(f"Unknown model_type: {model_type}")

    # ---------------- Optuna objective ---------------- #

    def _objective(self, trial, X_train, y_train, scale_pos_weight):
        model_type = trial.suggest_categorical("model_type", list(self.config.candidate_algorithms))
        params = self._suggest_hyperparams(trial, model_type, scale_pos_weight)
        model = self._build_model(model_type, params)

        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            model.fit(X_train[train_idx], y_train[train_idx])
            y_proba = model.predict_proba(X_train[val_idx])[:, 1]
            score = average_precision_score(y_train[val_idx], y_proba) \
                if self.config.cv_metric == "average_precision" \
                else roc_auc_score(y_train[val_idx], y_proba)
            fold_scores.append(score)

            # Real pruning: report the running mean after each fold so
            # MedianPruner can actually cut off unpromising trials early,
            # rather than being configured but never invoked (as in the
            # cross_val_score-based version, which returns one final number
            # with no intermediate reporting).
            if self.config.enable_pruning:
                trial.report(float(np.mean(fold_scores)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_type", model_type)
            mlflow.log_params(params)
            mlflow.log_metric(f"cv_{self.config.cv_metric}_mean", mean_score)
            mlflow.log_metric(f"cv_{self.config.cv_metric}_std", float(np.std(fold_scores)))

        return mean_score

    # ---------------- Main entrypoint ---------------- #

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training.")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            neg, pos = np.bincount(y_train.astype(int))
            scale_pos_weight = neg / max(pos, 1)
            logging.info(f"Class balance — negative: {neg}, positive: {pos}, scale_pos_weight: {scale_pos_weight:.3f}")

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            with mlflow.start_run(run_name="hyperparameter-search") as parent_run:
                pruner = optuna.pruners.MedianPruner() if self.config.enable_pruning else optuna.pruners.NopPruner()
                study = optuna.create_study(direction="maximize", pruner=pruner)
                study.optimize(
                    lambda trial: self._objective(trial, X_train, y_train, scale_pos_weight),
                    n_trials=self.config.n_trials,
                )

                best_params = dict(study.best_params)
                model_type = best_params.pop("model_type")
                # Remaining keys are prefixed (rf_/lgbm_/logreg_) to stay
                # unique within one Optuna search space — strip the prefix
                # before passing to the estimator constructor.
                prefix = {"random_forest": "rf_", "lightgbm": "lgbm_", "logistic_regression": "logreg_"}[model_type]
                final_params = {k[len(prefix):]: v for k, v in best_params.items() if k.startswith(prefix)}
                if model_type == "lightgbm":
                    final_params["scale_pos_weight"] = scale_pos_weight
                elif model_type in ("random_forest", "logistic_regression"):
                    final_params["class_weight"] = "balanced"
                final_params["random_state"] = self.config.random_state

                logging.info(f"Best algorithm: {model_type} | Best CV {self.config.cv_metric}: {study.best_value:.4f}")
                logging.info(f"Best hyperparameters: {final_params}")

                final_model = self._build_model(model_type, {**final_params, "n_jobs": -1} if model_type != "logistic_regression" else final_params)

                # Threshold tuned on a held-out split of the training data —
                # see optimize_threshold()'s docstring for the caveat about
                # refitting afterward.
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train, random_state=self.config.random_state,
                )
                final_model.fit(X_fit, y_fit)
                threshold = self.optimize_threshold(final_model, X_val, y_val)

                if self.config.calibrate_probabilities:
                    from sklearn.calibration import CalibratedClassifierCV
                    logging.info("Calibrating probabilities (isotonic, cv=3) before final fit.")
                    final_model = CalibratedClassifierCV(final_model, method="isotonic", cv=3)

                # Refit on the full training set for the artifact that
                # actually ships — see the threshold caveat above regarding
                # the resulting slight probability-surface shift.
                final_model.fit(X_train, y_train)

                train_metrics = self._compute_metrics(final_model, X_train, y_train, threshold)
                test_metrics = self._compute_metrics(final_model, X_test, y_test, threshold)
                logging.info(f"Train metrics: {train_metrics}")
                logging.info(f"Test metrics: {test_metrics}")

                # Log everything BEFORE the quality gates — a run that fails
                # a gate still shows up in MLflow with full metrics attached,
                # instead of an empty run with no clue why it was rejected.
                mlflow.set_tag("model_type", model_type)
                mlflow.log_params(final_params)
                mlflow.log_metric("decision_threshold", threshold)
                mlflow.log_metric("target_recall_floor", self.config.target_recall_floor)
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                # ---- Single source of truth for serving code: feature
                # columns, threshold, and algorithm — logged as one MLflow
                # artifact rather than a separate model_card.json file (the
                # old standalone train.py wrote that file; this pipeline
                # doesn't, so serving code must not depend on it existing).
                mlflow.log_dict(
                    {
                        "model_type": model_type,
                        "threshold": threshold,
                        "feature_columns": self.data_transformation_artifact.feature_columns,
                    },
                    "model_metadata.json",
                )

                # ---- Feature importance (RF/LGBM only — LogisticRegression
                # uses coefficients instead), logged separately for
                # clinical/interpretability review. ----
                importances = None
                if hasattr(final_model, "feature_importances_"):
                    importances = final_model.feature_importances_.tolist()
                elif hasattr(final_model, "coef_"):
                    importances = final_model.coef_[0].tolist()
                if importances is not None:
                    mlflow.log_dict({"feature_importances": importances}, "feature_importance.json")

                # ---- Confusion matrix at the tuned threshold, for clinical review ----
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, (final_model.predict_proba(X_test)[:, 1] >= threshold).astype(int))
                confusion_path = os.path.join(self.config.evaluation_dir, "confusion_matrix.json")

                with open(confusion_path,"w") as f:
                    json.dump(
                        {
                        "actual": y_test.tolist(),
                        "predicted": (final_model.predict_proba(X_test)[:, 1]).tolist()
                        },
                        f,
                        indent=4
                    )
                mlflow.log_dict({"confusion_matrix": cm.tolist(), "threshold": threshold}, "confusion_matrix.json")

                # ---- Clinical quality gates — config-driven, not magic numbers ----
                gate_failures = []
                if test_metrics["roc_auc"] < self.config.min_roc_auc:
                    gate_failures.append(f"roc_auc {test_metrics['roc_auc']} < {self.config.min_roc_auc}")
                if test_metrics["recall"] < self.config.min_recall:
                    gate_failures.append(f"recall {test_metrics['recall']} < {self.config.min_recall} (screening tools need high recall)")
                if test_metrics["f1_score"] < self.config.min_f1:
                    gate_failures.append(f"f1_score {test_metrics['f1_score']} < {self.config.min_f1}")

                score_gap = train_metrics[self.config.target_metric] - test_metrics[self.config.target_metric]
                if score_gap > self.config.overfitting_threshold:
                    gate_failures.append(
                        f"overfitting: train {self.config.target_metric} ({train_metrics[self.config.target_metric]}) "
                        f"exceeds test ({test_metrics[self.config.target_metric]}) by {score_gap:.4f}, "
                        f"over the {self.config.overfitting_threshold} threshold"
                    )

                mlflow.set_tag("quality_gate_passed", str(len(gate_failures) == 0))

                if gate_failures:
                    mlflow.set_tag("quality_gate_failures", "; ".join(gate_failures))
                    raise CustomException(f"Model failed quality gates: {gate_failures}", sys)

                # ---- Bundle the fitted preprocessor WITH the classifier as
                # one Pipeline before this gets logged/saved anywhere.
                #
                # Without this, /predict in app/main.py would receive raw
                # feature values and hand them straight to the classifier,
                # never applying the SimpleImputer that DataTransformation
                # fit — training and serving would silently use different
                # preprocessing the moment that imputer does anything beyond
                # a no-op (which it currently is, only because this specific
                # dataset has zero missing values). Bundling them into one
                # artifact means there is no second file to keep in sync —
                # `pipeline.predict_proba(raw_input)` always applies the
                # exact preprocessing fit during training, by construction.
                preprocessor = load_object(self.data_transformation_artifact.preprocessor_object_file_path)
                serving_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", final_model),
                ])

                signature = infer_signature(X_train, serving_pipeline.predict(X_train))
                mlflow.sklearn.log_model(
                    serving_pipeline,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train[:5],
                    registered_model_name=self.config.mlflow_registered_model_name,
                )
                run_id = parent_run.info.run_id

            # model_path = f"{self.config.trainer_dir}/{self.config.trained_model_file_name}"
            model_dir = self.config.trainer_dir
            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(model_dir, "model.joblib")
            metadata_path = os.path.join(model_dir, "metadata.json")

            joblib.dump(serving_pipeline, model_path)

            metadata = {
                "threshold": float(threshold),
                "model_type": model_type,
                "feature_columns": self.data_transformation_artifact.feature_columns,
                "mlflow_run_id": run_id,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            logging.info(f"Model training completed — algorithm: {model_type}, threshold: {threshold:.3f}")

            return ModelTrainerArtifact(
                trained_model_file_path=model_path,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                mlflow_run_id=run_id,
                best_hyperparameters=final_params,
                model_type=model_type,
                decision_threshold=threshold,
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model training: {e}", sys)