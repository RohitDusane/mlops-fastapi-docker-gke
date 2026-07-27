import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, train_test_split)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    
)

from lightgbm import LGBMClassifier

from diabetes_risk_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from diabetes_risk_prediction.entity.config_entity import (ModelTrainerConfig)
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging

from diabetes_risk_prediction.utils.common import (
    load_numpy_array_data,
    load_object
)

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
        tn,fp,fn,tp = confusion_matrix(y, y_pred).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall_value = round(recall_score(y, y_pred), 4)

        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": recall_value,
            "sensitivity": recall_value,
            "specificity": round(specificity, 4),
            "f1_score": round(f1_score(y, y_pred), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
            "pr_auc": round(average_precision_score(y, y_proba), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        }

    # ---------------- Threshold tuning ---------------- #

    def _threshold_from_probabilities(self, y_true, y_proba) -> float:
        """
        Shared core: picks the highest decision threshold that still keeps
        recall at or above `target_recall_floor` — maximizes precision
        subject to a recall floor, since in a screening context a false
        negative (missed at-risk patient) is worse than a false positive.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        valid = np.where(recall[:-1] >= self.config.target_recall_floor)[0]
        if len(valid):
            return float(thresholds[valid[-1]])
        logging.warning(
            f"No threshold achieves the target recall floor "
            f"({self.config.target_recall_floor}) — falling back to 0.5"
        )
        return 0.5
 
    def optimize_threshold(self, model, X_val, y_val) -> float:
        """Threshold tuning against a single held-out split (a model must already be fit)."""
        y_proba = model.predict_proba(X_val)[:, 1]
        return self._threshold_from_probabilities(y_val, y_proba)
 
    def optimize_threshold_from_probs(self, y_true, y_proba) -> float:
        """
        CHANGE: this method was being called but never defined — guaranteed
        AttributeError on every run. Added as a thin wrapper so the
        out-of-fold cross_val_predict approach (more rigorous than a single
        train/val split — this closes the exact gap the original
        docstring flagged as "worth upgrading") actually works.
        """
        return self._threshold_from_probabilities(y_true, y_proba)

    # ---------------- Multi-algorithm search space ---------------- #

    def _suggest_hyperparams(self, trial, model_type: str, scale_pos_weight: float) -> dict:
        if model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("rf_n_estimators", 100, 400, step=25),
                "max_depth": trial.suggest_int("rf_max_depth", 4, 20),
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
                "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 700, step=50),
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
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "force_col_wise": True,
                "random_state": self.config.random_state,
                "n_jobs": 1,
                "verbosity": -1,
            }
        if model_type == "logistic_regression":
            # Fast, interpretable baseline — useful as a sanity floor: if a
            # tuned RF/LGBM can't beat simple logistic regression by much,
            # that's worth knowing, not just worth beating.
            return {
                "C": trial.suggest_float("logreg_C", 0.001, 10.0, log=True),
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

        model_type = trial.suggest_categorical("model_type", self.config.candidate_algorithms)
        params = self._suggest_hyperparams(trial, model_type, scale_pos_weight)
        model = self._build_model(model_type, params)

        skf = StratifiedKFold(
            n_splits=self.config.cv_folds, 
            shuffle=True, 
            random_state=self.config.random_state)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]

            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_proba = model.predict_proba(X_val_fold)[:, 1]

            score = (
                average_precision_score(y_val_fold, y_proba)
                if self.config.cv_metric == "average_precision"
                else roc_auc_score(y_val_fold, y_proba)
            )
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
            mlflow.log_param("algorithm", model_type)
            mlflow.set_tag("model_type", model_type)
            mlflow.log_params(params)
            mlflow.log_metric("cv_fold_min", float(np.min(fold_scores)))
            mlflow.log_metric("cv_fold_max", float(np.max(fold_scores)))
            mlflow.log_metric(f"cv_{self.config.cv_metric}_mean", mean_score)
            mlflow.log_metric(f"cv_{self.config.cv_metric}_std", float(np.std(fold_scores)))

        return mean_score
    
    def _get_feature_importance_source(self, model):
        """
        CHANGE: fixed. When calibrate_probabilities=True, `final_model` is a
        CalibratedClassifierCV — its `.estimator` attribute after fit() is
        the original UNFITTED prototype (sklearn keeps the constructor arg
        as-is; the actually-fitted base estimators live inside
        `calibrated_classifiers_[i].estimator`). Calling
        `.feature_importances_` on the unfitted prototype raises
        NotFittedError. Previously dormant since calibrate_probabilities
        defaults to False, but a real, guaranteed break the moment someone
        flips that flag on.
        """
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            try:
                return model.calibrated_classifiers_[0].estimator
            except Exception:
                return None
        return model

    # ---------------- Main entrypoint ---------------- #

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training.")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            # X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            feature_names = self.data_transformation_artifact.feature_columns

            X_train = pd.DataFrame(train_arr[:, :-1], columns=feature_names)
            X_test = pd.DataFrame(test_arr[:, :-1], columns=feature_names)
            y_train = train_arr[:, -1].astype(int)
            y_test = test_arr[:, -1].astype(int)

            neg, pos = np.bincount(y_train.astype(int))
            scale_pos_weight = neg / max(pos, 1)
            logging.info(f"Class balance — negative: {neg}, positive: {pos}, scale_pos_weight: {scale_pos_weight:.3f}")

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            # CHANGE: verbosity now set BEFORE optimize() — previously set
            # after study.optimize() completed, so it had zero effect on the
            # actual search's log spam, only on anything logged afterward.
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            with mlflow.start_run(run_name="model-training") as parent_run:
                # Hyperparameter search
                pruner = optuna.pruners.MedianPruner() if self.config.enable_pruning else optuna.pruners.NopPruner()
                # study = optuna.create_study(direction="maximize", pruner=pruner)
                
                if self.config.optuna_storage_path:
                    os.makedirs(os.path.dirname(self.config.optuna_storage_path) or ".", exist_ok=True)
                    study = optuna.create_study(
                        study_name="diabetes-risk-model",
                        storage=f"sqlite:///{self.config.optuna_storage_path}",
                        load_if_exists=True,
                        direction="maximize",
                        pruner=pruner,
                    )
                else:
                    study = optuna.create_study(direction="maximize", pruner=pruner)  # in-memory, fresh every run
                
                study.optimize(
                    lambda trial: self._objective(trial, X_train, y_train, scale_pos_weight),
                    n_trials=self.config.n_trials,
                    n_jobs=self.config.optuna_n_jobs,
                )

                best_params = dict(study.best_params.copy())
                model_type = best_params.pop("model_type")
                # Remaining keys are prefixed (rf_/lgbm_/logreg_) to stay
                # unique within one Optuna search space — strip the prefix
                # before passing to the estimator constructor.
                prefix = {"random_forest": "rf_", "lightgbm": "lgbm_", "logistic_regression": "logreg_"}[model_type]
                final_params = {k[len(prefix):]: v for k, v in best_params.items() if k.startswith(prefix)}
                if model_type == "lightgbm":
                    final_params["scale_pos_weight"] = scale_pos_weight
                    final_params.update({
                        "objective": "binary",
                        "metric": "auc",
                        "boosting_type": "gbdt",
                        "force_col_wise": True,
                        "verbosity": -1,
                        "n_jobs": -1,
                    })
                # elif model_type in ("random_forest", "logistic_regression"):
                #     final_params["class_weight"] = "balanced"
                #     final_params.update({"n_jobs": -1})
                elif model_type == "random_forest":
                    final_params.update({
                        "class_weight": "balanced",
                        "n_jobs": -1,
                    })
                elif model_type == "logistic_regression":
                    final_params.update({"class_weight": "balanced",})

                final_params["random_state"] = self.config.random_state

                logging.info(f"Best algorithm: {model_type} | Best CV {self.config.cv_metric}: {study.best_value:.4f}")
                logging.info(f"Best hyperparameters: {final_params}")

                # final_model = self._build_model(model_type, {**final_params, "n_jobs": -1} if model_type != "logistic_regression" else final_params)
                final_model = self._build_model(model_type, final_params)

                if self.config.calibrate_probabilities:
                    from sklearn.calibration import CalibratedClassifierCV
                    logging.info("Calibrating probabilities (sigmoid, cv=5).")
                    final_model = CalibratedClassifierCV(estimator=final_model, method="sigmoid", cv=5, n_jobs=1)
 
                # Out-of-fold probabilities for threshold tuning — more
                # rigorous than a single train/val split, since every
                # training example contributes to exactly one out-of-fold
                # prediction rather than only 20% of the data ever being
                # "held out."
                oof_probabilities = cross_val_predict(
                    final_model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state),
                    method="predict_proba", n_jobs=1,
                )[:, 1]
 
                threshold = self.optimize_threshold_from_probs(y_train, oof_probabilities)  # CHANGE: now a real method
 
                # Final fit on the full training set for the artifact that actually ships.
                final_model.fit(X_train, y_train)
 
                train_metrics = self._compute_metrics(final_model, X_train, y_train, threshold)
                test_metrics = self._compute_metrics(final_model, X_test, y_test, threshold)
                logging.info(f"Train metrics: {train_metrics}")
                logging.info(f"Test metrics: {test_metrics}")

                # Save metrics for DVC tracking
                os.makedirs(self.config.trainer_dir, exist_ok=True)

                train_metrics_path = os.path.join(self.config.trainer_dir, "train_metrics.json")
                test_metrics_path = os.path.join(self.config.trainer_dir, "test_metrics.json")

                with open(train_metrics_path, "w") as f:
                    json.dump(train_metrics, f, indent=4)

                with open(test_metrics_path, "w") as f:
                    json.dump(test_metrics, f, indent=4)

                logging.info(f"Train metrics saved: {train_metrics_path}")
                logging.info(f"Test metrics saved: {test_metrics_path}")

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
                metadata = {
                    "model_type": model_type,
                    "threshold": float(threshold),
                    "feature_columns": self.data_transformation_artifact.feature_columns,
                }

                explain_source = self._get_feature_importance_source(final_model)  # CHANGE: fixed unwrapping
                importances = None
                if explain_source is not None:
                    if hasattr(explain_source, "feature_importances_"):
                        importances = explain_source.feature_importances_.tolist()
                    elif hasattr(explain_source, "coef_"):
                        importances = explain_source.coef_[0].tolist()
                if importances is not None:
                    mlflow.log_dict({"feature_importances": importances}, "feature_importance.json")
                    # CHANGE: also written locally so it's a real DVC output,
                    # not only an MLflow artifact.
                    fi_path = os.path.join(self.config.trainer_dir, "feature_importance.json")
                    os.makedirs(self.config.trainer_dir, exist_ok=True)
                    with open(fi_path, "w") as f:
                        json.dump({"feature_importances": importances, "feature_columns": feature_names}, f, indent=4)

                # ---- Confusion matrix at the tuned threshold, for clinical review ----
                os.makedirs(self.config.evaluation_dir, exist_ok=True)
                probs = final_model.predict_proba(X_test)[:,1]
                preds = (probs >= threshold).astype(int)
                cm = confusion_matrix(y_test, preds)
                confusion_path = os.path.join(self.config.evaluation_dir, "confusion_matrix.json")

                with open(confusion_path,"w") as f:
                    json.dump({
                        "TN": int(cm[0,0]),
                        "FP": int(cm[0,1]),
                        "FN": int(cm[1,0]),
                        "TP": int(cm[1,1]),
                        "threshold": float(threshold)
                    }, f, indent=4)
                mlflow.log_dict({"confusion_matrix": cm.tolist(), "threshold": threshold}, "confusion_matrix.json")

                # ---- Clinical quality gates — config-driven, not magic numbers ----
                # ---- Clinical quality gates ---- #
                gate_failures = []
                if test_metrics["roc_auc"] < self.config.min_roc_auc:
                    gate_failures.append(f"roc_auc {test_metrics['roc_auc']} < {self.config.min_roc_auc}")
                if test_metrics["recall"] < self.config.min_recall:  # CHANGE: "recall" key now exists again — this no longer KeyErrors
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
 
                preprocessor = load_object(self.data_transformation_artifact.preprocessor_object_file_path)
                serving_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", final_model)])
 
                signature = infer_signature(X_train, serving_pipeline.predict_proba(X_train.head()))
 
                metadata_path = os.path.join(self.config.trainer_dir, "model_metadata.json")
                os.makedirs(self.config.trainer_dir, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)

                mlflow.sklearn.log_model(
                    serving_pipeline,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.head(5),
                    metadata=metadata,
                )
                mlflow.log_artifact(metadata_path, artifact_path="model",)
 
                # # Local copy for DVC tracking / offline inspection — separate
                # # concern from the MLflow-side metadata= mechanism above.
                # metadata_path = os.path.join(self.config.trainer_dir, "model_metadata.json")
                # os.makedirs(self.config.trainer_dir, exist_ok=True)
                # with open(metadata_path, "w") as f:
                #     json.dump(metadata, f, indent=4)
 
                model_uri = f"runs:/{parent_run.info.run_id}/model"
                registered_model = mlflow.register_model(model_uri=model_uri, name=self.config.mlflow_registered_model_name)

                logging.info(
                    f"Registered model: {registered_model.name} "
                    f"Version: {registered_model.version}"
                )
                                
                # CHANGE (confirmed intentional, not a bug): no alias is set
                # here. Setting @champion at training time would bypass
                # ModelEvaluation's quality gate + ModelPusher's precision
                # guard entirely — every trained model would become champion
                # regardless of whether it's actually better. ModelPusher is
                # correctly the only place that assigns the alias, and only
                # after ModelEvaluation approves it.
 
                run_id = parent_run.info.run_id
                mlflow.set_tag("task", "diabetes-risk-classification")
                mlflow.set_tag("framework", "scikit-learn")
                mlflow.set_tag("deployment_ready", "true")
 
            logging.info(
                f"Model training completed | algorithm={model_type} | "
                f"threshold={threshold:.3f} | run_id={run_id}"
            )
 
            return ModelTrainerArtifact(
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                mlflow_run_id=run_id,
                best_hyperparameters=final_params,
                model_type=model_type,
                decision_threshold=threshold,
                registered_model_name=registered_model.name,
                model_version=str(registered_model.version),
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Error during model training: {e}", sys)