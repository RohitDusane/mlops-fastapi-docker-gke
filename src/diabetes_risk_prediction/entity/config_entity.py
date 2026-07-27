from dataclasses import dataclass, field
from datetime import datetime
import yaml
import os
from dotenv import load_dotenv

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/raw/raw_data.csv"
    train_test_split_ratio: float = 0.2
    random_state: int = 24
    ingestion_dir: str = "artifacts/ingestion"

@dataclass
class DataValidationConfig:
    schema_path: str = "configs/schema.yaml"
    drift_threshold: float = 0.05
    missing_value_threshold: float = 0.2
    drift_report_dir: str = "artifacts/data_validation"

@dataclass
class DataTransformationConfig:
    transformation_dir: str = "artifacts/data_transformation"
    preprocessor_file_name: str = "preprocessor.pkl"
    target_column: str = "Diabetes_binary"

# load params
# params = yaml.safe_load(open("configs/params.yaml"))
load_dotenv()

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)


def resolve_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value

@dataclass
class ModelTrainerConfig:
    trainer_dir: str = params["artifacts"]["trainer_dir"]
    trained_model_file_name: str = "model.pkl"
    random_state: int = 24
    evaluation_dir: str = params["artifacts"]["evaluation_dir"]

    candidate_algorithms: tuple = field(default_factory=lambda: tuple(params["model_training"]["candidate_algorithms"]))
    n_trials: int = params["model_training"]["n_trials"]
    cv_folds: int = params["model_training"]["cv_folds"]
    cv_metric: str = params["model_training"]["cv_metric"]
    enable_pruning: bool = params["model_training"]["enable_pruning"]
    optuna_n_jobs: int = params["model_training"]["optuna_n_jobs"]
    optuna_storage_path: str | None = None

    calibrate_probabilities: bool = params["model_training"]["calibrate_probabilities"]
    target_recall_floor: float = params["model_training"]["target_recall_floor"]

    min_roc_auc: float = params["quality_gate"]["min_roc_auc"]
    min_recall: float = params["quality_gate"]["min_recall"]
    min_f1: float = params["quality_gate"]["min_f1"]
    overfitting_threshold: float = params["quality_gate"]["overfitting_threshold"]
    target_metric: str = params["quality_gate"]["target_metric"]

    mlflow_tracking_uri: str = resolve_env(params["mlflow"]["tracking_uri"])
    mlflow_experiment_name: str = params["mlflow"]["experiment_name"]
    mlflow_registered_model_name: str = params["mlflow"]["registered_model_name"]
    mlflow_model_alias: str = params["mlflow"]["model_alias"]

# @dataclass
# class ModelTrainerConfig:
#     # trainer_dir: str = "artifacts/model_trainer"
#     # trained_model_file_name: str = "model.pkl"
#     trainer_dir=params["artifacts"]["trainer_dir"],
#     n_trials=params["model_training"]["optuna"]["n_trials"],
#     cv_folds=params["model_training"]["optuna"]["cv_folds"],
#     random_state: int = 24

#     # ---- Search space: algorithm family is itself an Optuna dimension,
#     # not a hardcoded model class — this is what actually makes RF vs LGBM
#     # (vs anything else added later) a genuine comparison in one study,
#     # rather than dead alternate code that's never instantiated.
#     candidate_algorithms: tuple = ("random_forest", "lightgbm", "logistic_regression")
#     n_trials: int = 15
#     cv_folds: int = 5
#     cv_metric: str = "average_precision"   # PR-AUC — more informative than ROC-AUC under class imbalance
#     enable_pruning: bool = True

#     # ---- Threshold tuning (decision threshold ≠ 0.5 for imbalanced screening) ----
#     target_recall_floor: float = 0.80      # tune threshold to the highest cutoff that still meets this recall
#     calibrate_probabilities: bool = False   # set True to wrap the final model in CalibratedClassifierCV first

#     # ---- Clinical quality gates (config-driven — no magic numbers in code) ----
#     min_roc_auc: float = 0.80
#     min_recall: float = 0.70               # false negatives are the costly error in a screening tool
#     min_f1: float = 0.40
#     overfitting_threshold: float = 0.10    # max allowed (train - test) gap on the primary metric

#     # Primary metric used for the overfitting gap check and for
#     # ModelEvaluation's champion-vs-challenger comparison downstream.
#     target_metric: str = "recall"

#     mlflow_tracking_uri: str = "http://localhost:5000"
#     mlflow_experiment_name: str = "diabetes-risk-screening"
#     mlflow_registered_model_name: str = "diabetes-risk-model"


@dataclass
class ModelEvaluationConfig:
    evaluation_dir: str = params["artifacts"]["evaluation_dir"]
    changed_threshold_score: float = params["evaluation"]["changed_threshold_score"]
    max_precision_regression: float = params["evaluation"]["max_precision_regression"]

    mlflow_tracking_uri: str = resolve_env(params["mlflow"]["tracking_uri"])
    mlflow_experiment_name: str = params["mlflow"]["experiment_name"]
    mlflow_registered_model_name: str = params["mlflow"]["registered_model_name"]
    mlflow_model_alias: str = params["mlflow"]["model_alias"]


# @dataclass
# class ModelPusherConfig:
#     saved_model_dir: str = "saved_models"
#     mlflow_tracking_uri: str = "http://localhost:5000"
#     mlflow_registered_model_name: str = "diabetes-risk-model"
#     mlflow_model_alias: str = "champion"

@dataclass
class ModelPusherConfig:
    mlflow_tracking_uri: str = resolve_env(params["mlflow"]["tracking_uri"])
    mlflow_registered_model_name: str = params["mlflow"]["registered_model_name"]
    mlflow_model_alias: str = params["mlflow"]["model_alias"]
    