from dataclasses import dataclass, field
from datetime import datetime
import yaml

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
params = yaml.safe_load(open("config/params.yaml"))
@dataclass
class ModelTrainerConfig:
    # trainer_dir: str = "artifacts/model_trainer"
    # trained_model_file_name: str = "model.pkl"
    trainer_dir=params["artifacts"]["trainer_dir"],
    n_trials=params["model_training"]["optuna"]["n_trials"],
    cv_folds=params["model_training"]["optuna"]["cv_folds"],
    random_state: int = 24

    # ---- Search space: algorithm family is itself an Optuna dimension,
    # not a hardcoded model class — this is what actually makes RF vs LGBM
    # (vs anything else added later) a genuine comparison in one study,
    # rather than dead alternate code that's never instantiated.
    candidate_algorithms: tuple = ("random_forest", "lightgbm", "logistic_regression")
    n_trials: int = 15
    cv_folds: int = 5
    cv_metric: str = "average_precision"   # PR-AUC — more informative than ROC-AUC under class imbalance
    enable_pruning: bool = True

    # ---- Threshold tuning (decision threshold ≠ 0.5 for imbalanced screening) ----
    target_recall_floor: float = 0.80      # tune threshold to the highest cutoff that still meets this recall
    calibrate_probabilities: bool = False   # set True to wrap the final model in CalibratedClassifierCV first

    # ---- Clinical quality gates (config-driven — no magic numbers in code) ----
    min_roc_auc: float = 0.80
    min_recall: float = 0.70               # false negatives are the costly error in a screening tool
    min_f1: float = 0.40
    overfitting_threshold: float = 0.10    # max allowed (train - test) gap on the primary metric

    # Primary metric used for the overfitting gap check and for
    # ModelEvaluation's champion-vs-challenger comparison downstream.
    target_metric: str = "recall"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "diabetes-risk-screening"
    mlflow_registered_model_name: str = "diabetes-risk-model"


@dataclass
class ModelEvaluationConfig:
    evaluation_dir: str = "artifacts/model_evaluation"
    changed_threshold_score: float = 0.01  # new model must beat current Production by at least this much
    # Guards against a composite score improving purely by recall gains
    # while precision quietly collapses (e.g. a near-always-positive model
    # scores high recall trivially) — reject the challenger if precision
    # regresses by more than this fraction versus current Production, even
    # if the weighted score looks better overall.
    max_precision_regression: float = 0.10
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_registered_model_name: str = "diabetes-risk-model"


@dataclass
class ModelPusherConfig:
    saved_model_dir: str = "saved_models"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_registered_model_name: str = "diabetes-risk-model"