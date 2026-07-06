from dataclasses import dataclass, field
from datetime import datetime


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


@dataclass
class ModelTrainerConfig:
    trainer_dir: str = "artifacts/model_trainer"
    trained_model_file_name: str = "model.pkl"
    n_trials: int = 15                     # Optuna trials
    cv_folds: int = 3
    target_metric: str = "roc_auc"
    expected_score: float = 0.70           # minimum acceptable roc_auc — below this, training fails loudly
    overfitting_threshold: float = 0.10    # max allowed (train_score - test_score) gap
    # mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "diabetes-risk-screening"
    mlflow_registered_model_name: str = "diabetes-risk-model"


@dataclass
class ModelEvaluationConfig:
    evaluation_dir: str = "artifacts/model_evaluation"
    changed_threshold_score: float = 0.01  # new model must beat current Production by at least this much
    # mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_registered_model_name: str = "diabetes-risk-model"


@dataclass
class ModelPusherConfig:
    saved_model_dir: str = "saved_models"
    # mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_registered_model_name: str = "diabetes-risk-model"