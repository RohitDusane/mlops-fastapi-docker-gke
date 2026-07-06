from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    valid_train_file_path: Optional[str] = None
    invalid_train_file_path: Optional[str] = None
    valid_test_file_path: Optional[str] = None
    invalid_test_file_path: Optional[str] = None
    drift_report_file_path: Optional[str] = None


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metrics: dict
    test_metrics: dict
    mlflow_run_id: str
    best_hyperparameters: dict


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_score: float
    trained_model_metrics: dict
    best_model_metrics: Optional[dict]
    trained_model_file_path: str


@dataclass
class ModelPusherArtifact:
    is_model_pushed: bool
    mlflow_model_version: Optional[str]
    saved_model_path: str