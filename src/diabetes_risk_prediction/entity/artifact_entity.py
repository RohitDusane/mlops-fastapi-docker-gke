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
    feature_columns: list


@dataclass
class ModelBundle:
    """
    What actually gets pickled and served — not just the raw estimator.
    Bundling the decision threshold with the model prevents the two from
    drifting apart (e.g. someone redeploying a new model.pkl while the
    serving code still hardcodes 0.5). Anything loading this artifact
    should read `bundle.threshold`, not assume 0.5.
    """
    model: object
    threshold: float
    model_type: str
    feature_columns: list
    mlflow_run_id: str


@dataclass
class ModelTrainerArtifact:
    # trained_model_file_path: str
    mlflow_run_id: str
    train_metrics: dict
    test_metrics: dict
    best_hyperparameters: dict
    model_type: str
    decision_threshold: float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_score: float
    trained_model_metrics: dict
    best_model_metrics: Optional[dict]
    # trained_model_file_path: str
    mlflow_run_id: str

@dataclass
class ModelPusherArtifact:
    is_model_pushed: bool
    mlflow_model_version: Optional[str]
    # saved_model_path: str