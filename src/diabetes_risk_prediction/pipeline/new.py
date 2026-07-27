from diabetes_risk_prediction.components.model_pusher import ModelPusher
from diabetes_risk_prediction.entity.config_entity import ModelPusherConfig
from diabetes_risk_prediction.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

trainer_artifact = ModelTrainerArtifact(
    mlflow_run_id="9f59955da6f54d4bb5c067b36885a272",
    train_metrics={},
    test_metrics={},
    best_hyperparameters={},
    model_type="lightgbm",
    decision_threshold=0.4791008620714077,
)

evaluation_artifact = ModelEvaluationArtifact(
    is_model_accepted=True,
    improved_score=0.7269,
    trained_model_metrics={},
    best_model_metrics=None,
    mlflow_run_id="9f59955da6f54d4bb5c067b36885a272",
)

config = ModelPusherConfig()

pusher = ModelPusher(
    trainer_artifact,
    evaluation_artifact,
    config,
)

print(pusher.initiate_model_pusher())