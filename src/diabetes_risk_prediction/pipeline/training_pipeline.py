"""
diabetes_risk_prediction/pipeline/training_pipeline.py

End-to-end orchestration:
  Data Ingestion -> Data Validation -> Data Transformation ->
  Model Trainer -> Model Evaluation -> Model Pusher

Run:
    python -m diabetes_risk_prediction.pipeline.training_pipeline
"""
import sys

from diabetes_risk_prediction.components.data_ingestion import DataIngestion
from diabetes_risk_prediction.components.data_validation import DataValidation
from diabetes_risk_prediction.components.data_transformation import DataTransformation
from diabetes_risk_prediction.components.model_trainer import ModelTrainer
from diabetes_risk_prediction.components.model_evaluation import ModelEvaluation
from diabetes_risk_prediction.components.model_pusher import ModelPusher

from diabetes_risk_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self):
        logging.info(">>>>> Stage 1: Data Ingestion <<<<<")
        ingestion = DataIngestion(self.data_ingestion_config)
        artifact = ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion artifact: {artifact}")
        return artifact

    def start_data_validation(self, ingestion_artifact):
        logging.info(">>>>> Stage 2: Data Validation <<<<<")
        validation = DataValidation(ingestion_artifact, self.data_validation_config)
        artifact = validation.initiate_data_validation()
        logging.info(f"Data validation artifact: {artifact}")
        return artifact

    def start_data_transformation(self, validation_artifact):
        logging.info(">>>>> Stage 3: Data Transformation <<<<<")
        transformation = DataTransformation(validation_artifact, self.data_transformation_config)
        artifact = transformation.initiate_data_transformation()
        logging.info(f"Data transformation artifact: {artifact}")
        return artifact

    def start_model_trainer(self, transformation_artifact):
        logging.info(">>>>> Stage 4: Model Training (Optuna + MLflow) <<<<<")
        trainer = ModelTrainer(transformation_artifact, self.model_trainer_config)
        artifact = trainer.initiate_model_trainer()
        logging.info(f"Model trainer artifact: {artifact}")
        return artifact

    def start_model_evaluation(self, trainer_artifact):
        logging.info(">>>>> Stage 5: Model Evaluation <<<<<")
        evaluation = ModelEvaluation(trainer_artifact, self.model_evaluation_config)
        artifact = evaluation.initiate_model_evaluation()
        logging.info(f"Model evaluation artifact: {artifact}")
        return artifact

    def start_model_pusher(self, trainer_artifact, evaluation_artifact):
        logging.info(">>>>> Stage 6: Model Pusher <<<<<")
        pusher = ModelPusher(trainer_artifact, evaluation_artifact, self.model_pusher_config)
        artifact = pusher.initiate_model_pusher()
        logging.info(f"Model pusher artifact: {artifact}")
        return artifact

    def run_pipeline(self):
        try:
            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)

            if not validation_artifact.validation_status:
                raise ValueError(validation_artifact.message)

            transformation_artifact = self.start_data_transformation(validation_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            evaluation_artifact = self.start_model_evaluation(trainer_artifact)
            pusher_artifact = self.start_model_pusher(trainer_artifact, evaluation_artifact)

            logging.info(">>>>> Pipeline completed successfully <<<<<")
            return pusher_artifact
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Pipeline failed: {e}", sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run_pipeline()
    print(result)