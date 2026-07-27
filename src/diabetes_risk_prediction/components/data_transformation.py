import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from diabetes_risk_prediction.entity.config_entity import DataTransformationConfig
from diabetes_risk_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging
from diabetes_risk_prediction.utils.common import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.config = data_transformation_config
        except Exception as e:
            raise CustomException(f"Error initializing DataTransformation: {e}", sys)

    def get_preprocessor(self) -> Pipeline:
        """
        Median imputation only — RandomForest doesn't require feature
        scaling, but the imputer is still necessary since the UCI source
        documents no missing values today; that could change if the data
        source is swapped later, and this is what would actually catch it.
        """
        try:
            return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        except Exception as e:
            raise CustomException(f"Error building preprocessor: {e}", sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise CustomException(f"Cannot transform data — validation failed: {self.data_validation_artifact.message}", sys,)

            logging.info("Starting data transformation.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            target_col = self.config.target_column
            feature_cols = [c for c in train_df.columns if c != target_col]

            X_train, y_train = train_df[feature_cols], train_df[target_col]
            X_test, y_test = test_df[feature_cols], test_df[target_col]

            preprocessor = self.get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            train_path = f"{self.config.transformation_dir}/transformed_train.npy"
            test_path = f"{self.config.transformation_dir}/transformed_test.npy"
            preprocessor_path = f"{self.config.transformation_dir}/{self.config.preprocessor_file_name}"

            save_numpy_array_data(train_path, train_arr)
            save_numpy_array_data(test_path, test_arr)
            save_object(preprocessor_path, preprocessor)

            logging.info("Data transformation completed.")

            return DataTransformationArtifact(
                transformed_train_file_path=train_path,
                transformed_test_file_path=test_path,
                preprocessor_object_file_path=preprocessor_path,
                feature_columns=feature_cols,
            )
        except Exception as e:
            raise CustomException(f"Error during data transformation: {e}", sys)