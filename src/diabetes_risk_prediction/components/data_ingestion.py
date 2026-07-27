import os
import sys
import pandas as pd
import numpy as np
# from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from diabetes_risk_prediction.entity.artifact_entity import DataIngestionArtifact
from diabetes_risk_prediction.entity.config_entity import DataIngestionConfig

from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging
from pathlib import Path

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise CustomException(f"Error initializing DataIngestion: {e}", sys)

    def download_data(self):
        try:
            logging.info("🔥 USING UPDATED DATA INGESTION FILE")
            if not os.path.exists(self.config.raw_data_path):
                raise CustomException(f"File not found: {self.config.raw_data_path}", sys)

            df = pd.read_csv(self.config.raw_data_path)

            logging.info(f"RAW unique values: {df['Diabetes_binary'].unique()}")
            logging.info(f"RAW distribution:\n{df['Diabetes_binary'].value_counts()}")

            # Convert to binary: 0,1 → 0 and 2 → 1
            df["Diabetes_binary"] = (df["Diabetes_binary"]).astype(int)

            vc = df["Diabetes_binary"].value_counts().sort_index()
            logging.info(f"Target distribution:\n{vc}")

            # SAFE CHECK (keep only this)
            if vc.shape[0] != 2:
                raise CustomException(
                    f"Invalid target distribution after mapping: {vc.to_dict()}",
                    sys
                )

            # Optional stronger check (recommended in production)
            if min(vc.values) < 100:  # prevent extreme imbalance bugs
                logging.warning("Severe class imbalance detected")

            logging.info(f"Data shape: {df.shape}")
            logging.info("Data loaded successfully.")

            return df

        except Exception as e:
            raise CustomException(f"Error downloading data: {e}", sys)

    def split_data(self, data: pd.DataFrame) -> DataIngestionArtifact:
        try:
            os.makedirs(self.config.ingestion_dir, exist_ok=True)

            # SAVE RAW
            # raw_path = os.path.join(self.config.ingestion_dir, "raw.csv")
            # data.to_csv(raw_path, index=False)

            raw_path = Path(self.config.ingestion_dir) / "raw.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(raw_path, index=False)
            
            # Split the data into train and test sets
            train_data, test_data = train_test_split(
                data,
                test_size=self.config.train_test_split_ratio,
                random_state=self.config.random_state,
            )
            logging.info("Data split into train and test sets successfully.")

            # Save the train and test sets to CSV files
            train_path = os.path.join(self.config.ingestion_dir, "train.csv")
            test_path = os.path.join(self.config.ingestion_dir, "test.csv")

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            logging.info(f"Train saved at: {train_path}")
            logging.info(f"Test saved at: {test_path}")
            logging.info("Data Ingestion.")

            return DataIngestionArtifact(train_file_path=train_path, test_file_path=test_path)
        except Exception as e:
            raise CustomException(f"Error splitting data: {e}", sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process.")

            # Download the data
            data = self.download_data()
            if os.path.getsize(self.config.raw_data_path) == 0:
                raise CustomException("Empty CSV file", sys)

            # Split the data into train and test sets
            artifact = self.split_data(data)

            return artifact
        except Exception as e:
            raise CustomException(f"Error in data ingestion: {e}", sys)