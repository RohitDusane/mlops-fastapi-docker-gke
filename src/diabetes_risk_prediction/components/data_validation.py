import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from diabetes_risk_prediction.entity.config_entity import DataValidationConfig
from diabetes_risk_prediction.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from diabetes_risk_prediction.exception.custom_exception import CustomException
from diabetes_risk_prediction.logger.logging import logging
from diabetes_risk_prediction.utils.common import read_yaml_file, save_json_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            logging.info("Initializing DataValidation class.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(self.data_validation_config.schema_path)
        except Exception as e:
            raise CustomException(f"Error initializing DataValidation: {e}", sys)

    # ---------------- BASIC DATA CHECKS ---------------- #
    def basic_checks(self, df: pd.DataFrame, schema: dict) -> list:
        try:
            logging.info("Running basic schema checks (columns, dtypes, missing values).")
            errors = []

            expected_columns = list(schema["columns"].keys())
            actual_columns = list(df.columns)

            missing_cols = [c for c in expected_columns if c not in actual_columns]
            extra_cols = [c for c in actual_columns if c not in expected_columns]

            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
            if extra_cols:
                errors.append(f"Extra/unexpected columns: {extra_cols}")

            missing_ratio = df.isnull().mean()
            high_missing = missing_ratio[missing_ratio > self.data_validation_config.missing_value_threshold]
            if len(high_missing) > 0:
                errors.append(f"Columns exceeding missing-value threshold: {high_missing.index.tolist()}")

            return errors
        except Exception as e:
            raise CustomException(f"Error during basic checks: {e}", sys)

    # ---------------- SCHEMA RULE VALIDATION (range/domain) ---------------- #
    def validate_schema_rules(self, df: pd.DataFrame, schema: dict) -> list:
        try:
            errors = []
            for col, rules in schema["columns"].items():
                if col not in df.columns:
                    continue  # already flagged by basic_checks

                if rules.get("dtype") == "int":
                    # Soft check: pandas often reads whole-number columns as
                    # float64 if any NaNs are present, so check value-level
                    # integrality rather than the raw pandas dtype.
                    non_null = df[col].dropna()
                    if not (non_null % 1 == 0).all():
                        errors.append(f"'{col}' expected integer values, found non-integer entries")

                if "range" in rules:
                    min_v, max_v = rules["range"]
                    if df[col].min() < min_v or df[col].max() > max_v:
                        errors.append(f"'{col}' out of expected range {rules['range']}")

                if "domain" in rules:
                    invalid = ~df[col].dropna().isin(rules["domain"])
                    if invalid.any():
                        errors.append(f"'{col}' has values outside allowed domain {rules['domain']}")

            return errors
        except Exception as e:
            raise CustomException(f"Error during schema rule validation: {e}", sys)

    # ---------------- DRIFT DETECTION (KS TEST) ---------------- #
    def detect_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame, schema: dict) -> dict:
        try:
            drift_report = {}
            for col in schema["columns"].keys():
                if col not in train_df.columns or col not in test_df.columns:
                    continue
                if train_df[col].dtype == "object":
                    continue

                stat, p_value = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
                drift_report[col] = {
                    "ks_stat": float(stat),
                    "p_value": float(p_value),
                    # Bug fix: was `self.config.drift_threshold` — self.config
                    # doesn't exist on this class, it's self.data_validation_config
                    "drift_detected": bool(p_value < self.data_validation_config.drift_threshold),
                }
            return drift_report
        except Exception as e:
            raise CustomException(f"Error during drift detection: {e}", sys)

    # ---------------- PSI SCORE (OPTIONAL, for numeric columns) ---------------- #
    def psi_score(self, expected, actual, bins: int = 10) -> float:
        try:
            import numpy as np

            expected_hist, bin_edges = np.histogram(expected, bins=bins)
            actual_hist, _ = np.histogram(actual, bins=bin_edges)

            expected_perc = expected_hist / max(len(expected), 1)
            actual_perc = actual_hist / max(len(actual), 1)

            psi = sum(
                (e - a) * np.log((e + 1e-6) / (a + 1e-6))
                for e, a in zip(expected_perc, actual_perc)
            )
            return float(psi)
        except Exception as e:
            raise CustomException(f"Error computing PSI score: {e}", sys)

    # ---------------- REPORT GENERATION ---------------- #
    def save_report(self, report: dict, report_path: str) -> str:
        """
        Bug fix: previously wrote to a hardcoded path while creating
        directories for a different (unused) `report_path` argument, and was
        called both with one arg and via a second nonexistent method name.
        Now there is exactly one save method, and it actually uses the path
        it's given.
        """
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            save_json_file(report_path, report)
            logging.info(f"Validation/drift report saved at: {report_path}")
            return report_path
        except Exception as e:
            raise CustomException(f"Error saving validation report: {e}", sys)

    # ---------------- FINAL PIPELINE ---------------- #
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating data validation process.")
            schema = self._schema_config

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            errors = []
            errors += self.basic_checks(train_df, schema)
            # Bug fix: was `self.validate_data(...)` — method didn't exist
            errors += self.validate_schema_rules(train_df, schema)

            errors += self.basic_checks(test_df, schema)
            errors += self.validate_schema_rules(test_df, schema)

            drift_report = self.detect_drift(train_df, test_df, schema)
            drift_detected_any = any(v.get("drift_detected") for v in drift_report.values())

            report_path = os.path.join(self.data_validation_config.drift_report_dir, "validation_report.json")
            # Bug fix: was calling save_report with only one argument, then a
            # second nonexistent save_drift_report method right after.
            saved_path = self.save_report(
                {"validation_errors": errors, "drift_report": drift_report},
                report_path,
            )

            # Distinguish hard schema failures (block the pipeline) from
            # drift-only findings (log and continue — drift on its own
            # doesn't mean the data is malformed, it means the model may
            # need retraining/monitoring attention).
            validation_status = len(errors) == 0
            if not validation_status:
                message = f"Data validation failed: {errors}"
            elif drift_detected_any:
                message = "Schema validation passed; data drift detected — see drift report."
            else:
                message = "Data validation completed successfully, no drift detected."

            logging.info(message)

            return DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path if validation_status else None,
                invalid_train_file_path=None if validation_status else self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path if validation_status else None,
                invalid_test_file_path=None if validation_status else self.data_ingestion_artifact.test_file_path,
                drift_report_file_path=saved_path,
            )
        except Exception as e:
            raise CustomException(f"Error during data validation: {e}", sys)