import os
import sys
import json

import joblib
import numpy as np
import yaml

from diabetes_risk_prediction.exception.custom_exception import CustomException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(f"Error reading YAML file {file_path}: {e}", sys)


def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise CustomException(f"Error writing YAML file {file_path}: {e}", sys)


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {e}", sys)


def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {e}", sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
    except Exception as e:
        raise CustomException(f"Error saving numpy array to {file_path}: {e}", sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise CustomException(f"Error loading numpy array from {file_path}: {e}", sys)


def save_json_file(file_path: str, content: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(content, f, indent=4, default=str)
    except Exception as e:
        raise CustomException(f"Error saving JSON to {file_path}: {e}", sys)