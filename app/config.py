"""
app/config.py

Centralized, typed configuration. Reads from environment variables (with
the same names you'd set in a K8s Deployment's `env:` block or a .env file
locally), validated once at startup instead of scattered `os.environ.get()`
calls with silent string defaults throughout the codebase.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # model_source: str = "local"
    # model_path: str = "artifacts/model_trainer/model.joblib"
    metadata_path: str = "artifacts/model_trainer/model_metadata.json"

    # mlflow_tracking_uri: str = ("http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow_tracking_uri:str = "http://mlflow:5000"
    mlflow_model_name: str = "diabetes-risk-model"
    mlflow_model_alias: str = "champion"
    # mlflow_model_stage: str = "Production"

    # MLFLOW_TRACKING_URI=http://mlflow:5000

    # MLFLOW_MODEL_NAME=diabetes-risk-model

    # MLFLOW_MODEL_ALIAS=champion

    cors_allowed_origins: str = ""

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",        # Ignore variables not defined here
        case_sensitive=False,
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

settings = Settings()

print("=" * 60)
print("Tracking URI :", settings.mlflow_tracking_uri)
print("Model Name   :", settings.mlflow_model_name)
print("Model Alias  :", settings.mlflow_model_alias)
print("=" * 60)