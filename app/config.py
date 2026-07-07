"""
app/config.py

Centralized, typed configuration. Reads from environment variables (with
the same names you'd set in a K8s Deployment's `env:` block or a .env file
locally), validated once at startup instead of scattered `os.environ.get()`
calls with silent string defaults throughout the codebase.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_source: str = "local"          # "local" or "mlflow"
    model_path: str = "artifacts/model_trainer/model.joblib"
    metadata_path: str = "artifacts/model_trainer/metadata.json"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_name: str = "diabetes-risk-model"
    mlflow_model_stage: str = "Production"

    # CORS — empty by default (same-origin only, since the UI is served by
    # this same app). Set to a comma-separated list of origins if a separate
    # frontend or external client needs to call this API directly.
    cors_allowed_origins: str = ""

    log_level: str = "INFO"

    class Config:
        env_file = ".env"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


settings = Settings()