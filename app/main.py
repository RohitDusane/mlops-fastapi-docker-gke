"""
app/main.py

Diabetes risk screening API. Returns a probability-based risk score, not a
bare boolean — the model's `predict_proba` output is the clinically useful
part; discarding it in favor of a hard 0/1 throws away information a real
screening tool would need.

Model loading: defaults to the local artifacts/diabetes_model.pkl (baked
into the Docker image at build time). Set MODEL_SOURCE=mlflow to instead
pull the current "Production"-stage model from the MLflow Model Registry at
container startup — this is the more production-realistic pattern, since it
lets you promote a new model version without rebuilding the image, but it
does add a hard runtime dependency on the MLflow tracking server being
reachable from the pod.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from app.schemas import DiabetesInput, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diabetes-api")

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "diabetes_model.pkl"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local")  # "local" or "mlflow"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "diabetes-risk-model")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Production")

DISCLAIMER = (
    "This is a statistical screening risk estimate based on self-reported "
    "health survey indicators. It is NOT a medical diagnosis. Consult a "
    "licensed healthcare professional for any clinical decision."
)

state = {"model": None, "model_version": "unknown", "feature_order": []}


def load_model_local():
    state["model"] = joblib.load(MODEL_PATH)
    with open(MODEL_CARD_PATH) as f:
        card = json.load(f)
    state["model_version"] = card.get("model_version", "unknown")
    state["feature_order"] = card.get("feature_columns", [])


def load_model_mlflow():
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
    state["model"] = mlflow.sklearn.load_model(model_uri)
    state["model_version"] = f"{MLFLOW_MODEL_NAME}:{MLFLOW_MODEL_STAGE}"

    # Feature order still needs to come from somewhere — the local model_card
    # (shipped alongside the image) is the simplest source of truth for this,
    # even when the model weights themselves come from MLflow.
    with open(MODEL_CARD_PATH) as f:
        card = json.load(f)
    state["feature_order"] = card.get("feature_columns", [])


def load_model():
    try:
        if MODEL_SOURCE == "mlflow":
            logger.info("Loading model from MLflow registry: %s/%s", MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
            load_model_mlflow()
        else:
            logger.info("Loading model from local artifact: %s", MODEL_PATH)
            load_model_local()
        logger.info("Model %s loaded with %d features", state["model_version"], len(state["feature_order"]))
    except Exception as e:
        # Fail loudly in logs, but don't crash the process — /health will
        # report unhealthy and K8s will hold traffic back via the readiness
        # probe rather than the whole pod crash-looping.
        logger.error("Failed to load model: %s", e)
        state["model"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Diabetes Risk Screening API",
    description="Screening-risk estimation model based on BRFSS health indicators. Not a diagnostic device.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"message": "Diabetes Risk Screening API is live", "model_version": state["model_version"]}


@app.get("/health")
def health():
    """
    Used by the K8s readiness/liveness probes. Checks the model is actually
    loaded, not just that the process is alive — a pod that's up but has no
    model loaded should not receive traffic.
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": state["model_version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: DiabetesInput):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded — service unavailable")

    input_dict = data.model_dump()
    try:
        input_array = np.array([[input_dict[col] for col in state["feature_order"]]])
    except KeyError as e:
        # Signals a mismatch between the deployed model's expected features
        # and the API schema — should never happen in practice if both are
        # versioned together, but fail with a clear error rather than a
        # silent wrong prediction if it does.
        raise HTTPException(status_code=500, detail=f"Feature mismatch with loaded model: {e}")

    start = time.time()
    proba = float(state["model"].predict_proba(input_array)[0][1])
    prediction = proba >= 0.5
    latency_ms = round((time.time() - start) * 1000, 2)

    risk_category = "low" if proba < 0.3 else "moderate" if proba < 0.6 else "high"

    # Structured audit log: every prediction traceable to model version,
    # input, output, and timestamp. This is the minimum audit trail expected
    # for a health-adjacent prediction system, and it's also what feeds
    # drift monitoring later (comparing incoming feature distributions over
    # time against the training distribution).
    logger.info(json.dumps({
        "event": "prediction",
        "model_version": state["model_version"],
        "risk_score": round(proba, 4),
        "risk_category": risk_category,
        "latency_ms": latency_ms,
        "input": input_dict,
    }))

    return PredictionResponse(
        diabetic_risk=prediction,
        risk_score=round(proba, 4),
        risk_category=risk_category,
        model_version=state["model_version"],
        disclaimer=DISCLAIMER,
    )