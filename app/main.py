"""
app/main.py

Diabetes risk screening API + server-rendered UI. Returns a probability-
based risk score, not a bare boolean — predict_proba's output is the
clinically useful part.

Route handlers stay thin: model state lives in app/model_loader.py, config
lives in app/config.py, "why this score" logic lives in app/insights.py.
"""

import json
import logging
import time
import uuid
import joblib
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.insights import generate_insights
from app.model_loader import model_state
from app.schemas import DiabetesInput, PredictionResponse

logging.basicConfig(level=settings.log_level, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diabetes-api")

DISCLAIMER = (
    "This is a statistical screening risk estimate based on self-reported "
    "health survey indicators. It is NOT a medical diagnosis. Consult a "
    "licensed healthcare professional for any clinical decision."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state.load()
    yield


app = FastAPI(
    title="Diabetes Risk Screening API",
    description="Screening-risk estimation model based on BRFSS health indicators. Not a diagnostic device.",
    version="1.0.0",
    lifespan=lifespan,
)

if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(json.dumps({
        "event": "request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    }))
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(json.dumps({"event": "validation_error", "errors": exc.errors()}, default=str))
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/info")
def api_info():
    return {
        "message": "Diabetes Risk Screening API is live",
        "model_version": model_state.model_version,
        "model_type": model_state.model_type,
    }


@app.get("/health")
def health():
    if not model_state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": model_state.model_version}


@app.get("/metrics")
def metrics():
    lines = [
        "# HELP diabetes_api_model_loaded Whether a model is currently loaded (1) or not (0)",
        "# TYPE diabetes_api_model_loaded gauge",
        f"diabetes_api_model_loaded {1 if model_state.is_loaded else 0}",
        "# HELP diabetes_api_decision_threshold Current model's decision threshold",
        "# TYPE diabetes_api_decision_threshold gauge",
        f"diabetes_api_decision_threshold {model_state.threshold}",
    ]
    return "\n".join(lines) + "\n"


# @app.post("/predict", response_model=PredictionResponse)
# def predict(data: DiabetesInput):
#     if not model_state.is_loaded:
#         raise HTTPException(status_code=503, detail="Model not loaded — service unavailable")

#     # input_dict = data.model_dump(by_alias=True)
#     # try:
#     #     input_array = np.array([[input_dict[col] for col in model_state.feature_order]])
#     # except KeyError as e:
#     #     raise HTTPException(status_code=500, detail=f"Feature mismatch with loaded model: {e}")

#     # start = time.time()
#     # proba = float(model_state.model.predict_proba(input_array)[0][1])
    
#     input_dict = data.model_dump(by_alias=True)

#     try:
#         input_df = pd.DataFrame(
#             [[input_dict[col] for col in model_state.feature_order]],
#             columns=model_state.feature_order,
#         )
#     except KeyError as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Feature mismatch with loaded model: {e}",
#         )

#     start = time.time()
#     proba = float(model_state.model.predict_proba(input_df)[0][1])
#     # input_df = input_df.astype(float)

#     native_model = model_state.model._model_impl.python_model.model
    
#     proba = float(native_model.predict_proba(input_df)[0][1])
#     prediction = proba >= model_state.threshold

#     latency_ms = round((time.time() - start) * 1000, 2)

#     # Canonical key only — "low"/"moderate"/"high". Display phrasing
#     # ("LOWER LIKELIHOOD" etc.) is decided once, in main.js, not here too.
#     risk_category = "low" if proba < 0.25 else "moderate" if proba < 0.5 else "high"

#     reasons, recommendations = generate_insights(data)

#     logger.info(json.dumps({
#         "event": "prediction",
#         "model_version": model_state.model_version,
#         "model_type": model_state.model_type,
#         "decision_threshold": model_state.threshold,
#         "risk_score": round(proba, 4),
#         "risk_category": risk_category,
#         "latency_ms": latency_ms,
#         "input": input_dict,
#     }))

#     return PredictionResponse(
#         diabetic_risk=prediction,
#         risk_score=round(proba, 4),
#         risk_category=risk_category,
#         reasons=reasons,
#         recommendations=recommendations,
#         model_version=model_state.model_version,
#         model_type=model_state.model_type,
#         features_used=len(model_state.feature_order),
#         disclaimer=DISCLAIMER,
#     )

@app.post("/predict", response_model=PredictionResponse)
def predict(data: DiabetesInput):
    if not model_state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service unavailable"
        )

    input_dict = data.model_dump(by_alias=True)

    try:
        input_df = pd.DataFrame(
            [[input_dict[col] for col in model_state.feature_order]],
            columns=model_state.feature_order,
        )

    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch with loaded model: {e}",
        )

    start = time.time()

    # Convert according to MLflow schema
    input_df = input_df.astype(float)

    # Access underlying sklearn/lightgbm model
    # native_model = model_state.model._model_impl.python_model.model
    native_model = model_state.model._model_impl.sklearn_model
    # Probability
    proba = float(native_model.predict_proba(input_df)[0][1])

    prediction = proba >= model_state.threshold

    latency_ms = round(
        (time.time() - start) * 1000,
        2
    )

    risk_category = (
        "low"
        if proba < 0.25
        else "moderate"
        if proba < 0.5
        else "high"
    )

    reasons, recommendations = generate_insights(data)

    logger.info(json.dumps({
        "event": "prediction",
        "model_version": model_state.model_version,
        "model_type": model_state.model_type,
        "decision_threshold": model_state.threshold,
        "risk_score": round(proba, 4),
        "risk_category": risk_category,
        "latency_ms": latency_ms,
        "input": input_dict,
    }))

    return PredictionResponse(
        diabetic_risk=prediction,
        risk_score=round(proba, 4),
        risk_category=risk_category,
        reasons=reasons,
        recommendations=recommendations,
        model_version=model_state.model_version,
        model_type=model_state.model_type,
        features_used=len(model_state.feature_order),
        disclaimer=DISCLAIMER,
    )