# app/schemas.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BinaryFlag = Literal[0, 1]


class DiabetesInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    high_bp: BinaryFlag = Field(..., alias="HighBP")
    high_chol: BinaryFlag = Field(..., alias="HighChol")
    chol_check: BinaryFlag = Field(..., alias="CholCheck")

    # Was missing ge/le entirely in the previous version — any value was
    # accepted and passed straight to the model.
    bmi: float = Field(..., alias="BMI", ge=10, le=100)

    smoker: BinaryFlag = Field(..., alias="Smoker")
    stroke: BinaryFlag = Field(..., alias="Stroke")
    heart_disease: BinaryFlag = Field(..., alias="HeartDiseaseorAttack")
    phys_activity: BinaryFlag = Field(..., alias="PhysActivity")
    fruits: BinaryFlag = Field(..., alias="Fruits")
    veggies: BinaryFlag = Field(..., alias="Veggies")
    alcohol: BinaryFlag = Field(..., alias="HvyAlcoholConsump")
    any_healthcare: BinaryFlag = Field(..., alias="AnyHealthcare")
    no_docbc_cost: BinaryFlag = Field(..., alias="NoDocbcCost")

    gen_hlth: int = Field(..., alias="GenHlth", ge=1, le=5)
    ment_hlth: int = Field(..., alias="MentHlth", ge=0, le=30)
    phys_hlth: int = Field(..., alias="PhysHlth", ge=0, le=30)

    diff_walk: BinaryFlag = Field(..., alias="DiffWalk")

    sex: BinaryFlag = Field(..., alias="Sex")
    age: int = Field(..., alias="Age", ge=1, le=13)
    education: int = Field(..., alias="Education", ge=1, le=6)
    income: int = Field(..., alias="Income", ge=1, le=8)


class PredictionResponse(BaseModel):
    diabetic_risk: bool
    risk_score: float

    # Canonical machine-readable key ("low"/"moderate"/"high") — the ONLY
    # place display phrasing ("LOWER LIKELIHOOD" etc.) gets decided is the
    # frontend. Backend and frontend both trying to own display text is
    # exactly what caused the doubled "LIKELIHOOD LIKELIHOOD" bug.
    risk_category: Literal["low", "moderate", "high"]

    reasons: list[str]
    recommendations: list[str]

    model_version: str
    model_type: str
    features_used: int

    disclaimer: str