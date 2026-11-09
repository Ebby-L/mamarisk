"""MamaRisk API.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs   Dashboard: http://127.0.0.1:8000/
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mamarisk.model import MODEL_PATH, load, predict_one

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
DISCLAIMER = ("Research prototype trained on a public dataset from Bangladesh. "
              "Not validated for clinical use; it supports, never replaces, "
              "clinical judgement.")

app = FastAPI(title="MamaRisk API", version="1.0.0",
              description="Maternal health risk classification from routine vital signs.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bundle: dict | None = None


def bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run `python train.py` first.")
        _bundle = load()
    return _bundle


class Vitals(BaseModel):
    """Ranges reject obvious entry errors, not unusual-but-real readings."""
    Age: int = Field(..., ge=10, le=70, description="Years")
    SystolicBP: int = Field(..., ge=60, le=220, description="mmHg")
    DiastolicBP: int = Field(..., ge=30, le=140, description="mmHg")
    BS: float = Field(..., ge=2.0, le=30.0, description="Blood glucose, mmol/L")
    BodyTemp: float = Field(..., ge=94.0, le=106.0, description="°F")
    HeartRate: int = Field(..., ge=30, le=200, description="Beats per minute")

    model_config = {"json_schema_extra": {"example": {
        "Age": 35, "SystolicBP": 140, "DiastolicBP": 90,
        "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 70}}}


class Prediction(BaseModel):
    risk_level: str
    probabilities: dict[str, float]
    model: str
    disclaimer: str


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model")
def model_info():
    b = bundle()
    return {"model": b["name"], "features": b["features"],
            "classes": b["classes"], "cv_scores": b["scores"]}


@app.post("/predict", response_model=Prediction)
def predict(v: Vitals):
    if v.DiastolicBP >= v.SystolicBP:
        raise HTTPException(422, "Diastolic BP must be lower than systolic BP.")
    b = bundle()
    return {**predict_one(b, v.model_dump()), "model": b["name"], "disclaimer": DISCLAIMER}
