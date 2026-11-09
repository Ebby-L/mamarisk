"""Model candidates, evaluation and persistence for MamaRisk."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASSES, FEATURES, TARGET

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_PATH = MODEL_DIR / "mamarisk.joblib"
SEED = 42


def candidates() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                class_weight="balanced", random_state=SEED)),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=SEED)),
        ]),
    }


# Missing a high-risk mother is the costly error, so we track its recall
# separately from overall performance.
SCORING = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(f1_score, average="macro"),
    "high_risk_recall": make_scorer(
        recall_score, labels=["high risk"], average="macro", zero_division=0),
}


def evaluate(df: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """Stratified k-fold scores for every candidate model."""
    X, y = df[FEATURES], df[TARGET]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, pipe in candidates().items():
        s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
        rows.append({
            "model": name,
            **{k: s[f"test_{k}"].mean() for k in SCORING},
            "macro_f1_sd": s["test_macro_f1"].std(),
        })
    return (pd.DataFrame(rows)
            .sort_values(["high_risk_recall", "macro_f1"], ascending=False)
            .reset_index(drop=True))


def train(df: pd.DataFrame, name: str) -> Pipeline:
    pipe = candidates()[name]
    pipe.fit(df[FEATURES], df[TARGET])
    return pipe


def save(pipe: Pipeline, name: str, scores: dict, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "name": name, "scores": scores,
                 "features": FEATURES, "classes": CLASSES}, path)
    return path


def load(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_one(bundle: dict, record: dict) -> dict:
    """Predict a single patient record. Returns label and class probabilities."""
    pipe = bundle["pipeline"]
    X = pd.DataFrame([record], columns=bundle["features"])
    proba = pipe.predict_proba(X)[0]
    probs = dict(zip(pipe.classes_, np.round(proba, 3).tolist()))
    label = pipe.classes_[int(np.argmax(proba))]
    return {"risk_level": label,
            "probabilities": {c: probs.get(c, 0.0) for c in bundle["classes"]}}
