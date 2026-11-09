"""Tests run on synthetic data with the same schema, so they need no download."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api
from mamarisk.data import CLASSES, FEATURES, TARGET, clean
from mamarisk.model import evaluate, predict_one, train


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 300
    risk = rng.choice(CLASSES, n)
    shift = pd.Series(risk).map({"low risk": 0, "mid risk": 1, "high risk": 2}).to_numpy()
    raw = pd.DataFrame({
        "Age": rng.integers(15, 50, n),
        "SystolicBP": 105 + 15 * shift + rng.integers(-8, 8, n),
        "DiastolicBP": 70 + 8 * shift + rng.integers(-5, 5, n),
        "BS": np.round(6.5 + 2.5 * shift + rng.normal(0, 1, n), 1),
        "BodyTemp": np.round(98 + 0.6 * shift + rng.normal(0, 0.4, n), 1),
        "HeartRate": 72 + 4 * shift + rng.integers(-6, 6, n),
        TARGET: [" " + r.upper() for r in risk],  # messy labels on purpose
    })
    return clean(pd.concat([raw, raw.head(20)]))  # include duplicates


def test_clean_drops_duplicates_and_normalises_labels(df):
    assert not df.duplicated().any()
    assert set(df[TARGET]) <= set(CLASSES)


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean(pd.DataFrame({"Age": [30]}))


def test_evaluate_and_predict(df):
    results = evaluate(df, folds=3)
    assert len(results) == 3 and results["macro_f1"].between(0, 1).all()
    pipe = train(df, results.iloc[0]["model"])
    bundle = {"pipeline": pipe, "name": "test", "features": FEATURES, "classes": CLASSES}
    out = predict_one(bundle, df[FEATURES].iloc[0].to_dict())
    assert out["risk_level"] in CLASSES
    assert abs(sum(out["probabilities"].values()) - 1) < 0.01


def test_api(df, monkeypatch):
    pipe = train(df, "Random Forest")
    monkeypatch.setattr(api, "_bundle", {"pipeline": pipe, "name": "Random Forest",
                                         "features": FEATURES, "classes": CLASSES, "scores": {}})
    client = TestClient(api.app)
    ok = {"Age": 35, "SystolicBP": 140, "DiastolicBP": 90, "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 70}
    r = client.post("/predict", json=ok)
    assert r.status_code == 200 and r.json()["risk_level"] in CLASSES
    assert client.post("/predict", json={**ok, "SystolicBP": 500}).status_code == 422
    assert client.post("/predict", json={**ok, "DiastolicBP": 150, "SystolicBP": 140}).status_code == 422
    assert client.get("/").status_code == 200
