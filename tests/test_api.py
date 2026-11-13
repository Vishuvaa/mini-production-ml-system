import pytest
from fastapi.testclient import TestClient

import serving.app as api

app = api.app

SAMPLE = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
}


def test_health_and_prediction():
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_loaded": True,
        }
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 200
        body = response.json()
        assert body["customerID"] == SAMPLE["customerID"]
        assert 0 <= body["churn_probability"] <= 1
        assert body["churn_prediction"] in (0, 1)
        assert 0 <= body["threshold"] <= 1
        assert body["model"]


def test_missing_field_rejected():
    payload = SAMPLE.copy()
    payload.pop("tenure")
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_extra_field_rejected():
    payload = SAMPLE | {"unexpected_feature": "oops"}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_invalid_ranges_and_types_rejected():
    for field, value in (("tenure", 73), ("SeniorCitizen", 2), ("gender", 1)):
        payload = SAMPLE | {field: value}
        with TestClient(app) as client:
            assert client.post("/predict", json=payload).status_code == 422


def test_missing_total_charges_is_supported():
    payload = SAMPLE | {"TotalCharges": None}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 200


def test_unknown_category_is_handled_by_pipeline():
    payload = SAMPLE | {"gender": "Other"}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 200


def test_unavailable_model_returns_503():
    with TestClient(app) as client:
        api.bundle = None
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 503
        assert "manual review" in response.json()["detail"]


def test_missing_artifact_fails_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "BUNDLE_PATH", tmp_path / "missing.joblib")
    with pytest.raises(RuntimeError, match="Model artifact not found"):
        with TestClient(app):
            pass
