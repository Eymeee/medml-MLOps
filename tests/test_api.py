from __future__ import annotations


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "model_missing"}


def test_model_info_endpoint(api_client):
    response = api_client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["best_model_name"] == "logreg"


def test_predict_endpoint(api_client):
    payload = {
        "age": 54,
        "bmi": 31.2,
        "glucose": 165.0,
        "systolic_bp": 138.0,
        "diastolic_bp": 86.0,
        "insulin": 120.0,
        "hba1c": 7.1,
        "gender": "female",
        "smoking_status": "former",
        "family_history": "yes",
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert body["result"]["prediction"] in [0, 1]
    assert 0.0 <= body["result"]["probability"] <= 1.0
