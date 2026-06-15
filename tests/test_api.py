from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "backend" in data
    assert "model_version" in data


def test_metadata_endpoint():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()
    assert data["model_path"]
    assert data["training_method"] == "SFT LoRA + DPO + merge"


def test_triage_endpoint_mock():
    # Le mode mock permet de tester l'API sans charger le modele complet.
    payload = {
        "patient_text": "Patient de 62 ans avec douleur thoracique brutale, essoufflement et sueurs.",
        "max_tokens": 128,
        "temperature": 0.2,
    }

    response = client.post("/triage", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["request_id"]
    assert data["response"]
    assert data["latency_ms"] >= 0
    assert data["backend"] == "mock"
    assert data["model_version"]
    assert data["timestamp"]
