from fastapi.testclient import TestClient

from logistics_ai.api import app

client = TestClient(app)


def test_route_constraints_and_baseline():
    response = client.post("/route", json={"seed": 42, "customers": 12, "capacity": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["solution"]["valid"]
    assert data["solution"]["distance"] <= data["nearest_neighbor_distance"] + 1e-8
    assert data["synthetic_data"] is True


def test_invalid_requests_are_rejected():
    assert client.post("/route", json={"customers": 10000}).status_code == 422
    assert client.post("/route", json={"capacity": -1}).status_code == 422
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_offline_question_returns_provenance():
    response = client.post("/ask", json={"question": "파손 화물을 접수할 때 사진을 몇 장 남기나요?"})
    assert response.status_code == 200
    data = response.json()
    assert "abstained" in data
    assert data["abstained"] is False
    assert data["evidence"]
