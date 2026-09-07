from fastapi.testclient import TestClient

from src.api import app


def test_healthz_and_endpoints():
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["n_chunks"] > 1000

        r = client.post("/search", json={"query": "vendor onboarding process", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 3
        assert body["latency_ms"] >= 0

        r = client.post("/ask", json={"question": "How many orders were placed in EMEA?"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "answered"
        assert body["columns"] == ["order_count"]
