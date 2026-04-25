from fastapi.testclient import TestClient
from server import app


client = TestClient(app)


def test_health_endpoints():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "healthy"}

    live = client.get("/health/liveness")
    assert live.status_code == 200
    assert live.json()["status"] == "alive"


def test_metrics_endpoints():
    session_metrics = client.get("/metrics/session")
    assert session_metrics.status_code == 200
    assert "step_count" in session_metrics.json()

    ops_metrics = client.get("/metrics/ops")
    assert ops_metrics.status_code == 200
    assert "requests" in ops_metrics.json()

