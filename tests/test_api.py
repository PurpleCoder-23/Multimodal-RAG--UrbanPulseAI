
import urllib.request
import json


BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Verify that the backend health endpoint works."""

    with urllib.request.urlopen(
        f"{BASE_URL}/health", timeout=10
    ) as response:

        assert response.status == 200

        data = json.loads(response.read().decode("utf-8"))

        assert data["status"] == "healthy"
        assert data["service"] == "UrbanPulse AI"
        assert data["agent_ready"] is True
