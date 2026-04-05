"""
Smoke tests: run against a live backend (not unit tests).
Start backend first: uvicorn backend.main:app --app-dir .
"""
import httpx
import pytest

BASE = "http://localhost:8000"


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=10)


def test_health(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_register_and_login(client):
    email = "smoketest@example.com"
    r = client.post("/auth/register", json={"email": email, "password": "testpass123"})
    assert r.status_code in (200, 400)  # 400 if already exists
    r2 = client.post("/auth/login", json={"email": email, "password": "testpass123"})
    assert r2.status_code == 200
    assert "access_token" in r2.json()


def test_stats_requires_auth(client):
    resp = client.get("/api/v1/stats")
    assert resp.status_code == 403
