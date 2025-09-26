import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_trends():
    resp = client.get("/api/trends")
    assert resp.status_code == 200
    data = resp.json()
    assert "trends" in data
    assert isinstance(data["trends"], list)
    assert any("id" in t and "name" in t for t in data["trends"])

def test_get_cases_for_trend():
    resp = client.get("/api/cases/1")
    assert resp.status_code == 200
    data = resp.json()
    assert "cases" in data
    assert isinstance(data["cases"], list)
    if data["cases"]:
        assert "case_id" in data["cases"][0]
        assert "order_id" in data["cases"][0]
