import pytest
from fastapi.testclient import TestClient
from main import app, db
import os
import csv

client = TestClient(app)

TEST_CSV = "test_data.csv"
TEST_COLLECTION = "test_deliveries"

def setup_module(module):
    with open(TEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "status", "customer"])
        writer.writeheader()
        writer.writerow({"order_id": "ORD1", "status": "Delivered", "customer": "Alice"})
        writer.writerow({"order_id": "ORD2", "status": "Delayed", "customer": "Bob"})
    db[TEST_COLLECTION].delete_many({})

def teardown_module(module):
    os.remove(TEST_CSV)
    db[TEST_COLLECTION].delete_many({})

def test_llm_query_nl_to_db(monkeypatch):
    # Patch LLM to return a simple MongoDB query
    monkeypatch.setattr("main.call_llm_api", lambda *a, **kw: {"choices": [{"text": "{'status': 'Delivered'}"}]})
    # Patch explanation to return dummy text
    monkeypatch.setattr("main.get_llm_explanation", lambda *a, **kw: "This query finds all delivered orders.")
    with open(TEST_CSV, "rb") as f:
        client.post("/api/ingest-csv-to-db", files={"file": (TEST_CSV, f, "text/csv")}, data={"collection": TEST_COLLECTION})
    resp = client.post("/api/llm-query", json={"prompt": "Show delivered orders", "collection": TEST_COLLECTION})
    assert resp.status_code == 200
    data = resp.json()
    assert "llm_result" in data and isinstance(data["llm_result"], list)
    assert any(row["status"] == "Delivered" for row in data["llm_result"])
    assert "explanation" in data and "delivered" in data["explanation"].lower()
    assert "query" in data and data["query"] == {'status': 'Delivered'}
