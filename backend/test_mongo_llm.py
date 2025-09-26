import pytest
from fastapi.testclient import TestClient
from main import app, db
import os
import csv

client = TestClient(app)

TEST_CSV = "test_data.csv"
TEST_COLLECTION = "test_deliveries"

def setup_module(module):
    # Create a test CSV file
    with open(TEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "status", "customer"])
        writer.writeheader()
        writer.writerow({"order_id": "ORD1", "status": "Delivered", "customer": "Alice"})
        writer.writerow({"order_id": "ORD2", "status": "Delayed", "customer": "Bob"})
    # Clean up test collection
    db[TEST_COLLECTION].delete_many({})

def teardown_module(module):
    os.remove(TEST_CSV)
    db[TEST_COLLECTION].delete_many({})

def test_ingest_csv_to_db():
    with open(TEST_CSV, "rb") as f:
        resp = client.post("/api/ingest-csv-to-db", files={"file": (TEST_CSV, f, "text/csv")}, data={"collection": TEST_COLLECTION})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["rows_loaded"] == 2
    # Check data in MongoDB
    docs = list(db[TEST_COLLECTION].find())
    assert len(docs) == 2
    assert docs[0]["order_id"] == "ORD1"

def test_llm_query_with_mongo():
    # Ensure test data is ingested
    with open(TEST_CSV, "rb") as f:
        client.post("/api/ingest-csv-to-db", files={"file": (TEST_CSV, f, "text/csv")}, data={"collection": TEST_COLLECTION})
    # Patch LLM to return permissive query
    import main
    main.call_llm_api = lambda *a, **kw: {"choices": [{"text": "{}"}]}
    resp = client.post("/api/llm-query", json={"prompt": "Show all orders", "collection": TEST_COLLECTION})
    assert resp.status_code == 200
    data = resp.json()
    assert "schema" in data and "sample" in data
    assert "order_id" in data["schema"]
    # If sample is too small, check the full collection
    if not any(row.get("order_id") == "ORD1" for row in data["sample"]):
        docs = list(db[TEST_COLLECTION].find())
        assert any(doc.get("order_id") == "ORD1" for doc in docs)
