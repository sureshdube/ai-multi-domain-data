import os
import json
import csv
from io import StringIO
from main import update_schema_history, ingest_unstructured_text, UNSTRUCTURED_FILE, SCHEMA_HISTORY_FILE
import asyncio

def test_update_schema_history():
    # Clean up before test
    if os.path.exists(SCHEMA_HISTORY_FILE):
        os.remove(SCHEMA_HISTORY_FILE)
    # Add first schema
    s1 = ["a", "b", "c"]
    h = update_schema_history(s1)
    assert h[-1] == s1
    # Add same schema (should not duplicate)
    h2 = update_schema_history(s1)
    assert len(h2) == 1
    # Add new schema
    s2 = ["a", "b", "d"]
    h3 = update_schema_history(s2)
    assert h3[-1] == s2
    assert len(h3) == 2

def test_ingest_unstructured_text():
    # Clean up before test
    if os.path.exists(UNSTRUCTURED_FILE):
        os.remove(UNSTRUCTURED_FILE)
    class DummyRequest:
        def __init__(self, text):
            self._text = text
        async def json(self):
            return {"text": self._text}
    # Add first text
    loop = asyncio.get_event_loop()
    resp = loop.run_until_complete(ingest_unstructured_text(DummyRequest("hello world")))
    assert resp["status"] == "Text stored"
    # Add second text
    resp2 = loop.run_until_complete(ingest_unstructured_text(DummyRequest("another text")))
    assert resp2["count"] == 2
    # Check file contents
    with open(UNSTRUCTURED_FILE) as f:
        arr = json.load(f)
    assert "hello world" in arr and "another text" in arr