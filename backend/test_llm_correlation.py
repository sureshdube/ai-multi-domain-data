import os
import json
import csv
from main import call_llm_api, correlate_events_by_order_id

def test_call_llm_api_priority():
    # Simulate LLM config with a dummy local endpoint (should fail and return error)
    result = call_llm_api("Test prompt", ["a", "b"], [{"a": 1, "b": 2}])
    assert "error" in result

def test_correlate_events_by_order_id():
    # Create two CSVs with order_id
    csv1 = "order_id,name\n1,Alice\n2,Bob\n"
    csv2 = "order_id,city\n1,NY\n2,LA\n1,Boston\n"
    f1 = "test1.csv"
    f2 = "test2.csv"
    with open(f1, "w") as f:
        f.write(csv1)
    with open(f2, "w") as f:
        f.write(csv2)
    result = correlate_events_by_order_id([f1, f2])
    assert "1" in result and "2" in result
    assert any(e["file"] == "test1.csv" for e in result["1"])
    assert any(e["file"] == "test2.csv" for e in result["1"])
    os.remove(f1)
    os.remove(f2)