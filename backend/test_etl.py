import csv
from io import StringIO
from main import streaming_etl

def test_streaming_etl_basic():
    csv_data = """name,age,city\nAlice,30,New York\nBob, ,Los Angeles\nAlice,30,New York\nCharlie,25,\nDavid,40,Chicago\n"""
    reader = csv.DictReader(StringIO(csv_data))
    result = list(streaming_etl(reader))
    # Should skip Bob (missing age), Alice duplicate, Charlie (missing city)
    assert {'name': 'ALICE', 'age': '30', 'city': 'NEW YORK'} in result
    assert {'name': 'DAVID', 'age': '40', 'city': 'CHICAGO'} in result
    assert len(result) == 2