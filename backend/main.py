
# --- Imports ---
import os
import csv
import json
from io import StringIO
import requests
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# --- FastAPI app and CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schema Versioning Utility ---
SCHEMA_HISTORY_FILE = "schema_history.json"

def load_schema_history():
    if os.path.exists(SCHEMA_HISTORY_FILE):
        with open(SCHEMA_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_schema_history(history):
    with open(SCHEMA_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def update_schema_history(new_fields):
    history = load_schema_history()
    if not history or set(history[-1]) != set(new_fields):
        history.append(list(new_fields))
        save_schema_history(history)
    return history

# --- ETL Utility ---
def streaming_etl(rows):
    """
    Streaming ETL: validate, cleanse, and transform each row.
    - Validation: skip rows with missing required fields
    - Cleansing: trim whitespace, remove duplicates
    - Transformation: uppercase all string fields
    """
    required_fields = set(rows.fieldnames)
    seen = set()
    for row in rows:
        # Validation: skip if any field is missing
        if any(row[f] is None or row[f] == '' for f in required_fields):
            continue
        # Cleansing: trim whitespace
        clean_row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        # Remove duplicates (by tuple of values)
        row_tuple = tuple(clean_row.values())
        if row_tuple in seen:
            continue
        seen.add(row_tuple)
        # Transformation: uppercase all string fields
        transformed = {k: (v.upper() if isinstance(v, str) else v) for k, v in clean_row.items()}
        yield transformed

# --- LLM API Integration ---
def load_llm_config():
    with open("llm_config.json") as f:
        return json.load(f)["providers"]

def get_llm_providers_by_priority():
    providers = load_llm_config()
    return sorted(providers, key=lambda p: p["priority"])

def call_llm_api(prompt, schema=None, data_sample=None):
    """
    Try LLM providers in priority order until one succeeds.
    Pass schema and data_sample in the prompt/context.
    """
    providers = get_llm_providers_by_priority()
    for provider in providers:
        try:
            if provider["name"] == "open_source_llm":
                # Example: POST to open source LLM
                resp = requests.post(provider["url"], json={"prompt": prompt, "schema": schema, "data": data_sample})
                if resp.ok:
                    return resp.json()
            elif provider["name"] == "openai":
                headers = {"Authorization": f"Bearer {provider['api_key']}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": f"Schema: {schema}\nSample: {data_sample}"},
                        {"role": "user", "content": prompt}
                    ]
                }
                resp = requests.post(provider["url"], headers=headers, json=payload)
                if resp.ok:
                    return resp.json()
            elif provider["name"] == "gemini":
                headers = {"Authorization": f"Bearer {provider['api_key']}"}
                payload = {
                    "contents": [{"parts": [{"text": f"Schema: {schema}\nSample: {data_sample}\nPrompt: {prompt}"}]}]
                }
                resp = requests.post(provider["url"], headers=headers, json=payload)
                if resp.ok:
                    return resp.json()
        except Exception as e:
            continue
    return {"error": "All LLM providers failed"}

# --- ML-driven event correlation using order_id ---
def correlate_events_by_order_id(csv_files):
    """
    Given a list of CSV file paths, correlate events by order_id.
    Returns a dict: order_id -> list of event dicts from all files.
    """
    order_events = {}
    for file_path in csv_files:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = row.get("order_id")
                if not oid:
                    continue
                if oid not in order_events:
                    order_events[oid] = []
                order_events[oid].append({"file": os.path.basename(file_path), **row})
    return order_events

# --- Unstructured Text File ---
UNSTRUCTURED_FILE = "unstructured_texts.json"

# --- API Endpoints ---
@app.get("/api/insights")
def get_insights():
    return {"insights": ["Dummy insight 1", "Dummy insight 2"]}

@app.post("/api/nlquery")
async def nl_query(request: Request):
    data = await request.json()
    return {"result": f"Dummy response to: {data.get('query', '')}"}

@app.get("/api/drilldown")
def get_drilldown():
    return {"case": "Dummy delivery case details"}

@app.get("/api/schedule")
def get_schedule():
    return {"schedule": "Dummy schedule config"}

@app.post("/api/schedule")
async def set_schedule(request: Request):
    data = await request.json()
    return {"status": "Schedule updated", "config": data}

@app.post("/api/ingest-csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Accepts a single CSV file upload, runs streaming ETL (validate, cleanse, transform),
    supports schema evolution/versioning, and returns a preview of the processed data.
    """
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are supported"})
    try:
        contents = await file.read()
        decoded = contents.decode('utf-8')
        reader = csv.DictReader(StringIO(decoded))
        # --- Schema evolution/versioning ---
        schema_history = update_schema_history(reader.fieldnames)
        processed = []
        for i, row in enumerate(streaming_etl(reader)):
            if i >= 5:
                break
            processed.append(row)
        return {
            "etl_csv_preview": {file.filename: processed},
            "schema_history": schema_history
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/unstructured-text")
async def ingest_unstructured_text(request: Request):
    """
    Accepts and stores unstructured text data for further analysis.
    """
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})
    # Store text in a JSON file (append mode)
    if os.path.exists(UNSTRUCTURED_FILE):
        with open(UNSTRUCTURED_FILE, "r") as f:
            texts = json.load(f)
    else:
        texts = []
    texts.append(text)
    with open(UNSTRUCTURED_FILE, "w") as f:
        json.dump(texts, f, indent=2)
    return {"status": "Text stored", "count": len(texts)}

@app.post("/api/llm-query")
async def llm_query(request: Request):
    """
    Accepts a prompt, uses schema and sample data, and calls LLMs in priority order.
    """
    data = await request.json()
    prompt = data.get("prompt", "")
    # Use latest schema and sample data
    schema_history = load_schema_history()
    schema = schema_history[-1] if schema_history else []
    # Try to get a sample from the latest CSV (if any)
    sample = []
    if os.path.exists("data_sample.csv"):
        with open("data_sample.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                sample.append(row)
    result = call_llm_api(prompt, schema, sample)
    return {"llm_result": result}

@app.post("/api/correlate-events")
async def correlate_events(request: Request):
    """
    Accepts a list of CSV file paths, correlates events by order_id, and returns the mapping.
    """
    data = await request.json()
    csv_files = data.get("csv_files", [])
    if not csv_files:
        return JSONResponse(status_code=400, content={"error": "No CSV files provided"})
    result = correlate_events_by_order_id(csv_files)
    return {"order_event_map": result}


# --- REMOVE DUPLICATE CODE BLOCKS ABOVE (utility functions, app instance, endpoints) ---
