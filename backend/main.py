# --- Field Mapping and Descriptions ---
from field_mapping import FIELD_MAPPINGS
# --- Imports ---
import os
import logging
import csv
import json
from io import StringIO
import requests
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pymongo import MongoClient

# --- FastAPI app and CORS ---
app = FastAPI()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("llm_query")
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
        # Validation: skip if any required field is missing or only whitespace
        if any(row[f] is None or str(row[f]).strip() == '' for f in required_fields):
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
        provider_name = provider.get("name", "unknown")
        try:
            if provider_name == "open_source_llm":
                try:
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": provider["model"],
                        "messages": [
                            {"role": "system", "content": f"Schema: {schema}\nSample: {data_sample}"},
                            {"role": "user", "content": prompt}
                        ]
                    }
                    resp = requests.post(provider["url"], headers=headers, json=payload)
                    if resp.ok:
                        return resp.json()
                except Exception as e:
                    logger.error(f"Error with open_source_llm provider: {e}")
            elif provider_name == "openai":
                try:
                    headers = {"Authorization": f"Bearer {provider['api_key']}", "Content-Type": "application/json"}
                    payload = {
                        "model": provider["model"],
                        "messages": [
                            {"role": "system", "content": f"Schema: {schema}\nSample: {data_sample}"},
                            {"role": "user", "content": prompt}
                        ]
                    }
                    resp = requests.post(provider["url"], headers=headers, json=payload)
                    if resp.ok:
                        return resp.json()
                except Exception as e:
                    logger.error(f"Error with openai provider: {e}")
            elif provider_name == "gemini":
                try:
                    headers = {"Authorization": f"Bearer {provider['api_key']}"}
                    payload = {
                        "contents": [{"parts": [{"text": f"Schema: {schema}\nSample: {data_sample}\nPrompt: {prompt}"}]}]
                    }
                    resp = requests.post(provider["url"], headers=headers, json=payload)
                    if resp.ok:
                        return resp.json()
                except Exception as e:
                    logger.error(f"Error with gemini provider: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with provider {provider_name}: {e}")
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

# --- MongoDB Connection ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["delivery_data"]  # Use a DB name relevant to your domain

# --- CSV to MongoDB Ingestion Utility ---
def ingest_csv_to_mongo(file_path, collection_name, max_rows=None):
    """
    Loads data from a CSV file into the specified MongoDB collection.
    If max_rows is set, only loads up to that many rows.
    """
    collection = db[collection_name]
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        docs = []
        for i, row in enumerate(reader):
            docs.append(row)
            if max_rows and i+1 >= max_rows:
                break
        if docs:
            collection.insert_many(docs)
    return len(docs)

# --- Unstructured Text File ---
UNSTRUCTURED_FILE = "unstructured_texts.json"

# --- Utility: Get MongoDB Collection Schema ---
def get_mongo_collection_schema(collection_name, sample_size=5):
    collection = db[collection_name]
    sample_docs = list(collection.find({}, limit=sample_size))
    # Remove MongoDB's _id field for LLM context
    for doc in sample_docs:
        doc.pop('_id', None)
    # Infer schema from all docs (union of keys)
    schema = set()
    for doc in sample_docs:
        schema.update(doc.keys())
    schema = list(schema)
    return schema, sample_docs

# --- Utility: LLM generates MongoDB query ---
def get_llm_mongo_query(prompt, schema, sample):
    """
    Calls LLM to generate a MongoDB query string from NL prompt, schema, and sample data.
    Returns the query as a Python dict (parsed from LLM output).
    """
    # Compose a prompt for the LLM to generate a MongoDB query
    llm_prompt = (
        f"Schema: {schema}\nSample: {sample}\n"
        f"User Question: {prompt}\n"
        "Generate a MongoDB find() query (as a Python dict) to answer the question. "
        "Return ONLY the query dict, no explanation."
    )
    llm_response = call_llm_api(llm_prompt, schema, sample)
    # Try to parse the LLM's response as a Python dict
    import ast
    try:
        query_dict = ast.literal_eval(llm_response.get('choices', [{}])[0].get('text', '{}'))
    except Exception:
        query_dict = {}
    return query_dict

# --- Utility: LLM generates explanation ---
def get_llm_explanation(prompt, schema, sample, query, result):
    llm_prompt = (
        f"Schema: {schema}\nSample: {sample}\nQuery: {query}\nResult: {result}\n"
        f"User Question: {prompt}\n"
        "You are an expert data analyst. Based on the above, provide a clear, concise answer to the user's question in plain language, as if explaining to a non-technical end user. Summarize the key findings from the result and directly answer the user's prompt."
    )
    llm_response = call_llm_api(llm_prompt, schema, sample)
    return llm_response

# --- API Endpoints ---

# --- Trends and Drill-down Endpoints ---
@app.get("/api/trends")
def get_trends():
    """
    Returns a list of delivery trends (e.g., delayed deliveries, on-time, failed, etc.)
    Each trend has an id, name, and count.
    """
    # Dummy data for trends
    trends = [
        {"id": 1, "name": "Delayed Deliveries", "count": 12},
        {"id": 2, "name": "On-Time Deliveries", "count": 34},
        {"id": 3, "name": "Failed Deliveries", "count": 3},
    ]
    return {"trends": trends}

@app.get("/api/cases/{trend_id}")
def get_cases_for_trend(trend_id: int):
    """
    Returns a list of delivery cases for a given trend.
    """
    # Dummy data for cases
    cases = {
        1: [
            {"case_id": "D001", "order_id": "ORD123", "status": "Delayed", "customer": "Alice", "details": "Weather delay"},
            {"case_id": "D002", "order_id": "ORD124", "status": "Delayed", "customer": "Bob", "details": "Traffic jam"},
        ],
        2: [
            {"case_id": "O001", "order_id": "ORD125", "status": "On-Time", "customer": "Carol", "details": "Delivered as scheduled"},
        ],
        3: [
            {"case_id": "F001", "order_id": "ORD126", "status": "Failed", "customer": "Dave", "details": "Address not found"},
        ],
    }
    return {"cases": cases.get(trend_id, [])}

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


from difflib import SequenceMatcher
from llm_utils import parse_llm_multi_collection, replace_date_placeholders

def extract_collection_from_prompt(prompt, default_collection="deliveries", threshold=0.6):
    """
    Improved heuristic: return all relevant collections based on fuzzy keyword matching.
    - Returns a list of collection names ranked by relevance.
    - Uses fuzzy matching and keyword overlap, not just exact names.
    """
    known_collections = db.list_collection_names()
    prompt_lower = prompt.lower()
    relevant = []
    for name in known_collections:
        # Fuzzy match: collection name or its words appear in prompt, or high similarity
        name_lower = name.lower()
        # Direct substring or word overlap
        if name_lower in prompt_lower or any(word in prompt_lower for word in name_lower.split("_")):
            relevant.append((name, 1.0))
            continue
        # Fuzzy similarity
        sim = SequenceMatcher(None, name_lower, prompt_lower).ratio()
        if sim >= threshold:
            relevant.append((name, sim))
    # Sort by similarity descending
    relevant.sort(key=lambda x: -x[1])
    if relevant:
        return [name for name, _ in relevant]
    return [default_collection]

@app.post("/api/llm-query")
async def llm_query(request: Request):
    """
    Accepts a prompt, uses full MongoDB schema and sample data, and calls LLMs in priority order.
    Handles multiple relevant collections. Aggregates schemas, samples, and results.
    """
    data = await request.json()
    prompt = data.get("prompt", "")
    collections = extract_collection_from_prompt(prompt, data.get("collection", "deliveries"))
    logger.info(f"Received llm-query request: prompt={prompt}, collections={collections}")

    # Aggregate schemas and samples from all relevant collections
    schemas = {}
    samples = {}
    for collection in collections:
        schema, sample = get_mongo_collection_schema(collection)
        schemas[collection] = schema
        samples[collection] = sample


    # --- Dynamic Date Placeholders ---
    from datetime import datetime, timedelta
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    last_30_days_start = today - timedelta(days=29)
    today_iso = today.strftime("%Y-%m-%dT00:00:00Z")
    last_30_days_start_iso = last_30_days_start.strftime("%Y-%m-%dT00:00:00Z")

    # Compose a focused, strict prompt for the LLM
    all_collections = db.list_collection_names()
    available_collections_str = "Available collections: " + ", ".join(all_collections)
    # List fields for each collection
    collection_fields_str = "\n".join([
        f"{c} fields: {', '.join(str(f) for f in schemas[c])}" for c in collections
    ])
    # Add field mapping/description section for each collection if available
    mapping_strs = []
    for c in collections:
        if c in FIELD_MAPPINGS:
            mapping_strs.append(f"Field Mapping for {c}:\n" + "\n".join([
                f"- '{k}' → '{v['field']}' : {v['desc']}" for k, v in FIELD_MAPPINGS[c].items()
            ]))
    field_mapping_section = "\n\n".join(mapping_strs)
    # Few-shot example for hallucinated collection correction
    few_shot_example = (
        "Example: If the user asks for deliveries, but only the 'orders' collection contains delivery details, use 'orders' in your query. Do NOT use a 'deliveries' collection if it is not listed above.\n"
        "Q: List all deliveries in the last 30 days.\n"
        "A: Use the 'orders' collection and the 'actual_delivery_date' field."
    )
    llm_prompt = (
        "IMPORTANT: Only use the collection names and field names listed below. Do not invent or assume any collections or fields.\n"
        f"{available_collections_str}\n"
        f"{collection_fields_str}\n"
        "Do NOT use 'deliveries' or any collection not listed above.\n"
        f"{few_shot_example}\n"
        + (f"\n{field_mapping_section}\n" if field_mapping_section else "")
        + f"User Question: {prompt}\n"
        "When referring to dynamic date ranges like 'last 30 days', use the placeholders {last_30_days_start} and {today} in your queries. Do not use hardcoded dates. The backend will replace these placeholders with the actual dates before executing the query. "
        "When the user asks for a specific month (e.g., 'August 2025'), use the first day of that month as the start date and the first day of the next month as the end date. For example, for 'August 2025', use '$gte': '2025-08-01T00:00:00Z', '$lt': '2025-09-01T00:00:00Z'. "
        "Always use only the field names provided in the schema and field mapping. Do not invent or guess field names. "
        "If the user asks 'why' something failed, ensure your query retrieves the failure reason (e.g., 'reason', 'details', or similar field) and include it in the output. "
        "\n\n"
        "EXAMPLES FOR MONTH QUERIES AND FAILURE REASONS:\n"
        "Q: Why did customer 'John Doe' order fail in August 2025?\n"
        "A: Use a query like: { 'customer_name': 'John Doe', 'status': 'Failed', 'order_date': { '$gte': '2025-08-01T00:00:00Z', '$lt': '2025-09-01T00:00:00Z' } } and return the 'failure_reason' or 'details' field.\n"
        "\n"
        "RESPONSE FORMAT INSTRUCTIONS (IMPORTANT):\n"
        "Respond ONLY in the following JSON format, with no extra text, markdown, or explanation. Do not use code fences.\n"
        "{\n"
        "  \"plan\": <string, step-by-step plan>,\n"
        "  \"queries\": { <collection_name>: <MongoDB query dict or aggregation pipeline>, ... },\n"
        "  \"python\": <optional Python code for post-processing, as a string>\n"
        "}\n"
        "Do not include any other text, explanation, or formatting. Do not use markdown or code blocks. Only output the JSON object as shown above."
    )
    logger.info(f"LLM llm_prompt: {llm_prompt}")
    # Call LLM for multi-collection reasoning
    llm_response = call_llm_api(llm_prompt, schemas, samples)

    logger.info(f"LLM multi-collection response: {llm_response}")


    # --- LLM Multi-Collection Parsing and Execution ---
    plan, queries, python_code = parse_llm_multi_collection(llm_response)
    queries = replace_date_placeholders(queries, last_30_days_start_iso, today_iso)
    logger.info(f"LLM queries: {queries}")

    # Execute queries as per LLM plan
    results = {}
    for collection, query in queries.items():
        if not isinstance(query, dict):
            logger.error(f"LLM did not return a valid query dict for {collection}: {query}")
            continue
        try:
            docs = list(db[collection].find(query, limit=100))  # Increase limit for more robust answers
            for doc in docs:
                doc.pop('_id', None)
            results[collection] = docs
        except Exception as e:
            logger.error(f"Error executing MongoDB query for {collection}: {e}")
            results[collection] = []

    # --- Deduplicate results by order_id before post-processing ---
    def deduplicate_docs(docs, key='order_id'):
        seen = set()
        unique_docs = []
        for doc in docs:
            val = doc.get(key)
            if val is not None and val not in seen:
                seen.add(val)
                unique_docs.append(doc)
        return unique_docs

    for collection in results:
        results[collection] = deduplicate_docs(results[collection], key='order_id')

    # If LLM provided python code for post-processing, try to execute it (optional, advanced)
    final_result = results
    if python_code:
        try:
            # Provide 'results' in local scope for the code
            local_vars = {'results': results}
            exec(python_code, {}, local_vars)
            if 'final_result' in local_vars:
                final_result = local_vars['final_result']
        except Exception as e:
            logger.error(f"Error executing LLM-provided python code: {e}")

    # --- Only pass requested fields to LLM for explanation ---
    def extract_requested_fields(prompt, schemas):
        # Naive: look for field names in prompt that match schema fields
        requested = set()
        prompt_lower = prompt.lower()
        for collection, fields in schemas.items():
            for f in fields:
                if f.lower() in prompt_lower:
                    requested.add(f)
        # Fallback: if none found, return all fields (to avoid empty)
        if not requested:
            for fields in schemas.values():
                requested.update(fields)
        return list(requested)

    requested_fields = extract_requested_fields(prompt, schemas)
    # Filter final_result to only requested fields and limit to 5 items per collection
    def filter_and_limit_fields(data, fields, limit=5):
        # Lowercase set for case-insensitive matching
        fields_lc = set(f.lower() for f in fields)
        def filter_doc(doc):
            filtered = {fk: v2 for fk, v2 in doc.items() if fk.lower() in fields_lc}
            # If filtering results in empty dict, return original doc
            return filtered if filtered else doc
        if isinstance(data, dict):
            return {k: [
                filter_doc(v) for v in vlist[:limit]
            ] for k, vlist in data.items()}
        if isinstance(data, list):
            return [
                filter_doc(v) for v in data[:limit]
            ]
        return data

    filtered_result = filter_and_limit_fields(final_result, requested_fields, limit=5)
    logger.info(f"LLM filtered_result for explanation: {filtered_result}")
    explanation = get_llm_explanation(prompt, schemas, samples, queries, filtered_result)
    logger.info(f"LLM explanation: {explanation}")

    # Defensive: If LLM explanation says 'no data' but results exist, re-prompt or override
    def has_data(res):
        if isinstance(res, dict):
            return any(isinstance(v, list) and len(v) > 0 for v in res.values())
        if isinstance(res, list):
            return len(res) > 0
        return False

    def explanation_says_no_data(expl):
        try:
            if isinstance(expl, dict) and 'choices' in expl:
                content = expl['choices'][0]['message']['content'].lower()
                return 'no data' in content or 'no deliveries' in content or 'no records' in content
            if isinstance(expl, str):
                return 'no data' in expl.lower() or 'no deliveries' in expl.lower() or 'no records' in expl.lower()
        except Exception:
            return False
        return False

    if has_data(final_result) and explanation_says_no_data(explanation):
        # Re-prompt LLM with a more direct instruction
        logger.info("Explanation contradicted data, re-prompting LLM for correct summary.")
        reprompt = (
            f"The following data was found in response to the user's question: {final_result}. "
            f"User Question: {prompt}\n"
            "Please summarize the key findings and answer the user's question in plain language. Do not say 'no data' if there is data present."
        )
        explanation = call_llm_api(reprompt, schemas, samples)

    return {
        "llm_result": explanation,
        "plan": plan,
        "explanation": explanation,
        "queries": queries,
        "schemas": schemas,
        "samples": samples,
        "collections": collections,
        "raw_result": final_result  # full, unfiltered result
    }


# --- LLM Multi-Collection Parsing and Execution ---
import re
import ast
import json as _json
from typing import Any, Dict, Optional, Tuple

def _clean_jsonish(text: str) -> str:
    """Remove JS-style comments and ISODate wrappers for JSON parsing."""
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    text = re.sub(r'ISODate\("([^"]+)"\)', r'"\1"', text)
    return text.strip()

def extract_result_dict(text: str):
    # Try fenced JSON first
    json_blocks = re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    for block in json_blocks:
        try:
            return _json.loads(_clean_jsonish(block))
        except Exception:
            pass
    # Try any fenced code block that might contain a dict
    code_blocks = re.findall(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    for block in code_blocks:
        match = re.search(r"result\s*=\s*({[\s\S]*?})", block)
        if match:
            dict_str = match.group(1)
            try:
                return _json.loads(_clean_jsonish(dict_str))
            except Exception:
                try:
                    return ast.literal_eval(dict_str)
                except Exception:
                    pass
    # Try to find a top-level {...} in plain text
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate = _clean_jsonish(m.group(0))
        try:
            return _json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    return None

# -----------------------------
# Helpers
# -----------------------------

def _get_assistant_text(llm_response: Any) -> str:
    """
    Extract the assistant text across OpenAI/Gemini-like responses.
    """
    if isinstance(llm_response, dict):
        # OpenAI Chat Completions-like
        choices = llm_response.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg.get("content") or ""
                # Legacy text field
                if "text" in ch0:
                    return ch0.get("text") or ""
    # Already a string or unknown
    return str(llm_response or "")

def _try_parse_json_object(s: str) -> Optional[dict]:
    """
    Try to parse a JSON object from the given string. Return None if it fails.
    """
    if not s:
        return None
    s = s.strip()
    # If content looks like a JSON object
    if s.startswith("{") and s.endswith("}"):
        try:
            return _json.loads(s)
        except Exception:
            return None
    # Some models wrap JSON in code fences
    m = re.search(r"```(?:json|javascript|)\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return _json.loads(m.group(1))
        except Exception:
            return None
    return None

def _extract_plan_from_text(text: str) -> str:
    """
    Pull "Plan:" section when content isn't structured JSON.
    """
    m = re.search(r'(?is)Plan:\s*(.*?)\n\s*(?:Queries?:|$)', text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'(?is)Plan:\s*(.*)$', text)
    return m2.group(1).strip() if m2 else ""

def _iter_code_blocks(text: str):
    """
    Yield (language, code) for all fenced code blocks.
    """
    for m in re.finditer(r'```([a-zA-Z0-9_+-]*)\s*(.*?)```', text, flags=re.DOTALL):
        lang = (m.group(1) or "").lower()
        code = m.group(2)
        yield lang, code

def _first_braced_literal(s: str) -> str:
    """
    Return the first {...} or [...] literal by brace matching.
    """
    # Prefer an assignment right-hand side if present
    if '=' in s:
        # keep RHS of the first '=' (common in "query = {...}" / "pipeline = [...]")
        s = s.split('=', 1)[1]

    # Find first '{' or '['
    start_brace_pos = None
    start_ch = None
    for i, ch in enumerate(s):
        if ch in '{[':
            start_brace_pos = i
            start_ch = ch
            break
    if start_brace_pos is None:
        raise ValueError("No dict/list literal found.")

    open_ch, close_ch = ('{', '}') if start_ch == '{' else ('[', ']')
    depth = 0
    end_pos = None
    for i in range(start_brace_pos, len(s)):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
    if end_pos is None:
        raise ValueError("Unbalanced braces.")
    return s[start_brace_pos:end_pos].strip()

def _parse_literal(s: str):
    """
    Parse a Python/JSON literal safely. Keeps Mongo $-keys intact.
    """
    # Try Python literal
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # Try JSON (normalize single quotes)
    try:
        js_like = s
        # Only do a conservative quote normalization when obviously JSON-like
        if "'" in js_like and '"' not in js_like:
            js_like = js_like.replace("'", '"')
        return _json.loads(js_like)
    except Exception:
        raise

def _parse_query_from_code_block(code: str):
    """
    From a code block, try to parse a dict (filter) or list (pipeline).
    Returns either a dict or a list.
    """
    literal = _first_braced_literal(code)
    return _parse_literal(literal)

def _extract_python_block_after(text: str, heading_regex: str) -> Optional[str]:
    """
    Grab the first code block that appears after a given heading (regex).
    """
    h = re.search(heading_regex, text, flags=re.IGNORECASE)
    if not h:
        return None
    tail = text[h.end():]
    m = re.search(r'```(?:[a-zA-Z0-9_+-]*)\s*(.*?)```', tail, flags=re.DOTALL)
    return m.group(1) if m else None

def _maybe_parse_db_call(line: str):
    """
    Handle forms like:
      db.orders.find({...})
      db.orders.aggregate([...])
    Return (collection, value, kind) where kind is "filter" or "pipeline".
    """
    m = re.search(r'db\.([A-Za-z0-9_.-]+)\.(find|aggregate)\s*\((.*)\)\s*', line, flags=re.DOTALL)
    if not m:
        return None
    coll = m.group(1)
    op = m.group(2).lower()
    inside = m.group(3).strip()

    if op == 'find':
        # find(filter[, projection])
        # take the first literal argument
        lit = _first_braced_literal(inside)
        return coll, _parse_literal(lit), "filter"
    else:
        # aggregate([ ... ])
        lit = _first_braced_literal(inside)
        return coll, _parse_literal(lit), "pipeline"

# -----------------------------
# Main parser
# -----------------------------

def parse_llm_multi_collection(llm_response: Any) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Returns: (plan, queries, python_code)

    - plan: string (may be empty)
    - queries: dict mapping collection -> filter OR {"$pipeline": [...]} when an aggregation is detected
    - python_code: optional Python post-processing block if present
    """
    text = _get_assistant_text(llm_response).strip()

    # 1) If the entire content is a JSON object (or fenced JSON), use it
    as_json = _try_parse_json_object(text)
    if isinstance(as_json, dict) and as_json:
        plan = as_json.get("plan", "")
        queries_field = as_json.get("queries", {})
        python_code = as_json.get("python") or as_json.get("post_processing")  # optional

        queries: Dict[str, Any] = {}
        if isinstance(queries_field, dict):
            for coll, val in queries_field.items():
                # If it looks like an aggregation pipeline (list), store as $pipeline
                if isinstance(val, list):
                    queries[coll] = {"$pipeline": val}
                else:
                    queries[coll] = val
        return plan, queries, python_code

    # 2) Otherwise, parse free-form text + code blocks
    plan = _extract_plan_from_text(text)
    queries: Dict[str, Any] = {}
    python_code: Optional[str] = None

    # Try to read any code block labeled "Python code for post-processing"
    python_code = _extract_python_block_after(text, r'Python code for post-processing\s*:')

    # 2a) Try numbered or bullet collection headings BEFORE a code block, like:
    #     "1. orders:" or "- orders:" or "orders:" (line)
    #     followed by a fenced block containing dict/list
    pattern = re.compile(
        r'(?m)^(?:\s*(?:\d+\.\s*|[-*]\s*)?)'           # optional bullet/number
        r'([A-Za-z0-9_.:-]+)\s*:\s*$'                  # collection name + colon
    )

    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            break
        coll = m.group(1).strip()
        # Find the FIRST code block after this heading
        mcode = re.search(r'```([a-zA-Z0-9_+-]*)\s*(.*?)```', text[m.end():], flags=re.DOTALL)
        if not mcode:
            pos = m.end()
            continue
        code = mcode.group(2)
        try:
            parsed = _parse_query_from_code_block(code)
            if isinstance(parsed, list):
                queries[coll] = {"$pipeline": parsed}
            else:
                queries[coll] = parsed
        except Exception:
            pass
        pos = m.end()

    # 2b) If nothing yet, take the first code block after a "Queries" section
    if not queries:
        qsec = re.search(r'(?is)Queries?\s*:\s*(.*)', text)
        if qsec:
            mcode = re.search(r'```([a-zA-Z0-9_+-]*)\s*(.*?)```', qsec.group(1), flags=re.DOTALL)
            if mcode:
                try:
                    parsed = _parse_query_from_code_block(mcode.group(2))
                    # Unknown collection name; store under default
                    coll = "__default_pipeline__" if isinstance(parsed, list) else "__default__"
                    queries[coll] = {"$pipeline": parsed} if isinstance(parsed, list) else parsed
                except Exception:
                    pass

    # 2c) As a last resort, scan lines for db.<coll>.find(...) / .aggregate([...])
    if not queries:
        for line in text.splitlines():
            try:
                parsed = _maybe_parse_db_call(line)
                if parsed:
                    coll, val, kind = parsed
                    if kind == "pipeline":
                        queries[coll] = {"$pipeline": val}
                    else:
                        queries[coll] = val
            except Exception:
                continue

    # 2d) If still nothing, but there is a code block, attempt a blind parse into __default__
    if not queries:
        for _, code in _iter_code_blocks(text):
            try:
                parsed = _parse_query_from_code_block(code)
                coll = "__default_pipeline__" if isinstance(parsed, list) else "__default__"
                queries[coll] = {"$pipeline": parsed} if isinstance(parsed, list) else parsed
                break
            except Exception:
                continue

    return plan, queries, python_code

# -----------------------------
# Optional: placeholder resolver
# -----------------------------

def resolve_placeholders(obj: Any, **vars) -> Any:
    """
    Replace strings like "{today}" / "{last_30_days_start}" inside the parsed
    queries with actual values. Call this before sending to MongoDB.
    Example:
        queries = resolve_placeholders(queries,
                    today=datetime.utcnow(),
                    last_30_days_start=datetime.utcnow() - timedelta(days=30))
    """
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, **vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(x, **vars) for x in obj]
    if isinstance(obj, str):
        # Only replace exact whole-string placeholders (avoid breaking $expr/$dateFromString etc.)
        if obj.startswith("{") and obj.endswith("}") and obj[1:-1] in vars:
            return vars[obj[1:-1]]
    return obj



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

@app.post("/api/ingest-csv-to-db")
async def ingest_csv_to_db(file: UploadFile = File(...), collection: str = Form(...)):
    """
    Accepts a CSV file upload and loads its data into the specified MongoDB collection.
    """
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are supported"})
    try:
        contents = await file.read()
        decoded = contents.decode('utf-8')
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "w", encoding='utf-8') as f:
            f.write(decoded)
        count = ingest_csv_to_mongo(temp_path, collection)
        os.remove(temp_path)
        return {"status": "success", "rows_loaded": count, "collection": collection}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/health")
def health_check():
    """Health check endpoint for container/service monitoring."""
    return {"status": "ok"}
