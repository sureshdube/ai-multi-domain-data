"""
llm_utils.py - Helper functions for LLM-driven MongoDB query endpoint
"""
import re
import ast
import json as _json
from datetime import datetime, timedelta

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

def parse_llm_multi_collection(llm_response):
    plan = ""
    queries = {}
    python_code = None
    try:
        # Extract assistant text
        if isinstance(llm_response, dict) and 'choices' in llm_response:
            choice = llm_response['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                text = choice['message']['content']
            else:
                text = choice.get('text', '{}')
        elif isinstance(llm_response, dict) and 'plan' in llm_response:
            text = _json.dumps(llm_response)
        else:
            text = str(llm_response)
        # First attempt: direct JSON (after cleaning)
        parsed = None
        try:
            parsed = _json.loads(_clean_jsonish(text))
        except Exception:
            parsed = extract_result_dict(text)
        if parsed is None:
            raise ValueError("Could not parse LLM response as JSON or Python dict")
        plan = parsed.get('plan', '')
        queries = parsed.get('queries', {})
        python_code = parsed.get('python', None)
    except Exception as e:
        plan = str(llm_response)
        queries = {}
    return plan, queries, python_code

def replace_date_placeholders(obj, last_30_days_start_iso, today_iso):
    if isinstance(obj, dict):
        return {k: replace_date_placeholders(v, last_30_days_start_iso, today_iso) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_date_placeholders(x, last_30_days_start_iso, today_iso) for x in obj]
    elif isinstance(obj, str):
        if obj == "{last_30_days_start}":
            return last_30_days_start_iso
        elif obj == "{today}":
            return today_iso
    return obj
