import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# --- NEW GOOGLE LIBRARY (V2) ---
from google import genai
from google.genai import types

# Load env vars
load_dotenv()

app = FastAPI(title="ArgScore SaaS (Gemini V2)")
templates = Jinja2Templates(directory="templates")

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

# Initialize Client
client = genai.Client(api_key=API_KEY)

# --- Data Models ---
class ArgumentRequest(BaseModel):
    text: str
    numeric_tolerance: float = 0.10

# --- Helper: Chat with JSON Enforcement ---
def chat_json(system_instruction: str, user_message: str) -> Dict:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        # If the model name is wrong, this helps us debug
        if "404" in str(e):
             raise HTTPException(status_code=404, detail=f"Model '{MODEL_NAME}' not found. Check your .env file.")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

# --- Prompts ---
def get_fact_gate_prompt(text: str, tol: float):
    return f"""
    TASK: FACT-CHECK GATE.
    INPUT: {text}
    RULES: 
    - Block ONLY if critical factual/numeric claims are CONTRADICTED.
    - Do NOT block on 'unclear' or 'normative'.
    - Tolerance: {tol}.
    OUTPUT JSON schema: 
    {{ "gate": "PASS"|"BLOCK", "message": "string", "claims": [{{ "text": "string", "verdict": "supported|contradicted" }}] }}
    """

def get_domain_prompt(text: str):
    return f"""
    TASK: HARD RUBRIC ANALYSIS.
    INPUT: {text}
    OUTPUT JSON schema: 
    {{
      "domain_primary": "string", 
      "confidence": 0.0,
      "domain_packet": {{ "allowed_weight_bounds": {{ "structure_validity": [0.1, 0.2] }} }}
    }}
    """

def get_scoring_prompt(domain_data: Dict, text: str):
    return f"""
    TASK: SCORE ARGUMENT.
    DOMAIN DATA: {json.dumps(domain_data)}
    INPUT: {text}
    OUTPUT JSON schema: 
    {{
      "dimension_scores": {{ "structure_validity": 0.0, "support_sufficiency": 0.0, "evidence_quality": 0.0, "causal_discipline": 0.0, "clarity_scope": 0.0, "counterarguments": 0.0, "uncertainty": 0.0 }},
      "weights": {{ "structure_validity": 0.0, "support_sufficiency": 0.0, "evidence_quality": 0.0, "causal_discipline": 0.0, "clarity_scope": 0.0, "counterarguments": 0.0, "uncertainty": 0.0 }},
      "top_strengths": [{{ "dim": "string", "reason": "string" }}],
      "top_weaknesses": [{{ "dim": "string", "reason": "string" }}],
      "improvements": [{{ "action": "string", "expected_score_gain": 0.0 }}]
    }}
    """

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# DEBUG ROUTE: Run this in browser /api/models to see what works
@app.get("/api/models")
async def list_models():
    try:
        # Pager object iteration
        model_list = []
        for m in client.models.list():
            model_list.append(m.name)
        return {"available_models": model_list, "current_target": MODEL_NAME}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/analyze")
async def analyze_argument(req: ArgumentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Argument text is empty")

    SYSTEM_PROMPT = "You are a precise evaluation engine. Output valid JSON."

    # 1. Fact Gate
    gate_res = chat_json(SYSTEM_PROMPT, get_fact_gate_prompt(req.text, req.numeric_tolerance))
    if gate_res.get("gate") == "BLOCK":
        return {
            "status": "BLOCKED", 
            "message": gate_res.get("message"), 
            "claims": gate_res.get("claims")
        }

    # 2. Domain
    domain_res = chat_json(SYSTEM_PROMPT, get_domain_prompt(req.text))
    
    # 3. Scoring
    score_res = chat_json(SYSTEM_PROMPT, get_scoring_prompt(domain_res.get("domain_packet", {}), req.text))

    # Calculate Score
    dims = score_res.get("dimension_scores", {})
    weights = score_res.get("weights", {})
    
    # Safety defaults
    required_keys = ["structure_validity", "support_sufficiency", "evidence_quality", "causal_discipline", "clarity_scope", "counterarguments", "uncertainty"]
    for k in required_keys:
        if k not in dims: dims[k] = 0.5
        if k not in weights: weights[k] = 1.0/7.0

    raw_score = sum(weights.get(k,0) * dims.get(k,0) for k in dims)
    final_score = 1 + (9 * raw_score)

    return {
        "status": "SUCCESS",
        "domain": domain_res.get("domain_primary", "General"),
        "final_score": round(final_score, 2),
        "raw_score": round(raw_score, 3),
        "dimension_scores": dims,
        "weights": weights,
        "strengths": score_res.get("top_strengths", []),
        "weaknesses": score_res.get("top_weaknesses", []),
        "improvements": score_res.get("improvements", [])
    }