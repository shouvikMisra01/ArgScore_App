import os
import json
import sys
from typing import Dict, Any, List, Tuple
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# --- OPENAI LIBRARY ---
from openai import OpenAI
import httpx
import traceback

# Load env vars
load_dotenv()

app = FastAPI(title="ArgScore MVP v0.4 (OpenAI)")
templates = Jinja2Templates(directory="templates")

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

# Initialize Client with manual httpx client to bypass version conflict
http_client = httpx.Client()
client = OpenAI(api_key=API_KEY, http_client=http_client)


# ============================================================
# ArgScore MVP v0.3 Constants & Validators
# ============================================================

HARD_DIMENSIONS = [
    "structure_validity",
    "support_sufficiency",
    "evidence_quality",
    "causal_discipline",
    "clarity_scope",
    "counterarguments",
    "uncertainty",
]

DEFAULT_WEIGHT_BOUNDS = {
    "structure_validity":   [0.12, 0.20],
    "support_sufficiency":  [0.10, 0.20],
    "evidence_quality":     [0.10, 0.22],
    "causal_discipline":    [0.08, 0.22],
    "clarity_scope":        [0.06, 0.16],
    "counterarguments":     [0.06, 0.18],
    "uncertainty":          [0.06, 0.16],
}

DOMAIN_TAXONOMY = [
    "physics", "engineering", "computer_science", "medicine", "biology", "economics",
    "political_science", "sociology", "law", "public_policy", "ethics", "philosophy",
    "business_strategy", "history", "journalism", "general",
]

# --- Validators ---

def die(msg: str, code: int = 500):
    # Updated to raise HTTPException instead of sys.exit
    raise HTTPException(status_code=code, detail=msg)

def validate_weight_bounds(bounds: Dict[str, List[float]]) -> None:
    for d in HARD_DIMENSIONS:
        if d not in bounds:
            die(f"allowed_weight_bounds missing dimension: {d}")
        lo, hi = bounds[d]
        if not (0 <= lo <= hi <= 1):
            die(f"Invalid bounds for {d}: {bounds[d]}")
    sum_mins = sum(bounds[d][0] for d in HARD_DIMENSIONS)
    sum_maxs = sum(bounds[d][1] for d in HARD_DIMENSIONS)
    if sum_mins > 1.0 + 1e-6:
        die(f"Sum of min bounds > 1: {sum_mins}")
    if sum_maxs < 1.0 - 1e-6:
        die(f"Sum of max bounds < 1: {sum_maxs}")

def validate_weights(weights: Dict[str, float], bounds: Dict[str, List[float]]) -> None:
    did_clamp = False
    for d in HARD_DIMENSIONS:
        if d not in weights:
             weights[d] = bounds[d][0] # Default to min if missing
        
        w = float(weights[d])
        lo, hi = bounds[d]
        
        # Clamp
        if w < lo:
            weights[d] = lo
            did_clamp = True
        elif w > hi:
            weights[d] = hi
            did_clamp = True
            
    # If we clamped, we must re-normalize to 1.0
    if did_clamp:
        total = sum(float(weights[d]) for d in HARD_DIMENSIONS)
        if total > 0:
            for d in HARD_DIMENSIONS:
                weights[d] = float(weights[d]) / total
    s = sum(float(weights[d]) for d in HARD_DIMENSIONS)
    if s <= 0:
        die(f"Weights sum to non-positive value: {s}")
    
    # Normalize if not 1.0
    if abs(s - 1.0) > 1e-9:
        for d in HARD_DIMENSIONS:
            weights[d] = float(weights[d]) / s

def validate_dimension_scores(scores: Dict[str, float]) -> None:
    for d in HARD_DIMENSIONS:
        if d not in scores:
            die(f"dimension_scores missing: {d}")
        v = float(scores[d])
        if not (0.0 <= v <= 1.0):
            die(f"dimension_scores[{d}] out of [0,1]: {v}")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def compute_final_score(weights: Dict[str, float], scores: Dict[str, float]) -> Tuple[float, float]:
    raw = 0.0
    for d in HARD_DIMENSIONS:
        raw += float(weights[d]) * float(scores[d])
    raw = clamp(raw, 0.0, 1.0)
    score_1to10 = 1.0 + 9.0 * raw
    score_1to10 = clamp(score_1to10, 1.0, 10.0)
    return raw, score_1to10


# ============================================================
# Prompts
# ============================================================

SYSTEM_JSON_STRICT = (
    "You are a precise evaluation engine. "
    "You MUST output ONLY valid JSON. No markdown. No commentary."
)

def prompt_fact_gate(argument: str, numeric_tolerance: float) -> str:
    return f"""
TASK: FACT-CHECK GATE (FIRST, BLOCKING)

IMPORTANT POLICY:
- This gate is NOT allowed to block due to "unclear" claims.
- This gate is NOT allowed to block due to normative/policy claims.
- This gate may block ONLY due to contradicted factual/numeric claims that are critical.

INPUT ARGUMENT:
{argument}

STEP 1: Extract claims
Return a list of claims with:
- id, text
- type: factual | numeric | causal | normative | definition | prediction
- verdict: supported | contradicted | unclear
- confidence in [0,1]
- criticality.centrality in [0,1]
- criticality.sensitivity in [0,1]
- numeric object only if type=numeric, else null

Numeric tolerance:
- tolerance = {numeric_tolerance:.2f}
- relative_error = |claimed - checked| / max(|checked|, 1e-9)
- within_tolerance = (relative_error <= tolerance)
- material: true if the mismatch would meaningfully change the conclusion, otherwise false
- If you cannot verify numeric truth value, set verdict="unclear" and numeric.checked_value=null

STEP 2: Decide gate using ONLY these rules:

Set gate="BLOCK" if and only if there exists at least one claim that satisfies:
(A) claim.type in ["factual","numeric","definition","prediction","causal"]
AND claim.verdict == "contradicted"
AND (claim.criticality.centrality >= 0.6 OR claim.criticality.sensitivity >= 0.6)

OR

(B) claim.type == "numeric"
AND claim.verdict == "contradicted"
AND claim.numeric.relative_error > tolerance
AND claim.numeric.material == true

Otherwise gate="PASS".

EXPLICIT FORBIDDEN BEHAVIOR:
- You MUST NOT set gate="BLOCK" if all contradictions are absent.
- You MUST NOT set gate="BLOCK" due to "unclear" claims.
- You MUST NOT set gate="BLOCK" due to any "normative" claim.

OUTPUT JSON SCHEMA:
{{
  "gate": "PASS" | "BLOCK",
  "message": "Facts passed." | "Check your facts.",
  "claims": [
    {{
      "id": "c1",
      "text": "...",
      "type": "factual|numeric|causal|normative|definition|prediction",
      "verdict": "supported|contradicted|unclear",
      "confidence": 0.0,
      "criticality": {{"centrality": 0.0, "sensitivity": 0.0}},
      "numeric": {{
        "claimed_value": 0.0,
        "checked_value": 0.0,
        "unit": "",
        "relative_error": 0.0,
        "within_tolerance": true,
        "material": false
      }}
    }}
  ],
  "notes": "short"
}}
"""

def prompt_hard_domain(argument: str) -> str:
    return f"""
TASK: HARD RUBRIC ANALYSIS (STATIC)
Determine:
1) domain_primary from taxonomy
2) domain_family
3) argument_type
4) question_type
5) evidence_profile
6) Create domain_packet with:
   - allowed_weight_bounds (start from DEFAULT and adjust slightly for domain)
   - required_domain_subtests (must attach to HARD_DIMENSIONS)

ARGUMENT:
{argument}

DOMAIN TAXONOMY:
{", ".join(DOMAIN_TAXONOMY)}

DEFAULT WEIGHT BOUNDS:
{json.dumps(DEFAULT_WEIGHT_BOUNDS, indent=2)}

OUTPUT JSON:
{{
  "domain_primary": "economics",
  "domain_family": "social_science|stem|law_policy|ethics_philosophy|business|history_media|general",
  "confidence": 0.0,
  "argument_type": "descriptive|causal|normative|prediction|definition|mixed",
  "question_type": "descriptive|causal|normative|policy|prediction|mixed",
  "evidence_profile": "anecdotal|observational_stats|experimental|theoretical|legal_precedent|mixed|none",
  "signals": ["...","..."],
  "domain_packet": {{
    "allowed_weight_bounds": {{
      "structure_validity":[0.0,0.0],
      "support_sufficiency":[0.0,0.0],
      "evidence_quality":[0.0,0.0],
      "causal_discipline":[0.0,0.0],
      "clarity_scope":[0.0,0.0],
      "counterarguments":[0.0,0.0],
      "uncertainty":[0.0,0.0]
    }},
    "required_domain_subtests": [
      {{"parent":"evidence_quality","name":"..."}},
      {{"parent":"causal_discipline","name":"..."}}
    ]
  }}
}}
CONSTRAINTS:
- Sum(mins) <= 1 <= Sum(maxs)
"""

def prompt_soft_rubric(domain_packet: Dict[str, Any], argument: str) -> str:
    return f"""
TASK: GENERATE SOFT RUBRIC (DOMAIN-SPECIFIC) + SCORE ARGUMENT

A) Generate weights within allowed bounds; sum to 1.0.
B) Include required_domain_subtests (and optionally add more under HARD_DIMENSIONS).
C) Score each hard dimension in [0,1].
D) final score: score_1to10 = 1 + 9*sum(w*dscore)

ARGUMENT:
{argument}

HARD DIMENSIONS:
{json.dumps(HARD_DIMENSIONS, indent=2)}

DOMAIN PACKET:
{json.dumps(domain_packet, indent=2)}

OUTPUT JSON:
{{
  "soft_rubric": {{
    "domain": "economics",
    "weights": {{
      "structure_validity": 0.0,
      "support_sufficiency": 0.0,
      "evidence_quality": 0.0,
      "causal_discipline": 0.0,
      "clarity_scope": 0.0,
      "counterarguments": 0.0,
      "uncertainty": 0.0
    }},
    "domain_subtests": [
      {{"parent":"evidence_quality","name":"...", "how_to_check":"short"}},
      {{"parent":"causal_discipline","name":"...", "how_to_check":"short"}}
    ]
  }},
  "dimension_scores": {{
    "structure_validity": 0.0,
    "support_sufficiency": 0.0,
    "evidence_quality": 0.0,
    "causal_discipline": 0.0,
    "clarity_scope": 0.0,
    "counterarguments": 0.0,
    "uncertainty": 0.0
  }},
  "raw": 0.0,
  "score_1to10": 0.0,
  "top_strengths": [
    {{"dim":"structure_validity","span":"...","reason":"..."}},
    {{"dim":"clarity_scope","span":"...","reason":"..."}},
    {{"dim":"uncertainty","span":"...","reason":"..."}}
  ],
  "top_weaknesses": [
    {{"dim":"evidence_quality","span":"...","reason":"..."}},
    {{"dim":"counterarguments","span":"...","reason":"..."}},
    {{"dim":"causal_discipline","span":"...","reason":"..."}}
  ],
  "improvements": [
    {{"action":"...", "expected_score_gain": 0.0}},
    {{"action":"...", "expected_score_gain": 0.0}}
  ]
}}
"""


# --- Helper: Chat with JSON Enforcement (OpenAI Version) ---
def chat_json(system_instruction: str, user_message: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        print(f"OpenAI Error: {e}")
        # If the model name is wrong, this helps us debug
        if "404" in str(e):
             raise HTTPException(status_code=404, detail=f"Model '{MODEL_NAME}' not found or access denied.")
        
        # Detailed error report with traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}\n\nTraceback:\n{tb_str}")

# --- Gemini Version (Commented Out) ---
# from google import genai
# from google.genai import types
# API_KEY_G = os.getenv("GOOGLE_API_KEY")
# MODEL_NAME_G = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
# client_g = genai.Client(api_key=API_KEY_G)
#
# def chat_json_gemini(system_instruction: str, user_message: str) -> Dict:
#     try:
#         response = client_g.models.generate_content(
#             model=MODEL_NAME_G,
#             contents=user_message,
#             config=types.GenerateContentConfig(
#                 system_instruction=system_instruction,
#                 response_mime_type="application/json"
#             )
#         )
#         return json.loads(response.text)
#     except Exception as e:
#         if "404" in str(e):
#              raise HTTPException(status_code=404, detail=f"Model '{MODEL_NAME_G}' not found.")
#         raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
# ----------------------------------------

# --- Data Models ---
class ArgumentRequest(BaseModel):
    text: str
    numeric_tolerance: float = 0.10

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def list_models():
    try:
        # OpenAI models listing is different, simplified for now
        return {"available_models": [MODEL_NAME], "current_target": MODEL_NAME}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/analyze")
async def analyze_argument(req: ArgumentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Argument text is empty")

    # 1. Fact Gate
    gate_res = chat_json(SYSTEM_JSON_STRICT, prompt_fact_gate(req.text, req.numeric_tolerance))
    
    if gate_res.get("gate") == "BLOCK":
        return {
            "status": "BLOCKED", 
            "message": gate_res.get("message"), 
            "claims": gate_res.get("claims"),
            "notes": gate_res.get("notes") # Forward notes
        }

    # 2. Hard Rubric -> Domain Packet
    domain_res = chat_json(SYSTEM_JSON_STRICT, prompt_hard_domain(req.text))
    
    domain_packet = domain_res.get("domain_packet")
    if not isinstance(domain_packet, dict):
        raise HTTPException(status_code=500, detail="AI produced invalid domain packet")
    
    validate_weight_bounds(domain_packet.get("allowed_weight_bounds", {}))

    # 3. Soft Rubric -> Scoring
    score_res = chat_json(SYSTEM_JSON_STRICT, prompt_soft_rubric(domain_packet, req.text))

    soft = score_res.get("soft_rubric", {})
    weights = soft.get("weights", {})
    dim_scores = score_res.get("dimension_scores", {})

    validate_weights(weights, domain_packet.get("allowed_weight_bounds", {}))
    validate_dimension_scores(dim_scores)

    raw, final_score = compute_final_score(weights, dim_scores)

    return {
        "status": "SUCCESS",
        "domain": domain_res.get("domain_primary", "General"),
        "confidence": domain_res.get("confidence", 0.0), # Added confidence
        "final_score": round(final_score, 2),
        "raw_score": round(raw, 3),
        "dimension_scores": dim_scores,
        "weights": weights,
        "strengths": score_res.get("top_strengths", []),
        "weaknesses": score_res.get("top_weaknesses", []),
        "improvements": score_res.get("improvements", []),
        "domain_subtests": soft.get("domain_subtests", []) # Detailed subtests
    }