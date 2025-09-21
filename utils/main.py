from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from google import genai

from config import gemini_api, prompt_normal, prompt_with_context
import base64
import tempfile
import os
from rx_generate import *
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append(r"C:\Users\shubh\Desktop\Projects\Multi-Agent-Workflow-for-Patient-Diagnostics")
import main

   
import os
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MEMORY_RETRIEVE_URL = os.getenv("MEMORY_RETRIEVE_URL", "http://127.0.0.1:8000/retrieve")
LITERATURE_SEARCH_URL = os.getenv("LITERATURE_SEARCH_URL", "https://5f43ca9de8bb.ngrok-free.app/literature")

app = FastAPI(title="Medical Image Analysis API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your friend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CasePayload(BaseModel):
    text: str
    source: str = "doc_pat_symptomDescription"
    metadata: dict

class ComposeResponse(BaseModel):
    final_diagnosis: dict

class ComposeJSON(BaseModel):
    symptomDescription: Optional[str] = None
    patient_info: Optional[Dict[str, Any]] = None
    doctor_opinion: Optional[str] = None
    abha_id: Optional[str] = None
    image: Optional[str] = None




@app.post("/analyze_image")
async def analyze_image(
    image: str = Form(...),
    conversation: str = Form(default="")
):
    import google.generativeai as genai
    try:
        if not gemini_api:
            return JSONResponse(content={"error": "GEMINI_API_KEY not configured"}, status_code=500)

        # Select appropriate prompt based on whether conversation is provided
        prompt = prompt_normal if not conversation.strip() else prompt_with_context.format(conversation=conversation)

        # Configure Gemini API and create model object
        genai.configure(api_key=gemini_api)
        model = genai.GenerativeModel("models/gemini-2.0-flash")

        # Prepare the content with text and image
        contents = [
            prompt,
            {
                "inline_data": {
                    "mime_type": "image/png",  # or "image/jpeg" if you expect JPEG
                    "data": image.split(",")[-1]  # Remove data URI prefix if present
                }
            }
        ]

        # Call Gemini Vision model
        response = model.generate_content(contents)

        result = getattr(response, "text", None) or "No analysis result returned"

        return JSONResponse(content={"analysis": result})

    except Exception as e:
        return JSONResponse(content={"error": f"Image analysis failed: {str(e)}"}, status_code=500)

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Medical Image Analysis API"}

# Optional: Add an endpoint to check supported image formats
@app.get("/supported_formats")
async def supported_formats():
    return {
        "supported_formats": ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"],
        "max_file_size": "20MB",
        "recommended_format": "image/jpeg"
    }


@app.post("/generate_report_image", response_model=GenerateImageResponse)
async def generate_report_image(req: GenerateImageRequest):
    print("[DEBUG] starting image generation")
    try:
        img_bytes = create_prescription_image(req.final_diagnosis)
        print("[DEBUG] image created successfully")
    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    file_name = f"prescription_{uuid.uuid4().hex}.png"
    try:
        res = supabase.storage.from_('GDHS').upload(file_name, img_bytes)
        print("Supabase upload response:", res)
        error = getattr(res, "error", None)
        if error:
            message = getattr(error, "message", str(error))
            raise Exception(message)
        url = f"{SUPABASE_URL}/storage/v1/object/public/GDHS/{file_name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uploading to Supabase failed: {e}")

    if not url or not isinstance(url, str):
        raise HTTPException(status_code=500, detail="Image upload did not return a valid URL")

    return GenerateImageResponse(image_url=url)
 

def build_prompt(symptomDescription: str, opinion: str, vitals_json: str, abha_id: str) -> str:
    return f"""
            You are a clinical reasoning assistant. Given the inputs below, write a single JSON object strictly in this structure:

            {{
                "text": "<one-paragraph, clinically-written case summary with symptoms, timeline, vitals, exam, provisional diagnosis, differentials, and plan>",
                "source": "doc_pat_symptomDescription",
                "metadata": {{
                    "age": <int if known else omit>,
                    "sex": "<male|female|unknown>",
                    "body_system": "<dominant system, e.g., respiratory, cardiology, neurology, etc.>",
                    "symptom_tags": ["tag1","tag2"],
                    "vitals": {{"temperature": <float|omit>, "heart_rate": <int|omit>, "blood_pressure": "<str|omit>", "oxygen_saturation": <int|omit>}},
                    "abha_id": "<exact input ABHA id>"
                }}
            }}

            Rules:
            - Output ONLY valid JSON. No markdown, no explanations, no extra keys.
            - If information is missing, omit that key (do not invent values).
            - Keep "text" as one cohesive medical paragraph, 3–6 sentences max.
            - Keep metadata concise and consistent with the text.
            - Choose body_system from: respiratory, cardiology, neurology, dermatology, genitourinary, gastroenterology, endocrine, ophthalmology, musculoskeletal, vascular, general.
            - Derive symptom_tags from the content.

            Inputs:
            symptomDescription:
            {symptomDescription}

            Doctor opinion:
            {opinion}

            Patient vitals (JSON if provided):
            {vitals_json}

            ABHA ID:
            {abha_id}
    """.strip()

def handle_case(case: dict) -> dict:
    text = case.get("text", "")
    meta = case.get("metadata", {}) or {}
    abha_id = meta.get("abha_id", "Unknown")
    return {"text": text, "meta": meta, "abha_id": abha_id, "text_len": len(text)}

def post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "url": url}

def route_case_to_apis(case: dict) -> dict:
    summary = {"memory_retrieve": None, "literature_search": None}
    query_text = case.get("text", "") or ""
    retrieval_payload = {
        "query": query_text,
        "k": 1,
        "return_chunks": True,
        "similarity_threshold": 0.8
    }
    summary["memory_retrieve"] = post_json(MEMORY_RETRIEVE_URL, retrieval_payload)
    literature_payload = {"query": query_text}
    summary["literature_search"] = post_json(LITERATURE_SEARCH_URL, literature_payload)
    return summary

def on_compose_case_response(compose_response: dict) -> dict:
    case = compose_response.get("case") or {}
    local_result = handle_case(case)
    api_results = route_case_to_apis(case)
    return {"local_result": local_result, "api_results": api_results}


@app.post("/compose_case", response_model=ComposeResponse)
async def compose_case(payload: ComposeJSON):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    symptomDescription = payload.symptomDescription or ""
    doctor_opinion = payload.doctor_opinion or ""
    patient_vitals = json.dumps(payload.patient_info) if payload.patient_info else None
    abha_id = payload.abha_id or ""
    image_note = "(An image was provided.)" if payload.image else ""

    vitals_json_str = ""
    if patient_vitals:
        try:
            _ = json.loads(patient_vitals)
            vitals_json_str = patient_vitals
        except Exception:
            vitals_json_str = patient_vitals

    prompt = build_prompt(symptomDescription + "\n" + image_note, doctor_opinion, vitals_json_str, abha_id)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw_text = resp.text or ""
        print("[DEBUG] LLM raw response:")
        print(raw_text)
    except Exception as e:
        print(f"[ERROR] Gemini error: {e}")
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    try:
        obj = json.loads(raw_text)
        print("[DEBUG] Parsed summary JSON:")
        print(obj)
    except Exception:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(raw_text[start:end+1])
                print("[DEBUG] Fallback parsed summary JSON:")
                print(obj)
            except Exception as e:
                print(f"[ERROR] Invalid JSON from model: {e}")
                raise HTTPException(status_code=500, detail=f"Invalid JSON from model: {e}")
        else:
            print("[ERROR] Model did not return JSON")
            raise HTTPException(status_code=500, detail="Model did not return JSON")

    if "text" not in obj or not isinstance(obj["text"], str) or not obj["text"].strip():
        print("[ERROR] Model output missing 'text']")
        raise HTTPException(status_code=500, detail="Model output missing 'text'")

    obj["source"] = "doc_pat_symptomDescription"
    obj.setdefault("metadata", {})
    if isinstance(obj["metadata"], dict):
        obj["metadata"]["abha_id"] = abha_id

    combined = on_compose_case_response({"case": obj})
    api_results = dict(combined["api_results"]) if combined.get("api_results") else {}
    literature_search = api_results.get("literature_search", {})
    filtered_literature = literature_search.get("llm_response") if isinstance(literature_search, dict) else None
    filtered_api_results = {
        "memory_retrieve": api_results.get("memory_retrieve"),
        "literature_search_llm_response": filtered_literature
    }
    print("[DEBUG] Filtered API results:")
    print(json.dumps(filtered_api_results, indent=2))

    summary_kv = json.dumps(obj, indent=2)
    api_results_kv = json.dumps(filtered_api_results, indent=2)
    diagnosis_prompt = f"""
            You are a clinical diagnosis assistant. Given the following patient summary and external retrievals, provide a comprehensive clinical assessment with diagnoses listed in order of likelihood. Output ONLY a JSON object in the exact format specified below. No explanations, no markdown.
            Note: no need of excessive thinking, just do what is asked.
            Patient summary:
            {summary_kv}


            External retrievals:
            {api_results_kv}


            Required JSON format:
            {{
                "final_diagnosis": {{
                    "patient_problem": "Brief summary of the patient's presentation and clinical findings",
                    "diagnosis": ["Most likely diagnosis", "Second most likely", "Third most likely", "etc."],
                    "diagnosis_simplified": ["for patient understanding, in simple language"],
                    "treatment_plan": {{
                        "medications": ["specific medications with dosing if appropriate"],
                        "lifestyle_modifications": ["dietary, activity, or behavioral recommendations"]
                    }},
                    "metadata": {{
                        "body_system": "primary body system affected",
                        "symptom_tags": ["list", "of", "key", "symptoms"],
                        "vitals": {{
                            "temperature": float_value_or_null,
                            "heart_rate": int_value_or_null,
                            "blood_pressure": "systolic/diastolic_or_null",
                            "oxygen_saturation": int_value_or_null
                        }},
                        "abha_id": "patient_identifier"
                    }},
                    "references": ["list of links of sources used from pubmed or who"]
                }}
            }}
            also at the end, give the accuracy score of the diagnosis parsed from the above information
    """
    diagnosis = []
    diagnosis_simplified = []
    treatment_plan = None
    llm_metadata = None
    references = []
    
    try:
        diag_resp = client.models.generate_content(model=GEMINI_MODEL, contents=diagnosis_prompt)
        diag_text = diag_resp.text or ""
        print("[DEBUG] LLM diagnosis response:")
        print(diag_text)
        
        try:
            parsed = json.loads(diag_text)
        except Exception:
            start = diag_text.find("{")
            end = diag_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(diag_text[start:end+1])
            else:
                raise Exception("No valid JSON found in response")

        if isinstance(parsed, dict) and "final_diagnosis" in parsed:
            fd = parsed["final_diagnosis"]
            diagnosis = fd.get("diagnosis", [])
            diagnosis_simplified = fd.get("diagnosis_simplified", [])
            treatment_plan = fd.get("treatment_plan")
            references = fd.get("references", [])
        elif isinstance(parsed, dict) and "diagnosis" in parsed:
            diagnosis = parsed["diagnosis"]
            diagnosis_simplified = parsed.get("diagnosis_simplified", [])
            treatment_plan = parsed.get("treatment_plan")
            references = parsed.get("references", [])
        elif isinstance(parsed, list):
            diagnosis = parsed
            
    except Exception as e:
        print(f"[ERROR] Diagnosis ranking failed: {e}")
        diagnosis = obj["metadata"].get("condition", [])
        pass

    print("[DEBUG] Final response structure:")
    final_diagnosis = {
        "patient_problem": obj["text"],
        "diagnosis": diagnosis,
        "diagnosis_simplified": diagnosis_simplified,
        "metadata": obj["metadata"]
 
    }
    
    if treatment_plan:
        final_diagnosis["treatment_plan"] = treatment_plan
    if references:
        final_diagnosis["references"] = references
        
    print(json.dumps(final_diagnosis, indent=2))
    return {"final_diagnosis": final_diagnosis}