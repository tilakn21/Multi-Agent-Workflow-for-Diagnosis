import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import redis
import whisper  # Requires pip install openai-whisper
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langgraph.graph import StateGraph
import boto3  # For S3 storage; requires pip install boto3
from botocore.exceptions import NoCredentialsError

# Setup logging for debugging and traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection (for storing outputs to pass to next agents)
r = redis.Redis(host="localhost", port=6379, db=0)

# S3/MinIO setup (replace with your credentials/endpoints)
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',  # For MinIO; use AWS endpoint for real S3
    aws_access_key_id='your_access_key',
    aws_secret_access_key='your_secret_key'
)
BUCKET_NAME = 'medical-audio-bucket'

# Initialize LangChain LLM for entity extraction
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Prompt template for extracting structured info from transcript
extraction_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
    You are a medical entity extractor. From the following doctor-patient conversation transcript, extract structured information:

    - Patient demographics: age, sex/gender, comorbidities (e.g., diabetes, hypertension).
    - Symptoms: list with name, duration, severity (e.g., mild/moderate/severe).
    - Context: travel history, risk factors, family history.
    - Intent: diagnosis, treatment, follow-up, or other.

    Normalize symptoms to standard terms if possible (e.g., 'high temperature' → 'fever').

    Output in JSON with keys: demographics, symptoms, context, intent, raw_transcript.

    Transcript: {transcript}

    Example Output:
    {{
      "demographics": {{"age": 45, "gender": "male", "comorbidities": ["diabetes"]}},
      "symptoms": [{{"name": "fever", "duration": "2 weeks", "severity": "moderate"}},
                   {{"name": "cough", "duration": "2 weeks", "severity": "severe"}},
                   {{"name": "weight loss", "duration": "3 kg", "severity": "mild"}}],
      "context": {{"travel_history": "Recent trip to India", "risk_factors": ["smoking"]}},
      "intent": "diagnosis",
      "raw_transcript": "Doctor, I have fever and cough for 2 weeks..."
    }}
    """
)

# Initialize LangChain chain for extraction
extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)

app = FastAPI(title="Audio-to-Text Agent")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    try:
        model = whisper.load_model("base")  # Use 'large' for better accuracy in production
        result = model.transcribe(audio_path)
        transcript = result["text"]
        logger.info(f"Transcribed audio: {transcript[:100]}...")  # Log snippet
        return transcript
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise ValueError("Failed to transcribe audio")

def extract_entities(transcript: str) -> Dict[str, Any]:
    """Extract structured info using LLM chain."""
    try:
        response = extraction_chain.run(transcript=transcript)
        extracted = json.loads(response)
        extracted["raw_transcript"] = transcript
        logger.info("Extracted entities successfully")
        return extracted
    except json.JSONDecodeError:
        logger.error("Invalid JSON from extraction chain")
        raise ValueError("Extraction failed: Invalid output")
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise ValueError("Failed to extract entities")

def store_audio_in_s3(audio_path: str, patient_id: str) -> str:
    """Store raw audio in S3/MinIO for traceability."""
    try:
        key = f"audio/{patient_id}/{datetime.utcnow().isoformat()}.mp3"
        s3_client.upload_file(audio_path, BUCKET_NAME, key)
        logger.info(f"Stored audio in S3: {key}")
        return key
    except NoCredentialsError:
        logger.warning("S3 credentials missing; skipping storage")
        return ""
    except Exception as e:
        logger.error(f"S3 upload error: {str(e)}")
        return ""

def store_output_in_redis(patient_id: str, output: Dict[str, Any]) -> None:
    """Store JSON output in Redis for next agent."""
    try:
        r.set(f"{patient_id}:audio_to_text", json.dumps(output))
        logger.info(f"Stored AudioToText output in Redis for {patient_id}")
    except Exception as e:
        logger.error(f"Redis storage error: {str(e)}")

@app.post("/process_audio/{patient_id}")
async def process_audio(patient_id: str, audio: UploadFile = File(...)):
    """FastAPI endpoint to receive audio, process, and output JSON."""
    try:
        # Save uploaded audio temporarily
        audio_path = f"/tmp/{audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        # Store raw audio in S3
        s3_key = store_audio_in_s3(audio_path, patient_id)
        
        # Transcribe
        transcript = transcribe_audio(audio_path)
        
        # Extract entities
        extracted = extract_entities(transcript)
        extracted["patient_id"] = patient_id
        extracted["audio_storage"] = s3_key  # Add storage ref for traceability
        extracted["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Clean up temp file
        os.remove(audio_path)
        
        # Store in Redis for Memory Agent
        store_output_in_redis(patient_id, extracted)
        
        return JSONResponse(status_code=200, content=extracted)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def audio_to_text_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for AudioToText Agent (placeholder; actual trigger via API)."""
    patient_id = state["patient_id"]
    try:
        # In full workflow, assume API has been called; fetch from Redis
        raw = r.get(f"{patient_id}:audio_to_text")
        if raw:
            intake = json.loads(raw)
            state["audio_to_text"] = intake
            state["status"] = "success"
        else:
            raise ValueError("No AudioToText output found")
    except Exception as e:
        state["audio_to_text"] = {"error": str(e)}
        state["status"] = "error"
    return state

# LangGraph workflow integration (partial; add to main graph)
workflow = StateGraph()
workflow.add_node("AudioToTextAgent", audio_to_text_node)
# Edges would be added in the full orchestrator, e.g., START -> AudioToTextAgent -> MemoryAgent

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)