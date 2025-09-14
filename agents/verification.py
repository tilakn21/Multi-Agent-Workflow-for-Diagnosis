import json
import logging
from datetime import datetime
from typing import Dict, Any
import psycopg2
import redis
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Setup logging for debugging and traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connections (shared with Decision Agent)
r = redis.Redis(host="localhost", port=6379, db=0)
pg_conn = psycopg2.connect("dbname=meddb user=postgres password=secret")
pg_cursor = pg_conn.cursor()

# Initialize LangChain LLM (shared)
llm = ChatOpenAI(model="gpt-4", temperature=0.0)  # Low temp for determinism

# Prompt for CrossVerifying Agent
verification_prompt = PromptTemplate(
    input_variables=["query", "memory_cases", "literature", "scraped_data", "decision"],
    template="""
    You are a clinical verification AI. Review the patient's query, supporting data, and initial decision for consistency, accuracy, and completeness. Cross-check diagnoses against evidence, flag inconsistencies, and provide a verified output.

    Always output valid JSON matching the schema: 
    {"verified_diagnoses": [{"condition": str, "confidence": float (0-1)}], 
     "verified_tests": [str], "verified_treatment": [str], "verification_explanation": str, 
     "inconsistencies": [str], "final_risk_assessment": {"severity": "low|medium|high", "urgency": "routine|soon|immediate"}, 
     "corrections": [str], "verified": bool, "confidence_method": str}.

    If inconsistencies are found, set "verified": false, suggest corrections, and lower confidences. If all aligns, set "verified": true.

    Patient query and details: {query}
    Similar cases: {memory_cases}
    Literature evidence: {literature}
    Live scraped data: {scraped_data}
    Initial decision: {decision}

    Example (few-shot):
    Input: Query: Fever, cough. Memory: TB case. Literature: PubMed TB. Scraped: Outbreak. Decision: TB 0.8, X-ray, etc.
    Output: {{"verified_diagnoses": [{{"condition": "TB", "confidence": 0.85}}], "verified_tests": ["X-ray"], "verified_treatment": ["Antibiotics"], "verification_explanation": "Decision aligns with evidence; minor confidence adjustment.", "inconsistencies": [], "final_risk_assessment": {{"severity": "high", "urgency": "immediate"}}, "corrections": [], "verified": true, "confidence_method": "Cross-verified evidence synthesis"}}

    Based on this:
    1. Verify 2–3 diagnoses with adjusted confidence scores.
    2. Verify/suggest tests and treatments.
    3. Explain verification, citing evidence.
    4. List any inconsistencies (e.g., 'Literature contradicts symptom match').
    5. Assess final risk/urgency.
    6. List corrections if needed.
    7. Set verified to true/false.
    8. Describe confidence method.
    Output ONLY the JSON.
    """
)

# Initialize LangChain chain for CrossVerifying Agent
verification_chain = LLMChain(llm=llm, prompt=verification_prompt)

def fetch_verification_inputs(patient_id: str) -> Dict[str, Any]:
    """Collect all inputs including decision for CrossVerifying Agent."""
    data = {}
    required_agents = ["audio_to_text", "memory", "literature", "web_scraping"]
    
    # Fetch prior agent outputs
    for agent in required_agents:
        raw = r.get(f"{patient_id}:{agent}")
        if raw:
            try:
                data[agent] = json.loads(raw)
                logger.info(f"Fetched {agent} for {patient_id}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {agent} for {patient_id}")
                data[agent] = {}
        else:
            logger.warning(f"Missing {agent} for {patient_id}")
            data[agent] = {}  # Fallback empty dict
    
    # Fetch decision from Redis or Postgres
    raw_decision = r.get(f"{patient_id}:decision")
    if raw_decision:
        try:
            data["decision"] = json.loads(raw_decision)
            logger.info(f"Fetched decision from Redis for {patient_id}")
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON for decision in Redis for {patient_id}")
            data["decision"] = {}
    else:
        # Fallback to Postgres
        try:
            pg_cursor.execute("""
                SELECT decision FROM decisions WHERE patient_id = %s 
                ORDER BY timestamp DESC LIMIT 1
            """, (patient_id,))
            result = pg_cursor.fetchone()
            if result:
                data["decision"] = json.loads(result[0])
                logger.info(f"Fetched decision from Postgres for {patient_id}")
            else:
                logger.warning(f"No decision found for {patient_id}")
                data["decision"] = {}
        except Exception as e:
            logger.error(f"Error fetching decision for {patient_id}: {str(e)}")
            data["decision"] = {}
    
    # Validate: Ensure decision exists
    if not data.get("decision") or "error" in data["decision"]:
        logger.error(f"No valid decision provided for {patient_id} - cannot verify")
        raise ValueError("Initial decision is required for verification")
    
    # Validate: Ensure query exists
    if not data.get("audio_to_text", {}).get("query"):
        logger.error(f"No query provided for {patient_id}")
        raise ValueError("Patient query is required for verification")
    
    return data

def store_verification_db(patient_id: str, verification: Dict[str, Any]) -> None:
    """Store verification JSON in Postgres for audit."""
    try:
        pg_cursor.execute("""
            INSERT INTO verifications (patient_id, verification, timestamp)
            VALUES (%s, %s, %s)
        """, (patient_id, json.dumps(verification), datetime.utcnow()))
        pg_conn.commit()
        logger.info(f"Stored verification for {patient_id} in Postgres")
    except Exception as e:
        logger.error(f"Failed to store verification for {patient_id}: {str(e)}")
        pg_conn.rollback()
        raise

def run_verification_agent(patient_id: str) -> Dict[str, Any]:
    """Gather all inputs including decision, run verification LLM chain, enhance, and store."""
    try:
        # Gather inputs
        inputs = fetch_verification_inputs(patient_id)
        
        # Run chain
        response = verification_chain.run(
            query=inputs.get("audio_to_text", {}).get("query", ""),
            memory_cases=json.dumps(inputs.get("memory", [])),
            literature=json.dumps(inputs.get("literature", {})),
            scraped_data=json.dumps(inputs.get("web_scraping", {})),
            decision=json.dumps(inputs.get("decision", {}))
        )
        
        # Parse JSON output
        verification = json.loads(response)
        
        # Validate: Ensure core keys exist
        required_keys = ["verified_diagnoses", "verified_tests", "verified_treatment", 
                         "verification_explanation", "inconsistencies", "final_risk_assessment", 
                         "corrections", "verified", "confidence_method"]
        for key in required_keys:
            if key not in verification:
                raise ValueError(f"Missing required key: {key}")
        
        # Enhance schema
        verification["patient_id"] = patient_id
        verification["verification_timestamp"] = datetime.utcnow().isoformat() + "Z"
        verification["model_version"] = "gpt-4-32k-2025-08"
        verification["final_risk_assessment"] = verification.get("final_risk_assessment", {"severity": "medium", "urgency": "soon"})
        verification["inconsistencies"] = verification.get("inconsistencies", [])
        verification["corrections"] = verification.get("corrections", [])
        verification["confidence_method"] = verification.get("confidence_method", "Cross-verified evidence synthesis")
        
        # Format explanation for frontend (basic Markdown)
        verification["formatted_verification_explanation"] = verification["verification_explanation"].replace(". ", ".\n\n").replace(", ", ", \n")
        
        # Store in Postgres
        store_verification_db(patient_id, verification)
        
        logger.info(f"Generated verification for {patient_id}")
        return verification
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response for {patient_id} in verification")
        return {"patient_id": patient_id, "error": "Invalid LLM output", "raw_output": response}
    except Exception as e:
        logger.error(f"Error in run_verification_agent for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def get_verification(patient_id: str) -> Dict[str, Any]:
    """Retrieve stored verification from Postgres."""
    try:
        pg_cursor.execute("""
            SELECT verification FROM verifications WHERE patient_id = %s 
            ORDER BY timestamp DESC LIMIT 1
        """, (patient_id,))
        result = pg_cursor.fetchone()
        if result:
            logger.info(f"Retrieved verification for {patient_id}")
            return json.loads(result[0])
        logger.warning(f"No verification found for {patient_id}")
        return {"patient_id": patient_id, "error": "No verification found"}
    except Exception as e:
        logger.error(f"Error retrieving verification for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def verification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for CrossVerifying Agent."""
    patient_id = state["patient_id"]
    try:
        verification = run_verification_agent(patient_id)
        state["verification"] = verification
        state["status"] = "success" if "error" not in verification else "error"
    except Exception as e:
        state["verification"] = {"patient_id": patient_id, "error": str(e)}
        state["status"] = "error"
    return state

# Example usage (for testing standalone)
if __name__ == "__main__":
    # Mock test (assuming prior agents and decision are in Redis/Postgres)
    test_state = {"patient_id": "PAT-101"}
    result = verification_node(test_state)
    print(json.dumps(result["verification"], indent=2))