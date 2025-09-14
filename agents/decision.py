import json
import logging
from datetime import datetime
from typing import Dict, Any
import psycopg2
import redis
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langgraph.graph import StateGraph

# Setup logging for debugging and traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connections
r = redis.Redis(host="localhost", port=6379, db=0)
pg_conn = psycopg2.connect("dbname=meddb user=postgres password=secret")
pg_cursor = pg_conn.cursor()

# Initialize LangChain LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.0)  # Low temp for determinism

# Enhanced prompt with few-shot example, updated for new inputs
decision_prompt = PromptTemplate(
    input_variables=["query", "memory_cases", "literature", "scraped_data"],
    template="""
    You are a clinical decision support AI. Always output valid JSON matching the schema: 
    {"differential_diagnoses": [{"condition": str, "confidence": float (0-1)}], 
     "recommended_tests": [str], "suggested_treatment": [str], "explanation": str, 
     "evidence_sources": [str], "risk_assessment": {"severity": "low|medium|high", "urgency": "routine|soon|immediate"}, 
     "alternative_explanations": [str], "confidence_method": str}.

    If data is missing or insufficient, assign low confidence (<0.5), suggest more tests, and note in explanation.

    Patient query and details: {query}
    Similar cases: {memory_cases}
    Literature evidence: {literature}
    Live scraped data: {scraped_data}

    Example (few-shot):
    Input: Query: Fever, cough. Memory: TB case (sim 0.8). Literature: PubMed TB link. Scraped: Recent outbreak data.
    Output: {{"differential_diagnoses": [{{"condition": "TB", "confidence": 0.8}}], "recommended_tests": ["X-ray"], "suggested_treatment": ["Antibiotics if confirmed"], "explanation": "Matches TB symptoms; cite PubMed.", "evidence_sources": ["PubMed:123"], "risk_assessment": {{"severity": "high", "urgency": "immediate"}}, "alternative_explanations": ["Viral if no bacteria"], "confidence_method": "Evidence weighting"}}

    Based on this:
    1. Propose 2–3 differential diagnoses with confidence scores (0-1).
    2. Suggest 2-4 diagnostic tests.
    3. Suggest treatment options (guideline-aligned, conditional).
    4. Provide a clear explanation citing evidence.
    5. Assess risk/urgency.
    6. List 1-2 alternatives.
    7. Describe confidence method (e.g., 'Bayesian based on matches').
    Output ONLY the JSON.
    """
)

# Initialize LangChain chain
decision_chain = LLMChain(llm=llm, prompt=decision_prompt)

def fetch_agent_outputs(patient_id: str) -> Dict[str, Any]:
    """Collect agent outputs from Redis with fallbacks and validation."""
    data = {}
    required_agents = ["audio_to_text", "memory", "literature", "web_scraping"]
    
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
    
    # Validate: Ensure query exists
    if not data.get("audio_to_text", {}).get("query"):
        logger.error(f"No query provided for {patient_id} - cannot proceed")
        raise ValueError("Patient query is required for decision making")
    
    return data

def store_decision_db(patient_id: str, decision: Dict[str, Any]) -> None:
    """Store decision JSON in Postgres for audit."""
    try:
        pg_cursor.execute("""
            INSERT INTO decisions (patient_id, decision, timestamp)
            VALUES (%s, %s, %s)
        """, (patient_id, json.dumps(decision), datetime.utcnow()))
        pg_conn.commit()
        logger.info(f"Stored decision for {patient_id} in Postgres")
    except Exception as e:
        logger.error(f"Failed to store decision for {patient_id}: {str(e)}")
        pg_conn.rollback()
        raise

def run_decision_agent(patient_id: str) -> Dict[str, Any]:
    """Gather inputs, run LLM chain, enhance output, and store decision."""
    try:
        # Gather inputs
        inputs = fetch_agent_outputs(patient_id)
        
        # Run chain
        response = decision_chain.run(
            query=inputs.get("audio_to_text", {}).get("query", ""),
            memory_cases=json.dumps(inputs.get("memory", [])),
            literature=json.dumps(inputs.get("literature", {})),
            scraped_data=json.dumps(inputs.get("web_scraping", {}))
        )
        
        # Parse JSON output
        decision = json.loads(response)
        
        # Validate: Ensure core keys exist
        required_keys = ["differential_diagnoses", "recommended_tests", "suggested_treatment", 
                        "explanation", "evidence_sources", "risk_assessment", 
                        "alternative_explanations", "confidence_method"]
        for key in required_keys:
            if key not in decision:
                raise ValueError(f"Missing required key: {key}")
        
        # Enhance schema
        decision["patient_id"] = patient_id
        decision["decision_timestamp"] = datetime.utcnow().isoformat() + "Z"
        decision["model_version"] = "gpt-4-32k-2025-08"
        decision["risk_assessment"] = decision.get("risk_assessment", {"severity": "medium", "urgency": "soon"})
        decision["alternative_explanations"] = decision.get("alternative_explanations", [])
        decision["confidence_method"] = decision.get("confidence_method", "LLM evidence synthesis")
        
        # Format evidence sources (add URLs for PubMed)
        for i, source in enumerate(decision["evidence_sources"]):
            if "PubMed" in source and ": " in source:
                pmid = source.split(": ")[1]
                decision["evidence_sources"][i] = f"{source} (https://pubmed.ncbi.nlm.nih.gov/{pmid})"
        
        # Format explanation for frontend (basic Markdown)
        decision["formatted_explanation"] = decision["explanation"].replace(". ", ".\n\n").replace(", ", ", \n")
        
        # Store in Postgres
        store_decision_db(patient_id, decision)
        
        logger.info(f"Generated decision for {patient_id}")
        return decision
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response for {patient_id}")
        return {"patient_id": patient_id, "error": "Invalid LLM output", "raw_output": response}
    except Exception as e:
        logger.error(f"Error in run_decision_agent for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def get_decision(patient_id: str) -> Dict[str, Any]:
    """Retrieve stored decision from Postgres."""
    try:
        pg_cursor.execute("""
            SELECT decision FROM decisions WHERE patient_id = %s 
            ORDER BY timestamp DESC LIMIT 1
        """, (patient_id,))
        result = pg_cursor.fetchone()
        if result:
            logger.info(f"Retrieved decision for {patient_id}")
            return json.loads(result[0])
        logger.warning(f"No decision found for {patient_id}")
        return {"patient_id": patient_id, "error": "No decision found"}
    except Exception as e:
        logger.error(f"Error retrieving decision for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for Decision Agent."""
    patient_id = state["patient_id"]
    try:
        decision = run_decision_agent(patient_id)
        state["decision"] = decision
        state["status"] = "success" if "error" not in decision else "error"
    except Exception as e:
        state["decision"] = {"patient_id": patient_id, "error": str(e)}
        state["status"] = "error"
    return state

# LangGraph workflow setup (updated for sequential workflow)
workflow = StateGraph()
workflow.add_node("AudioToTextAgent", lambda state: state)  # Placeholder for AudioToText node
workflow.add_node("MemoryAgent", lambda state: state)  # Placeholder
workflow.add_node("LiteratureAgent", lambda state: state)  # Placeholder
workflow.add_node("WebScrapingAgent", lambda state: state)  # Placeholder
workflow.add_node("DecisionAgent", decision_node)
workflow.add_node("CrossVerifyingAgent", lambda state: state)  # Placeholder for cross-verification

# Sequential edges
workflow.add_edge("START", "AudioToTextAgent")
workflow.add_edge("AudioToTextAgent", "MemoryAgent")
workflow.add_edge("MemoryAgent", "LiteratureAgent")
workflow.add_edge("LiteratureAgent", "WebScrapingAgent")
workflow.add_edge("WebScrapingAgent", "DecisionAgent")
workflow.add_edge("DecisionAgent", "CrossVerifyingAgent")
workflow.add_edge("CrossVerifyingAgent", "END")

if __name__ == "__main__":
    # Example usage (for testing)
    test_state = {"patient_id": "PAT-101"}
    result = decision_node(test_state)
    print(json.dumps(result["decision"], indent=2))
