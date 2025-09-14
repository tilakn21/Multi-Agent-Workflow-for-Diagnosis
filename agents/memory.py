import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import psycopg2
import redis
from pinecone import Pinecone, ServerlessSpec  # Requires pip install pinecone-client
from langchain.embeddings import OpenAIEmbeddings  # For vector embeddings
from langgraph.graph import StateGraph

# Setup logging for debugging and traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connections
r = redis.Redis(host="localhost", port=6379, db=0)
pg_conn = psycopg2.connect("dbname=meddb user=postgres password=secret")
pg_cursor = pg_conn.cursor()

# Pinecone setup (replace with your API key and environment)
pc = Pinecone(api_key="your_pinecone_api_key")
INDEX_NAME = "patient-cases"
EMBEDDING_DIM = 1536  # For OpenAI text-embedding-ada-002; adjust if using other models

# Check/create Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")  # Adjust region as needed
    )
index = pc.Index(INDEX_NAME)

# Initialize embeddings model (OpenAI; requires OPENAI_API_KEY env var)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def fetch_input_from_redis(patient_id: str) -> Dict[str, Any]:
    """Fetch structured JSON from AudioToText Agent in Redis."""
    try:
        raw = r.get(f"{patient_id}:audio_to_text")
        if raw:
            data = json.loads(raw)
            logger.info(f"Fetched AudioToText input for {patient_id}")
            return data
        else:
            logger.warning(f"No AudioToText input found for {patient_id}")
            return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from AudioToText for {patient_id}")
        return {}
    except Exception as e:
        logger.error(f"Redis fetch error: {str(e)}")
        return {}

def store_in_postgres(patient_id: str, data: Dict[str, Any]) -> None:
    """Store patient case in Postgres for persistence."""
    try:
        # Enhanced: Store demographics, symptoms, etc., as JSONB for flexibility
        pg_cursor.execute("""
            INSERT INTO patient_cases (patient_id, demographics, symptoms, context, intent, raw_transcript, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (patient_id) DO UPDATE SET
                demographics = EXCLUDED.demographics,
                symptoms = EXCLUDED.symptoms,
                context = EXCLUDED.context,
                intent = EXCLUDED.intent,
                raw_transcript = EXCLUDED.raw_transcript,
                timestamp = EXCLUDED.timestamp
        """, (
            patient_id,
            json.dumps(data.get("demographics", {})),
            json.dumps(data.get("symptoms", [])),
            json.dumps(data.get("context", {})),
            data.get("intent", ""),
            data.get("raw_transcript", ""),
            datetime.utcnow()
        ))
        pg_conn.commit()
        logger.info(f"Stored patient case in Postgres for {patient_id}")
    except Exception as e:
        logger.error(f"Postgres storage error for {patient_id}: {str(e)}")
        pg_conn.rollback()
        raise

def generate_embedding(data: Dict[str, Any]) -> List[float]:
    """Generate vector embedding from symptoms, demographics, and context."""
    try:
        # Concatenate key info for embedding (enhanced: normalize and weigh sections)
        text = f"Demographics: {json.dumps(data.get('demographics', {}))}. " \
               f"Symptoms: {json.dumps(data.get('symptoms', []))}. " \
               f"Context: {json.dumps(data.get('context', {}))}. " \
               f"Intent: {data.get('intent', '')}."
        
        # Embed (handle long text by truncating if needed)
        vector = embeddings.embed_query(text)
        logger.info(f"Generated embedding for {data.get('patient_id')}")
        return vector
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        raise ValueError("Failed to generate embedding")

def store_in_pinecone(patient_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
    """Upsert embedding and metadata to Pinecone."""
    try:
        # Enhanced metadata: Include searchable fields like age, symptoms list
        enhanced_meta = {
            "patient_id": patient_id,
            "demographics": metadata.get("demographics", {}),
            "symptoms": metadata.get("symptoms", []),
            "context": metadata.get("context", {}),
            "intent": metadata.get("intent", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        index.upsert(vectors=[(patient_id, vector, enhanced_meta)])
        logger.info(f"Stored embedding in Pinecone for {patient_id}")
    except Exception as e:
        logger.error(f"Pinecone upsert error for {patient_id}: {str(e)}")
        raise

def retrieve_similar_cases(patient_id: str, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k similar cases from Pinecone."""
    try:
        results = index.query(vector=vector, top_k=top_k, include_metadata=True)
        similar_cases = []
        for match in results['matches']:
            similar_cases.append({
                "similar_patient_id": match['id'],
                "similarity_score": match['score'],
                "metadata": match['metadata']
            })
        logger.info(f"Retrieved {len(similar_cases)} similar cases for {patient_id}")
        return similar_cases
    except Exception as e:
        logger.error(f"Pinecone query error for {patient_id}: {str(e)}")
        return []

def store_output_in_redis(patient_id: str, output: Dict[str, Any]) -> None:
    """Store similar cases JSON in Redis for next agent."""
    try:
        r.set(f"{patient_id}:memory", json.dumps(output))
        logger.info(f"Stored Memory output in Redis for {patient_id}")
    except Exception as e:
        logger.error(f"Redis storage error: {str(e)}")

def run_memory_agent(patient_id: str) -> Dict[str, Any]:
    """Full workflow: Store new case and retrieve similar ones."""
    try:
        data = fetch_input_from_redis(patient_id)
        if not data:
            raise ValueError("No input data from AudioToText")
        
        # Store in Postgres
        store_in_postgres(patient_id, data)
        
        # Generate and store embedding
        vector = generate_embedding(data)
        store_in_pinecone(patient_id, vector, data)
        
        # Retrieve similar cases (enhanced: exclude self if score=1)
        similar_cases = retrieve_similar_cases(patient_id, vector)
        similar_cases = [case for case in similar_cases if case['similar_patient_id'] != patient_id]
        
        # Output JSON
        output = {
            "patient_id": patient_id,
            "similar_cases": similar_cases,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Store in Redis for Literature Agent
        store_output_in_redis(patient_id, output)
        
        return output
    except Exception as e:
        logger.error(f"Error in run_memory_agent for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def get_memory_output(patient_id: str) -> Dict[str, Any]:
    """Retrieve stored memory output from Redis (fallback to Pinecone query if needed)."""
    try:
        raw = r.get(f"{patient_id}:memory")
        if raw:
            return json.loads(raw)
        else:
            # Fallback: Re-run retrieval if not in Redis (enhanced for robustness)
            data = fetch_input_from_redis(patient_id)
            if data:
                vector = generate_embedding(data)
                similar_cases = retrieve_similar_cases(patient_id, vector)
                return {"patient_id": patient_id, "similar_cases": similar_cases}
            return {"patient_id": patient_id, "error": "No memory output found"}
    except Exception as e:
        logger.error(f"Error retrieving memory for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for Memory Agent."""
    patient_id = state["patient_id"]
    try:
        memory_output = run_memory_agent(patient_id)
        state["memory"] = memory_output
        state["status"] = "success" if "error" not in memory_output else "error"
    except Exception as e:
        state["memory"] = {"patient_id": patient_id, "error": str(e)}
        state["status"] = "error"
    return state

# LangGraph workflow setup (partial; integrate with full graph)
workflow = StateGraph()
workflow.add_node("MemoryAgent", memory_node)
# Example edges: AudioToTextAgent -> MemoryAgent -> LiteratureAgent

if __name__ == "__main__":
    # Example usage (for testing)
    test_patient_id = "PAT-101"
    result = run_memory_agent(test_patient_id)
    print(json.dumps(result, indent=2))