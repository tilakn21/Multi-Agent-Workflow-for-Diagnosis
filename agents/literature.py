import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import psycopg2
import redis
from pinecone import Pinecone, ServerlessSpec  # Requires pip install pinecone-client
from langchain.embeddings import OpenAIEmbeddings  # For vector embeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langgraph.graph import StateGraph
import boto3  # For S3 document access; requires pip install boto3
from botocore.exceptions import NoCredentialsError

# Setup logging for debugging and traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connections
r = redis.Redis(host="localhost", port=6379, db=0)
pg_conn = psycopg2.connect("dbname=meddb user=postgres password=secret")
pg_cursor = pg_conn.cursor()

# Pinecone setup for literature chunks (replace with your API key)
pc = Pinecone(api_key="your_pinecone_api_key")
LIT_INDEX_NAME = "medical-literature"
EMBEDDING_DIM = 1536  # For OpenAI text-embedding-ada-002

# Check/create Pinecone index for literature
if LIT_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=LIT_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
lit_index = pc.Index(LIT_INDEX_NAME)

# S3/MinIO setup for full documents (replace credentials/endpoints)
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',  # For MinIO; use AWS for production
    aws_access_key_id='your_access_key',
    aws_secret_access_key='your_secret_key'
)
DOC_BUCKET_NAME = 'medical-literature-bucket'

# Initialize embeddings model (OpenAI)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize LLM for summarization
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Prompt template for summarizing retrieved chunks
summary_prompt = PromptTemplate(
    input_variables=["query", "chunks"],
    template="""
    You are a medical evidence summarizer. Given the patient query and retrieved literature chunks, summarize the relevant evidence into concise notes. Focus on key findings, guidelines, and citations. Avoid hallucinations; stick to provided chunks.

    Patient query: {query}

    Retrieved chunks:
    {chunks}

    Output in JSON with keys: evidence_snippets (list of {"snippet": str, "source": str, "relevance_score": float}), summary (str).

    Example:
    {{
      "evidence_snippets": [
        {{"snippet": "TB treatment: HRZE for 6 months.", "source": "WHO Guideline 2023, page 45", "relevance_score": 0.92}},
        {{"snippet": "Sputum test sensitivity 85%.", "source": "PubMed PMID:123456", "relevance_score": 0.85}}
      ],
      "summary": "Guidelines recommend HRZE regimen for TB; sputum testing is first-line."
    }}
    """
)

# Initialize LangChain chain for summarization
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

def fetch_input_from_redis(patient_id: str) -> Dict[str, Any]:
    """Fetch inputs from previous agents (AudioToText query + Memory similar cases) in Redis."""
    try:
        data = {}
        # Fetch patient query from AudioToText
        raw_query = r.get(f"{patient_id}:audio_to_text")
        if raw_query:
            data["query_data"] = json.loads(raw_query)
            logger.info(f"Fetched AudioToText query for {patient_id}")
        
        # Fetch similar cases from Memory (optional but enhances context)
        raw_memory = r.get(f"{patient_id}:memory")
        if raw_memory:
            data["memory_data"] = json.loads(raw_memory)
            logger.info(f"Fetched Memory data for {patient_id}")
        
        if not data.get("query_data"):
            logger.warning(f"No query data found for {patient_id}")
            return {}
        return data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from previous agents for {patient_id}")
        return {}
    except Exception as e:
        logger.error(f"Redis fetch error: {str(e)}")
        return {}

def generate_query_embedding(data: Dict[str, Any]) -> List[float]:
    """Generate embedding for the patient query + context from memory."""
    try:
        query = data["query_data"].get("raw_transcript", "")  # Use transcript or structured symptoms
        context = json.dumps(data.get("memory_data", {}).get("similar_cases", []))  # Add similar cases for better relevance
        
        # Enhanced: Combine query with memory context
        text = f"Patient query: {query}. Similar cases context: {context}."
        
        vector = embeddings.embed_query(text)
        logger.info(f"Generated query embedding for {data['query_data'].get('patient_id')}")
        return vector
    except Exception as e:
        logger.error(f"Query embedding error: {str(e)}")
        raise ValueError("Failed to generate query embedding")

def retrieve_from_pinecone(vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant chunks from Pinecone literature index."""
    try:
        results = lit_index.query(vector=vector, top_k=top_k, include_metadata=True)
        chunks = []
        for match in results['matches']:
            chunks.append({
                "chunk_text": match['metadata'].get('text', ''),
                "source": match['metadata'].get('source', 'Unknown'),
                "citation": match['metadata'].get('citation', ''),
                "relevance_score": match['score']
            })
        logger.info(f"Retrieved {len(chunks)} literature chunks")
        return chunks
    except Exception as e:
        logger.error(f"Pinecone query error: {str(e)}")
        return []

def summarize_evidence(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize retrieved chunks using LLM chain."""
    try:
        chunks_str = json.dumps(chunks)  # Pass as string to prompt
        response = summary_chain.run(query=query, chunks=chunks_str)
        summary = json.loads(response)
        logger.info("Summarized literature evidence")
        return summary
    except json.JSONDecodeError:
        logger.error("Invalid JSON from summary chain")
        raise ValueError("Summarization failed: Invalid output")
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise ValueError("Failed to summarize evidence")

def fetch_full_document(source: str) -> str:
    """Fetch full document text from S3 if needed (enhanced for deep dives)."""
    try:
        obj = s3_client.get_object(Bucket=DOC_BUCKET_NAME, Key=source)
        text = obj['Body'].read().decode('utf-8')
        logger.info(f"Fetched full document from S3: {source}")
        return text
    except NoCredentialsError:
        logger.warning("S3 credentials missing; skipping full doc fetch")
        return ""
    except Exception as e:
        logger.error(f"S3 fetch error for {source}: {str(e)}")
        return ""

def store_output_in_redis(patient_id: str, output: Dict[str, Any]) -> None:
    """Store literature evidence JSON in Redis for next agent."""
    try:
        r.set(f"{patient_id}:literature", json.dumps(output))
        logger.info(f"Stored Literature output in Redis for {patient_id}")
    except Exception as e:
        logger.error(f"Redis storage error: {str(e)}")

def run_literature_agent(patient_id: str) -> Dict[str, Any]:
    """Full workflow: Retrieve and summarize literature based on query."""
    try:
        data = fetch_input_from_redis(patient_id)
        if not data:
            raise ValueError("No input data from previous agents")
        
        query = data["query_data"].get("raw_transcript", "")  # Or use structured symptoms
        
        # Generate query embedding
        vector = generate_query_embedding(data)
        
        # Retrieve chunks
        chunks = retrieve_from_pinecone(vector)
        
        # Enhanced: If high-relevance chunks reference full docs, fetch one (e.g., top chunk)
        if chunks and chunks[0]["relevance_score"] > 0.8:
            full_text = fetch_full_document(chunks[0]["source"])
            if full_text:
                chunks[0]["full_text"] = full_text[:2000]  # Truncate to avoid token limits
        
        # Summarize
        summary = summarize_evidence(query, chunks)
        
        # Output JSON
        output = {
            "patient_id": patient_id,
            "evidence": summary.get("evidence_snippets", []),
            "summary": summary.get("summary", ""),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Store in Redis for Web Scraping Agent
        store_output_in_redis(patient_id, output)
        
        return output
    except Exception as e:
        logger.error(f"Error in run_literature_agent for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def get_literature_output(patient_id: str) -> Dict[str, Any]:
    """Retrieve stored literature output from Redis."""
    try:
        raw = r.get(f"{patient_id}:literature")
        if raw:
            return json.loads(raw)
        return {"patient_id": patient_id, "error": "No literature output found"}
    except Exception as e:
        logger.error(f"Error retrieving literature for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}

def literature_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for Literature Agent."""
    patient_id = state["patient_id"]
    try:
        lit_output = run_literature_agent(patient_id)
        state["literature"] = lit_output
        state["status"] = "success" if "error" not in lit_output else "error"
    except Exception as e:
        state["literature"] = {"patient_id": patient_id, "error": str(e)}
        state["status"] = "error"
    return state

# LangGraph workflow setup (partial)
workflow = StateGraph()
workflow.add_node("LiteratureAgent", literature_node)
# Edges: MemoryAgent -> LiteratureAgent -> WebScrapingAgent

# Optional: Function to ingest new documents (for setup/hackathon prep)
def ingest_document(doc_path: str, metadata: Dict[str, Any]) -> None:
    """Ingest a new document: Chunk, embed, store in Pinecone and S3/Postgres."""
    try:
        # Upload to S3
        s3_key = f"docs/{metadata.get('title', 'untitled')}.txt"
        s3_client.upload_file(doc_path, DOC_BUCKET_NAME, s3_key)
        
        # Store metadata in Postgres
        pg_cursor.execute("""
            INSERT INTO literature_docs (doc_id, title, authors, year, journal, citation, s3_key, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            metadata.get("doc_id"),
            metadata.get("title"),
            json.dumps(metadata.get("authors", [])),
            metadata.get("year"),
            metadata.get("journal"),
            metadata.get("citation"),
            s3_key,
            datetime.utcnow()
        ))
        pg_conn.commit()
        
        # Chunk and embed (simple: split by paragraphs; enhance with langchain splitters)
        with open(doc_path, 'r') as f:
            text = f.read()
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]  # Basic chunking
        
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            chunk_meta = {
                "text": chunk,
                "source": s3_key,
                "citation": metadata.get("citation"),
                "doc_id": metadata.get("doc_id"),
                "chunk_id": i,
                "tags": metadata.get("tags", [])
            }
            lit_index.upsert(vectors=[(f"{metadata['doc_id']}_{i}", vector, chunk_meta)])
        
        logger.info(f"Ingested document: {metadata.get('title')}")
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        pg_conn.rollback()

if __name__ == "__main__":
    # Example usage (for testing)
    test_patient_id = "PAT-101"
    result = run_literature_agent(test_patient_id)
    print(json.dumps(result, indent=2))
    
    # Example ingestion (run once for setup)
    # ingest_document("path/to/guideline.pdf.txt", {"doc_id": "WHO-TB-2023", "title": "WHO TB Guidelines 2023", "authors": ["WHO"], "year": 2023, "journal": "WHO", "citation": "WHO Guideline NG123", "tags": ["tuberculosis", "guidelines"]})