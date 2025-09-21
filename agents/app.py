# app.py
import os
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory import BioClinicalMemoryAgent

# ---------- Config ----------
MEMORY_JSON_PATH = os.getenv("MEMORY_JSON_PATH", "bioclinical_memory_new.json")
# FAISS path is derived automatically by save_memory: base + ".faiss"

# ---------- Schemas ----------
class Vitals(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure: Optional[str] = None
    oxygen_saturation: Optional[int] = None

class AddDocumentMetadata(BaseModel):
    condition: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    body_system: Optional[str] = None
    symptom_tags: Optional[List[str]] = None
    severity: Optional[str] = None
    treatment: Optional[str] = None
    vitals: Optional[Vitals] = None
    setting: Optional[str] = None
    patient_id: Optional[int] = None
    # Accept passthrough extras
    extra: Optional[Dict[str, Any]] = None

class AddDocumentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field("api")
    metadata: Optional[AddDocumentMetadata] = None

class AddDocumentResponse(BaseModel):
    doc_id: int
    chunks_added: int
    condition: Optional[str]
    body_system: Optional[str]
    total_documents: int
    total_chunks: int

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=100)
    return_chunks: bool = True
    similarity_threshold: float = Field(0.1, ge=0.0, le=1.0)
    use_metadata_filters: bool = False
    apply_lexical_rerank: bool = False

class ChunkInfo(BaseModel):
    chunk_index: Optional[int]
    total_chunks: Optional[int]
    chunk_id: Optional[int]

class RetrievalItem(BaseModel):
    text: str
    score: float
    source: str
    doc_id: int
    rank: int
    metadata: Dict[str, Any]
    chunk_info: Optional[ChunkInfo] = None

class RetrieveResponse(BaseModel):
    results: List[RetrievalItem]
    avg_similarity: float
    count: int

# ---------- App / Lifespan ----------
agent: Optional[BioClinicalMemoryAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = BioClinicalMemoryAgent()
    # Try to load an existing snapshot if present
    try:
        if os.path.exists(MEMORY_JSON_PATH) and os.path.exists(os.path.splitext(MEMORY_JSON_PATH)[0] + ".faiss"):
            agent.load_memory(MEMORY_JSON_PATH)
        else:
            # Fresh memory if no files yet
            agent.clear_memory()
    except Exception as e:
        # Start clean if load fails
        agent.clear_memory()
        print(f"[WARN] Failed to load memory, starting fresh: {e}")
    yield
    # Optionally save on shutdown (safety)
    try:
        agent.save_memory(MEMORY_JSON_PATH)
    except Exception as e:
        print(f"[WARN] Failed to save memory on shutdown: {e}")

app = FastAPI(title="BioClinical Memory API", version="1.0.0", lifespan=lifespan)

# ---------- Helpers ----------
def _serialize_result(r: Dict[str, Any]) -> RetrievalItem:
    chunk_info = r.get("chunk_info")
    return RetrievalItem(
        text=r["text"],
        score=r["score"],
        source=r["source"],
        doc_id=r["doc_id"],
        rank=r["rank"],
        metadata=r["metadata"],
        chunk_info=ChunkInfo(**chunk_info) if chunk_info else None
    )

# ---------- Endpoints ----------
@app.post("/add_document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    meta_dict: Optional[Dict[str, Any]] = None
    if req.metadata:
        meta_dict = req.metadata.dict(exclude_none=True)
        # Flatten vitals into dict
        if "vitals" in meta_dict and isinstance(meta_dict["vitals"], dict):
            pass  # already a dict via .dict()
        # Merge 'extra' into root level, if provided
        extra = meta_dict.pop("extra", None)
        if extra and isinstance(extra, dict):
            meta_dict.update(extra)

    try:
        before = agent.get_stats()
        doc_id = agent.add_document(text=req.text, source=req.source, metadata=meta_dict)
        after = agent.get_stats()
        # Persist immediately
        agent.save_memory(MEMORY_JSON_PATH)
        # Determine chunks added by delta
        chunks_added = int(after.get("total_chunks", 0) - before.get("total_chunks", 0))
        # Fetch doc-level metadata
        doc_meta = agent.doc_metadata[doc_id] if 0 <= doc_id < len(agent.doc_metadata) else {}
        return AddDocumentResponse(
            doc_id=doc_id,
            chunks_added=chunks_added,
            condition=doc_meta.get("condition"),
            body_system=doc_meta.get("body_system"),
            total_documents=after.get("total_documents", 0),
            total_chunks=after.get("total_chunks", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {e}")

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        results = agent.retrieve_similar_cases(
            query=req.query,
            k=req.k,
            return_chunks=req.return_chunks,
            similarity_threshold=req.similarity_threshold,
            use_metadata_filters=req.use_metadata_filters,
            apply_lexical_rerank=req.apply_lexical_rerank
        )
        avg = 0.0
        if results:
            avg = sum(r["score"] for r in results) / len(results)
        return RetrieveResponse(
            results=[_serialize_result(r) for r in results],
            avg_similarity=avg,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")