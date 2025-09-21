"""
Simplified Vector DB for Scraped Articles
Focuses on chunking, embedding, FAISS indexing, and semantic search for relevant content based on prompts.
Updated to load only the 'content' entity from each JSON item, ignoring other entities.
Persists the vector DB to disk for reuse without reloading into memory every time.
Approaches and models filtered from the original script: uses SentenceTransformer for embeddings (simplified from BioBERT), RecursiveCharacterTextSplitter for chunking, FAISS for indexing, no medical-specific NLP or graphs.
Updated based on optimization suggestions: Increased chunk_size to 1000 chars, overlap to 250 (25%), default top_k to 10, similarity_threshold to 0.4. Added optional context expansion in search for ±1 chunk.
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Core dependencies (filtered from original)
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class ArticleDocument:
    """Simplified document representation for scraped articles (filtered from MedicalDocument)"""
    doc_id: str
    title: Optional[str]
    content: str
    chunks: List[Dict[str, Any]]

@dataclass
class SearchResult:
    """Simplified search result (filtered from original)"""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class SimpleVectorDB:
    """Simplified FAISS-based vector database for article search (filtered from MedicalGraphRAG)"""
    
    def __init__(self, storage_path: str = "./vector_db_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize paths (similar to original)
        self.vector_db_path = self.storage_path / "vector_db"
        self.metadata_path = self.storage_path / "metadata.json"
        self.embeddings_cache = self.storage_path / "embeddings_cache.pickle"
        
        # Initialize embedding model (simplified to general-purpose SentenceTransformer instead of BioBERT)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize splitter (updated: larger chunk size and overlap for better context)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 500 to 1000 for more context
            chunk_overlap=250,  # 25% overlap for continuity
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector database (FAISS, same as original)
        self.index = None
        self.document_store = {}
        self.chunk_metadata = []
        
        # Load existing data if available (same persistence approach)
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load previously ingested data from local storage (disk persistence, same as original)"""
        try:
            # Load vector index from disk
            if (self.vector_db_path / "index.faiss").exists():
                self.index = faiss.read_index(str(self.vector_db_path / "index.faiss"))
                print(f"Loaded FAISS index with {self.index.ntotal} vectors from disk")
            
            # Load metadata from disk
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.document_store = metadata.get('documents', {})
                    self.chunk_metadata = metadata.get('chunks', [])
                print(f"Loaded metadata for {len(self.document_store)} documents from disk")
            
            # Load embeddings cache from disk
            if self.embeddings_cache.exists():
                with open(self.embeddings_cache, 'rb') as f:
                    self.embeddings_cache_dict = pickle.load(f)
            else:
                self.embeddings_cache_dict = {}
                
        except Exception as e:
            print(f"Error loading existing data: {e}")
            self.embeddings_cache_dict = {}
        
        # Deserialize document store into ArticleDocument objects (simplified)
        self.document_store = {
            doc_id: ArticleDocument(
                doc_id=doc['doc_id'],
                title=doc.get('title'),
                content=doc['content'],
                chunks=doc['chunks']
            )
            for doc_id, doc in self.document_store.items()
        }
    
    def save_data(self):
        """Save all data locally to disk for persistence (same as original)"""
        # Create vector db directory
        self.vector_db_path.mkdir(exist_ok=True)

        # Save FAISS index to disk
        if self.index:
            faiss.write_index(self.index, str(self.vector_db_path / "index.faiss"))

        # Convert ArticleDocument objects to dictionaries for JSON serialization
        document_store_serializable = {
            doc_id: {
                'doc_id': doc.doc_id,
                'title': doc.title,
                'content': doc.content,
                'chunks': doc.chunks
            }
            for doc_id, doc in self.document_store.items()
        }

        # Save metadata to disk
        metadata = {
            'documents': document_store_serializable,
            'chunks': self.chunk_metadata
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings cache to disk
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump(self.embeddings_cache_dict, f)

        print(f"Data saved to {self.storage_path} on disk")
    
    def ingest_json(self, json_path: str, force_reprocess: bool = False) -> None:
        """Ingest scraped articles from JSON file, loading only the 'content' entity (filtered from ingest_pdf and ingest_text_dataset)"""
        with open(json_path, 'r') as f:
            items = json.load(f)  # Assume list of dicts
        
        for item in items:
            if 'content' not in item or not item['content'].strip():
                print(f"Skipping item without valid 'content' key")
                continue

            content = item['content']
            title = item.get('title', 'Untitled')
            doc_id = hashlib.md5(content.encode()).hexdigest()

            if doc_id in self.document_store and not force_reprocess:
                print(f"Article '{title}' already processed.")
                continue

            print(f"Processing article content: {title}")

            doc = ArticleDocument(
                doc_id=doc_id,
                title=title,
                content=content,
                chunks=[]
            )

            # Chunk the content (updated splitter params)
            chunks = self.chunk_text(content, doc_id)
            if not chunks:
                print(f"Skipping article '{title}' (doc_id={doc_id}) due to empty chunks.")
                continue

            doc.chunks = chunks

            # Generate embeddings
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)

            # Add to vector DB
            self.add_to_vector_db(embeddings, chunks)

            # Store document
            self.document_store[doc_id] = doc

        # Save after ingestion
        self.save_data()
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces (filtered from chunk_text_medical_aware, no entities)"""
        chunks = self.text_splitter.split_text(text)
        processed_chunks = []
        
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_data = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'content': chunk_content,
                'position': i,
                'metadata': {'char_count': len(chunk_content)}
            }
            processed_chunks.append(chunk_data)
            self.chunk_metadata.append(chunk_data)
        
        return processed_chunks
    
    def add_to_vector_db(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Add embeddings to FAISS index (same as original, with normalization)"""
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be a 2D array, but got shape {embeddings.shape}")
        
        # Initialize index if needed
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        print(f"Added {len(chunks)} chunks to vector database")
    
    def search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.4, expand_context: bool = False) -> List[SearchResult]:
        """Search for relevant chunks based on prompt/query (updated: higher top_k, lower threshold, optional context expansion)"""
        if self.index is None or self.index.ntotal == 0:
            print("No documents in database. Please ingest first.")
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_metadata) and dist >= similarity_threshold:
                chunk = self.chunk_metadata[idx]
                content = chunk['content']
                
                # Optional: Expand context with ±1 chunk
                if expand_context:
                    prev_chunk = next((c for c in self.chunk_metadata if c['doc_id'] == chunk['doc_id'] and c['position'] == chunk['position'] - 1), None)
                    next_chunk = next((c for c in self.chunk_metadata if c['doc_id'] == chunk['doc_id'] and c['position'] == chunk['position'] + 1), None)
                    if prev_chunk:
                        content = prev_chunk['content'] + "\n" + content
                    if next_chunk:
                        content = content + "\n" + next_chunk['content']
                
                results.append(SearchResult(
                    doc_id=chunk['doc_id'],
                    chunk_id=chunk['chunk_id'],
                    content=content,
                    score=float(dist),
                    metadata=chunk['metadata']
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results

# Example usage (updated: higher top_k, enable context expansion)
import csv

def filter_scraped_data(vector_db_path: str, scraped_json_path: str, filtering_prompt: str, pmc_csv_path: str) -> Dict[str, Any]:
    vdb = SimpleVectorDB(storage_path=vector_db_path)
    vdb.ingest_json(scraped_json_path, force_reprocess=True)
    results = vdb.search(filtering_prompt, top_k=10, expand_context=True)

    # Load original articles for metadata lookup
    with open(scraped_json_path, "r") as f:
        articles = json.load(f)
        articles_by_doc_id = {
            hashlib.md5(article['content'].encode()).hexdigest(): article
            for article in articles if 'content' in article and article['content'].strip()
        }

    # Prepare output for top 10 results
    output = []
    for result in results:
        article = articles_by_doc_id.get(result.doc_id, {})
        output.append({
            "filtered_content": result.content,
            "score": result.score
        })

    # Load scrapped resources from CSV
    scrapped_resources = []
    with open(pmc_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            scrapped_resources.append({
                "pmcid": row["pmcid"],
                "url": row["url"]
            })

    # Final result
    return {
        "results": output,
        "scrapped_resources": scrapped_resources
    }

    # Save top 10 results to JSON
    # with open("top10_filtered_results.json", "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    # for result in results:
    #     print(f"Score: {result.score}\nContent: {result.content[:200]}...\n")
    # print("✅ Top 10 results (with expanded context) saved to top10_filtered_results.json")