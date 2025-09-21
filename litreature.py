"""
Medical Document Ingestion and Retrieval System with GraphRAG + Vector DB
Combines graph-based reasoning with semantic search for medical diagnosis workflows
"""

import os
import pickle
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pathlib import Path

# Core dependencies
import torch
import faiss
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import pymupdf  # fitz
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import scispacy
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Medical NLP libraries
from transformers import AutoTokenizer, AutoModel
import torch

# Set up Groq API key
os.environ["GROQ_API_KEY"] = "o"

class BioBERTEmbedding:
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

@dataclass
class MedicalDocument:
    """Structured medical document representation"""
    doc_id: str
    title: str
    content: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    entities: List[Dict[str, Any]]
    embeddings: Optional[np.ndarray] = None

@dataclass
class SearchResult:
    """Search result with confidence and context"""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    graph_context: Optional[Dict[str, Any]] = None

class MedicalEmbeddingModel:
    """Medical-specific embedding model wrapper"""
    
    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        """
        Initialize medical embedding model
        Options:
        - dmis-lab/biobert-v1.1
        - emilyalsentzer/Bio_ClinicalBERT
        - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
        """
        # Use MPS (Metal Performance Shaders) if available, otherwise fallback to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using device: MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Replace BiobertEmbedding with BioBERTEmbedding
        self.bio_bert = BioBERTEmbedding(model_name)
        
    def embed_text(self, text: str, use_sentence_transformer: bool = False) -> np.ndarray:
        """Generate embeddings for text"""
        if use_sentence_transformer:
            return self.sentence_model.encode(text, convert_to_numpy=True)

        # Use BioBERTEmbedding for embedding
        return self.bio_bert.embed(text)

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Batch embedding for efficiency"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.embed_text(text) for text in batch]
            embeddings.extend(batch_embeddings)
        
        # Ensure embeddings is a 2D NumPy array
        embeddings = np.array(embeddings)
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be a 2D array, but got shape {embeddings.shape}")
        return embeddings

class MedicalGraphRAG:
    """Graph-based Retrieval Augmented Generation for medical documents"""
    
    def __init__(self, storage_path: str = "./medical_rag_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize paths
        self.vector_db_path = self.storage_path / "vector_db"
        self.graph_path = self.storage_path / "knowledge_graph.pickle"
        self.metadata_path = self.storage_path / "metadata.json"
        self.embeddings_cache = self.storage_path / "embeddings_cache.pickle"
        
        # Initialize models
        print("Loading medical embedding model...")
        self.embedding_model = MedicalEmbeddingModel()
        
        # Initialize medical NLP using scispacy
        try:
            print("Loading medical NLP models using scispacy...")
            # Use a more advanced scispaCy model for better entity extraction
            self.nlp = spacy.load("en_core_sci_md")  # Load large scispaCy model

            # Adjust chunking parameters for better context
            self.chunk_size = 512  # Reduce chunk size for finer granularity
            self.chunk_overlap = 256  # Increase overlap for better context preservation

            # Enhance scoring mechanism
            self.entity_overlap_weight = 0.2  # Increase weight for entity overlap
            self.graph_bonus_weight = 0.3  # Increase weight for graph relationships
        except OSError as e:
            print("Error loading spaCy model 'en_core_sci_md'. Ensure it is installed.")
            raise e
        
        # Initialize graph
        self.knowledge_graph = nx.DiGraph()
        
        # Initialize vector database (FAISS)
        self.index = None
        self.document_store = {}
        self.chunk_metadata = []
        
        # Load existing data if available
        self.load_existing_data()
        
        # Load external medical ontologies (e.g., UMLS, SNOMED CT) to enhance the knowledge graph
        self.load_medical_ontology("umls")
        self.load_medical_ontology("snomed_ct")

        # Fine-tune BioBERT or switch to PubMedBERT for embeddings
        self.embedding_model = MedicalEmbeddingModel(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

        # Adjust scoring weights for precision
        self.entity_overlap_weight = 0.3  # Increase weight for entity overlap
        self.graph_bonus_weight = 0.5  # Increase weight for graph relationships
    
    def load_existing_data(self):
        """Load previously ingested data from local storage"""
        try:
            # Load vector index
            if (self.vector_db_path / "index.faiss").exists():
                self.index = faiss.read_index(str(self.vector_db_path / "index.faiss"))
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load graph
            if self.graph_path.exists():
                with open(self.graph_path, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                print(f"Loaded knowledge graph with {len(self.knowledge_graph.nodes)} nodes")
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.document_store = metadata.get('documents', {})
                    self.chunk_metadata = metadata.get('chunks', [])
                print(f"Loaded metadata for {len(self.document_store)} documents")
            
            # Load embeddings cache
            if self.embeddings_cache.exists():
                with open(self.embeddings_cache, 'rb') as f:
                    self.embeddings_cache_dict = pickle.load(f)
            else:
                self.embeddings_cache_dict = {}
                
        except Exception as e:
            print(f"Error loading existing data: {e}")
            self.embeddings_cache_dict = {}
        
        # Deserialize document store into MedicalDocument objects
        self.document_store = {
            doc_id: MedicalDocument(
                doc_id=doc['doc_id'],
                title=doc['title'],
                content=doc['content'],
                chunks=doc['chunks'],
                metadata=doc['metadata'],
                entities=doc['entities'],
                embeddings=np.array(doc['embeddings']) if doc['embeddings'] else None
            )
            for doc_id, doc in self.document_store.items()
        }
    
    def save_data(self):
        """Save all data locally"""
        # Create vector db directory
        self.vector_db_path.mkdir(exist_ok=True)

        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, str(self.vector_db_path / "index.faiss"))

        # Save knowledge graph
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.knowledge_graph, f)

        # Convert MedicalDocument objects to dictionaries for JSON serialization
        document_store_serializable = {
            doc_id: {
                'doc_id': doc.doc_id,
                'title': doc.title,
                'content': doc.content,
                'chunks': doc.chunks,
                'metadata': doc.metadata,
                'entities': doc.entities,
                'embeddings': doc.embeddings.tolist() if doc.embeddings is not None else None
            }
            for doc_id, doc in self.document_store.items()
        }

        # Save metadata
        metadata = {
            'documents': document_store_serializable,
            'chunks': self.chunk_metadata,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings cache
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump(self.embeddings_cache_dict, f)

        print(f"Data saved to {self.storage_path}")
    
    def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities and relationships"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(
                ent.text.lower(),
                label=ent.label_,
                frequency=self.knowledge_graph.nodes.get(ent.text.lower(), {}).get('frequency', 0) + 1
            )
            entities.append(entity_data)
        
        # Extract relationships between entities
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                if abs(ent1['start'] - ent2['start']) < 200:  # Proximity-based relationship
                    self.knowledge_graph.add_edge(
                        ent1['text'].lower(),
                        ent2['text'].lower(),
                        weight=self.knowledge_graph.edges.get(
                            (ent1['text'].lower(), ent2['text'].lower()), {}
                        ).get('weight', 0) + 1
                    )
        
        return entities
    
    def ingest_pdf(self, pdf_path: str, force_reprocess: bool = False) -> MedicalDocument:
        """Ingest PDF document with medical-aware processing"""
        
        # Check if already processed
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()
        if doc_id in self.document_store and not force_reprocess:
            print(f"Document {pdf_path} already processed. Use force_reprocess=True to reprocess.")
            return self.document_store[doc_id]
        
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        pdf_document = pymupdf.open(pdf_path)
        full_text = ""
        for page in pdf_document:
            full_text += page.get_text()
        
        print(f"Extracted {len(full_text)} characters from PDF.")
        
        # Create document
        doc = MedicalDocument(
            doc_id=doc_id,
            title=os.path.basename(pdf_path),
            content=full_text,
            chunks=[],
            metadata={
                'path': pdf_path,
                'pages': len(pdf_document),
                'processed_at': datetime.now().isoformat()
            },
            entities=[]
        )
        
        # Extract medical entities
        print("Extracting medical entities...")
        chunk_size = 100_000  # Define chunk size
        full_text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        all_entities = []
        for idx, chunk in enumerate(full_text_chunks):
            print(f"Processing chunk {idx+1}/{len(full_text_chunks)}...")
            chunk_entities = self.extract_medical_entities(chunk)
            print(f"Extracted {len(chunk_entities)} entities from chunk {idx+1}.")
            all_entities.extend(chunk_entities)
        doc.entities = all_entities

        print(f"Total entities extracted: {len(all_entities)}")
        
        # Chunk text with better context awareness
        print("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100_000,  # Define chunk size
            chunk_overlap=10_000,  # Allow some overlap for context
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        full_text_chunks = splitter.split_text(full_text)
        print(f"Split text into {len(full_text_chunks)} chunks.")

        # Chunk text with medical awareness
        print("Chunking text...")
        chunks = self.chunk_text_medical_aware(full_text, doc_id)
        doc.chunks = chunks
        
        # Generate embeddings for chunks
        print("Generating embeddings...")
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(chunk_texts)
        
        # Add to vector database
        print("Adding embeddings to vector database...")
        self.add_to_vector_db(embeddings, chunks)
        
        # Store document
        self.document_store[doc_id] = doc
        
        # Save data
        self.save_data()
        
        print(f"Successfully ingested {pdf_path}")
        return doc

    def chunk_text_medical_aware(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Chunk text with medical context awareness"""
        
        # Use different strategies for chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        chunks = splitter.split_text(text)
        
        # Process chunks
        processed_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Extract entities for this chunk
            chunk_entities = self.extract_medical_entities(chunk_content)
            
            chunk_data = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'content': chunk_content,
                'position': i,
                'entities': chunk_entities,
                'metadata': {
                    'char_count': len(chunk_content),
                    'entity_count': len(chunk_entities)
                }
            }
            processed_chunks.append(chunk_data)
            self.chunk_metadata.append(chunk_data)
        
        return processed_chunks
    
    def add_to_vector_db(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Add embeddings to FAISS vector database"""
        
        # Validate embeddings shape
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be a 2D array, but got shape {embeddings.shape}")

        # Initialize index if needed
        if self.index is None:
            dimension = embeddings.shape[1]
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        print(f"Added {len(chunks)} chunks to vector database")
    
    def search(self, query: str, top_k: int = 5, use_graph: bool = True, 
               similarity_threshold: float = 0.5) -> List[SearchResult]:
        """
        Advanced medical search with graph enhancement
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_graph: Whether to use graph relationships
            similarity_threshold: Minimum similarity score
        """
        
        if self.index is None or self.index.ntotal == 0:
            print("No documents in database. Please ingest documents first.")
            return []
        
        print(f"Performing search for query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search vector database
        distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
        print(f"Retrieved {len(indices[0])} results from vector database.")
        
        # Extract query entities
        query_entities = self.extract_medical_entities(query)
        query_entity_texts = [e['text'].lower() for e in query_entities]
        print(f"Extracted {len(query_entities)} entities from query: {query_entity_texts}")
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx]
                
                # Calculate enhanced score
                base_score = float(dist)
                
                # Entity overlap bonus
                chunk_entity_texts = [e['text'].lower() for e in chunk.get('entities', [])]
                entity_overlap = len(set(query_entity_texts) & set(chunk_entity_texts))
                entity_bonus = entity_overlap * 0.1
                
                # Graph relationship bonus
                graph_bonus = 0
                if use_graph and query_entity_texts:
                    graph_bonus = self.calculate_graph_relevance(
                        query_entity_texts, 
                        chunk_entity_texts
                    )
                
                final_score = base_score + entity_bonus + graph_bonus
                
                print(f"Chunk ID: {chunk['chunk_id']}, Base Score: {base_score}, Entity Bonus: {entity_bonus}, Graph Bonus: {graph_bonus}, Final Score: {final_score}")
                
                if final_score >= similarity_threshold:
                    result = SearchResult(
                        doc_id=chunk['doc_id'],
                        chunk_id=chunk['chunk_id'],
                        content=chunk['content'],
                        score=final_score,
                        metadata={
                            'position': chunk['position'],
                            'entities': chunk.get('entities', []),
                            'base_score': base_score,
                            'entity_bonus': entity_bonus,
                            'graph_bonus': graph_bonus
                        },
                        graph_context=self.get_graph_context(chunk_entity_texts) if use_graph else None
                    )
                    results.append(result)
        
        print(f"Search completed. Found {len(results)} relevant results.")
        
        # Sort by final score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def calculate_graph_relevance(self, query_entities: List[str], 
                                  chunk_entities: List[str]) -> float:
        """Calculate relevance using knowledge graph"""
        relevance = 0
        
        for q_entity in query_entities:
            for c_entity in chunk_entities:
                if q_entity in self.knowledge_graph and c_entity in self.knowledge_graph:
                    try:
                        # Check if connected
                        if nx.has_path(self.knowledge_graph, q_entity, c_entity):
                            path_length = nx.shortest_path_length(
                                self.knowledge_graph, q_entity, c_entity
                            )
                            # Shorter paths = higher relevance
                            relevance += 1 / (path_length + 1) * 0.1
                    except:
                        pass
        
        return relevance
    
    def get_graph_context(self, entities: List[str]) -> Dict[str, Any]:
        """Get graph context for entities"""
        context = {
            'related_entities': set(),
            'relationships': []
        }
        
        for entity in entities:
            if entity in self.knowledge_graph:
                # Get neighbors
                neighbors = list(self.knowledge_graph.neighbors(entity))
                context['related_entities'].update(neighbors[:5])
                
                # Get relationships
                for neighbor in neighbors[:3]:
                    if self.knowledge_graph.has_edge(entity, neighbor):
                        edge_data = self.knowledge_graph[entity][neighbor]
                        context['relationships'].append({
                            'from': entity,
                            'to': neighbor,
                            'weight': edge_data.get('weight', 1)
                        })
        
        context['related_entities'] = list(context['related_entities'])
        return context
    
    def get_structured_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate structured prompt for medical diagnosis
        """
        prompt = f"""<medical_context>
You are a medical AI assistant analyzing patient information and medical literature to assist in diagnosis.

## Query
{query}

## Relevant Medical Information
"""
        
        for i, result in enumerate(search_results, 1):
            prompt += f"""
### Source {i} (Relevance: {result.score:.3f})
**Content:** {result.content}

**Medical Entities Identified:**
"""
            for entity in result.metadata.get('entities', []):
                prompt += f"- {entity['text']} ({entity['label']})\n"
            
            if result.graph_context:
                prompt += "\n**Related Medical Concepts:**\n"
                for related in result.graph_context.get('related_entities', [])[:5]:
                    prompt += f"- {related}\n"
        
        prompt += """
</medical_context>

<instructions>
Based on the medical context provided above:

1. **Analyze** the relevant medical information in relation to the query.
2. **Identify** key medical concepts, symptoms, conditions, or treatments mentioned.
3. **Synthesize** the information to provide a comprehensive and structured response.
4. **Consider** differential diagnoses if applicable, and highlight any potential comorbidities.
5. **Note** any important contraindications, warnings, or drug interactions.
6. **Cite** specific sources when making medical claims, ensuring traceability.
7. **Provide Recommendations** for further investigation, tests, or treatments based on the context.

Important Guidelines:
- Be precise and use appropriate medical terminology.
- Acknowledge any limitations or uncertainties in the available information.
- Recommend consultation with healthcare professionals for actual medical decisions.
- Do not provide definitive diagnoses without sufficient evidence.
- Consider patient safety as the top priority.
- Ensure the response is empathetic and patient-centric.
</instructions>

Please provide your detailed medical analysis:"""
        
        return prompt

    def call_llm(self, query: str, top_chunks: list) -> str:
        """Call Groq OSS LLM for generating responses based on the query and top chunks."""
        from groq import Groq

        # Format prompt with query and top chunks
        prompt = f"Medical Query: {query}\n\nTop Chunks:\n"
        for i, chunk in enumerate(top_chunks, 1):
            prompt += f"\nChunk {i}:\n{chunk['content']}\n"
        # Truncate if needed
        max_prompt_length = 4096  # Increase the maximum prompt length
        if len(prompt) > max_prompt_length:
            print(f"Prompt length ({len(prompt)}) exceeds max length ({max_prompt_length}). Adjusting logic to include all relevant chunks...")
            # Instead of truncating, prioritize chunks based on their scores
            top_chunks = sorted(top_chunks, key=lambda x: x['score'], reverse=True)
            prompt = f"Medical Query: {query}\n\nTop Chunks:\n"
            for i, chunk in enumerate(top_chunks):
                prompt += f"\nChunk {i+1}:\n{chunk['content']}\n"
                if len(prompt) > max_prompt_length:
                    break

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        print("Calling Groq OSS LLM (llama-3.3-70b-versatile)...")
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        # Collect streamed output
        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""

        # Strip <think> block if present
        return self.strip_think(response_text)

    @staticmethod
    def strip_think(text: str) -> str:
        """Strip the <think> block from the response."""
        import re
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S | re.I).strip()
    
    def get_structured_prompt_with_llm(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate structured prompt and use LLM to get a detailed response."""
        # Extract top chunks
        top_chunks = [
            {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata
            }
            for result in search_results
        ]
        # Call the LLM with the query and top chunks
        llm_response = self.call_llm(query, top_chunks)
        return llm_response

    def get_answer_from_prompt(self, query: str, top_chunks: list) -> str:
        """Use the LLM to get an answer directly from the query and top chunks."""
        llm_response = self.call_llm(query, top_chunks)
        return llm_response

    def load_medical_ontology(self, ontology_name: str):
        """Placeholder for loading external medical ontologies like UMLS or SNOMED CT."""
        print(f"Loading medical ontology: {ontology_name} (placeholder implementation)")
        # TODO: Implement actual ontology loading logic here
        pass

    def ingest_text_dataset(self, dataset, text_column: str = "text", title_column: str = None, metadata_columns: list = None):
        """Ingest a HuggingFace text dataset into the RAG and GraphRAG system."""
        print(f"Ingesting dataset with {len(dataset)} samples...")
        print("MedRAG/textbooks columns:", dataset.column_names)
        text_col = "content"
        bm25_col = "contents"
        title_col = "title"
        metadata_cols = ["id", "title"]
        print(f"BM25 will use column: {bm25_col}")
        print(f"Embedding will use column: {text_col}")
        for i, sample in enumerate(dataset):
            text = sample[text_col]
            title = sample[title_col] if title_col and title_col in sample else f"Sample {i+1}"
            metadata = {col: sample[col] for col in metadata_cols if col in sample}
            doc_id = hashlib.md5(f"medrag_{i}_{title}".encode()).hexdigest()
            if doc_id in self.document_store:
                continue  # Skip already ingested
            # Extract entities from the full document and add to knowledge graph
            print(f"Extracting entities for document {title}...")
            doc_entities = self.extract_medical_entities(text)
            doc = MedicalDocument(
                doc_id=doc_id,
                title=title,
                content=text,
                chunks=[],
                metadata=metadata,
                entities=doc_entities
            )
            # Chunk text and extract entities for each chunk (updates graph)
            chunks = self.chunk_text_medical_aware(text, doc_id)
            doc.chunks = chunks
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.embed_batch(chunk_texts)
            self.add_to_vector_db(embeddings, chunks)
            # Aggregate all chunk entities into doc.entities
            doc.entities.extend([entity for chunk in chunks for entity in chunk['entities']])
            self.document_store[doc_id] = doc
        self.save_data()
        print("Dataset ingestion complete.")
    
    def build_bm25_index(self, dataset, bm25_col="contents"):
        """Build BM25 index from the dataset's 'contents' column."""
        print(f"Building BM25 index using column: {bm25_col}")
        self.bm25_corpus = [doc for doc in dataset[bm25_col]]
        self.bm25_tokenized = [doc.split() for doc in self.bm25_corpus]
        self.bm25 = BM25Okapi(self.bm25_tokenized)
        print(f"BM25 index built for {len(self.bm25_corpus)} documents.")

    def bm25_search(self, query, top_k=5):
        """Retrieve top_k documents using BM25."""
        if not hasattr(self, 'bm25'):
            print("BM25 index not built. Call build_bm25_index() first.")
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "index": idx,
                "score": scores[idx],
                "content": self.bm25_corpus[idx]
            })
        return results

# Example usage and testing
def main():
    """Example usage of the Medical GraphRAG system"""
    
    # Initialize system
    rag_system = MedicalGraphRAG(storage_path="./medical_rag_storage")
    
    # Example: Ingest PDFs
    pdf_paths = [
        "sources/Differential_Diagnosis_Clinical_Medicine.pdf",
        # "path/to/medical_paper1.pdf",
        # "path/to/clinical_guidelines.pdf"
    ]
    
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            rag_system.ingest_pdf(pdf_path)
    
    # Ingest MedRAG/textbooks dataset from HuggingFace
    from datasets import load_dataset
    ds = load_dataset("MedRAG/textbooks", split="train")
    # Print available columns in the dataset for debugging
    print("MedRAG/textbooks columns:", ds.column_names)
    # Use the correct text column name
    text_col = ds.column_names[0] if "text" not in ds.column_names else "text"
    title_col = "title" if "title" in ds.column_names else None
    metadata_cols = [col for col in ["source", "book", "chapter"] if col in ds.column_names]
    rag_system.ingest_text_dataset(ds, text_column=text_col, title_column=title_col, metadata_columns=metadata_cols)

    # Example search queries with medical context
    example_queries = [
        "What are the treatment options for type 2 diabetes with cardiovascular complications?",
        "Differential diagnosis for patient presenting with chest pain and shortness of breath",
        "Drug interactions between metformin and ACE inhibitors",
        "Latest guidelines for hypertension management in elderly patients"
    ]

    # Initialize results storage
    results_data = []

    for query in example_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        # Perform search to get top chunks
        results = rag_system.search(query, top_k=3)

        if results:
            # Extract top 3 chunks
            top_chunks = [
                {
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata
                }
                for result in results
            ]

            # Get the answer from the LLM using the query and top chunks
            llm_response = rag_system.get_answer_from_prompt(query, top_chunks)

            # Store query, top chunks, and LLM response in results
            results_data.append({
                "query": query,
                "top_chunks": top_chunks,
                "llm_response": llm_response
            })
        else:
            print("No results found.")

    # Save results to JSON file
    with open("query_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print("\nResults saved to query_results.json")

    # Save the system state
    rag_system.save_data()
    print("\nSystem data saved successfully!")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    top_chunks: list
    llm_response: str

# Initialize the MedicalGraphRAG system once at the start of the application
rag_system = MedicalGraphRAG(storage_path="./medical_rag_storage")

@app.on_event("startup")
def initialize_rag_system():
    print("Initializing MedicalGraphRAG system...")
    # Ensure the system is ready for queries
    if not rag_system.index or rag_system.index.ntotal == 0:
        print("Warning: No documents in the database. Please ingest documents before querying.")

@app.post("/literature", response_model=QueryResponse)
def search_query(request: QueryRequest):
    query = request.query
    print(f"Received query: {query}")

    # Perform search to get top chunks
    results = rag_system.search(query, top_k=3)

    if not results:
        raise HTTPException(status_code=404, detail="No results found.")

    # Extract top 3 chunks
    top_chunks = [
        {
            "chunk_id": result.chunk_id,
            "content": result.content,
            "score": result.score,
            "metadata": result.metadata
        }
        for result in results
    ]

    # Get the answer from the LLM using the query and top chunks
    llm_response = rag_system.get_answer_from_prompt(query, top_chunks)

    # Return the response
    return QueryResponse(
        query=query,
        top_chunks=top_chunks,
        llm_response=llm_response
    )
