# memory.py - FULLY CORRECTED AND OPTIMIZED VERSION
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os
import re
import warnings


class BioClinicalMemoryAgent:
    """
    BioClinicalBERT retrieval agent with intelligent chunking, metadata-aware filtering,
    and lexical reranking for medical case matching.
    
    FULLY FIXED: Now uses proper IndexFlatIP for true cosine similarity calculation.
    All thresholds, scoring, and ranking are calibrated for accurate medical diagnosis.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 chunk_size: int = 300, overlap: int = 50):
        """
        Initialize BioClinicalMemoryAgent with corrected similarity calculation.
        
        Args:
            model_name: HuggingFace model name for embeddings
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_dim = 768
        
        # Storage for documents and chunks
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        
        # FIXED: Use IndexFlatIP for direct cosine similarity
        import faiss
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_chunk_id_map: List[int] = []
        
        # Initialize components
        self._init_embedding_system()
        self._compile_query_patterns()
        
        print(f"✅ BioClinicalMemoryAgent Initialized (FULLY CORRECTED)")
        print(f"   Model: {model_name}")
        print(f"   Chunking: {chunk_size} tokens with {overlap} overlap")
        print(f"   Index: FAISS IndexFlatIP with TRUE cosine similarity")
        print(f"   Embeddings: {self.embedding_dim}D normalized vectors")

    def _init_embedding_system(self):
        """Initialize BioClinicalBERT model for medical domain embeddings."""
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"🔄 Loading {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.use_transformers = True
            
            print("✅ BioClinicalBERT loaded successfully")
            print("✅ Medical domain embedding system ready")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load BioClinicalBERT model: {e}")

    def _compile_query_patterns(self):
        """Compile regex patterns for medical domain and symptom detection."""
        self._patterns = {
            "ophthalmology": re.compile(r"\b(eye|ocular|conjunct|red eye|photophobia|vision|visual)\b", re.I),
            "vascular": re.compile(r"\b(raynaud|color change|triphasic|white|blue|blanch|rewarm|tingl|vasospasm)\w*", re.I),
            "respiratory": re.compile(r"\b(cough|hemoptysis|sputum|wheeze|dyspnea|shortness of breath|pleuritic|respiratory|pulmonary)\b", re.I),
            "gastroenterology": re.compile(r"\b(abdominal|ruq|nausea|vomit|biliary|gallstone|reflux|heartburn|gastro)\b", re.I),
            "neurology": re.compile(r"\b(weakness|numb|tingl|droop|slurred|aphasia|syncope|seizure|neurologic)\b", re.I),
            "dermatology": re.compile(r"\b(rash|itch|pruritus|scaly|lesion|eczema|dermatitis|alopecia|hair loss|scalp|skin)\b", re.I),
            "genitourinary": re.compile(r"\b(dysuria|frequency|urgency|flank|suprapubic|hematuria|urinary|renal)\b", re.I),
            "cardiology": re.compile(r"\b(angina|exertional chest|chest pressure|palpit|nitro|cardiac|heart)\b", re.I),
            "endocrine": re.compile(r"\b(polyuria|polydipsia|hyperglyc|hba1c|weight loss|diabetes|thyroid|hormone)\b", re.I),
            "musculoskeletal": re.compile(r"\b(knee|shoulder|hip|ankle|runner|run|overuse|joint|ligament|menisc|tendon|patell|bone|muscle)\b", re.I),
        }
        
        self._boost_terms = {
            "ophthalmology": ["conjunctivitis", "red_eye", "itching", "no_discharge", "photophobia", "vision_change", "allergic_history"],
            "vascular": ["raynaud", "vasospasm", "acrocyanosis", "color_change", "blanching", "rewarming", "digits", "circulation"],
            "respiratory": ["hemoptysis", "sputum", "wheeze", "dyspnea", "bronchitis", "pneumonia", "tb", "flu", "influenza", "viral"],
            "gastroenterology": ["ruq", "biliary_colic", "gallstones", "ultrasound", "murphy", "fatty_meal", "peptic", "gerd"],
            "neurology": ["TIA", "transient_deficit", "dysarthria", "facial_droop", "arm_weakness", "stroke", "migraine"],
            "dermatology": ["atopic", "eczema", "pruritus", "flexural", "emollients", "alopecia", "hair_loss", "scalp_thinning", "bald", "minoxidil"],
            "genitourinary": ["dysuria", "frequency", "suprapubic_pain", "pyuria", "nitrite", "cystitis", "uti", "kidney"],
            "cardiology": ["stable_angina", "exertional_chest_pressure", "relief_with_rest", "stress_test", "nitroglycerin", "myocardial"],
            "endocrine": ["hyperglycemia", "elevated_hba1c", "metformin", "polyuria", "polydipsia", "insulin", "glucose"],
            "musculoskeletal": ["knee_pain", "running_trigger", "overuse", "no_trauma", "patellofemoral", "it_band", "arthritis", "fracture"],
        }

    def add_document(self, text: str, source: str = "manual",
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a medical document to the knowledge base with proper indexing.
        
        Args:
            text: Medical case text
            source: Source identifier
            metadata: Document metadata including condition, body_system, etc.
            
        Returns:
            Document ID
        """
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty")
            
        doc_id = len(self.documents)
        doc_meta = {"doc_id": doc_id, "source": source, "added_timestamp": datetime.now().isoformat()}
        
        if metadata:
            doc_meta.update(metadata)
        
        # Enrich with inferred metadata if missing
        if "body_system" not in doc_meta or "symptom_tags" not in doc_meta:
            inferred_system, tags = self._infer_system_and_tags(text)
            doc_meta.setdefault("body_system", inferred_system)
            doc_meta.setdefault("symptom_tags", tags)
        
        self.documents.append(text)
        self.doc_metadata.append(doc_meta)
        
        # Create chunks with improved chunking
        chunks = self._chunk_document(text)
        print(f"📄 Document {doc_id}: {len(chunks)} chunks created")
        
        # Add chunks to index
        embeddings_batch = []
        chunk_metas_batch = []
        
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "doc_id": doc_id,
                "chunk_id": len(self.chunks) + i,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "condition": doc_meta.get("condition"),
                "body_system": doc_meta.get("body_system"),
                "symptom_tags": doc_meta.get("symptom_tags", []),
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            }
            
            # Generate embedding
            embedding = self._get_embedding(chunk).astype('float32')
            
            # Add to batch
            embeddings_batch.append(embedding)
            chunk_metas_batch.append(chunk_meta)
            
            # Add to storage
            self.chunks.append(chunk)
            self.chunk_metadata.append(chunk_meta)
            self.faiss_chunk_id_map.append(len(self.chunks) - 1)
        
        # Batch add to FAISS index for efficiency
        if embeddings_batch:
            embeddings_array = np.vstack(embeddings_batch)
            self.faiss_index.add(embeddings_array)
        
        condition_name = doc_meta.get('condition', 'unknown')
        body_system = doc_meta.get('body_system', 'unknown')
        print(f"✅ Added document {doc_id} ({condition_name} - {body_system}) with {len(chunks)} chunks")
        
        return doc_id

    def _infer_system_and_tags(self, text: str) -> Tuple[str, List[str]]:
        """Infer medical body system and symptom tags from text."""
        text_lower = text.lower()
        
        # Calculate system scores
        system_scores = {}
        for system, pattern in self._patterns.items():
            matches = pattern.findall(text_lower)
            system_scores[system] = len(matches)
        
        # Get dominant system
        inferred_system = max(system_scores, key=system_scores.get) if system_scores else "general"
        if system_scores[inferred_system] == 0:
            inferred_system = "general"
        
        # Extract symptom tags
        tags = []
        for system, terms in self._boost_terms.items():
            for term in terms:
                # Check both with and without underscores
                term_spaced = term.replace("_", " ")
                if term in text_lower or term_spaced in text_lower:
                    tags.append(term)
        
        # Remove duplicates and limit
        tags = list(dict.fromkeys(tags))[:15]  # Increased limit
        
        return inferred_system, tags

    def _chunk_document(self, text: str) -> List[str]:
        """
        Improved document chunking for medical texts.
        Preserves medical context and handles various text formats.
        """
        # Split on major breaks first
        major_sections = re.split(r'\n\s*\n+|\n(?=[A-Z][a-z]+(?:\s[A-Z][a-z]+)*:)', text.strip())
        
        chunks = []
        
        for section in major_sections:
            section = section.strip()
            if not section:
                continue
                
            # If section is already small enough, use as is
            if len(section.split()) <= self.chunk_size:
                chunks.append(section)
                continue
            
            # Split into sentences for larger sections
            sentences = re.split(r'(?<=[.!?;])\s+(?=[A-Z])', section)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                # Fallback: split by punctuation
                sentences = re.split(r'[.!?;]+', section)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            # Build chunks from sentences
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = len(sentence.split())
                
                # If adding this sentence exceeds limit and we have content
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Create overlap with previous chunk
                    overlap_sentences = max(1, self.overlap // 25)  # Roughly overlap sentences
                    if len(current_chunk) > overlap_sentences:
                        overlap_content = current_chunk[-overlap_sentences:]
                    else:
                        overlap_content = current_chunk
                    
                    # Start new chunk with overlap + current sentence
                    current_chunk = overlap_content + [sentence]
                    current_tokens = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # Add final chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding for text using BioClinicalBERT."""
        return self._get_transformer_embedding(text)

    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate BioClinicalBERT embedding with proper normalization."""
        try:
            import torch
            
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512,
                padding=False
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over sequence length
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Normalize for cosine similarity (CRITICAL for IndexFlatIP)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                # Handle zero vector edge case
                embedding = embedding / (norm + 1e-8)
                
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def retrieve_similar_cases(
        self,
        query: str,
        k: int = 5,
        return_chunks: bool = True,
        similarity_threshold: float = 0.1,
        use_metadata_filters: bool = False,
        apply_lexical_rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar medical cases using true cosine similarity.
        
        Args:
            query: Medical query text
            k: Number of results to return
            return_chunks: Return chunks (True) or full documents (False)
            similarity_threshold: Minimum cosine similarity (0-1)
            use_metadata_filters: Apply domain-specific filtering
            apply_lexical_rerank: Apply lexical boosting
            
        Returns:
            List of similar cases with metadata and scores
        """
        if not self.chunks:
            print("⚠️ No medical cases in memory")
            return []
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Extract filters if requested
        filters = None
        if use_metadata_filters:
            filters = self._extract_filters_from_query(query)
        
        # Generate query embedding
        try:
            query_embedding = self._get_embedding(query).astype('float32').reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")
        
        # Search with expanded pool for better results
        search_pool = max(k * 10, 100)  # Increased pool size
        search_pool = min(search_pool, len(self.chunks))  # Don't exceed available chunks
        
        # FIXED: IndexFlatIP returns cosine similarity directly for normalized vectors
        D, I = self.faiss_index.search(query_embedding, search_pool)
        similarities = D[0]  # Direct cosine similarity scores
        indices = I[0]
        
        # Process results
        preliminary_results = []
        for similarity, idx in zip(similarities, indices):
            if idx == -1:  # Invalid index
                continue
            if similarity < similarity_threshold:
                continue  # Skip low similarity results
                
            chunk_meta = self.chunk_metadata[idx]
            preliminary_results.append({
                "chunk_idx": idx,
                "text": self.chunks[idx],
                "score": float(similarity),
                "metadata": chunk_meta,
                "doc_id": chunk_meta["doc_id"],
                "source": chunk_meta["source"]
            })
        
        # Apply metadata filtering
        if filters and preliminary_results:
            preliminary_results = self._apply_metadata_filters(preliminary_results, filters)
        
        # Apply lexical reranking
        if apply_lexical_rerank and preliminary_results:
            preliminary_results = self._lexical_rerank(query, preliminary_results)
        
        # Format final results
        final_results = []
        seen_docs = set() if not return_chunks else set()
        
        for result in sorted(preliminary_results, key=lambda x: x["score"], reverse=True):
            doc_id = result["doc_id"]
            
            # Skip duplicate documents if returning full documents
            if not return_chunks and doc_id in seen_docs:
                continue
                
            if not return_chunks:
                seen_docs.add(doc_id)
                text = self.documents[doc_id]
                metadata = self.doc_metadata[doc_id]
            else:
                text = result["text"]
                metadata = result["metadata"]
            
            # Build output record
            output_record = {
                "text": text,
                "score": float(result["score"]),
                "source": metadata["source"],
                "doc_id": doc_id,
                "rank": len(final_results) + 1,
                "metadata": metadata
            }
            
            # Add chunk info if returning chunks
            if return_chunks:
                output_record["chunk_info"] = {
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"],
                    "chunk_id": metadata["chunk_id"]
                }
            
            final_results.append(output_record)
            
            if len(final_results) >= k:
                break
        
        # Log retrieval stats
        avg_similarity = np.mean([r["score"] for r in final_results]) if final_results else 0.0
        result_type = "chunks" if return_chunks else "documents"
        print(f"🔍 Retrieved {len(final_results)} {result_type} (avg similarity: {avg_similarity:.3f})")
        
        return final_results

    def find_best_match(self, query: str) -> Dict[str, Any]:
        """
        Find the best matching medical case for diagnosis.
        Uses proper cosine similarity thresholds for accurate matching.
        
        Args:
            query: Medical query describing symptoms/case
            
        Returns:
            Dictionary with query, diagnosis, and optional note
        """
        # Thresholds calibrated for TRUE cosine similarity (0-1 range)
        similarity_threshold = 0.75  # High threshold for initial retrieval
        domain_confidence_min = 0.65  # Minimum confidence for diagnosis
        
        # Retrieve best candidates
        results = self.retrieve_similar_cases(
            query,
            k=1,
            return_chunks=True,
            similarity_threshold=similarity_threshold,
            use_metadata_filters=True,
            apply_lexical_rerank=True
        )
        
        # Extract domain expectations from query
        filters = self._extract_filters_from_query(query)
        expected_systems = set(filters.get("body_system", []))
        
        # Check if we have any results
        if not results:
            return {
                "query": query,
                "diagnosis": None,
                "confidence": 0.0,
                "note": "No high-confidence matches found. Consider expanding patient history or examination findings."
            }
        
        best_match = results[0]
        best_system = {best_match['metadata'].get('body_system')}
        best_score = best_match['score']
        
        # Domain consistency check
        if expected_systems and not (best_system & expected_systems):
            return {
                "query": query,
                "diagnosis": None,
                "confidence": best_score,
                "note": f"No matches found in expected domain ({', '.join(expected_systems)}). Consider broader differential diagnosis."
            }
        
        # Confidence threshold check
        if best_score < domain_confidence_min:
            return {
                "query": query,
                "diagnosis": None,
                "confidence": best_score,
                "note": f"Low confidence match (score: {best_score:.3f}). Consider additional diagnostic workup."
            }
        
        # Return successful diagnosis
        diagnosis = best_match['metadata'].get('condition', 'Unknown condition')
        return {
            "query": query,
            "diagnosis": diagnosis,
            "confidence": best_score,
            "body_system": best_match['metadata'].get('body_system'),
            "source": best_match['source'],
            "doc_id": best_match['doc_id']
        }

    def _extract_filters_from_query(self, query: str) -> Dict[str, List[str]]:
        """Extract medical domain and symptom filters from query."""
        query_lower = query.lower()
        
        # Calculate system relevance scores
        system_scores = {}
        for system, pattern in self._patterns.items():
            matches = pattern.findall(query_lower)
            system_scores[system] = len(matches)
        
        # Find dominant system
        dominant_system = max(system_scores, key=system_scores.get) if system_scores else None
        
        filters = {"body_system": [], "symptom_tags": []}
        
        # Add dominant system if it has matches
        if dominant_system and system_scores[dominant_system] > 0:
            filters["body_system"].append(dominant_system)
        
        # Extract symptom tags
        for system, terms in self._boost_terms.items():
            for term in terms:
                term_spaced = term.replace("_", " ")
                if term in query_lower or term_spaced in query_lower:
                    filters["symptom_tags"].append(term)
        
        return filters

    def _apply_metadata_filters(self, candidates: List[Dict[str, Any]], 
                              filters: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Apply metadata-based filtering to candidates."""
        if not candidates:
            return candidates
        
        expected_systems = set(filters.get("body_system", []))
        expected_tags = set(filters.get("symptom_tags", []))
        
        # Apply filters
        filtered_candidates = []
        for candidate in candidates:
            metadata = candidate["metadata"]
            
            # System filter
            candidate_system = metadata.get("body_system")
            system_match = (not expected_systems) or (candidate_system in expected_systems)
            
            # Tag filter
            candidate_tags = set(metadata.get("symptom_tags", []))
            tag_match = (not expected_tags) or bool(candidate_tags & expected_tags)
            
            if system_match and tag_match:
                filtered_candidates.append(candidate)
        
        # Fallback: if no matches with full filters, try system-only
        if not filtered_candidates and expected_systems:
            for candidate in candidates:
                if candidate["metadata"].get("body_system") in expected_systems:
                    filtered_candidates.append(candidate)
        
        return filtered_candidates if filtered_candidates else candidates

    def _lexical_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply lexical boosting based on medical domain keywords."""
        query_lower = query.lower()
        
        # Determine dominant system for boosting
        system_scores = {sys: len(p.findall(query_lower)) for sys, p in self._patterns.items()}
        dominant_system = max(system_scores, key=system_scores.get) if system_scores else None
        
        boost_terms = self._boost_terms.get(dominant_system, []) if dominant_system else []
        
        # Cross-domain penalty terms (avoid confusion)
        penalty_terms = []
        if dominant_system == "vascular":
            penalty_terms = self._boost_terms.get("respiratory", [])
        elif dominant_system == "respiratory":
            penalty_terms = self._boost_terms.get("vascular", [])
        elif dominant_system == "cardiology":
            penalty_terms = self._boost_terms.get("respiratory", [])
        
        def calculate_lexical_adjustment(text: str) -> float:
            """Calculate lexical score adjustment."""
            text_lower = text.lower()
            
            # Positive boosting for relevant terms
            boost_score = sum(1.5 for term in boost_terms 
                            if term in text_lower or term.replace("_", " ") in text_lower)
            
            # Negative penalty for cross-domain terms
            penalty_score = sum(0.8 for term in penalty_terms 
                              if term in text_lower or term.replace("_", " ") in text_lower)
            
            # Length normalization (slightly favor concise, relevant matches)
            length_factor = 1.0 / (1.0 + np.log1p(len(text_lower) / 1000))
            
            return (boost_score * 0.15) - (penalty_score * 0.1) + (length_factor * 0.05)
        
        # Apply lexical adjustments
        for candidate in candidates:
            lexical_adjustment = calculate_lexical_adjustment(candidate["text"])
            candidate["score"] = candidate["score"] + lexical_adjustment
        
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory agent."""
        if not self.chunks:
            return {"status": "empty", "total_documents": 0, "total_chunks": 0}
        
        # Calculate statistics
        chunk_words = [meta.get("word_count", 0) for meta in self.chunk_metadata]
        avg_chunk_words = np.mean(chunk_words) if chunk_words else 0
        
        conditions = [meta.get("condition", "unknown") for meta in self.doc_metadata]
        body_systems = [meta.get("body_system", "unknown") for meta in self.doc_metadata]
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "avg_chunks_per_doc": len(self.chunks) / len(self.documents) if self.documents else 0,
            "avg_chunk_words": round(avg_chunk_words, 1),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "index_type": "IndexFlatIP",
            "similarity_method": "true_cosine_similarity",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.overlap,
            "sources": list(set(meta["source"] for meta in self.doc_metadata)),
            "conditions": list(set(conditions)),
            "body_systems": list(set(body_systems)),
            "conditions_count": len(set(conditions)),
            "body_systems_count": len(set(body_systems)),
            "use_transformers": getattr(self, 'use_transformers', False),
            "version": "fully_corrected_v1.0"
        }

    def analyze_similarity_distribution(self, query: str) -> Dict[str, Any]:
        """Analyze similarity distribution for a query across all chunks."""
        if not self.chunks:
            return {"error": "No chunks available for analysis"}
        
        try:
            query_embedding = self._get_embedding(query).astype('float32').reshape(1, -1)
            
            # Get similarities to all chunks
            D, I = self.faiss_index.search(query_embedding, len(self.chunks))
            similarities = D[0]  # Direct cosine similarities
            
            return {
                "query": query,
                "total_chunks": len(similarities),
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "mean_similarity": float(np.mean(similarities)),
                "median_similarity": float(np.median(similarities)),
                "std_similarity": float(np.std(similarities)),
                "quartiles": {
                    "q25": float(np.percentile(similarities, 25)),
                    "q50": float(np.percentile(similarities, 50)),
                    "q75": float(np.percentile(similarities, 75))
                },
                "similarity_range": float(np.max(similarities) - np.min(similarities)),
                "above_threshold_count": {
                    "0.5": int(np.sum(similarities > 0.5)),
                    "0.7": int(np.sum(similarities > 0.7)),
                    "0.8": int(np.sum(similarities > 0.8)),
                    "0.9": int(np.sum(similarities > 0.9))
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze similarity distribution: {e}"}

    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        if doc_id < 0 or doc_id >= len(self.documents):
            raise ValueError(f"Document ID {doc_id} out of range [0, {len(self.documents)})")
        
        doc_chunks = []
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata["doc_id"] == doc_id:
                doc_chunks.append({
                    "chunk_index": metadata["chunk_index"],
                    "chunk_id": metadata["chunk_id"],
                    "text": self.chunks[i],
                    "word_count": metadata.get("word_count", 0),
                    "char_count": metadata.get("char_count", 0),
                    "metadata": metadata
                })
        
        return sorted(doc_chunks, key=lambda x: x["chunk_index"])

    def save_memory(self, filepath: str = "bioclinical_memory.json"):
        """Save the complete memory state with correct indexing information."""
        import faiss
        
        base, ext = os.path.splitext(filepath)
        faiss_path = base + ".faiss"
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, faiss_path)
        
        # Prepare save data with complete metadata
        save_data = {
            "documents": self.documents,
            "doc_metadata": self.doc_metadata,
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "faiss_chunk_id_map": self.faiss_chunk_id_map,
            "config": {
                "model_name": self.model_name,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "embedding_dim": self.embedding_dim,
                "index_type": "IndexFlatIP",
                "similarity_method": "true_cosine_similarity",
                "use_transformers": getattr(self, 'use_transformers', False)
            },
            "saved_timestamp": datetime.now().isoformat(),
            "version": "fully_corrected_v1.0",
            "stats": self.get_stats()
        }
        
        # Save to JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Memory saved successfully!")
        print(f"   Data: {filepath}")
        print(f"   Index: {faiss_path}")
        print(f"   Documents: {len(self.documents)}, Chunks: {len(self.chunks)}")
        print("   ✅ Using TRUE cosine similarity with IndexFlatIP")

    def load_memory(self, filepath: str = "bioclinical_memory.json"):
        """Load memory state with compatibility checking."""
        import faiss
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Memory file not found: {filepath}")
        
        base, ext = os.path.splitext(filepath)
        faiss_path = base + ".faiss"
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")
        
        try:
            # Load JSON data
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check version compatibility
            config = data.get("config", {})
            version = data.get("version", "unknown")
            index_type = config.get("index_type", "unknown")
            
            if index_type != "IndexFlatIP":
                warnings.warn(
                    f"Loading index from older version (index_type: {index_type}). "
                    f"This may produce incorrect similarity scores. "
                    f"Consider re-indexing your documents with the corrected version.",
                    UserWarning
                )
            
            # Load data
            self.documents = data["documents"]
            self.doc_metadata = data["doc_metadata"]
            self.chunks = data["chunks"]
            self.chunk_metadata = data["chunk_metadata"]
            self.faiss_chunk_id_map = data.get("faiss_chunk_id_map", list(range(len(self.chunks))))
            
            # Load configuration
            self.model_name = config.get("model_name", self.model_name)
            self.chunk_size = config.get("chunk_size", self.chunk_size)
            self.overlap = config.get("overlap", self.overlap)
            self.use_transformers = config.get("use_transformers", False)
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(faiss_path)
            
            # Verify index dimensions
            if self.faiss_index.d != self.embedding_dim:
                raise ValueError(f"Index dimension mismatch: expected {self.embedding_dim}, got {self.faiss_index.d}")
            
            print(f"✅ Memory loaded successfully!")
            print(f"   Source: {filepath}")
            print(f"   Version: {version}")
            print(f"   Documents: {len(self.documents)}, Chunks: {len(self.chunks)}")
            print(f"   Index Type: {index_type}")
            print(f"   Saved: {data.get('saved_timestamp', 'Unknown time')}")
            
            if index_type == "IndexFlatIP":
                print("   ✅ Using correct cosine similarity calculation")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load memory: {e}")

    def clear_memory(self):
        """Clear all stored documents and reset the index."""
        import faiss
        
        self.chunks.clear()
        self.chunk_metadata.clear()
        self.documents.clear()
        self.doc_metadata.clear()
        self.faiss_chunk_id_map.clear()
        
        # Reset FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        print("🗑️ Memory cleared successfully")

    def get_medical_conditions(self) -> List[str]:
        """Get list of all medical conditions in the knowledge base."""
        conditions = [meta.get("condition", "unknown") for meta in self.doc_metadata]
        return sorted(list(set(conditions)))

    def get_body_systems(self) -> List[str]:
        """Get list of all body systems in the knowledge base."""
        systems = [meta.get("body_system", "unknown") for meta in self.doc_metadata]
        return sorted(list(set(systems)))

    def search_by_condition(self, condition: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for cases by medical condition."""
        matching_docs = []
        for i, meta in enumerate(self.doc_metadata):
            if condition.lower() in meta.get("condition", "").lower():
                matching_docs.append({
                    "doc_id": i,
                    "condition": meta.get("condition"),
                    "body_system": meta.get("body_system"),
                    "source": meta.get("source"),
                    "text": self.documents[i][:200] + "..." if len(self.documents[i]) > 200 else self.documents[i],
                    "metadata": meta
                })
        
        return matching_docs[:k]

    def __repr__(self) -> str:
        """String representation of the memory agent."""
        stats = self.get_stats()
        return (f"BioClinicalMemoryAgent(docs={stats['total_documents']}, "
                f"chunks={stats['total_chunks']}, "
                f"conditions={stats['conditions_count']}, "
                f"systems={stats['body_systems_count']})")