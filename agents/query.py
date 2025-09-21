import os
import sys
from memory import BioClinicalMemoryAgent

MEMORY_JSON = "bioclinical_memory_new.json" 

agent_ew = BioClinicalMemoryAgent()
ok = agent_ew.load_memory(MEMORY_JSON)
query = "56-year-old female presents with shortness of breath, fever, and headache for several days; mild upper respiratory symptoms"
results = agent_ew.retrieve_similar_cases(
        query=query,
        k=1,
        similarity_threshold=0.9,   # get everything, then we’ll inspect scores
        return_chunks=True,
        use_metadata_filters=True,
        apply_lexical_rerank=True   # keep scores raw for debugging
    )
print(results)