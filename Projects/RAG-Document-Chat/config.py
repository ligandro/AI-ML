"""
Configuration constants for RAG Document Chat
Can be imported by both main app and ingest modules
"""

MODEL_NAME = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"

VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "chroma_db"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# Retrieval Configuration
RETRIEVAL_TYPE = "mmr"  # Options: "mmr" or "multi_query"
MMR_K = 12  # Number of documents to return
MMR_FETCH_K = 60  # Number of documents to fetch before MMR filtering
MMR_LAMBDA = 0.5  # Diversity factor (0 = max diversity, 1 = max relevance)

# LLM Configuration (Anti-Hallucination Settings)
LLM_TEMPERATURE = 0  # Deterministic responses (0 = no creativity, 1 = creative)
LLM_MAX_TOKENS = 500  # Maximum response length
