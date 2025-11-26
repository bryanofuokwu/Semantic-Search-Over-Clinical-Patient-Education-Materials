"""Configuration constants for the application."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent  # project root
INDEX_DIR = BASE_DIR / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index_flatip.bin"
METADATA_PATH = INDEX_DIR / "chunk_metadata.parquet"

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search configuration
RERANKER_TOP_K = 20  # Retrieve more candidates for reranking
DEFAULT_RESULTS_COUNT = 1  # Always return top-1 result

