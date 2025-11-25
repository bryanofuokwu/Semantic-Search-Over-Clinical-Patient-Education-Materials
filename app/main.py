import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths / Config
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

INDEX_DIR = BASE_DIR / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index_flatip.bin"
METADATA_PATH = INDEX_DIR / "chunk_metadata.parquet"

# Must match the model used in embed_and_index.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5


# -----------------------------
# Startup / Shutdown
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, index, meta_df, device_str
    
    try:
        # Disable multiprocessing to avoid issues in web server context
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # 1) Run pipeline if needed
        run_full_pipeline_if_needed()

        # 2) Select device for query-time embeddings
        device_str = select_device()

        # 3) Load model, index, metadata
        print("[INFO] Loading model, index, and metadata...")
        model = load_model()
        # Ensure model is in eval mode
        if hasattr(model, 'eval'):
            model.eval()
        index = load_faiss_index(FAISS_INDEX_PATH)
        meta_df = load_metadata(METADATA_PATH)

        print("[INFO] API startup complete.")
    except Exception as e:
        print(f"[ERROR] Startup failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown (cleanup if needed)
    print("[INFO] Shutting down...")


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="MeduSearch API",
    description=(
        "Semantic search API over synthetic clinical patient education materials. "
        "Embeddings are computed with SentenceTransformers and a GPU (CUDA) when available. "
        "On startup, the full data → preprocess → embed → index pipeline runs automatically "
        "if no index is found."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# -----------------------------
# Request / Response models
# -----------------------------

class SearchFilters(BaseModel):
    condition: Optional[str] = None
    category: Optional[str] = None
    min_score: Optional[float] = None  # cosine similarity threshold (0–1)


class SearchRequest(BaseModel):
    query: str
    k: int = DEFAULT_TOP_K
    filters: Optional[SearchFilters] = None


class SearchResult(BaseModel):
    doc_id: int
    chunk_id: int
    condition: str
    category: Optional[str]
    title: str
    reading_level: Optional[str]
    score: float
    chunk_text: str


class SearchResponse(BaseModel):
    query: str
    k: int
    device: str
    results: List[SearchResult]


# -----------------------------
# Global state (model, index, metadata)
# -----------------------------

model: SentenceTransformer = None
index: faiss.Index = None
meta_df: pd.DataFrame = None
device_str: str = "cpu"
_model_lock = threading.Lock()  # Lock for thread-safe model encoding


# -----------------------------
# Pipeline automation
# -----------------------------

def run_full_pipeline_if_needed() -> None:
    """
    Run generate_data -> preprocess -> embed_and_index if the index/metadata
    do not exist, or if MEDUSEARCH_FORCE_REBUILD=1 is set in the environment.
    """
    force_rebuild = os.getenv("MEDUSEARCH_FORCE_REBUILD", "0") == "1"

    index_exists = FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()

    if index_exists and not force_rebuild:
        print("[PIPELINE] Existing index + metadata found. Skipping rebuild.")
        return

    print("[PIPELINE] No index found or rebuild forced. Running full pipeline...")

    # Ensure data/index dirs exist
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    # 1) Generate synthetic data
    cmd_generate = [
        "python",
        str(BASE_DIR / "pipelines" / "generate_data.py"),
        "--n_docs",
        "500",
    ]
    print(f"[PIPELINE] Running: {' '.join(cmd_generate)}")
    subprocess.run(cmd_generate, check=True)

    # 2) Preprocess & chunk
    cmd_preprocess = [
        "python",
        str(BASE_DIR / "pipelines" / "preprocess.py"),
        "--chunk_size",
        "1200",
        "--chunk_overlap",
        "200",
    ]
    print(f"[PIPELINE] Running: {' '.join(cmd_preprocess)}")
    subprocess.run(cmd_preprocess, check=True)

    # 3) Embed with CUDA (if available) and build FAISS index
    cmd_embed = [
        "python",
        str(BASE_DIR / "pipelines" / "embed_and_index.py"),
        "--device",
        "cuda",
        "--batch_size",
        "256",
        "--model",
        MODEL_NAME,
    ]
    print(f"[PIPELINE] Running: {' '.join(cmd_embed)}")
    subprocess.run(cmd_embed, check=True)

    print("[PIPELINE] Full pipeline completed.")


# -----------------------------
# Device / loading helpers
# -----------------------------

def select_device() -> str:
    if torch.cuda.is_available():
        print("[INFO] CUDA is available. Using GPU for query embeddings.")
        return "cuda"
    print("[INFO] CUDA is not available. Using CPU for query embeddings.")
    return "cpu"


def load_model() -> SentenceTransformer:
    print(f"[INFO] Loading SentenceTransformer model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(MODEL_NAME, device=device)
    print(f"[INFO] Model loaded on device: {device}")
    return m


def load_faiss_index(path: Path) -> faiss.Index:
    print(f"[INFO] Loading FAISS index from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {path}")
    idx = faiss.read_index(str(path))
    print(f"[INFO] Loaded FAISS index with {idx.ntotal} vectors.")
    return idx


def load_metadata(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading chunk metadata from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Metadata parquet not found at: {path}")
    df = pd.read_parquet(path)
    expected_cols = {"doc_id", "chunk_id", "condition", "title", "chunk_text"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    print(f"[INFO] Loaded metadata for {len(df)} chunks.")
    return df




# -----------------------------
# Utility: search helpers
# -----------------------------

def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string using the global model on the selected device.
    Returns a (1, dim) float32 numpy array normalized to unit length.
    Thread-safe.
    """
    if model is None:
        raise RuntimeError("Model is not initialized.")

    # Use lock to ensure thread-safe encoding
    # Disable multiprocessing to avoid issues in web server context
    with _model_lock:
        try:
            emb = model.encode(
                [text],
                device=device_str,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=1,  # Use small batch size
            )
        except Exception as e:
            print(f"[ERROR] Embedding failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    return emb.astype("float32")


def apply_filters(
    indices: np.ndarray,
    scores: np.ndarray,
    filters: Optional[SearchFilters],
) -> List[int]:
    """
    Given raw FAISS indices and scores, optionally filter by:
      - condition (exact match, case-insensitive)
      - category (exact match, case-insensitive)
      - min_score (similarity threshold)
    Returns a list of positions into the original arrays that pass filters, in order.
    """
    if filters is None:
        return list(range(len(indices)))

    keep_indices: List[int] = []

    for i, (idx, score) in enumerate(zip(indices, scores)):
        row = meta_df.iloc[idx]

        if filters.condition is not None:
            if str(row["condition"]).lower() != filters.condition.lower():
                continue

        if filters.category is not None:
            cat = row.get("category", None)
            if cat is None or str(cat).lower() != filters.category.lower():
                continue

        if filters.min_score is not None:
            if float(score) < float(filters.min_score):
                continue

        keep_indices.append(i)

    return keep_indices


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    """
    Simple healthcheck endpoint.
    """
    return {
        "status": "ok",
        "device": device_str,
        "n_vectors": int(index.ntotal) if index is not None else 0,
    }


@app.get("/test-embed")
def test_embed():
    """
    Test endpoint to check if embedding works.
    """
    try:
        if model is None:
            return {"error": "Model not loaded", "model_is_none": True}
        
        print("[TEST] Starting embedding test...")
        test_query = "test query"
        print(f"[TEST] Query: {test_query}")
        print(f"[TEST] Device: {device_str}")
        print(f"[TEST] Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'unknown'}")
        
        # Try direct model call first
        print("[TEST] Calling model.encode directly...")
        emb = model.encode(
            [test_query],
            device=device_str,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print(f"[TEST] Embedding successful: {emb.shape}")
        
        return {
            "status": "ok",
            "query": test_query,
            "embedding_shape": list(emb.shape),
            "embedding_dtype": str(emb.dtype),
        }
    except Exception as e:
        print(f"[TEST ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "type": type(e).__name__,
        }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Semantic search over patient education chunks.

    - Embeds the query on GPU (CUDA) if available.
    - Uses FAISS to retrieve top-k most similar chunks.
    - Applies optional filters (condition, category, min_score).
    """
    try:
        if index is None or model is None or meta_df is None:
            raise RuntimeError("Service not fully initialized. Check startup logs.")

        query = request.query.strip()
        if not query:
            return SearchResponse(query=query, k=request.k, device=device_str, results=[])

        # 1. Embed query
        query_emb = embed_query(query)

        # 2. FAISS search
        k = max(1, min(request.k, index.ntotal))  # Don't search for more than available
        distances, indices = index.search(query_emb, k)  # shapes: (1, k)
        distances = distances[0]
        indices = indices[0]

        # 3. Apply filters
        keep_positions = apply_filters(indices, distances, request.filters)
        results: List[SearchResult] = []

        for pos in keep_positions:
            idx = int(indices[pos])
            score = float(distances[pos])

            # Validate index bounds
            if idx < 0 or idx >= len(meta_df):
                print(f"[WARNING] Invalid index {idx}, skipping")
                continue

            row = meta_df.iloc[idx]

            result = SearchResult(
                doc_id=int(row["doc_id"]),
                chunk_id=int(row["chunk_id"]),
                condition=str(row["condition"]),
                category=str(row.get("category", "")) if "category" in row else None,
                title=str(row["title"]),
                reading_level=str(row.get("reading_level", "")) if "reading_level" in row else None,
                score=score,
                chunk_text=str(row["chunk_text"]),
            )
            results.append(result)

        return SearchResponse(
            query=query,
            k=k,
            device=device_str,
            results=results,
        )
    except Exception as e:
        print(f"[ERROR] Search failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise