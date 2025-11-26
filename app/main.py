"""FastAPI application for MeduSearch semantic search API."""
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    BASE_DIR,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    SCRAPE_MAX_TOPICS,
    SCRAPE_DELAY,
    USE_SCRAPED_DATA,
)
from services.filter_service import FilterService, SearchFilters as ServiceSearchFilters
from services.index_service import IndexService
from services.model_service import ModelService
from services.search_service import SearchService


# -----------------------------
# Pipeline automation
# -----------------------------

def run_full_pipeline_if_needed() -> None:
    """Run full pipeline if index/metadata don't exist."""
    force_rebuild = os.getenv("MEDUSEARCH_FORCE_REBUILD", "0") == "1"
    index_exists = FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()

    if index_exists and not force_rebuild:
        print("[PIPELINE] Existing index + metadata found. Skipping rebuild.")
        return

    print("[PIPELINE] No index found or rebuild forced. Running full pipeline...")

    # Ensure data/index dirs exist
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "index").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    # 1) Scrape or generate data
    raw_data_path = BASE_DIR / "data" / "raw" / "patient_education.parquet"
    
    if USE_SCRAPED_DATA and not raw_data_path.exists():
        print("[PIPELINE] Scraping health data from WebMD...")
        cmd_scrape = [
            "python",
            str(BASE_DIR / "pipelines" / "scrape_webmd.py"),
            "--output",
            str(raw_data_path),
        ]
        if SCRAPE_MAX_TOPICS:
            cmd_scrape.extend(["--max-topics", str(SCRAPE_MAX_TOPICS)])
        cmd_scrape.extend(["--delay", str(SCRAPE_DELAY)])
        print(f"[PIPELINE] Running: {' '.join(cmd_scrape)}")
        subprocess.run(cmd_scrape, check=True)
    elif not raw_data_path.exists():
        print("[PIPELINE] Generating synthetic data...")
        cmd_generate = [
            "python",
            str(BASE_DIR / "pipelines" / "generate_data.py"),
            "--n_docs",
            "500",
        ]
        print(f"[PIPELINE] Running: {' '.join(cmd_generate)}")
        subprocess.run(cmd_generate, check=True)
    else:
        print(f"[PIPELINE] Raw data already exists at {raw_data_path}, skipping data generation/scraping.")

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
        EMBEDDING_MODEL_NAME,
    ]
    print(f"[PIPELINE] Running: {' '.join(cmd_embed)}")
    subprocess.run(cmd_embed, check=True)

    print("[PIPELINE] Full pipeline completed.")


# -----------------------------
# Global services
# -----------------------------

model_service: Optional[ModelService] = None
index_service: Optional[IndexService] = None
filter_service: Optional[FilterService] = None
search_service: Optional[SearchService] = None


# -----------------------------
# Startup / Shutdown
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_service, index_service, filter_service, search_service
    
    try:
        # Disable multiprocessing to avoid issues in web server context
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # 1) Run pipeline if needed
        run_full_pipeline_if_needed()

        # 2) Initialize services
        print("[INFO] Initializing services...")
        model_service = ModelService()
        model_service.initialize()
        
        index_service = IndexService()
        index_service.initialize()
        
        filter_service = FilterService(index_service)
        
        search_service = SearchService(
            model_service=model_service,
            index_service=index_service,
            filter_service=filter_service,
        )

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
        "Results are reranked using a cross-encoder reranker and only the top-1 match is returned. "
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
    """Filter criteria for search requests."""
    condition: Optional[str] = None
    category: Optional[str] = None
    min_score: Optional[float] = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    k: Optional[int] = None  # Ignored - always returns top-1 result
    filters: Optional[SearchFilters] = None


class SearchResult(BaseModel):
    """Search result model."""
    doc_id: int
    chunk_id: int
    condition: str
    category: Optional[str]
    title: str
    reading_level: Optional[str]
    score: float
    chunk_text: str


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    k: int
    device: str
    results: List[SearchResult]


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": model_service.device if model_service else "unknown",
        "n_vectors": index_service.total_vectors if index_service else 0,
        "model_loaded": model_service.is_initialized() if model_service else False,
        "index_loaded": index_service.is_initialized() if index_service else False,
        "n_metadata_rows": index_service.total_chunks if index_service else 0,
    }


@app.get("/test-embed")
def test_embed():
    """Test endpoint to check if embedding works."""
    try:
        if model_service is None or not model_service.is_initialized():
            return {"error": "Model service not initialized", "model_is_none": True}
        
        test_query = "test query"
        emb = model_service.embed_query(test_query)
        
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
    Semantic search endpoint with reranking.
    
    Returns only the top-1 result after reranking.
    """
    if search_service is None:
        raise RuntimeError("Search service not initialized. Check startup logs.")

    # Convert API filters to service filters
    service_filters = None
    if request.filters:
        service_filters = ServiceSearchFilters(
            condition=request.filters.condition,
            category=request.filters.category,
            min_score=request.filters.min_score,
        )

    # Perform search
    result = search_service.search(query=request.query, filters=service_filters)

    # Convert to API response
    if result is None:
        return SearchResponse(
            query=request.query,
            k=1,
            device=model_service.device if model_service else "unknown",
            results=[],
        )

    return SearchResponse(
        query=request.query,
        k=1,
        device=model_service.device if model_service else "unknown",
        results=[SearchResult(**result.to_dict())],
    )
