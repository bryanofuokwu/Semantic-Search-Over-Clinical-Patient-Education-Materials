# MeduSearch 🩺  
Semantic Search Over Clinical Patient Education Materials (CUDA-Accelerated)

MeduSearch is an end-to-end semantic search project that:

- Generates **synthetic clinical patient education materials** (non-diagnostic, plain-language).
- Preprocesses and **chunks** long documents into smaller units.
- Uses **SentenceTransformers + PyTorch (CUDA)** to compute embeddings on a GPU.
- Indexes vectors with **FAISS** for fast similarity search.
- **Reranks** candidates using a **CrossEncoder** for improved relevance.
- Returns the **top-1 most relevant result** for each query.
- Exposes a **FastAPI** endpoint for natural language queries like:

> "What is high blood pressure and how do I manage it?"  
> "Explain Type 2 diabetes in simple terms."  
> "What are common side effects of asthma treatment?"

The focus is on **GPU-accelerated embeddings (CUDA)**, **reranking for accuracy**, and a pipeline that looks and feels production-ish.

---

## 🔧 Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch + SentenceTransformers
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder)
- **GPU Acceleration:** CUDA (tested on RTX 3060)
- **Vector Search:** FAISS (IndexFlatIP)
- **API:** FastAPI + Uvicorn
- **Data Format:** Parquet (via pandas + pyarrow)

---

## 📁 Project Structure

```text
medu-search/
├── app/
│   └── main.py                      # FastAPI application (API routes only)
├── services/                        # Service layer (business logic)
│   ├── model_service.py             # ML model management (embedding + reranker)
│   ├── index_service.py             # FAISS index and metadata management
│   ├── filter_service.py            # Search result filtering
│   └── search_service.py            # Main search orchestration
├── config.py                        # Configuration constants
├── configs/
│   └── config.yaml                  # Pipeline configuration
├── data/
│   ├── raw/                         # Synthetic source data (patient_education.parquet)
│   ├── processed/                   # full_text docs + chunks
│   └── index/                       # embeddings, FAISS index, metadata
├── metrics/
│   └── embedding_benchmarks.json   # CPU / CUDA embedding benchmarks
├── pipelines/
│   ├── generate_data.py             # synth patient education documents
│   ├── preprocess.py                # build full_text and chunk into smaller pieces
│   └── embed_and_index.py           # CUDA embeddings + FAISS index build
├── requirements.txt
└── README.md

---

## 🔍 How It Works

The search pipeline follows a two-stage retrieval and reranking approach:

1. **Initial Retrieval (FAISS)**: 
   - Query is embedded using SentenceTransformers
   - FAISS retrieves top 20 candidate chunks based on cosine similarity

2. **Reranking (CrossEncoder)**:
   - All candidates are reranked using a cross-encoder model
   - Cross-encoder scores query-document pairs directly for better relevance
   - Only the **top-1 result** is returned

This hybrid approach combines the speed of FAISS with the accuracy of cross-encoder reranking.

### Architecture

The application follows a clean **service layer architecture**:

- **API Layer** (`app/main.py`): FastAPI routes that handle HTTP requests/responses
- **Service Layer** (`services/`): Business logic separated into focused services:
  - `ModelService`: Manages embedding model and reranker initialization and operations
  - `IndexService`: Handles FAISS index operations and metadata management
  - `FilterService`: Applies search filters (condition, category, min_score)
  - `SearchService`: Orchestrates the complete search flow (embed → retrieve → filter → rerank)
- **Configuration** (`config.py`): Centralized configuration constants

This separation ensures clean code organization, testability, and maintainability.

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv semsearch-env
source semsearch-env/bin/activate  # On Windows: semsearch-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Running the API

```bash
# Start the FastAPI server
uvicorn app.main:app --reload

# The API will automatically run the full pipeline on first startup if no index exists
```

### API Usage

```bash
# Search endpoint
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is high blood pressure?",
    "filters": {
      "condition": "hypertension"
    }
  }'

# Health check
curl http://localhost:8000/health
```

The search endpoint returns only the **top-1 most relevant result** after reranking.

---

## 🏗️ Architecture

The codebase is organized with a **service layer pattern** for clean separation of concerns:

- **API Layer**: Thin FastAPI routes that delegate to services
- **Service Layer**: Reusable business logic components
- **Configuration**: Centralized constants and settings

This architecture makes the codebase:
- ✅ **Testable**: Services can be unit tested independently
- ✅ **Maintainable**: Clear separation between API and business logic
- ✅ **Scalable**: Easy to add new features or modify existing ones
- ✅ **Clean**: Each service has a single, well-defined responsibility