# MeduSearch 🩺  
Semantic Search Over Clinical Patient Education Materials (CUDA-Accelerated)

MeduSearch is an end-to-end semantic search project that:

- Generates **synthetic clinical patient education materials** (non-diagnostic, plain-language).
- Preprocesses and **chunks** long documents into smaller units.
- Uses **SentenceTransformers + PyTorch (CUDA)** to compute embeddings on a GPU.
- Indexes vectors with **FAISS** for fast similarity search.
- Exposes a **FastAPI** endpoint for natural language queries like:

> “What is high blood pressure and how do I manage it?”  
> “Explain Type 2 diabetes in simple terms.”  
> “What are common side effects of asthma treatment?”

The focus is on **GPU-accelerated embeddings (CUDA)** and a pipeline that looks and feels production-ish.

---

## 🔧 Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch + SentenceTransformers
- **GPU Acceleration:** CUDA (tested on RTX 3060)
- **Vector Search:** FAISS (IndexFlatIP)
- **API:** FastAPI + Uvicorn
- **Data Format:** Parquet (via pandas + pyarrow)

---

## 📁 Project Structure

```text
medu-search/
├── app/
│   └── main.py                      # FastAPI app (semantic search API)
├── configs/
│   └── config.yaml                  # Central config (paths, model, chunking, etc.)
├── data/
│   ├── raw/                         # Synthetic source data (patient_education.parquet)
│   ├── processed/                   # full_text docs + chunks
│   └── index/                       # embeddings, FAISS index, metadata
├── metrics/
│   └── embedding_benchmarks.json    # CPU / CUDA embedding benchmarks
├── pipelines/
│   ├── generate_data.py             # synth patient education documents
│   ├── preprocess.py                # build full_text and chunk into smaller pieces
│   └── embed_and_index.py           # CUDA embeddings + FAISS index build
├── requirements.txt
└── README.md