import os
import time
import json
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Default paths
# -----------------------------

DEFAULT_CHUNK_INPUT_PATH = os.path.join(
    "data", "processed", "patient_education_chunks.parquet"
)
DEFAULT_INDEX_DIR = os.path.join("data", "index")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 256
DEFAULT_METRICS_PATH = os.path.join("metrics", "embedding_benchmarks.json")


# -----------------------------
# CLI / Args
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunked patient education docs with CUDA (if available) "
                    "and build a FAISS index."
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=DEFAULT_CHUNK_INPUT_PATH,
        help=f"Path to chunk-level Parquet (default: {DEFAULT_CHUNK_INPUT_PATH})",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory to save embeddings, index, and metadata (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for encoding (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for embeddings: 'cuda', 'cpu', or 'auto' (prefers cuda if available). "
             "Default: auto",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=DEFAULT_METRICS_PATH,
        help=f"Path to save embedding benchmark metrics JSON (default: {DEFAULT_METRICS_PATH})",
    )
    return parser.parse_args()


# -----------------------------
# Core functions
# -----------------------------

def select_device(device_arg: str) -> str:
    """
    Decide whether to use 'cuda' or 'cpu' based on user preference and availability.
    """
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] --device cuda requested but CUDA is not available. Falling back to CPU.")
            return "cpu"
        return "cuda"
    elif device_arg == "cpu":
        return "cpu"
    else:  # auto
        if torch.cuda.is_available():
            print("[INFO] CUDA is available. Using GPU.")
            return "cuda"
        else:
            print("[INFO] CUDA not available. Using CPU.")
            return "cpu"


def load_chunks(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading chunks from: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded {len(df)} chunks.")
    if "chunk_text" not in df.columns:
        raise ValueError("Expected 'chunk_text' column in chunks parquet.")
    return df


def encode_texts(
    model: SentenceTransformer,
    texts,
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Encode texts with SentenceTransformers on the given device, measure performance,
    and return embeddings + metrics.
    """
    print(f"\n[INFO] Encoding {len(texts)} chunks on device='{device}' "
          f"with batch_size={batch_size}...")

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        device=device,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    embeds_per_sec = len(texts) / total_time if total_time > 0 else 0.0
    ms_per_1k = (total_time / (len(texts) / 1000)) * 1000 if len(texts) > 0 else 0.0

    print(
        f"[INFO] Embedding done in {total_time:.2f}s "
        f"({embeds_per_sec:.1f} embeds/sec, {ms_per_1k:.2f} ms per 1k docs)"
    )

    return {
        "embeddings": embeddings.astype("float32"),
        "total_time": total_time,
        "embeds_per_sec": embeds_per_sec,
        "ms_per_1k": ms_per_1k,
    }


def build_faiss_index(embeddings: np.ndarray, out_path: str) -> None:
    """
    Build a FAISS IndexFlatIP from the given embeddings and persist it to disk.
    Assumes embeddings are L2-normalized if cosine-like similarity is desired.
    """
    dim = embeddings.shape[1]
    print(f"\n[INFO] Building FAISS IndexFlatIP with dim={dim} on CPU...")
    index = faiss.IndexFlatIP(dim)

    t0 = time.perf_counter()
    index.add(embeddings)
    t1 = time.perf_counter()
    build_time = t1 - t0

    print(f"[INFO] FAISS index built in {build_time:.3f}s. Total vectors: {index.ntotal}")
    faiss.write_index(index, out_path)
    print(f"[INFO] FAISS index saved to: {out_path}")


def save_metrics(
    metrics_path: str,
    device: str,
    model_name: str,
    n_vectors: int,
    total_time: float,
    embeds_per_sec: float,
    ms_per_1k: float,
) -> None:
    """
    Append or create a JSON file containing embedding benchmark metrics.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    record = {
        "device": device,
        "model_name": model_name,
        "n_vectors": n_vectors,
        "total_time_sec": total_time,
        "embeds_per_sec": embeds_per_sec,
        "ms_per_1k_docs": ms_per_1k,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # If file exists, load and append; else, create new list
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        except Exception:
            data = []
    else:
        data = []

    data.append(record)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[INFO] Saved embedding metrics to: {metrics_path}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()

    # Resolve device
    device = select_device(args.device)

    # Ensure index directory exists
    os.makedirs(args.index_dir, exist_ok=True)

    # 1. Load chunks
    df_chunks = load_chunks(args.chunks)
    texts = df_chunks["chunk_text"].tolist()

    # 2. Load model
    print(f"[INFO] Loading SentenceTransformer model: {args.model}")
    model = SentenceTransformer(args.model)

    # 3. Encode with CUDA (or CPU if fallback)
    result = encode_texts(
        model=model,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
    )

    embeddings = result["embeddings"]
    total_time = result["total_time"]
    embeds_per_sec = result["embeds_per_sec"]
    ms_per_1k = result["ms_per_1k"]

    # 4. Save embeddings
    emb_path = os.path.join(args.index_dir, f"embeddings_{device}.npy")
    np.save(emb_path, embeddings)
    print(f"[INFO] Saved embeddings to: {emb_path}")

    # 5. Save chunk-level metadata aligned with embeddings
    #    We only keep what's useful at query time (doc_id, chunk_id, title, condition, etc.)
    meta_cols = [
        "doc_id",
        "chunk_id",
        "condition",
        "category",
        "title",
        "reading_level",
        "chunk_text",
    ]
    meta_cols = [c for c in meta_cols if c in df_chunks.columns]  # guard
    meta_path = os.path.join(args.index_dir, "chunk_metadata.parquet")
    df_chunks[meta_cols].to_parquet(meta_path, index=False)
    print(f"[INFO] Saved chunk metadata to: {meta_path}")

    # 6. Build FAISS index from embeddings (CPU)
    index_path = os.path.join(args.index_dir, "faiss_index_flatip.bin")
    build_faiss_index(embeddings, index_path)

    # 7. Save benchmark metrics
    save_metrics(
        metrics_path=args.metrics_path,
        device=device,
        model_name=args.model,
        n_vectors=len(texts),
        total_time=total_time,
        embeds_per_sec=embeds_per_sec,
        ms_per_1k=ms_per_1k,
    )

    print("\n[INFO] Embedding + indexing pipeline completed successfully.")


if __name__ == "__main__":
    main()