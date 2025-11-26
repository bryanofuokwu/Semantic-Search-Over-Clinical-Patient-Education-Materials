"""Service for managing FAISS index and metadata."""
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from config import FAISS_INDEX_PATH, METADATA_PATH


class IndexService:
    """Service for managing FAISS index and chunk metadata."""
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[pd.DataFrame] = None
    
    def load_index(self, path: Path) -> faiss.Index:
        """Load FAISS index from disk."""
        print(f"[INFO] Loading FAISS index from: {path}")
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {path}")
        idx = faiss.read_index(str(path))
        print(f"[INFO] Loaded FAISS index with {idx.ntotal} vectors.")
        return idx
    
    def load_metadata(self, path: Path) -> pd.DataFrame:
        """Load chunk metadata from disk."""
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
    
    def initialize(self):
        """Initialize index and metadata."""
        self.index = self.load_index(FAISS_INDEX_PATH)
        self.metadata = self.load_metadata(METADATA_PATH)
    
    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to retrieve
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise RuntimeError("Index is not initialized.")
        
        k = max(1, min(k, self.index.ntotal))
        distances, indices = self.index.search(query_embedding, k)
        distances = distances[0]
        indices = indices[0]
        
        # Filter out invalid indices (-1) that FAISS returns when there aren't enough results
        valid_mask = indices >= 0
        indices = indices[valid_mask]
        distances = distances[valid_mask]
        
        return distances, indices
    
    def get_chunk_by_index(self, idx: int) -> Optional[dict]:
        """
        Get chunk metadata by index.
        
        Args:
            idx: Index into the metadata dataframe
            
        Returns:
            Dictionary with chunk metadata or None if invalid
        """
        if self.metadata is None:
            raise RuntimeError("Metadata is not initialized.")
        
        if idx < 0 or idx >= len(self.metadata):
            return None
        
        row = self.metadata.iloc[idx]
        return {
            "doc_id": int(row["doc_id"]),
            "chunk_id": int(row["chunk_id"]),
            "condition": str(row["condition"]),
            "category": str(row.get("category", "")) if "category" in row else None,
            "title": str(row["title"]),
            "reading_level": str(row.get("reading_level", "")) if "reading_level" in row else None,
            "chunk_text": str(row["chunk_text"]),
        }
    
    def is_initialized(self) -> bool:
        """Check if index and metadata are initialized."""
        return self.index is not None and self.metadata is not None
    
    @property
    def total_vectors(self) -> int:
        """Get total number of vectors in the index."""
        return int(self.index.ntotal) if self.index is not None else 0
    
    @property
    def total_chunks(self) -> int:
        """Get total number of chunks in metadata."""
        return len(self.metadata) if self.metadata is not None else 0

