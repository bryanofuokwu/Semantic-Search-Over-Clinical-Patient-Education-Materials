"""Service for managing ML models (embedding model and reranker)."""
import threading
from typing import Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME


class ModelService:
    """Service for managing embedding and reranker models."""
    
    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.device: str = self._select_device()
        self._model_lock = threading.Lock()
        
    def _select_device(self) -> str:
        """Select the best available device (CUDA or CPU)."""
        if torch.cuda.is_available():
            print("[INFO] CUDA is available. Using GPU for models.")
            return "cuda"
        print("[INFO] CUDA is not available. Using CPU for models.")
        return "cpu"
    
    def load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model."""
        print(f"[INFO] Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        if hasattr(model, 'eval'):
            model.eval()
        print(f"[INFO] Embedding model loaded on device: {self.device}")
        return model
    
    def load_reranker(self) -> CrossEncoder:
        """Load the cross-encoder reranker model."""
        print(f"[INFO] Loading CrossEncoder reranker: {RERANKER_MODEL_NAME}")
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=self.device)
        if hasattr(reranker_model, 'model') and hasattr(reranker_model.model, 'eval'):
            reranker_model.model.eval()
        print(f"[INFO] Reranker loaded on device: {self.device}")
        return reranker_model
    
    def initialize(self):
        """Initialize all models."""
        self.embedding_model = self.load_embedding_model()
        self.reranker = self.load_reranker()
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            Normalized embedding vector as numpy array
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not initialized.")
        
        with self._model_lock:
            try:
                emb = self.embedding_model.encode(
                    [text],
                    device=self.device,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1,
                )
            except Exception as e:
                print(f"[ERROR] Embedding failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise
        return emb.astype("float32")
    
    def rerank(self, query: str, documents: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Rerank documents using the cross-encoder reranker.
        
        Args:
            query: Query text
            documents: List of document texts to rerank
            batch_size: Batch size for reranking
            
        Returns:
            Array of reranker scores
        """
        if self.reranker is None:
            raise RuntimeError("Reranker is not initialized.")
        
        if not documents:
            return np.array([])
        
        pairs = [[query, doc] for doc in documents]
        
        with self._model_lock:
            if hasattr(self.reranker, 'model') and hasattr(self.reranker.model, 'eval'):
                self.reranker.model.eval()
            
            scores = self.reranker.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        
        if scores.ndim == 0:
            scores = np.array([scores])
        
        if scores.ndim > 1:
            scores = scores.flatten()
        
        return scores
    
    def is_initialized(self) -> bool:
        """Check if models are initialized."""
        return self.embedding_model is not None and self.reranker is not None

