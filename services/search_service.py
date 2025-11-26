"""Main search service that orchestrates search operations."""
from typing import Optional

import numpy as np

from config import RERANKER_TOP_K
from services.filter_service import FilterService, SearchFilters
from services.index_service import IndexService
from services.model_service import ModelService


class SearchResult:
    """Represents a single search result."""
    
    def __init__(
        self,
        doc_id: int,
        chunk_id: int,
        condition: str,
        category: Optional[str],
        title: str,
        reading_level: Optional[str],
        score: float,
        chunk_text: str,
    ):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.condition = condition
        self.category = category
        self.title = title
        self.reading_level = reading_level
        self.score = score
        self.chunk_text = chunk_text
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "condition": self.condition,
            "category": self.category,
            "title": self.title,
            "reading_level": self.reading_level,
            "score": self.score,
            "chunk_text": self.chunk_text,
        }


class SearchService:
    """Main service for performing semantic search with reranking."""
    
    def __init__(
        self,
        model_service: ModelService,
        index_service: IndexService,
        filter_service: FilterService,
    ):
        self.model_service = model_service
        self.index_service = index_service
        self.filter_service = filter_service
    
    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
    ) -> Optional[SearchResult]:
        """
        Perform semantic search with reranking.
        
        Args:
            query: Search query text
            filters: Optional filter criteria
            
        Returns:
            Top-1 search result or None if no results found
        """
        query = query.strip()
        if not query:
            return None
        
        # 1. Embed query
        query_emb = self.model_service.embed_query(query)
        
        # 2. Search FAISS index
        rerank_k = max(1, min(RERANKER_TOP_K, self.index_service.total_vectors))
        distances, indices = self.index_service.search(query_emb, rerank_k)
        
        if len(indices) == 0:
            print(f"[WARNING] No valid results from FAISS search")
            return None
        
        # 3. Apply filters
        keep_positions = self.filter_service.apply_filters(indices, distances, filters)
        
        if not keep_positions:
            print(f"[WARNING] All candidates filtered out. Total candidates: {len(indices)}")
            return None
        
        # 4. Prepare candidates for reranking
        candidates = []
        valid_candidates = []
        
        for pos in keep_positions:
            idx = int(indices[pos])
            chunk_data = self.index_service.get_chunk_by_index(idx)
            
            if chunk_data is None:
                print(f"[WARNING] Invalid index {idx}, skipping")
                continue
            
            chunk_text = str(chunk_data["chunk_text"]).strip()
            if not chunk_text:
                print(f"[WARNING] Skipping candidate with empty chunk_text")
                continue
            
            candidate = {
                **chunk_data,
                "faiss_score": float(distances[pos]),
            }
            candidates.append(candidate)
            valid_candidates.append(chunk_text)
        
        if not candidates:
            return None
        
        # 5. Rerank candidates
        print(f"[DEBUG] Reranking {len(candidates)} valid candidates")
        try:
            rerank_scores = self.model_service.rerank(query, valid_candidates)
            
            # Handle NaN/invalid scores
            valid_mask = np.isfinite(rerank_scores)
            if not np.all(valid_mask):
                print(f"[WARNING] Found {np.sum(~valid_mask)} invalid scores (NaN/inf) from reranker")
                if not np.any(valid_mask):
                    print(f"[WARNING] All reranker scores are invalid, falling back to FAISS scores")
                    rerank_scores = np.array([cand["faiss_score"] for cand in candidates])
                else:
                    min_valid = np.min(rerank_scores[valid_mask])
                    rerank_scores[~valid_mask] = min_valid - 1.0
            
            # 6. Get top-1 result
            top_idx = int(np.argmax(rerank_scores))
            top_candidate = candidates[top_idx]
            score = float(rerank_scores[top_idx])
            
            # Validate score is JSON-compliant
            if not np.isfinite(score):
                print(f"[WARNING] Invalid score {score}, using 0.0 instead")
                score = 0.0
            
            print(f"[DEBUG] Selected top result at index {top_idx} with score {score}")
            
            return SearchResult(
                doc_id=top_candidate["doc_id"],
                chunk_id=top_candidate["chunk_id"],
                condition=top_candidate["condition"],
                category=top_candidate["category"],
                title=top_candidate["title"],
                reading_level=top_candidate["reading_level"],
                score=score,
                chunk_text=top_candidate["chunk_text"],
            )
            
        except Exception as e:
            print(f"[ERROR] Reranking failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use first candidate
            print(f"[WARNING] Falling back to first candidate")
            top_candidate = candidates[0]
            return SearchResult(
                doc_id=top_candidate["doc_id"],
                chunk_id=top_candidate["chunk_id"],
                condition=top_candidate["condition"],
                category=top_candidate["category"],
                title=top_candidate["title"],
                reading_level=top_candidate["reading_level"],
                score=float(top_candidate.get("faiss_score", 0.0)),
                chunk_text=top_candidate["chunk_text"],
            )

