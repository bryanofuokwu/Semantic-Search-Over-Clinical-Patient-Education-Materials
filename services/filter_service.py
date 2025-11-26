"""Service for filtering search results."""
from typing import List, Optional

import numpy as np
import pandas as pd

from services.index_service import IndexService


class SearchFilters:
    """Filter criteria for search results."""
    
    def __init__(
        self,
        condition: Optional[str] = None,
        category: Optional[str] = None,
        min_score: Optional[float] = None
    ):
        self.condition = condition
        self.category = category
        self.min_score = min_score


class FilterService:
    """Service for applying filters to search results."""
    
    def __init__(self, index_service: IndexService):
        self.index_service = index_service
    
    def apply_filters(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        filters: Optional[SearchFilters],
    ) -> List[int]:
        """
        Filter search results based on criteria.
        
        Args:
            indices: Array of FAISS indices
            scores: Array of similarity scores
            filters: Optional filter criteria
            
        Returns:
            List of positions that pass the filters
        """
        if filters is None:
            return list(range(len(indices)))
        
        if not self.index_service.is_initialized():
            raise RuntimeError("Index service is not initialized.")
        
        metadata = self.index_service.metadata
        keep_indices: List[int] = []
        
        for i, (idx, score) in enumerate(zip(indices, scores)):
            # Validate index bounds
            if idx < 0 or idx >= len(metadata):
                print(f"[WARNING] Invalid index {idx} in filter, skipping")
                continue
            
            row = metadata.iloc[idx]
            
            # Filter by condition
            if filters.condition is not None:
                if str(row["condition"]).lower() != filters.condition.lower():
                    continue
            
            # Filter by category
            if filters.category is not None:
                cat = row.get("category", None)
                if cat is None or str(cat).lower() != filters.category.lower():
                    continue
            
            # Filter by minimum score
            if filters.min_score is not None:
                if float(score) < float(filters.min_score):
                    continue
            
            keep_indices.append(i)
        
        return keep_indices

