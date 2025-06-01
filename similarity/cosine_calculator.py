import numpy as np
from typing import Dict, Any
from .base_calculator import BaseSimilarityCalculator


class CosineSimilarityCalculator(BaseSimilarityCalculator):
    """
    Calculates cosine similarity between vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    providing a similarity score between -1 and 1 (or 0 and 1 for non-negative vectors).
    """
    
    def calculate_similarity(self, query_vector: np.ndarray, vector: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            query_vector: The query vector
            vector: The vector to compare against
            
        Returns:
            Cosine similarity score (higher means more similar)
        """
        # Both vectors should already be normalized by preprocess_vector
        # For normalized vectors, cosine similarity is the same as dot product
        return float(np.dot(query_vector, vector))
    
    def preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize the vector to unit length for cosine similarity.
        
        Args:
            vector: The vector to normalize
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for the calculator.
        
        Returns:
            Dictionary of configuration parameters
        """
        config = super().get_config()
        config.update({
            "description": "Cosine similarity (normalized dot product)",
            "range": "0 to 1 (higher is more similar)"
        })
        return config
