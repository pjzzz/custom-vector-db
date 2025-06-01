import numpy as np
from typing import Dict, Any
from .base_calculator import BaseSimilarityCalculator


class EuclideanSimilarityCalculator(BaseSimilarityCalculator):
    """
    Calculates similarity based on Euclidean distance.
    
    Converts Euclidean distance to a similarity score where higher values
    indicate more similarity (closer vectors).
    """
    
    def calculate_similarity(self, query_vector: np.ndarray, vector: np.ndarray) -> float:
        """
        Calculate similarity based on Euclidean distance.
        
        Args:
            query_vector: The query vector
            vector: The vector to compare against
            
        Returns:
            Similarity score based on Euclidean distance (higher means more similar)
        """
        # Convert distance to similarity (1 / (1 + distance))
        # This maps distance [0, inf) to similarity (0, 1]
        distance = np.linalg.norm(query_vector - vector)
        return float(1.0 / (1.0 + distance))
    
    def preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Preprocess vector for Euclidean distance calculation.
        
        For Euclidean distance, no preprocessing is typically needed,
        but this method is required by the interface.
        
        Args:
            vector: The vector to preprocess
            
        Returns:
            The same vector (no preprocessing needed)
        """
        return vector
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for the calculator.
        
        Returns:
            Dictionary of configuration parameters
        """
        config = super().get_config()
        config.update({
            "description": "Euclidean distance converted to similarity",
            "range": "0 to 1 (higher is more similar)"
        })
        return config
