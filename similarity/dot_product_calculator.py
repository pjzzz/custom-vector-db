import numpy as np
from typing import Dict, Any
from .base_calculator import BaseSimilarityCalculator


class DotProductSimilarityCalculator(BaseSimilarityCalculator):
    """
    Calculates similarity using dot product.
    
    Dot product measures the product of the magnitudes and the cosine of the angle,
    providing a similarity score that depends on both angle and magnitude.
    """
    
    def __init__(self, normalize_output: bool = True):
        """
        Initialize the dot product calculator.
        
        Args:
            normalize_output: Whether to normalize the output to a 0-1 range
        """
        self.normalize_output = normalize_output
    
    def calculate_similarity(self, query_vector: np.ndarray, vector: np.ndarray) -> float:
        """
        Calculate dot product similarity between two vectors.
        
        Args:
            query_vector: The query vector
            vector: The vector to compare against
            
        Returns:
            Dot product similarity score (higher means more similar)
        """
        dot_product = np.dot(query_vector, vector)
        
        if self.normalize_output:
            # Ensure the result is non-negative for consistency with other metrics
            return float(max(0, dot_product))
        
        return float(dot_product)
    
    def preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Preprocess vector for dot product calculation.
        
        For dot product, no preprocessing is typically needed,
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
            "description": "Dot product similarity",
            "normalize_output": self.normalize_output,
            "range": "0 to inf (higher is more similar)" if not self.normalize_output else "0 to 1 (higher is more similar)"
        })
        return config
