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
