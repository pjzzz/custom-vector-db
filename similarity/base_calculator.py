from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class BaseSimilarityCalculator(ABC):
    """
    Abstract base class for similarity calculators.
    
    This defines the common interface that all similarity calculators must implement.
    """
    
    @abstractmethod
    def calculate_similarity(self, query_vector: np.ndarray, vector: np.ndarray) -> float:
        """
        Calculate similarity between two vectors.
        
        Args:
            query_vector: The query vector
            vector: The vector to compare against
            
        Returns:
            Similarity score (higher means more similar)
        """
        pass
    
    @abstractmethod
    def preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Preprocess a vector before storage or comparison.
        
        Args:
            vector: The vector to preprocess
            
        Returns:
            Preprocessed vector
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for the calculator.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {"type": self.__class__.__name__}
