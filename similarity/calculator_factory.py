from typing import Dict, Type, Optional, Any
from .base_calculator import BaseSimilarityCalculator
from .calculators import (
    CosineSimilarityCalculator,
    EuclideanSimilarityCalculator,
    DotProductSimilarityCalculator
)
import logging

logger = logging.getLogger(__name__)


class SimilarityCalculatorFactory:
    """
    Factory class for creating similarity calculators.
    
    This factory provides a centralized way to create and manage different
    similarity calculator implementations.
    """
    
    # Registry of available calculator types
    _calculators: Dict[str, Type[BaseSimilarityCalculator]] = {
        'cosine': CosineSimilarityCalculator,
        'euclidean': EuclideanSimilarityCalculator,
        'dot': DotProductSimilarityCalculator
    }
    
    # Descriptions for each calculator type
    _descriptions: Dict[str, str] = {
        'cosine': "Cosine similarity: Measures the cosine of the angle between vectors. "
                 "Range: 0 to 1, where 1 means identical direction.",
        'euclidean': "Euclidean similarity: Based on Euclidean distance, converted to similarity. "
                    "Range: 0 to 1, where 1 means identical vectors.",
        'dot': "Dot product similarity: Measures the product of magnitudes and cosine of angle. "
              "Sensitive to both direction and magnitude."
    }
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """
        Get a list of available similarity calculator types.
        
        Returns:
            List of calculator type names
        """
        return list(cls._calculators.keys())
    
    @classmethod
    def get_description(cls, calculator_type: str) -> str:
        """
        Get the description for a specific calculator type.
        
        Args:
            calculator_type: The type of calculator
            
        Returns:
            Description of the calculator
            
        Raises:
            ValueError: If calculator_type is not recognized
        """
        if calculator_type not in cls._descriptions:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
        
        return cls._descriptions[calculator_type]
    
    @classmethod
    def create(cls, calculator_type: str = 'cosine', **kwargs) -> BaseSimilarityCalculator:
        """
        Create a similarity calculator instance based on the specified type.
        
        Args:
            calculator_type: The type of calculator to create
            **kwargs: Additional parameters to pass to the calculator constructor
            
        Returns:
            Instance of the specified calculator
            
        Raises:
            ValueError: If calculator_type is not recognized
        """
        if calculator_type not in cls._calculators:
            raise ValueError(
                f"Unknown calculator type: {calculator_type}. "
                f"Available types: {', '.join(cls._calculators.keys())}"
            )
        
        calculator_class = cls._calculators[calculator_type]
        logger.info(f"Creating similarity calculator of type: {calculator_type}")
        return calculator_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, calculator_class: Type[BaseSimilarityCalculator], 
                description: Optional[str] = None):
        """
        Register a new similarity calculator type.
        
        Args:
            name: Name to register the calculator under
            calculator_class: The calculator class to register
            description: Optional description of the calculator
            
        Raises:
            ValueError: If the calculator doesn't implement BaseSimilarityCalculator
        """
        # Ensure the calculator implements the BaseSimilarityCalculator interface
        if not issubclass(calculator_class, BaseSimilarityCalculator):
            raise ValueError(
                f"Calculator class {calculator_class.__name__} must implement "
                "BaseSimilarityCalculator interface"
            )
        
        cls._calculators[name] = calculator_class
        
        if description:
            cls._descriptions[name] = description
        else:
            cls._descriptions[name] = f"{calculator_class.__name__}: Custom similarity calculator"
        
        logger.info(f"Registered new similarity calculator type: {name}")
    
    @classmethod
    def get_calculator_config(cls, calculator_type: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific calculator type.
        
        Args:
            calculator_type: The type of calculator
            
        Returns:
            Configuration dictionary for the calculator
            
        Raises:
            ValueError: If calculator_type is not recognized
        """
        if calculator_type not in cls._calculators:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
        
        # Create an instance to get its configuration
        calculator = cls.create(calculator_type)
        return calculator.get_config()
