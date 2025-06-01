from .base_calculator import BaseSimilarityCalculator
from .calculators import (
    CosineSimilarityCalculator,
    EuclideanSimilarityCalculator,
    DotProductSimilarityCalculator
)
from .calculator_factory import SimilarityCalculatorFactory

# For backward compatibility and convenience
CALCULATORS = {
    'cosine': CosineSimilarityCalculator,
    'euclidean': EuclideanSimilarityCalculator,
    'dot': DotProductSimilarityCalculator
}

__all__ = [
    'BaseSimilarityCalculator',
    'CosineSimilarityCalculator',
    'EuclideanSimilarityCalculator',
    'DotProductSimilarityCalculator',
    'SimilarityCalculatorFactory',
    'CALCULATORS'
]
