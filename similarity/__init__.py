from .base_calculator import BaseSimilarityCalculator
from .cosine_calculator import CosineSimilarityCalculator
from .euclidean_calculator import EuclideanSimilarityCalculator
from .dot_product_calculator import DotProductSimilarityCalculator
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
