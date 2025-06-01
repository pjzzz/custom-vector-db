import unittest
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from similarity import (
    SimilarityCalculatorFactory,
    BaseSimilarityCalculator,
    CosineSimilarityCalculator,
    EuclideanSimilarityCalculator,
    DotProductSimilarityCalculator
)


class TestSimilarityCalculatorFactory(unittest.TestCase):
    """Test cases for the SimilarityCalculatorFactory."""

    def test_available_calculator_types(self):
        """Test that the factory provides the expected calculator types."""
        available_types = SimilarityCalculatorFactory.get_available_types()
        self.assertEqual(set(available_types), {"cosine", "euclidean", "dot"})

    def test_calculator_descriptions(self):
        """Test that the factory provides descriptions for all calculator types."""
        for calculator_type in SimilarityCalculatorFactory.get_available_types():
            description = SimilarityCalculatorFactory.get_description(calculator_type)
            self.assertIsNotNone(description)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)

    def test_create_cosine_calculator(self):
        """Test creating a CosineSimilarityCalculator through the factory."""
        calculator = SimilarityCalculatorFactory.create("cosine")
        self.assertIsInstance(calculator, CosineSimilarityCalculator)
        self.assertIsInstance(calculator, BaseSimilarityCalculator)

    def test_create_euclidean_calculator(self):
        """Test creating an EuclideanSimilarityCalculator through the factory."""
        calculator = SimilarityCalculatorFactory.create("euclidean")
        self.assertIsInstance(calculator, EuclideanSimilarityCalculator)
        self.assertIsInstance(calculator, BaseSimilarityCalculator)

    def test_create_dot_product_calculator(self):
        """Test creating a DotProductSimilarityCalculator through the factory."""
        calculator = SimilarityCalculatorFactory.create("dot")
        self.assertIsInstance(calculator, DotProductSimilarityCalculator)
        self.assertIsInstance(calculator, BaseSimilarityCalculator)

    def test_default_calculator(self):
        """Test that the default calculator is CosineSimilarityCalculator."""
        calculator = SimilarityCalculatorFactory.create()
        self.assertIsInstance(calculator, CosineSimilarityCalculator)

    def test_invalid_calculator_type(self):
        """Test that an invalid calculator type raises a ValueError."""
        with self.assertRaises(ValueError):
            SimilarityCalculatorFactory.create("nonexistent_type")

    def test_register_custom_calculator(self):
        """Test registering and creating a custom calculator."""
        # Define a simple custom calculator
        class CustomSimilarityCalculator(BaseSimilarityCalculator):
            def calculate_similarity(self, query_vector: np.ndarray, vector: np.ndarray) -> float:
                # Simple implementation: average of cosine and euclidean
                cosine = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                euclidean = 1.0 / (1.0 + np.linalg.norm(query_vector - vector))
                return float((cosine + euclidean) / 2.0)
            
            def preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
                # Normalize the vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    return vector / norm
                return vector
        
        # Register the custom calculator
        SimilarityCalculatorFactory.register(
            name="custom",
            calculator_class=CustomSimilarityCalculator,
            description="A custom similarity calculator for testing"
        )
        
        # Verify it was registered
        self.assertIn("custom", SimilarityCalculatorFactory.get_available_types())
        self.assertEqual(
            SimilarityCalculatorFactory.get_description("custom"), 
            "A custom similarity calculator for testing"
        )
        
        # Create an instance
        calculator = SimilarityCalculatorFactory.create("custom")
        self.assertIsInstance(calculator, CustomSimilarityCalculator)
        self.assertIsInstance(calculator, BaseSimilarityCalculator)

    def test_calculator_functionality(self):
        """Test that calculators created by the factory have the expected functionality."""
        # Test vectors
        query_vector = np.array([1.0, 0.0, 0.0])
        similar_vector = np.array([0.9, 0.1, 0.0])
        dissimilar_vector = np.array([0.0, 0.0, 1.0])
        
        # Test each calculator type
        for calculator_type in ["cosine", "euclidean", "dot"]:
            calculator = SimilarityCalculatorFactory.create(calculator_type)
            
            # Preprocess vectors
            query_vector_processed = calculator.preprocess_vector(query_vector)
            similar_vector_processed = calculator.preprocess_vector(similar_vector)
            dissimilar_vector_processed = calculator.preprocess_vector(dissimilar_vector)
            
            # Calculate similarities
            similar_score = calculator.calculate_similarity(
                query_vector_processed, similar_vector_processed
            )
            dissimilar_score = calculator.calculate_similarity(
                query_vector_processed, dissimilar_vector_processed
            )
            
            # Similar vector should have higher score than dissimilar vector
            self.assertGreater(
                similar_score, 
                dissimilar_score,
                f"Expected similar vector to have higher score with {calculator_type} calculator"
            )
            
            # Scores should be in the expected range (0 to 1 for normalized metrics)
            self.assertGreaterEqual(similar_score, 0.0)
            self.assertGreaterEqual(dissimilar_score, 0.0)
            
            if calculator_type in ["cosine", "euclidean"]:
                self.assertLessEqual(similar_score, 1.0)
                self.assertLessEqual(dissimilar_score, 1.0)

    def test_dot_product_calculator_config(self):
        """Test that the dot product calculator can be configured."""
        # Default configuration (normalize_output=True)
        calculator = SimilarityCalculatorFactory.create("dot")
        self.assertTrue(calculator.normalize_output)
        
        # Custom configuration (normalize_output=False)
        calculator = SimilarityCalculatorFactory.create("dot", normalize_output=False)
        self.assertFalse(calculator.normalize_output)
        
        # Test the effect of normalize_output
        vector1 = np.array([2.0, 0.0, 0.0])
        vector2 = np.array([2.0, 0.0, 0.0])
        
        # With normalize_output=False, the dot product can exceed 1.0
        calculator.normalize_output = False
        similarity = calculator.calculate_similarity(vector1, vector2)
        self.assertEqual(similarity, 4.0)  # dot product of [2,0,0] and [2,0,0] is 4
        
        # With normalize_output=True, the result should be capped at max(0, dot_product)
        calculator.normalize_output = True
        similarity = calculator.calculate_similarity(vector1, vector2)
        self.assertEqual(similarity, 4.0)  # Still 4.0 because it's positive
        
        # Test with negative dot product
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([-1.0, 0.0, 0.0])
        
        calculator.normalize_output = False
        similarity = calculator.calculate_similarity(vector1, vector2)
        self.assertEqual(similarity, -1.0)  # dot product of [1,0,0] and [-1,0,0] is -1
        
        calculator.normalize_output = True
        similarity = calculator.calculate_similarity(vector1, vector2)
        self.assertEqual(similarity, 0.0)  # Capped at 0 because dot product is negative


if __name__ == "__main__":
    unittest.main()
