import numpy as np
from typing import Dict, List, Optional, Any
import threading
import logging
from similarity import SimilarityCalculatorFactory, BaseSimilarityCalculator

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    A custom vector similarity search service that implements efficient
    similarity search algorithms without relying on external libraries.

    Supports multiple distance metrics:
    - Cosine similarity (default)
    - Euclidean distance
    - Dot product

    Thread-safe implementation with optimized search algorithms.
    """

    # Distance metric types
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"

    def __init__(self, distance_metric: str = COSINE, **calculator_kwargs):
        """
        Initialize the similarity service with the specified distance metric.

        Args:
            distance_metric: The distance metric to use (cosine, euclidean, dot)
            **calculator_kwargs: Additional parameters to pass to the calculator constructor
        """
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.distance_metric = distance_metric

        # Create the appropriate similarity calculator using the factory
        self.calculator = SimilarityCalculatorFactory.create(
            calculator_type=distance_metric, 
            **calculator_kwargs
        )

        # Add thread safety with locks
        self._vector_lock = threading.RLock()  # Reentrant lock for vector operations

        logger.info(f"SimilarityService initialized with {distance_metric} distance metric")

    def upsert(self, id: str, vector: List[float], metadata: Optional[Dict] = None) -> Dict:
        """
        Insert or update a vector with its metadata.
        Thread-safe implementation.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector as a list of floats
            metadata: Optional metadata associated with the vector

        Returns:
            Dict with status and id
        """
        # Convert to numpy array for efficient operations
        vector_array = np.array(vector, dtype=np.float32)

        # Preprocess the vector using the calculator
        vector_array = self.calculator.preprocess_vector(vector_array)

        # Store with lock protection
        with self._vector_lock:
            self.vectors[id] = vector_array
            self.metadata[id] = metadata or {}

        return {"status": "success", "id": id}

    def search(self, query_vector: np.ndarray, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors.
        Thread-safe implementation with snapshot-based search.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of results with id, score, and metadata
        """
        # Convert query vector to numpy array if it's a list
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        # Create a snapshot of vectors and metadata to avoid holding the lock during computation
        with self._vector_lock:
            # Create copies to avoid race conditions
            vectors_snapshot = self.vectors.copy()
            metadata_snapshot = self.metadata.copy()

        # Filter by metadata if needed
        if filter_metadata:
            filtered_ids = []
            for id, meta in metadata_snapshot.items():
                if all(meta.get(k) == v for k, v in filter_metadata.items()):
                    filtered_ids.append(id)

            # Only keep vectors that match the filter
            vectors_snapshot = {id: vectors_snapshot[id] for id in filtered_ids if id in vectors_snapshot}

        # Calculate similarities using the calculator
        results = []
        for id, vector in vectors_snapshot.items():
            # Use the calculator to compute similarity
            similarity = self.calculator.calculate_similarity(query_vector, vector)

            results.append({
                "id": id,
                "score": float(similarity),  # Convert numpy types to native Python types
                "metadata": metadata_snapshot.get(id, {})
            })

        # Sort by score (descending) and take top_k
        # Higher scores are better for all metrics now
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_vectors(self) -> Dict[str, Any]:
        """
        Get all vectors and metadata for persistence.

        Returns:
            Dict containing vectors and metadata
        """
        with self._vector_lock:
            # Convert numpy arrays to lists for JSON serialization
            vectors_data = {}
            for id, vector in self.vectors.items():
                vectors_data[id] = vector.tolist()

            # Get calculator configuration for persistence
            calculator_kwargs = {}
            if hasattr(self.calculator, "normalize_output"):
                calculator_kwargs["normalize_output"] = self.calculator.normalize_output

            return {
                "vectors": vectors_data,
                "metadata": self.metadata,
                "distance_metric": self.distance_metric,
                "calculator_kwargs": calculator_kwargs
            }

    def load_vectors(self, vectors_data: Dict[str, Any]) -> None:
        """
        Load vectors and metadata from persistence.

        Args:
            vectors_data: Dict containing vectors and metadata
        """
        with self._vector_lock:
            # Convert lists back to numpy arrays
            self.vectors = {}
            for id, vector_list in vectors_data["vectors"].items():
                self.vectors[id] = np.array(vector_list, dtype=np.float32)

            self.metadata = vectors_data["metadata"]
            self.distance_metric = vectors_data["distance_metric"]
            
            # Recreate the calculator with the loaded distance metric
            calculator_kwargs = vectors_data.get("calculator_kwargs", {})
            self.calculator = SimilarityCalculatorFactory.create(
                calculator_type=self.distance_metric,
                **calculator_kwargs
            )

            logger.info(f"Loaded {len(self.vectors)} vectors with {self.distance_metric} distance metric")

    def delete(self, ids: List[str]) -> Dict:
        """
        Delete vectors by their ids.
        Thread-safe implementation.

        Args:
            ids: List of vector ids to delete

        Returns:
            Dict with status and deleted count
        """
        deleted_count = 0

        with self._vector_lock:
            for id in ids:
                if id in self.vectors:
                    del self.vectors[id]
                    del self.metadata[id]
                    deleted_count += 1

        return {"status": "success", "deleted": deleted_count}

    def _get_calculator_info(self) -> Dict[str, Any]:
        """
        Get information about the current calculator.

        Returns:
            Dictionary with calculator information
        """
        return {
            "type": self.distance_metric,
            "config": self.calculator.get_config()
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vectors stored in the service.

        Returns:
            Dict with statistics
        """
        with self._vector_lock:
            vector_count = len(self.vectors)

            # Calculate dimension if vectors exist
            dimension = 0
            if vector_count > 0:
                # Get the first vector's dimension
                first_id = next(iter(self.vectors))
                dimension = len(self.vectors[first_id])

        stats = {
            "vector_count": vector_count,
            "dimension": dimension,
            "distance_metric": self.distance_metric
        }
        
        # Add calculator information
        stats["calculator"] = self._get_calculator_info()
        
        return stats
