import numpy as np
from typing import Dict, List, Tuple, Optional
import threading
import logging
from models.chunk import Chunk

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

    def __init__(self, distance_metric: str = COSINE):
        """
        Initialize the similarity service with the specified distance metric.

        Args:
            distance_metric: The distance metric to use (cosine, euclidean, dot)
        """
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.distance_metric = distance_metric

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

        # Normalize the vector if using cosine similarity
        if self.distance_metric == self.COSINE:
            vector_array = self._normalize_vector(vector_array)

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

        # Calculate distances
        results = []
        for id, vector in vectors_snapshot.items():
            # Calculate distance based on the selected metric
            if self.distance_metric == self.COSINE:
                # Cosine similarity (1 - cosine distance)
                similarity = 1 - np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            elif self.distance_metric == self.EUCLIDEAN:
                # Convert Euclidean distance to similarity score (1 / (1 + distance))
                similarity = 1 / (1 + np.linalg.norm(query_vector - vector))
            elif self.distance_metric == self.DOT:
                # Dot product
                similarity = np.dot(query_vector, vector)
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")

            results.append({
                "id": id,
                "score": float(similarity),  # Convert numpy types to native Python types
                "metadata": metadata_snapshot.get(id, {})
            })

        # Sort by score (descending) and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_vectors(self):
        """
        Get all vectors and metadata for persistence.
        Thread-safe implementation.

        Returns:
            Dict containing vectors and metadata
        """
        with self._vector_lock:
            # Convert numpy arrays to lists for JSON serialization
            vectors_data = {}
            for id, vector in self.vectors.items():
                vectors_data[id] = vector.tolist()

            return {
                "vectors": vectors_data,
                "metadata": self.metadata,
                "distance_metric": self.distance_metric
            }

    def load_vectors(self, vectors_data):
        """
        Load vectors and metadata from persistence.
        Thread-safe implementation.

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

    def _calculate_similarity(self, query_vector: np.ndarray,
                             vector: np.ndarray) -> float:
        """
        Calculate similarity between two vectors based on the distance metric.

        Args:
            query_vector: The query vector
            vector: The vector to compare with

        Returns:
            Similarity score (higher is better)
        """
        if self.distance_metric == self.COSINE:
            # For normalized vectors, dot product is equivalent to cosine similarity
            # Convert numpy.float64 to Python float for type compatibility
            return float(np.dot(query_vector, vector))

        elif self.distance_metric == self.EUCLIDEAN:
            # Convert Euclidean distance to similarity score (higher is better)
            distance = np.linalg.norm(query_vector - vector)
            return float(1.0 / (1.0 + distance))  # Convert to similarity score

        elif self.distance_metric == self.DOT:
            return float(np.dot(query_vector, vector))

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            vector: The vector to normalize

        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """
        Check if metadata matches the filter criteria.

        Args:
            metadata: The metadata to check
            filter_dict: The filter criteria

        Returns:
            True if metadata matches the filter, False otherwise
        """
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def get_stats(self) -> Dict:
        """
        Get statistics about the vectors stored in the service.
        Thread-safe implementation.

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

        return {
            "vector_count": vector_count,
            "dimension": dimension,
            "distance_metric": self.distance_metric
        }
