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
    
    def search(self, query_vector: List[float], top_k: int = 10, 
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for the most similar vectors to the query vector.
        Thread-safe implementation.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter criteria
            
        Returns:
            List of dicts with id, score, and metadata
        """
        # Convert to numpy array
        query_array = np.array(query_vector, dtype=np.float32)
        
        # Normalize the query vector if using cosine similarity
        if self.distance_metric == self.COSINE:
            query_array = self._normalize_vector(query_array)
        
        # Create a snapshot of vectors and metadata with lock protection
        with self._vector_lock:
            # Make a shallow copy of the dictionaries
            vector_ids = list(self.vectors.keys())
            vectors_snapshot = {id: self.vectors[id] for id in vector_ids}
            metadata_snapshot = {id: self.metadata[id] for id in vector_ids}
        
        # Calculate similarities
        similarities = []
        
        for id, vector in vectors_snapshot.items():
            # Apply metadata filter if provided
            if filter_dict and not self._matches_filter(metadata_snapshot[id], filter_dict):
                continue
                
            # Calculate similarity score based on the distance metric
            score = self._calculate_similarity(query_array, vector)
            similarities.append((id, score, metadata_snapshot[id]))
        
        # Sort by similarity score (higher is better)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = [
            {"id": id, "score": float(score), "metadata": metadata}
            for id, score, metadata in similarities[:top_k]
        ]
        
        return results
    
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
