import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import logging
from services.vector_index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class Vector:
    id: str
    embedding: np.ndarray
    metadata: Dict


class VectorStore:
    def __init__(self, dimension: int = 1536, use_index: bool = True, n_trees: int = 10, max_top_k: int = 100):
        """Initialize the vector store."""
        self.dimension = dimension
        self.vectors: Dict[str, Vector] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Indexing configuration
        self.use_index = use_index
        self.n_trees = n_trees
        self.max_top_k = max_top_k
        self.index = VectorIndex(dimension=dimension, n_trees=n_trees) if use_index else None
        self._index_lock = RLock()  # Separate lock for index operations

    async def _validate_vector(self, vector: np.ndarray) -> None:
        """Validate vector dimensions."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector must be {self.dimension} dimensions")

    async def upsert(self, id: str, vector: List[float], metadata: Optional[Dict] = None) -> Dict:
        """Upsert a single vector."""
        try:
            np_vector = np.array(vector)
            await self._validate_vector(np_vector)

            with self._lock:
                self.vectors[id] = Vector(
                    id=id,
                    embedding=np_vector,
                    metadata=metadata or {}
                )

            # Update the index if enabled
            if self.use_index and self.index is not None:
                with self._index_lock:
                    self.index.add_item(id, np_vector, metadata)

            return {"status": "success", "id": id}
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise

    async def bulk_upsert(self, vectors: List[Dict]) -> Dict:
        """Bulk upsert multiple vectors."""
        try:
            # Validate all vectors first
            for item in vectors:
                vector = np.array(item['values'])
                await self._validate_vector(vector)

            # Update the vector store
            with self._lock:
                for item in vectors:
                    vector = np.array(item['values'])
                    self.vectors[item['id']] = Vector(
                        id=item['id'],
                        embedding=vector,
                        metadata=item.get('metadata', {})
                    )

            # Update the index if enabled
            if self.use_index and self.index is not None:
                with self._index_lock:
                    for item in vectors:
                        vector = np.array(item['values'])
                        self.index.add_item(item['id'], vector, item.get('metadata', {}))

            return {"status": "success", "count": len(vectors)}
        except Exception as e:
            logger.error(f"Bulk upsert failed: {str(e)}")
            raise

    async def delete(self, ids: List[str]) -> Dict:
        """Delete vectors by their IDs."""
        try:
            deleted_count = 0

            # Remove from vector store
            with self._lock:
                for id in ids:
                    if id in self.vectors:
                        del self.vectors[id]
                        deleted_count += 1

            # Remove from index if enabled
            if self.use_index and self.index is not None:
                with self._index_lock:
                    for id in ids:
                        self.index.remove_item(id)

            return {"status": "success", "count": deleted_count}
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise

    async def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None, use_index: Optional[bool] = None) -> Dict:
        """
        Search for the nearest vectors to the query vector.
        Thread-safe implementation.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter: Filter by metadata
            use_index: Whether to use the index. If None, use the default setting.

        Returns:
            Dictionary with search results
        """
        try:
            # Validate input
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector)

            if len(query_vector) != self.dimension:
                raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}")

            if top_k > self.max_top_k:
                raise ValueError(f"top_k exceeds maximum allowed value of {self.max_top_k}")

            # Normalize the query vector
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm

            # Check vector count to determine if index would be beneficial
            with self._lock:
                vector_count = len(self.vectors)

            # Determine whether to use index
            if vector_count <= 1000:
                # For small datasets, linear search is faster
                use_index = False
            elif vector_count <= 10000 and top_k < 50:
                # For medium datasets with small result sets, linear search is faster
                use_index = False
            elif use_index is None:
                # Use default setting if not explicitly specified
                use_index = self.use_index and self.index is not None

            # Try using index if conditions are met
            if use_index and self.index is not None and not filter and vector_count > 1000:
                with self._index_lock:
                    # Build the index if needed
                    if not self.index.is_built and vector_count > top_k:
                        self.index.build()

                    # Search using the index if it's built
                    if self.index.is_built:
                        results = self.index.search(query_vector, k=top_k)
                        return {
                            "matches": results,
                            "total": vector_count,
                            "indexed": True
                        }

            # Fall back to linear search
            with self._lock:
                # Create a snapshot of the vectors to avoid race conditions
                vectors_snapshot = dict(self.vectors)

            # Apply metadata filtering if needed
            if filter:
                filtered_vectors = {}
                for vector_id, vector_obj in vectors_snapshot.items():
                    if all(vector_obj.metadata.get(key) == value
                           for key, value in filter.items()):
                        filtered_vectors[vector_id] = vector_obj
                vectors_snapshot = filtered_vectors

            # Calculate similarities
            results = []
            for vector_id, vector_obj in vectors_snapshot.items():
                # Get the embedding from the Vector object
                vector = vector_obj.embedding

                # Normalize vector
                vector_norm = np.linalg.norm(vector)
                if vector_norm > 0:
                    vector = vector / vector_norm

                # Calculate cosine similarity
                similarity = np.dot(query_vector, vector)

                results.append({
                    "id": vector_id,
                    "score": float(similarity),
                    "metadata": vector_obj.metadata
                })

            # Sort by similarity (higher is better)
            results.sort(key=lambda x: x["score"], reverse=True)

            return {
                "matches": results[:top_k],
                "total": len(vectors_snapshot),
                "indexed": False
            }

        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            raise

    async def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        with self._lock:
            vector_count = len(self.vectors)
            memory_usage = sum(v.embedding.nbytes for v in self.vectors.values())

        stats = {
            "total_vectors": vector_count,
            "dimension": self.dimension,
            "memory_usage": memory_usage,
            "use_index": self.use_index
        }

        # Add index stats if enabled
        if self.use_index and self.index is not None:
            with self._index_lock:
                index_stats = self.index.get_stats()
                stats["index_stats"] = index_stats

        return stats

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
