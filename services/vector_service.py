from typing import List, Dict, Optional
from config import settings
import logging
from .vector_store import VectorStore
import numpy as np

logger = logging.getLogger(__name__)

class VectorService:
    """Service class for vector operations using custom vector store."""

    def __init__(self) -> None:
        """Initialize custom vector store."""
        self.store = VectorStore(dimension=settings.EMBEDDING_DIMENSION)

    async def validate_vector(self, vector: List[float]) -> bool:
        """Validate vector dimensions."""
        if len(vector) != settings.EMBEDDING_DIMENSION:
            raise ValueError(f"Vector must be {settings.EMBEDDING_DIMENSION} dimensions")
        return True

    async def upsert(self, id: str, vector: List[float], metadata: Optional[Dict] = None) -> Dict:
        """Upsert a single vector."""
        try:
            return await self.store.upsert(id, vector, metadata)
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise

    async def bulk_upsert(self, vectors: List[Dict]) -> Dict:
        """Bulk upsert multiple vectors."""
        try:
            for vector in vectors:
                await self.validate_vector(vector['values'])
            return await self.store.bulk_upsert(vectors)
        except Exception as e:
            logger.error(f"Bulk upsert failed: {str(e)}")
            raise

    async def delete(self, ids: List[str]) -> Dict:
        """Delete vectors by their IDs."""
        try:
            return await self.store.delete(ids)
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise

    async def search(self, query_vector: List[float], top_k: int = settings.DEFAULT_TOP_K, 
              filter: Optional[Dict] = None) -> Dict:
        """Search for similar vectors using cosine similarity."""
        try:
            if top_k > settings.MAX_TOP_K:
                raise ValueError(f"top_k must be less than or equal to {settings.MAX_TOP_K}")
            return await self.store.search(np.array(query_vector), top_k, filter)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    async def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        try:
            return await self.store.get_stats()
        except Exception as e:
            logger.error(f"Get stats failed: {str(e)}")
            raise
