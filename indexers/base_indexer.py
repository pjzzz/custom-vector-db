from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from models import Chunk


class BaseIndexer(ABC):
    """
    Abstract base class for all indexers.
    
    This class defines the interface that all indexers must implement.
    It ensures that all indexers have the same basic functionality
    regardless of their specific implementation details.
    """
    
    @abstractmethod
    def add_chunk(self, chunk: Chunk) -> None:
        """
        Add a chunk to the index.
        
        Args:
            chunk: The chunk to add to the index
        """
        pass
    
    @abstractmethod
    def search(self, query: str) -> List[Tuple[str, str, int]]:
        """
        Search for chunks matching the query.
        
        Args:
            query: The search query
            
        Returns:
            List of tuples containing (document_id, chunk_id, position)
        """
        pass
    
    @abstractmethod
    def remove_chunk(self, chunk_id: str) -> None:
        """
        Remove a chunk from the index.
        
        Args:
            chunk_id: ID of the chunk to remove
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary containing index statistics
        """
        # Default implementation that can be overridden
        return {
            "type": self.__class__.__name__,
            "description": "Base indexer implementation"
        }
        
    @abstractmethod
    def get_serializable_data(self) -> Dict[str, Any]:
        """
        Get serializable data for persistence.

        Returns:
            Dict containing serializable data
        """
        pass
        
    @abstractmethod
    def load_serializable_data(self, data: Dict[str, Any]) -> None:
        """
        Load serializable data from persistence.

        Args:
            data: Dict containing serializable data
        """
        pass
