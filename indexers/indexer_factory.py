from typing import Dict, Type, Optional
from .inverted_index import InvertedIndex
from .trie_index import TrieIndex
from .suffix_array_index import SuffixArrayIndex
import logging

logger = logging.getLogger(__name__)


class IndexerFactory:
    """
    Factory class for creating indexers based on the specified type.
    
    This class implements the Factory pattern to create the appropriate indexer
    instance based on the configuration. It provides a centralized way to manage
    indexer creation and makes it easy to add new indexer types in the future.
    """
    
    # Registry of available indexer types
    _indexers: Dict[str, Type] = {
        'inverted': InvertedIndex,
        'trie': TrieIndex,
        'suffix': SuffixArrayIndex
    }
    
    # Descriptions of each indexer type for documentation
    _descriptions: Dict[str, str] = {
        'inverted': """
        Inverted Index:
        - Space: O(T) where T is total number of terms
        - Build: O(T)
        - Search: O(Q + k) where Q is query size and k is number of results
        - Best for: Exact word matching, boolean queries
        - Strengths: Simple, efficient for exact matches
        - Weaknesses: No prefix or substring matching
        """,
        'trie': """
        Trie Index:
        - Space: O(T) where T is total number of characters
        - Build: O(T)
        - Search: O(P + k) where P is prefix length and k is number of results
        - Best for: Prefix matching, autocomplete
        - Strengths: Efficient prefix matching, space-efficient for prefixes
        - Weaknesses: No suffix matching
        """,
        'suffix': """
        Suffix Array Index:
        - Space: O(T) where T is total length of all text
        - Build: O(T log T)
        - Search: O(P log T + k) where P is pattern length and k is number of results
        - Best for: Substring matching, fuzzy matching
        - Strengths: Powerful substring matching, supports fuzzy queries
        - Weaknesses: Higher build time complexity
        """
    }
    
    @classmethod
    def create(cls, indexer_type: str = 'inverted'):
        """
        Create an indexer instance based on the specified type.
        
        Args:
            indexer_type: Type of indexer to create ('inverted', 'trie', or 'suffix')
                          Defaults to 'inverted' if not specified
        
        Returns:
            An instance of the specified indexer type
            
        Raises:
            ValueError: If the specified indexer type is not supported
        """
        if indexer_type not in cls._indexers:
            supported_types = list(cls._indexers.keys())
            raise ValueError(f"Unsupported indexer type: {indexer_type}. "
                             f"Supported types are: {supported_types}")
        
        indexer_class = cls._indexers[indexer_type]
        logger.info(f"Creating indexer of type: {indexer_type}")
        return indexer_class()
    
    @classmethod
    def register(cls, name: str, indexer_class: Type, description: Optional[str] = None):
        """
        Register a new indexer type.
        
        Args:
            name: Name of the indexer type
            indexer_class: Class of the indexer
            description: Optional description of the indexer
        """
        cls._indexers[name] = indexer_class
        if description:
            cls._descriptions[name] = description
        logger.info(f"Registered new indexer type: {name}")
    
    @classmethod
    def get_available_types(cls):
        """
        Get a list of available indexer types.
        
        Returns:
            List of available indexer type names
        """
        return list(cls._indexers.keys())
    
    @classmethod
    def get_description(cls, indexer_type: str):
        """
        Get the description of an indexer type.
        
        Args:
            indexer_type: Type of indexer
            
        Returns:
            Description of the indexer type, or None if not found
        """
        return cls._descriptions.get(indexer_type)
