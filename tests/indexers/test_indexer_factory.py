import unittest
import sys
import os
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the modules using the full path
from indexers.indexer_factory import IndexerFactory
from indexers.base_indexer import BaseIndexer
from indexers.inverted_index import InvertedIndex
from indexers.trie_index import TrieIndex
from indexers.suffix_array_index import SuffixArrayIndex
from models import Chunk


class TestIndexerFactory(unittest.TestCase):
    """Test cases for the IndexerFactory and BaseIndexer implementation."""

    def setUp(self):
        """Set up test data."""
        self.test_chunks = [
            Chunk(
                id="chunk_1",
                document_id="doc_1",
                text="Python is a high-level programming language with easy syntax.",
                position=0,
                metadata={"source": "documentation"}
            ),
            Chunk(
                id="chunk_2",
                document_id="doc_1",
                text="Java is another popular programming language used for enterprise applications.",
                position=1,
                metadata={"source": "documentation"}
            ),
            Chunk(
                id="chunk_3",
                document_id="doc_2",
                text="Python has a large standard library and active community support.",
                position=0,
                metadata={"source": "documentation"}
            )
        ]

    def test_available_indexer_types(self):
        """Test that the factory provides the expected indexer types."""
        available_types = IndexerFactory.get_available_types()
        self.assertEqual(set(available_types), {"inverted", "trie", "suffix"})

    def test_indexer_descriptions(self):
        """Test that the factory provides descriptions for all indexer types."""
        for indexer_type in IndexerFactory.get_available_types():
            description = IndexerFactory.get_description(indexer_type)
            self.assertIsNotNone(description)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)

    def test_create_inverted_index(self):
        """Test creating an InvertedIndex through the factory."""
        indexer = IndexerFactory.create("inverted")
        self.assertIsInstance(indexer, InvertedIndex)
        self.assertIsInstance(indexer, BaseIndexer)

    def test_create_trie_index(self):
        """Test creating a TrieIndex through the factory."""
        indexer = IndexerFactory.create("trie")
        self.assertIsInstance(indexer, TrieIndex)
        self.assertIsInstance(indexer, BaseIndexer)

    def test_create_suffix_array_index(self):
        """Test creating a SuffixArrayIndex through the factory."""
        indexer = IndexerFactory.create("suffix")
        self.assertIsInstance(indexer, SuffixArrayIndex)
        self.assertIsInstance(indexer, BaseIndexer)

    def test_default_indexer(self):
        """Test that the default indexer is InvertedIndex."""
        indexer = IndexerFactory.create()
        self.assertIsInstance(indexer, InvertedIndex)

    def test_invalid_indexer_type(self):
        """Test that an invalid indexer type raises a ValueError."""
        with self.assertRaises(ValueError):
            IndexerFactory.create("nonexistent_type")

    def test_register_custom_indexer(self):
        """Test registering and creating a custom indexer."""
        # Define a simple custom indexer
        class SimpleIndexer(BaseIndexer):
            def __init__(self):
                self.index = {}
                
            def add_chunk(self, chunk: Chunk) -> None:
                self.index[chunk.id] = chunk
                
            def search(self, query: str) -> List[tuple[str, str, int]]:
                results = []
                for chunk_id, chunk in self.index.items():
                    if query.lower() in chunk.text.lower():
                        results.append((chunk.document_id, chunk_id, chunk.position))
                return results
                
            def remove_chunk(self, chunk_id: str) -> None:
                if chunk_id in self.index:
                    del self.index[chunk_id]
                    
            def get_serializable_data(self) -> Dict[str, Any]:
                chunk_dict = {}
                for chunk_id, chunk in self.index.items():
                    chunk_dict[chunk_id] = chunk.model_dump()
                return {"chunks": chunk_dict}
                
            def load_serializable_data(self, data: Dict[str, Any]) -> None:
                self.index = {}
                for chunk_id, chunk_data in data.get("chunks", {}).items():
                    self.index[chunk_id] = Chunk(**chunk_data)
        
        # Register the custom indexer
        IndexerFactory.register(
            name="simple",
            indexer_class=SimpleIndexer,
            description="A simple indexer for testing"
        )
        
        # Verify it was registered
        self.assertIn("simple", IndexerFactory.get_available_types())
        self.assertEqual(IndexerFactory.get_description("simple"), "A simple indexer for testing")
        
        # Create an instance
        indexer = IndexerFactory.create("simple")
        self.assertIsInstance(indexer, SimpleIndexer)
        self.assertIsInstance(indexer, BaseIndexer)

    def test_indexer_functionality(self):
        """Test that indexers created by the factory have the expected functionality."""
        for indexer_type in IndexerFactory.get_available_types():
            # Skip if we're testing a custom indexer from another test
            if indexer_type == "simple":
                continue
                
            # Create the indexer
            indexer = IndexerFactory.create(indexer_type)
            
            # Add test chunks
            for chunk in self.test_chunks:
                indexer.add_chunk(chunk)
            
            # Test search functionality
            python_results = indexer.search("Python")
            self.assertGreaterEqual(len(python_results), 1, 
                                   f"Expected at least 1 result for 'Python' with {indexer_type} indexer")
            
            # Test remove functionality
            indexer.remove_chunk("chunk_1")
            results_after_remove = indexer.search("syntax")
            self.assertEqual(len(results_after_remove), 0, 
                            f"Expected 0 results for 'syntax' after removal with {indexer_type} indexer")

    def test_persistence(self):
        """Test that indexers can be serialized and deserialized."""
        for indexer_type in IndexerFactory.get_available_types():
            # Skip if we're testing a custom indexer from another test
            if indexer_type == "simple":
                continue
                
            # Create and populate the indexer
            indexer = IndexerFactory.create(indexer_type)
            for chunk in self.test_chunks:
                indexer.add_chunk(chunk)
            
            # Serialize
            serialized_data = indexer.get_serializable_data()
            self.assertIsInstance(serialized_data, dict)
            
            # Create a new indexer and deserialize
            new_indexer = IndexerFactory.create(indexer_type)
            new_indexer.load_serializable_data(serialized_data)
            
            # Test that the new indexer works as expected
            python_results = new_indexer.search("Python")
            self.assertGreaterEqual(len(python_results), 1, 
                                   f"Expected at least 1 result for 'Python' with deserialized {indexer_type} indexer")

    def test_optimized_inverted_index_remove(self):
        """Test that the InvertedIndex's optimized remove_chunk method works correctly."""
        # Create an inverted index
        indexer = IndexerFactory.create("inverted")
        
        # Add test chunks
        for chunk in self.test_chunks:
            indexer.add_chunk(chunk)
        
        # Verify initial state
        self.assertGreaterEqual(len(indexer.search("Python")), 2)
        self.assertGreaterEqual(len(indexer.search("syntax")), 1)
        
        # Remove a chunk that contains "syntax"
        indexer.remove_chunk("chunk_1")
        
        # Verify that "syntax" is no longer found
        self.assertEqual(len(indexer.search("syntax")), 0)
        
        # But "Python" should still be found in chunk_3
        python_results = indexer.search("Python")
        self.assertGreaterEqual(len(python_results), 1)
        
        # The remaining result should be from chunk_3
        found_chunk_3 = False
        for doc_id, chunk_id, _ in python_results:
            if chunk_id == "chunk_3":
                found_chunk_3 = True
                break
        self.assertTrue(found_chunk_3, "chunk_3 should still be found after removing chunk_1")


if __name__ == "__main__":
    unittest.main()
