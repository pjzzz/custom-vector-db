import unittest
import asyncio
from services.content_service import ContentService
from services.embedding_service import EmbeddingService
from services.similarity_service import SimilarityService
from models import Chunk, Document, Library
import random
import string
import threading
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVectorSearch(unittest.TestCase):
    """Test the custom vector search implementation with thread safety."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a content service with custom vector search
        self.content_service = ContentService(
            indexer_type='suffix',  # Use suffix array index for testing
            embedding_dimension=1536
        )
        
        # Fit the embedding model on sample texts
        sample_texts = [
            "Machine learning is a field of artificial intelligence",
            "Natural language processing focuses on text understanding",
            "Vector search enables semantic similarity queries",
            "Embeddings are numerical representations of text or images",
            "Thread safety ensures concurrent operations don't cause data corruption",
            "Locks prevent race conditions in multi-threaded environments",
            "Suffix arrays enable efficient substring searches",
            "Tries are tree data structures for prefix matching",
            "Inverted indices map terms to documents containing them",
            "Content management systems organize and store digital content"
        ]
        self.content_service.embedding_service.fit(sample_texts)
        
        # Create a library for testing
        self.library = Library(
            id="test-library",
            name="Test Library",
            description="Library for vector search testing",
            created_at=datetime.now()
        )
        
        # Create a document for testing
        self.document = Document(
            id="test-document",
            library_id="test-library",
            title="Test Document",
            content="This is a test document for vector search",
            created_at=datetime.now(),
            metadata={"type": "test"}
        )
    
    def generate_random_text(self, length=50):
        """Generate random text for testing."""
        words = [
            "vector", "search", "embedding", "similarity", "cosine",
            "euclidean", "distance", "index", "query", "result",
            "thread", "safe", "lock", "concurrent", "parallel",
            "suffix", "array", "trie", "inverted", "index",
            "machine", "learning", "artificial", "intelligence", "natural",
            "language", "processing", "semantic", "syntactic", "token"
        ]
        return " ".join(random.choices(words, k=length//5))
    
    def generate_random_id(self, prefix="chunk", length=8):
        """Generate a random ID for testing."""
        random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        return f"{prefix}-{random_part}"
    
    async def create_test_data(self):
        """Create test data for vector search."""
        # Create library
        await self.content_service.create_library(self.library)
        
        # Create document
        await self.content_service.create_document(self.document)
        
        # Create chunks
        chunks = []
        for i in range(20):
            chunk = Chunk(
                id=self.generate_random_id(),
                document_id="test-document",
                text=self.generate_random_text(),
                position=i,
                created_at=datetime.now(),
                metadata={"index": i}
            )
            await self.content_service.create_chunk(chunk)
            chunks.append(chunk)
        
        return chunks
    
    def test_vector_search_single_thread(self):
        """Test vector search in a single thread."""
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create test data
        chunks = loop.run_until_complete(self.create_test_data())
        
        # Perform vector search
        query = "vector similarity search with thread safety"
        results = loop.run_until_complete(
            self.content_service.vector_search(query, top_k=5)
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertLessEqual(len(results), 5)
        for result in results:
            self.assertIn("chunk", result)
            self.assertIn("score", result)
            self.assertGreaterEqual(result["score"], 0)
            self.assertLessEqual(result["score"], 1)
        
        # Clean up
        loop.close()
    
    def test_vector_search_multi_thread(self):
        """Test vector search with multiple concurrent threads."""
        # Create event loop for setup
        setup_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(setup_loop)
        
        # Create test data
        chunks = setup_loop.run_until_complete(self.create_test_data())
        
        # Define worker function for threads
        def worker(thread_id, results):
            """Worker function for concurrent vector search."""
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Generate a query
                query = f"thread {thread_id} searching for vector similarity"
                
                # Perform vector search
                thread_results = loop.run_until_complete(
                    self.content_service.vector_search(query, top_k=3)
                )
                
                # Store results
                results[thread_id] = thread_results
                
            except Exception as e:
                logger.error(f"Thread {thread_id} error: {str(e)}")
                results[thread_id] = str(e)
            finally:
                loop.close()
        
        # Create and start threads
        threads = []
        thread_results = {}
        num_threads = 5
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, thread_results))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results from all threads
        self.assertEqual(len(thread_results), num_threads)
        for thread_id, results in thread_results.items():
            self.assertIsInstance(results, list)
            for result in results:
                self.assertIn("chunk", result)
                self.assertIn("score", result)
        
        # Clean up
        setup_loop.close()
    
    def test_concurrent_operations(self):
        """Test concurrent vector operations (create, search, delete)."""
        # Create event loop for setup
        setup_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(setup_loop)
        
        # Create initial test data
        chunks = setup_loop.run_until_complete(self.create_test_data())
        
        # Define worker functions for different operations
        def create_worker(thread_id, results):
            """Worker for creating chunks."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create new chunks
                new_chunks = []
                for i in range(3):
                    chunk = Chunk(
                        id=self.generate_random_id(prefix=f"thread-{thread_id}"),
                        document_id="test-document",
                        text=f"Thread {thread_id} creating chunk {i} with vector embedding",
                        position=thread_id * 10 + i,
                        created_at=datetime.now(),
                        metadata={"thread": thread_id, "index": i}
                    )
                    loop.run_until_complete(self.content_service.create_chunk(chunk))
                    new_chunks.append(chunk.id)
                
                results[f"create-{thread_id}"] = new_chunks
                
            except Exception as e:
                logger.error(f"Create thread {thread_id} error: {str(e)}")
                results[f"create-{thread_id}"] = str(e)
            finally:
                loop.close()
        
        def search_worker(thread_id, results):
            """Worker for searching chunks."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Perform vector search
                query = f"Thread {thread_id} searching for similar content"
                search_results = loop.run_until_complete(
                    self.content_service.vector_search(query, top_k=5)
                )
                
                results[f"search-{thread_id}"] = len(search_results)
                
            except Exception as e:
                logger.error(f"Search thread {thread_id} error: {str(e)}")
                results[f"search-{thread_id}"] = str(e)
            finally:
                loop.close()
        
        def delete_worker(thread_id, results, chunk_ids):
            """Worker for deleting chunks."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Delete a chunk if available
                if chunk_ids and len(chunk_ids) > thread_id:
                    chunk_id = chunk_ids[thread_id]
                    loop.run_until_complete(self.content_service.delete_chunk(chunk_id))
                    results[f"delete-{thread_id}"] = chunk_id
                else:
                    results[f"delete-{thread_id}"] = "No chunk to delete"
                
            except Exception as e:
                logger.error(f"Delete thread {thread_id} error: {str(e)}")
                results[f"delete-{thread_id}"] = str(e)
            finally:
                loop.close()
        
        # Create and start threads for different operations
        threads = []
        thread_results = {}
        num_threads = 3
        
        # Get some chunk IDs for deletion
        chunk_ids = [chunk.id for chunk in chunks[:num_threads]]
        
        # Create threads for each operation type
        for i in range(num_threads):
            create_t = threading.Thread(target=create_worker, args=(i, thread_results))
            search_t = threading.Thread(target=search_worker, args=(i, thread_results))
            delete_t = threading.Thread(target=delete_worker, args=(i, thread_results, chunk_ids))
            
            threads.extend([create_t, search_t, delete_t])
            
            create_t.start()
            search_t.start()
            delete_t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        for thread_id in range(num_threads):
            # Verify create results
            self.assertIn(f"create-{thread_id}", thread_results)
            create_result = thread_results[f"create-{thread_id}"]
            self.assertIsInstance(create_result, list)
            self.assertEqual(len(create_result), 3)
            
            # Verify search results
            self.assertIn(f"search-{thread_id}", thread_results)
            search_result = thread_results[f"search-{thread_id}"]
            self.assertIsInstance(search_result, int)
            
            # Verify delete results
            self.assertIn(f"delete-{thread_id}", thread_results)
        
        # Clean up
        setup_loop.close()

if __name__ == "__main__":
    unittest.main()
