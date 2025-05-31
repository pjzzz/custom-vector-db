from typing import List, Dict, Optional, Set, Union, Any
from models import Chunk, Document, Library
from services.vector_service import VectorService
from services.similarity_service import SimilarityService
from services.embedding_service import EmbeddingService
from config import settings
from indexers import INDEXERS
import logging
import json
import asyncio
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ContentService:
    """Service class for managing libraries, documents, and chunks."""

    def __init__(self, vector_service: Optional[VectorService] = None, 
             indexer_type: str = 'inverted',
             embedding_dimension: int = 1536,
             data_dir: str = "./data",
             enable_persistence: bool = True,
             snapshot_interval: int = 300) -> None:
        """Initialize with vector service.
        
        Args:
            vector_service: Vector service instance (optional)
            indexer_type: Type of indexer to use ('inverted', 'trie', or 'suffix')
            embedding_dimension: Dimension of embedding vectors (default: 1536)
            data_dir: Directory to store persistence files
            enable_persistence: Whether to enable persistence to disk
            snapshot_interval: Seconds between automatic snapshots
        """
        self.vector_service = vector_service
        self.content_store = {}
        self.embedding_dimension = embedding_dimension
        self.indexer_type = indexer_type
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(vector_size=embedding_dimension, min_word_freq=2)
        logger.info("Using custom TF-IDF embeddings")
        
        # Initialize similarity service
        self.similarity_service = SimilarityService(distance_metric=SimilarityService.COSINE)
        logger.info("Using custom vector search implementation")
        
        # Initialize the selected indexer
        if indexer_type not in INDEXERS:
            raise ValueError(f"Invalid indexer type: {indexer_type}. Must be one of: {list(INDEXERS.keys())}")
        self.indexer = INDEXERS[indexer_type]()  # In-memory storage with persistence
        
        # Initialize locks for concurrency control
        self._library_lock = asyncio.Lock()
        self._document_lock = asyncio.Lock()
        self._chunk_lock = asyncio.Lock()
        self._indexer_lock = asyncio.Lock()
        self._vector_lock = asyncio.Lock()  # Lock for vector operations
        
        # Initialize persistence service if enabled
        self.enable_persistence = enable_persistence
        if enable_persistence:
            from services.persistence_service import PersistenceService
            self.persistence_service = PersistenceService(
                data_dir=data_dir,
                snapshot_interval=snapshot_interval,
                enable_auto_persist=True
            )
            logger.info(f"Initialized persistence service with data directory: {data_dir}")
        else:
            self.persistence_service = None
            logger.warning("Persistence is disabled - data will not be saved to disk")
        
    @asynccontextmanager
    async def _library_write_lock(self):
        """Context manager for library write operations."""
        async with self._library_lock:
            yield
            
    @asynccontextmanager
    async def _document_write_lock(self):
        """Context manager for document write operations."""
        async with self._document_lock:
            yield
            
    @asynccontextmanager
    async def _chunk_write_lock(self):
        """Context manager for chunk write operations."""
        async with self._chunk_lock:
            yield
            
    @asynccontextmanager
    async def _indexer_write_lock(self):
        """Context manager for indexer write operations."""
        async with self._indexer_lock:
            yield
            
    @asynccontextmanager
    async def _vector_write_lock(self):
        """Context manager for vector operations."""
        async with self._vector_lock:
            yield

    async def _persist_changes(self):
        """Persist changes to disk if persistence is enabled."""
        if not self.enable_persistence or not self.persistence_service:
            return
        
        try:
            # Create a snapshot in the background
            await self.persistence_service.create_snapshot(self)
        except Exception as e:
            logger.error(f"Error persisting changes: {str(e)}")

    async def create_library(self, library: Library) -> Dict:
        """
        Create a new library.
        
        Args:
            library: Library object to create
            
        Returns:
            Dict containing success message and library data
        """
        try:
            # Validate library ID
            if not library.id:
                raise ValueError("Library ID is required")
            
            async with self._library_write_lock():
                # Check if library already exists
                if library.id in self.content_store.get('libraries', {}):
                    raise ValueError(f"Library with ID {library.id} already exists")
                    
                # Store the library
                self.content_store.setdefault('libraries', {})[library.id] = library
                # Create a copy to avoid race conditions
                library_data = library.model_dump()
            
            # No external vector index creation needed - using custom implementation
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": "Library created successfully",
                "library": library_data
            }
        except Exception as e:
            logger.error(f"Create library failed: {str(e)}")
            raise

    async def get_library(self, library_id: str) -> Dict:
        """
        Get a library by ID.
        
        Args:
            library_id: ID of the library to retrieve
            
        Returns:
            Dict containing library data
        
        Raises:
            ValueError: If library not found
        """
        try:
            async with self._library_write_lock():
                libraries = self.content_store.get('libraries', {})
                if library_id not in libraries:
                    raise ValueError(f"Library {library_id} not found")
                
                # Create a copy to avoid race conditions
                library_data = libraries[library_id].model_dump()
            
            return {
                "message": "Library retrieved successfully",
                "library": library_data
            }
        except Exception as e:
            logger.error(f"Get library failed: {str(e)}")
            raise

    async def update_library(self, library_id: str, library: Library) -> Dict:
        """
        Update an existing library.
        
        Args:
            library_id: ID of the library to update
            library: Updated library data
            
        Returns:
            Dict containing success message and updated library data
        
        Raises:
            ValueError: If library not found
        """
        try:
            async with self._library_write_lock():
                libraries = self.content_store.get('libraries', {})
                if library_id not in libraries:
                    raise ValueError(f"Library {library_id} not found")
                    
                # Update the library
                libraries[library_id] = library
                # Create a copy to avoid race conditions
                library_data = library.model_dump()
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": "Library updated successfully",
                "library": library_data
            }
        except Exception as e:
            logger.error(f"Update library failed: {str(e)}")
            raise

    async def delete_library(self, library_id: str) -> Dict:
        """
        Delete a library by ID.
        
        Args:
            library_id: ID of the library to delete
            
        Returns:
            Dict containing success message
        
        Raises:
            ValueError: If library not found
        """
        try:
            # Get the library
            async with self._library_write_lock():
                libraries = self.content_store.get('libraries', {})
                if library_id not in libraries:
                    raise ValueError(f"Library {library_id} not found")
                
                # Get all documents in this library
                documents = self.content_store.get('documents', {})
                library_documents = [doc_id for doc_id, doc in documents.items() if doc.library_id == library_id]
                
                # Delete the library
                del libraries[library_id]
            
            # Delete all documents in this library
            for doc_id in library_documents:
                await self.delete_document(doc_id)
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": f"Library {library_id} deleted successfully"
            }
        except Exception as e:
            logger.error(f"Delete library failed: {str(e)}")
            raise

    async def create_document(self, document: Document) -> Dict:
        """
        Create a new document.
        
        Args:
            document: Document object to create
            
        Returns:
            Dict containing success message and document data
        """
        try:
            # Validate document ID
            if not document.id:
                raise ValueError("Document ID is required")
            
            # Validate library ID
            if not document.library_id:
                raise ValueError("Library ID is required")
            
            # Check if library exists
            async with self._library_write_lock():
                libraries = self.content_store.get('libraries', {})
                if document.library_id not in libraries:
                    raise ValueError(f"Library {document.library_id} not found")
            
            # Store the document
            async with self._document_write_lock():
                documents = self.content_store.setdefault('documents', {})
                if document.id in documents:
                    raise ValueError(f"Document {document.id} already exists")
                
                # Add the document
                documents[document.id] = document
                # Create a copy to avoid race conditions
                document_data = document.model_dump()
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": "Document created successfully",
                "document": document_data
            }
        except Exception as e:
            logger.error(f"Create document failed: {str(e)}")
            raise

    async def get_document(self, document_id: str) -> Dict:
        """
        Get a document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Dict containing document data
        
        Raises:
            ValueError: If document not found
        """
        try:
            async with self._document_write_lock():
                documents = self.content_store.get('documents', {})
                if document_id not in documents:
                    raise ValueError(f"Document {document_id} not found")
                
                # Create a copy to avoid race conditions
                document_data = documents[document_id].model_dump()
            
            return {
                "message": "Document retrieved successfully",
                "document": document_data
            }
        except Exception as e:
            logger.error(f"Get document failed: {str(e)}")
            raise

    async def update_document(self, document_id: str, document: Document) -> Dict:
        """
        Update an existing document.
        
        Args:
            document_id: ID of the document to update
            document: Updated document data
            
        Returns:
            Dict containing success message and updated document data
        
        Raises:
            ValueError: If document not found
        """
        try:
            async with self._document_write_lock():
                documents = self.content_store.get('documents', {})
                if document_id not in documents:
                    raise ValueError(f"Document {document_id} not found")
                    
                # Update the document
                documents[document_id] = document
                # Create a copy to avoid race conditions
                document_data = document.model_dump()
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": "Document updated successfully",
                "document": document_data
            }
        except Exception as e:
            logger.error(f"Update document failed: {str(e)}")
            raise

    async def delete_document(self, document_id: str) -> Dict:
        """
        Delete a document by ID.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Dict containing success message
        
        Raises:
            ValueError: If document not found
        """
        try:
            # Get the document
            async with self._document_write_lock():
                documents = self.content_store.get('documents', {})
                if document_id not in documents:
                    raise ValueError(f"Document {document_id} not found")
                
                # Get all chunks in this document
                chunks = self.content_store.get('chunks', {})
                document_chunks = [chunk_id for chunk_id, chunk in chunks.items() if chunk.document_id == document_id]
                
                # Delete the document
                del documents[document_id]
            
            # Delete all chunks in this document
            for chunk_id in document_chunks:
                await self.delete_chunk(chunk_id)
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": f"Document {document_id} deleted successfully"
            }
        except Exception as e:
            logger.error(f"Delete document failed: {str(e)}")
            raise

    async def create_chunk(self, chunk: Chunk) -> Dict:
        """
        Create a new chunk.
        
        Args:
            chunk: Chunk object to create
            
        Returns:
            Dict containing success message and chunk data
        """
        try:
            # Validate chunk ID
            if not chunk.id:
                raise ValueError("Chunk ID is required")
            
            # Validate document ID
            if not chunk.document_id:
                raise ValueError("Document ID is required")
            
            # Check if document exists
            async with self._document_write_lock():
                documents = self.content_store.get('documents', {})
                if chunk.document_id not in documents:
                    raise ValueError(f"Document {chunk.document_id} not found")
            
            # Generate the vector outside the lock for performance
            vector = self.embedding_service.get_embedding(chunk.text)
            
            # Store the chunk
            async with self._chunk_write_lock():
                chunks = self.content_store.setdefault('chunks', {})
                if chunk.id in chunks:
                    raise ValueError(f"Chunk {chunk.id} already exists")
                
                # Add the chunk
                chunks[chunk.id] = chunk
                # Create a copy to avoid race conditions
                chunk_data = chunk.model_dump()
            
            # Add to indexer with a separate lock
            async with self._indexer_write_lock():
                self.indexer.add_chunk(chunk)
            
            # Add to vector store
            async with self._vector_write_lock():
                # Add the vector to our similarity service
                self.similarity_service.upsert(
                    id=chunk.id,
                    vector=vector,
                    metadata={
                        "document_id": chunk.document_id,
                        "position": chunk.position
                    }
                )
            
            # Persist changes if enabled
            await self._persist_changes()
            
            return {
                "message": "Chunk created successfully",
                "chunk": chunk_data
            }
        except Exception as e:
            logger.error(f"Create chunk failed: {str(e)}")
            raise

    async def get_chunk(self, chunk_id: str) -> Dict:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Dict containing chunk data
        
        Raises:
            ValueError: If chunk not found
        """
        try:
            async with self._chunk_write_lock():
                chunks = self.content_store.get('chunks', {})
                if chunk_id not in chunks:
                    raise ValueError(f"Chunk {chunk_id} not found")
                
                # Create a copy to avoid race conditions
                chunk_data = chunks[chunk_id].model_dump()
            
            return {
                "message": "Chunk retrieved successfully",
                "chunk": chunk_data
            }
        except Exception as e:
            logger.error(f"Get chunk failed: {str(e)}")
            raise

    async def update_chunk(self, chunk_id: str, chunk: Chunk) -> Dict:
        """
        Update an existing chunk.
        
        Args:
            chunk_id: ID of the chunk to update
            chunk: Updated chunk data
            
        Returns:
            Dict containing success message and updated chunk data
        
        Raises:
            ValueError: If chunk not found
        """
        try:
            # Generate the vector outside the lock for performance
            vector = self.embedding_service.get_embedding(chunk.text)
            
            async with self._chunk_write_lock():
                chunks = self.content_store.get('chunks', {})
                if chunk_id not in chunks:
                    raise ValueError(f"Chunk {chunk_id} not found")
                    
                # Update the chunk
                chunks[chunk_id] = chunk
                # Create a copy to avoid race conditions
                chunk_data = chunk.model_dump()
            
            # Update the indexer with a separate lock
            async with self._indexer_write_lock():
                # Remove old chunk from index and add new one
                self.indexer.add_chunk(chunk)
            
            # Update the vector
            if vector is not None:
                # Use our custom similarity service with a lock
                async with self._vector_write_lock():
                    self.similarity_service.upsert(
                        id=chunk_id,
                        vector=vector,
                        metadata={
                            "document_id": chunk.document_id,
                            "position": chunk.position
                        }
                    )
            
            return {
                "message": "Chunk updated successfully",
                "chunk": chunk_data
            }
        except Exception as e:
            logger.error(f"Update chunk failed: {str(e)}")
            raise

    async def delete_chunk(self, chunk_id: str) -> Dict:
        """
        Delete a chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to delete
            
        Returns:
            Dict containing success message
        
        Raises:
            ValueError: If chunk not found
        """
        try:
            # Store chunk data before deletion for indexer removal
            chunk_to_delete = None
            
            async with self._chunk_write_lock():
                chunks = self.content_store.get('chunks', {})
                if chunk_id not in chunks:
                    raise ValueError(f"Chunk {chunk_id} not found")
                
                # Store chunk data before deletion for indexer removal
                chunk_to_delete = chunks[chunk_id]
                
                # Delete the chunk
                del chunks[chunk_id]
            
            # Remove from indexer with a separate lock
            if chunk_to_delete:
                async with self._indexer_write_lock():
                    # Use the remove_chunk method if available
                    if hasattr(self.indexer, 'remove_chunk'):
                        self.indexer.remove_chunk(chunk_id)
                    # Otherwise, we'll rely on the fact that deleted chunks
                    # won't be returned in search results even if they're in the index
            
            # Delete the vector
            async with self._vector_write_lock():
                self.similarity_service.delete([chunk_id])
            
            return {
                "message": "Chunk deleted successfully",
                "chunk_id": chunk_id
            }
        except Exception as e:
            logger.error(f"Delete chunk failed: {str(e)}")
            raise

    async def vector_search(self, query_text: str, top_k: int = 10, 
                          filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for chunks similar to the query text using vector similarity.
        Thread-safe implementation using custom vector search.
        
        Args:
            query_text: The text to search for
            top_k: Maximum number of results to return
            filter_dict: Optional filter to apply to the search
            
        Returns:
            List of dicts with chunk data and similarity scores
        """
        try:
            # Generate embedding for the query text
            query_embedding = self.embedding_service.get_embedding(query_text)
            
            # Use our custom similarity service
            search_results = self.similarity_service.search(
                query_vector=query_embedding,
                top_k=top_k,
                filter_metadata=filter_dict
            )
            
            # Get the full chunk data for each result
            chunks = self.content_store.get('chunks', {})
            enriched_results = []
            
            for result in search_results:
                chunk_id = result.get('id')
                if chunk_id in chunks:
                    # Create a copy of the chunk data
                    chunk_data = chunks[chunk_id].model_dump()
                    # Add the similarity score
                    chunk_data['score'] = result.get('score', 0.0)
                    enriched_results.append(chunk_data)
            
            return enriched_results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

    async def search(self, query: str, indexer_type: Optional[str] = None) -> List[Dict]:
        """
        Search for chunks containing the query text.
        
        Args:
            query: Search query string
            indexer_type: Optional indexer type to use for this search
                If not provided, uses the default indexer from initialization
                
        Returns:
            List of search results with chunk information
        """
        try:
            # Use provided indexer type if specified, otherwise use default
            if indexer_type and indexer_type in INDEXERS:
                indexer = INDEXERS[indexer_type]()
                # Add existing chunks to the indexer
                async with self._chunk_write_lock():
                    chunks = self.content_store.get('chunks', {})
                    for chunk in chunks.values():
                        indexer.add_chunk(chunk)
            else:
                indexer = self.indexer
            
            # Get results from indexer with a read lock
            results = []
            async with self._indexer_write_lock():
                results = indexer.search(query)
            
            # Convert results to detailed format with a read lock on chunks
            formatted_results = []
            async with self._chunk_write_lock():
                chunks = self.content_store.get('chunks', {})
                for doc_id, chunk_id, position in results:
                    if chunk_id in chunks:
                        chunk = chunks[chunk_id]
                        # Create a copy to avoid race conditions
                        formatted_results.append({
                            "document_id": doc_id,
                            "chunk_id": chunk_id,
                            "position": position,
                            "text": chunk.text,
                            "metadata": chunk.metadata
                        })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
            
    async def load_from_disk(self):
        """Load data from disk if persistence is enabled."""
        if not self.enable_persistence or not self.persistence_service:
            logger.warning("Persistence is disabled - cannot load from disk")
            return False
        
        try:
            # Load the latest snapshot
            result = await self.persistence_service.load_latest_snapshot(self)
            if result:
                logger.info("Successfully loaded data from disk")
            else:
                logger.info("No data found on disk to load")
            return result
        except Exception as e:
            logger.error(f"Error loading data from disk: {str(e)}")
            return False
            
    async def get_libraries(self):
        """Get all libraries.
        
        Returns:
            List of libraries as dictionaries
        """
        async with self._library_lock:
            libraries = self.content_store.get('libraries', {})
            # Create a copy to avoid race conditions
            return [lib.model_dump() for lib in libraries.values()]
    
    async def get_documents(self, library_id=None):
        """Get all documents, optionally filtered by library_id.
        
        Args:
            library_id: Optional library ID to filter by
            
        Returns:
            List of documents as dictionaries
        """
        async with self._document_lock:
            documents = self.content_store.get('documents', {})
            # Create a copy to avoid race conditions
            if library_id:
                return [doc.model_dump() for doc_id, doc in documents.items() 
                        if doc.library_id == library_id]
            else:
                return [doc.model_dump() for doc in documents.values()]
    
    async def get_chunks(self, document_id=None):
        """Get all chunks, optionally filtered by document_id.
        
        Args:
            document_id: Optional document ID to filter by
            
        Returns:
            List of chunks as dictionaries
        """
        async with self._chunk_lock:
            chunks = self.content_store.get('chunks', {})
            # Create a copy to avoid race conditions
            if document_id:
                return [chunk.model_dump() for chunk_id, chunk in chunks.items() 
                        if chunk.document_id == document_id]
            else:
                return [chunk.model_dump() for chunk in chunks.values()]
