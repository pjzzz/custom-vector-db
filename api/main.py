from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import logging
from datetime import datetime

import os

from models import Chunk, Document, Library
from services.content_service import ContentService


# Create a singleton instance of ContentService
_content_service = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vector Content Management API",
    description="API for thread-safe vector content management with custom vector search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get ContentService instance
async def get_content_service():
    """Dependency to get ContentService instance."""
    global _content_service

    # Initialize ContentService only once (singleton pattern)
    if _content_service is None:
        # Get persistence configuration from environment variables
        data_dir = os.environ.get("DATA_DIR", "./data")
        enable_persistence = os.environ.get("ENABLE_PERSISTENCE", "true").lower() == "true"
        snapshot_interval = int(os.environ.get("SNAPSHOT_INTERVAL", "300"))
        
        # Initialize ContentService with custom vector search
        _content_service = ContentService(
            indexer_type=os.environ.get("INDEXER_TYPE", "suffix"),
            embedding_dimension=1536,  # Default embedding dimension
            data_dir=data_dir,
            enable_persistence=enable_persistence,
            snapshot_interval=snapshot_interval
        )
        
        # Try to load data from disk first
        if enable_persistence:
            logger.info("Attempting to load data from disk...")
            loaded = await _content_service.load_from_disk()
            if loaded:
                logger.info("Successfully loaded data from disk")
            else:
                logger.info("No data found on disk, initializing with sample data")
                # If we have sample texts, fit the embedding model
                if os.environ.get("FIT_EMBEDDING_MODEL", "true").lower() == "true":
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
                    _content_service.embedding_service.fit(sample_texts)
        else:
            logger.info("Persistence is disabled, initializing with sample data")
            # If we have sample texts, fit the embedding model
            if os.environ.get("FIT_EMBEDDING_MODEL", "true").lower() == "true":
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
                _content_service.embedding_service.fit(sample_texts)

    return _content_service


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Library endpoints
@app.post("/libraries", response_model=Dict)
async def create_library(library: Library, content_service: ContentService = Depends(get_content_service)):
    """Create a new library."""
    try:
        result = await content_service.create_library(library)
        return result
    except Exception as e:
        logger.error(f"Create library failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/libraries/{library_id}", response_model=Dict)
async def get_library(library_id: str, content_service: ContentService = Depends(get_content_service)):
    """Get a library by ID."""
    try:
        result = await content_service.get_library(library_id)
        return result
    except Exception as e:
        logger.error(f"Get library failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/libraries/{library_id}", response_model=Dict)
async def delete_library(library_id: str, content_service: ContentService = Depends(get_content_service)):
    """Delete a library by ID."""
    try:
        result = await content_service.delete_library(library_id)
        return result
    except Exception as e:
        logger.error(f"Delete library failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


# Document endpoints
@app.post("/documents", response_model=Dict)
async def create_document(document: Document, content_service: ContentService = Depends(get_content_service)):
    """Create a new document."""
    try:
        result = await content_service.create_document(document)
        return result
    except Exception as e:
        logger.error(f"Create document failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/documents/{document_id}", response_model=Dict)
async def get_document(document_id: str, content_service: ContentService = Depends(get_content_service)):
    """Get a document by ID."""
    try:
        result = await content_service.get_document(document_id)
        return result
    except Exception as e:
        logger.error(f"Get document failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/documents/{document_id}", response_model=Dict)
async def delete_document(document_id: str, content_service: ContentService = Depends(get_content_service)):
    """Delete a document by ID."""
    try:
        result = await content_service.delete_document(document_id)
        return result
    except Exception as e:
        logger.error(f"Delete document failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


# Chunk endpoints
@app.post("/chunks", response_model=Dict)
async def create_chunk(chunk: Chunk, content_service: ContentService = Depends(get_content_service)):
    """Create a new chunk."""
    try:
        result = await content_service.create_chunk(chunk)
        return result
    except Exception as e:
        logger.error(f"Create chunk failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chunks/{chunk_id}", response_model=Dict)
async def get_chunk(chunk_id: str, content_service: ContentService = Depends(get_content_service)):
    """Get a chunk by ID."""
    try:
        result = await content_service.get_chunk(chunk_id)
        return result
    except Exception as e:
        logger.error(f"Get chunk failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/chunks/{chunk_id}", response_model=Dict)
async def delete_chunk(chunk_id: str, content_service: ContentService = Depends(get_content_service)):
    """Delete a chunk by ID."""
    try:
        result = await content_service.delete_chunk(chunk_id)
        return result
    except Exception as e:
        logger.error(f"Delete chunk failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


# Search endpoints
@app.post("/search/text", response_model=List[Dict])
async def text_search(
    query: str,
    indexer_type: Optional[str] = None,
    content_service: ContentService = Depends(get_content_service)
):
    """Search for chunks using text search."""
    try:
        results = await content_service.search(query, indexer_type)
        return results
    except Exception as e:
        logger.error(f"Text search failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search/vector", response_model=List[Dict])
async def vector_search(
    query_text: str,
    top_k: int = 10,
    filter_dict: Optional[Dict] = None,
    content_service: ContentService = Depends(get_content_service)
):
    """Search for chunks using vector similarity search."""
    try:
        results = await content_service.vector_search(query_text, top_k, filter_dict)
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Utility endpoints
@app.get("/stats")
async def get_stats(content_service: ContentService = Depends(get_content_service)):
    """Get statistics about the content service."""
    try:
        # Get embedding model info
        embedding_info = content_service.embedding_service.get_model_info()
        
        # Get similarity service stats if available
        similarity_stats = content_service.similarity_service.get_stats()
        
        # Count items in content store
        content_stats = {
            "libraries": len(content_service.content_store.get("libraries", {})),
            "documents": len(content_service.content_store.get("documents", {})),
            "chunks": len(content_service.content_store.get("chunks", {}))
        }
        
        return {
            "embedding_service": embedding_info,
            "similarity_service": similarity_stats,
            "content_service": content_stats,
            "indexer_type": content_service.indexer.__class__.__name__
        }
    except Exception as e:
        logger.error(f"Get stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
