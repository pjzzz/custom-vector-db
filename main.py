from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from models import Chunk, Document, Library, SearchRequest, SearchResponse, UpsertRequest, DeleteRequest, TextEmbeddingRequest
from services.embedding_service import EmbeddingService
from services.content_service import ContentService
from config import settings
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Vector Similarity Search Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService(vector_size=settings.EMBEDDING_DIMENSION)
content_service = ContentService(indexer_type=settings.INDEXER_TYPE)


class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    text: str


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for similar vectors using a text query.
    """
    try:
        # Use content_service.vector_search instead of vector_service.search
        results = await content_service.vector_search(
            query_text=request.query,
            top_k=request.top_k,
            metadata_filter=request.filter
        )
        # Get the embedding for the response
        query_vector = embedding_service.get_embedding(request.query)
        return SearchResponse(
            matches=results,
            query_vector=query_vector,
            top_k=request.top_k
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/text")
async def search_text(query: str, indexer_type: str = "inverted"):
    """
    Search for text using the specified indexer.
    """
    try:
        results = await content_service.search(query, indexer_type)
        
        # Add a score field to each result for consistency with vector search
        for result in results:
            # Add a score of 1.0 for exact matches
            result["score"] = 1.0
            
        return results
    except Exception as e:
        logger.error(f"Text search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/vector")
async def vector_search(query_text: str, top_k: int = 5, filter_metadata: Optional[Dict] = None):
    """
    Search for vectors using semantic similarity.
    """
    try:
        results = await content_service.vector_search(
            query_text=query_text,
            top_k=top_k,
            filter_dict=filter_metadata
        )
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upsert")
async def upsert(request: UpsertRequest):
    """
    Add or update a vector in the index.
    """
    try:
        # Create a chunk with the vector data
        chunk = Chunk(
            id=request.id,
            document_id="direct_upsert",  # Default document for direct upserts
            text=request.metadata.get("text", ""),
            position=0,
            created_at=datetime.now(),
            metadata=request.metadata
        )
        # Store the chunk and its embedding
        await content_service.create_chunk(chunk)
        # Return success response
        return {"status": "success", "id": request.id}
    except Exception as e:
        logger.error(f"Upsert failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bulk-upsert")
async def bulk_upsert(vectors: List[UpsertRequest]):
    """
    Bulk upsert multiple vectors at once.
    """
    try:
        results = []
        for vec in vectors:
            # Create a chunk with the vector data
            chunk = Chunk(
                id=vec.id,
                document_id="direct_upsert",  # Default document for direct upserts
                text=vec.metadata.get("text", ""),
                position=0,
                created_at=datetime.now(),
                metadata=vec.metadata
            )
            # Store the chunk and its embedding
            await content_service.create_chunk(chunk)
            results.append({"id": vec.id, "status": "success"})
        return {"status": "success", "upserted_count": len(results), "vectors": results}
    except Exception as e:
        logger.error(f"Bulk upsert failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete(request: DeleteRequest):
    """
    Delete vectors by their IDs.
    """
    try:
        deleted_count = 0
        for chunk_id in request.ids:
            try:
                await content_service.delete_chunk(chunk_id)
                deleted_count += 1
            except Exception as chunk_error:
                logger.warning(f"Failed to delete chunk {chunk_id}: {str(chunk_error)}")
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the index.
    """
    try:
        # Get statistics from embedding service
        embedding_stats = {
            "vector_size": embedding_service.vector_size,
            "vocabulary_size": len(embedding_service.vocabulary) if hasattr(embedding_service, 'vocabulary') else 0,
            "document_count": embedding_service.document_count,
            "is_fitted": embedding_service.is_fitted
        }
        
        # Get statistics from similarity service
        similarity_stats = {
            "vector_count": len(content_service.similarity_service.vectors),
            "dimension": content_service.similarity_service.dimension,
            "distance_metric": content_service.similarity_service.distance_metric
        }
        
        # Get statistics from content service
        libraries = await content_service.get_libraries()
        library_count = len(libraries)
        
        document_count = 0
        chunk_count = 0
        for library in libraries:
            documents = await content_service.get_documents(library['id'])
            document_count += len(documents)
            for document in documents:
                chunks = await content_service.get_chunks(document['id'])
                chunk_count += len(chunks)
        
        content_stats = {
            "libraries": library_count,
            "documents": document_count,
            "chunks": chunk_count
        }
        
        return {
            "embedding_service": embedding_stats,
            "similarity_service": similarity_stats,
            "content_service": content_stats,
            "indexer_type": content_service.indexer.__class__.__name__
        }
    except Exception as e:
        logger.error(f"Get stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding")
async def get_embedding(request: TextEmbeddingRequest):
    """
    Get embedding for a given text.
    """
    try:
        embedding = embedding_service.get_embedding(request.text)
        return {
            "embedding": embedding,
            "text": request.text
        }
    except Exception as e:
        logger.error(f"Get embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bulk-embed")
async def bulk_get_embeddings(texts: List[str]):
    """
    Get embeddings for multiple texts.
    """
    try:
        embeddings = embedding_service.get_embeddings(texts)
        return {"embeddings": embeddings, "count": len(embeddings)}
    except Exception as e:
        logger.error(f"Bulk embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check content service is accessible
        await content_service.get_libraries()
        
        # Check embedding service
        embedding_service.get_embedding("test")
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        return {"status": "healthy", "timestamp": timestamp}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/libraries")
async def create_library(library: Library):
    """
    Create a new library.
    """
    try:
        return await content_service.create_library(library)
    except Exception as e:
        logger.error(f"Create library failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/libraries/{library_id}")
async def get_library(library_id: str):
    """
    Get a library by ID.
    """
    try:
        return await content_service.get_library(library_id)
    except Exception as e:
        logger.error(f"Get library failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/libraries/{library_id}")
async def update_library(library_id: str, library: Library):
    """
    Update a library.
    """
    try:
        return await content_service.update_library(library_id, library)
    except Exception as e:
        logger.error(f"Update library failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/libraries/{library_id}")
async def delete_library(library_id: str):
    """
    Delete a library by ID.
    """
    try:
        return await content_service.delete_library(library_id)
    except Exception as e:
        logger.error(f"Delete library failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def create_document(document: Document):
    """
    Create a new document.
    """
    try:
        return await content_service.create_document(document)
    except Exception as e:
        logger.error(f"Create document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get a document by ID.
    """
    try:
        return await content_service.get_document(document_id)
    except Exception as e:
        logger.error(f"Get document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/documents/{document_id}")
async def update_document(document_id: str, document: Document):
    """
    Update a document.
    """
    try:
        return await content_service.update_document(document_id, document)
    except Exception as e:
        logger.error(f"Update document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document by ID.
    """
    try:
        return await content_service.delete_document(document_id)
    except Exception as e:
        logger.error(f"Delete document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chunks")
async def create_chunk(chunk: Chunk):
    """
    Create a new chunk.
    """
    try:
        return await content_service.create_chunk(chunk)
    except Exception as e:
        logger.error(f"Create chunk failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """
    Get a chunk by ID.
    """
    try:
        return await content_service.get_chunk(chunk_id)
    except Exception as e:
        logger.error(f"Get chunk failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/chunks/{chunk_id}")
async def update_chunk(chunk_id: str, chunk: Chunk):
    """
    Update a chunk.
    """
    try:
        return await content_service.update_chunk(chunk_id, chunk)
    except Exception as e:
        logger.error(f"Update chunk failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """
    Delete a chunk by ID.
    """
    try:
        return await content_service.delete_chunk(chunk_id)
    except Exception as e:
        logger.error(f"Delete chunk failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
