from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from models import Chunk, Document, Library, SearchRequest, SearchResponse, UpsertRequest, DeleteRequest, TextEmbeddingRequest
from services.vector_service import VectorService
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
vector_service = VectorService()
embedding_service = EmbeddingService(vector_size=settings.EMBEDDING_DIMENSION)
content_service = ContentService(vector_service)


class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    text: str


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for similar vectors using a text query.
    """
    try:
        query_vector = embedding_service.get_embedding(request.query)
        results = await vector_service.search(
            query_vector=query_vector,
            top_k=request.top_k,
            filter=request.filter
        )
        return SearchResponse(
            matches=results['matches'],
            query_vector=query_vector,
            top_k=request.top_k
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upsert")
async def upsert(request: UpsertRequest):
    """
    Add or update a vector in the index.
    """
    try:
        return await vector_service.upsert(
            id=request.id,
            vector=request.values,
            metadata=request.metadata
        )
    except Exception as e:
        logger.error(f"Upsert failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bulk-upsert")
async def bulk_upsert(vectors: List[UpsertRequest]):
    """
    Bulk upsert multiple vectors at once.
    """
    try:
        pinecone_vectors = [
            {
                'id': vec.id,
                'values': vec.values,
                'metadata': vec.metadata
            }
            for vec in vectors
        ]
        return await vector_service.bulk_upsert(pinecone_vectors)
    except Exception as e:
        logger.error(f"Bulk upsert failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete(request: DeleteRequest):
    """
    Delete vectors by their IDs.
    """
    try:
        return await vector_service.delete(request.ids)
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the index.
    """
    try:
        return await vector_service.get_stats()
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
        # Check Pinecone connection
        stats = vector_service.get_stats()

        # Check OpenAI connection
        embedding_service.get_embedding("test")

        return {"status": "healthy", "stats": stats}
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
