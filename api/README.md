# Vector Content Management API

This API provides a RESTful interface to our thread-safe vector content management system with custom vector search capabilities. It's designed to be efficient, scalable, and robust against concurrent operations.

## Features

- **Thread-safe operations**: All API endpoints are designed to handle concurrent requests safely
- **Custom vector search**: Uses our custom TF-IDF based embedding and similarity search
- **Multiple indexing algorithms**: Choose between SuffixArrayIndex, TrieIndex, and InvertedIndex
- **Comprehensive API**: Manage libraries, documents, chunks, and perform various search operations
- **Containerized**: Runs in a Docker container for easy deployment

## API Endpoints

### Health Check

- `GET /health`: Check if the API is running

### Libraries

- `POST /libraries`: Create a new library
- `GET /libraries/{library_id}`: Get a library by ID
- `DELETE /libraries/{library_id}`: Delete a library by ID

### Documents

- `POST /documents`: Create a new document
- `GET /documents/{document_id}`: Get a document by ID
- `DELETE /documents/{document_id}`: Delete a document by ID

### Chunks

- `POST /chunks`: Create a new chunk
- `GET /chunks/{chunk_id}`: Get a chunk by ID
- `DELETE /chunks/{chunk_id}`: Delete a chunk by ID

### Search

- `POST /search/text`: Search for chunks using text search
- `POST /search/vector`: Search for chunks using vector similarity search

### Utility

- `GET /stats`: Get statistics about the content service

## Running with Docker

The API is containerized using Docker for easy deployment. You can run it using Docker Compose:

```bash
docker-compose up -d
```

This will start the API on port 8000. You can access the API documentation at http://localhost:8000/docs.

## Environment Variables

The API can be configured using the following environment variables:

- `USE_PINECONE`: Whether to use Pinecone for vector storage (default: false)
- `INDEXER_TYPE`: Type of indexer to use (default: suffix, options: suffix, trie, inverted)
- `FIT_EMBEDDING_MODEL`: Whether to fit the embedding model on startup (default: true)

## Example Requests

### Create a Library

```bash
curl -X POST "http://localhost:8000/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-library",
    "name": "Test Library",
    "description": "Test library for vector search",
    "created_at": "2025-05-29T01:00:00"
  }'
```

### Create a Document

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-document",
    "library_id": "test-library",
    "title": "Test Document",
    "content": "This is a test document for vector search",
    "created_at": "2025-05-29T01:00:00",
    "metadata": {"type": "test"}
  }'
```

### Create a Chunk

```bash
curl -X POST "http://localhost:8000/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-chunk",
    "document_id": "test-document",
    "text": "This is a test chunk for vector similarity search",
    "position": 0,
    "created_at": "2025-05-29T01:00:00",
    "metadata": {"type": "test"}
  }'
```

### Vector Search

```bash
curl -X POST "http://localhost:8000/search/vector" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "similar vector search",
    "top_k": 5
  }'
```

### Text Search

```bash
curl -X POST "http://localhost:8000/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test chunk",
    "indexer_type": "suffix"
  }'
```

## Implementation Details

The API is built on top of our custom vector search implementation, which includes:

1. **Custom EmbeddingService**:
   - Uses TF-IDF with random projection for dimensionality reduction
   - Thread-safe with RLock for concurrent model access
   - Supports fitting on custom document corpora

2. **SimilarityService**:
   - Implements efficient vector similarity search algorithms
   - Supports multiple distance metrics (cosine, euclidean, dot product)
   - Thread-safe operations with snapshot-based search

3. **ContentService**:
   - Manages libraries, documents, and chunks
   - Implements fine-grained locking for concurrent operations
   - Provides both text search and vector search capabilities

4. **Indexing Algorithms**:
   - SuffixArrayIndex: Optimized for substring matching
   - TrieIndex: Optimized for prefix matching
   - InvertedIndex: Optimized for exact word matching

All components implement robust concurrency control mechanisms to ensure thread safety during concurrent operations.
