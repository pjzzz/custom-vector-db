# Thread-Safe Vector Content Management System

This project implements a robust, thread-safe content management service with advanced vector search capabilities, focusing on comprehensive concurrency control mechanisms to ensure data integrity and prevent race conditions. The system is fully self-contained with no external vector search dependencies (like Pinecone, Cohere, or FAISS).

## Project Objective

The primary goal of this project was to develop a comprehensive, self-contained vector search system with custom embedding and indexing services that provides thread-safe, high-performance content management capabilities. Key objectives included:

1. Removing external vector dependencies (Pinecone, Cohere)
2. Implementing a fully thread-safe vector search implementation
3. Creating custom embedding and similarity search algorithms
4. Developing a robust REST API for content management
5. Ensuring high performance and concurrent access
6. Simplifying codebase by eliminating unnecessary external libraries
7. Providing comprehensive testing and benchmarking capabilities

## Features

- Thread-safe indexing algorithms (SuffixArrayIndex, TrieIndex, InvertedIndex)
- Custom vector embedding and similarity search without external dependencies
- Fine-grained locking mechanisms for concurrent operations
- Multiple search capabilities: text search, vector search, and hybrid search
- Efficient data structures for high-performance search operations
- Comprehensive benchmarking tools and load testing framework
- RESTful API with FastAPI for easy integration
- Python client library for simplified interaction
- Docker containerization for easy deployment
- Jupyter notebook demos for interactive exploration
- **Persistence to disk** with automatic snapshots and recovery
- Configurable indexing strategies via environment variables

## Architecture

### Indexing Algorithms

The system implements three different thread-safe indexing algorithms:

1. **SuffixArrayIndex**: Optimized for substring matching
   - Build time complexity: O(T log T) where T is the total text length
   - Search time complexity: O(P log T + k) where P is the pattern length and k is the number of matches
   - Supports efficient substring searches

2. **TrieIndex**: Optimized for prefix matching
   - Build time complexity: O(T) where T is the total text length
   - Search time complexity: O(P + k) where P is the pattern length and k is the number of matches
   - Excellent for autocomplete and prefix-based queries

3. **InvertedIndex**: Optimized for exact word matching
   - Build time complexity: O(T) where T is the total text length
   - Search time complexity: O(Q + k) where Q is the number of query terms and k is the number of matches
   - Best for boolean queries and exact word matching
   - Optimized `remove_chunk` method that only processes terms actually appearing in the chunk being removed, significantly improving efficiency for large indexes

## Configuration Options

### Docker Environment Variables

The system can be configured using environment variables when running with Docker:

```bash
# Basic usage with default settings
docker run -p 8000:8000 vector-db-image

# Specify a different indexer type
docker run -p 8000:8000 -e INDEXER_TYPE=trie vector-db-image

# Configure embedding dimension
docker run -p 8000:8000 -e EMBEDDING_DIMENSION=768 vector-db-image

# Disable automatic persistence
docker run -p 8000:8000 -e ENABLE_PERSISTENCE=false vector-db-image

# Configure snapshot interval (in seconds)
docker run -p 8000:8000 -e SNAPSHOT_INTERVAL=600 vector-db-image

# Specify data directory for persistence
docker run -p 8000:8000 -e DATA_DIR=/data -v ./my_data:/data vector-db-image
```

### Available Configuration Options

| Environment Variable | Description | Default Value | Options |
|----------------------|-------------|---------------|----------|
| `INDEXER_TYPE` | Type of indexer to use | `inverted` | `inverted`, `trie`, `suffix` |
| `EMBEDDING_DIMENSION` | Dimension of embedding vectors | `1536` | Any positive integer |
| `ENABLE_PERSISTENCE` | Whether to enable persistence to disk | `true` | `true`, `false` |
| `SNAPSHOT_INTERVAL` | Seconds between automatic snapshots | `300` | Any positive integer |
| `DATA_DIR` | Directory to store persistence files | `./data` | Any valid path |
| `FIT_EMBEDDING_MODEL` | Whether to fit the embedding model on startup | `true` | `true`, `false` |

### Indexer Selection Guide

- **InvertedIndex** (`INDEXER_TYPE=inverted`): Best for traditional keyword search and boolean queries. Most memory-efficient option.
- **TrieIndex** (`INDEXER_TYPE=trie`): Excellent for autocomplete features and prefix-based searches. Moderate memory usage.
- **SuffixArrayIndex** (`INDEXER_TYPE=suffix`): Best for substring searches and fuzzy matching. Higher memory usage but most flexible search capabilities.

## Usage Examples

### Creating and Managing Content

```python
from services.content_service import ContentService
from models import Library, Document, Chunk
from datetime import datetime
import asyncio

async def main():
    # Initialize with desired indexer type
    content_service = ContentService(indexer_type='inverted')
    
    # Create a library
    library = Library(
        id="my-library",
        name="My Knowledge Base",
        description="Collection of technical documents",
        created_at=datetime.now()
    )
    await content_service.create_library(library)
    
    # Create a document
    document = Document(
        id="doc-1",
        library_id="my-library",
        title="Vector Database Guide",
        created_at=datetime.now()
    )
    await content_service.create_document(document)
    
    # Add chunks to the document
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="Vector databases store and retrieve data based on similarity.",
        position=0,
        created_at=datetime.now(),
        metadata={"topic": "introduction"}
    )
    await content_service.create_chunk(chunk)

# Run the example
asyncio.run(main())
```

### Performing Searches

```python
async def search_example():
    content_service = ContentService(indexer_type='suffix')
    
    # Text search using different indexers
    inverted_results = await content_service.search("vector database", indexer_type="inverted")
    trie_results = await content_service.search("vec", indexer_type="trie")  # Prefix search
    suffix_results = await content_service.search("base", indexer_type="suffix")  # Suffix search
    
    # Vector search (semantic search)
    vector_results = await content_service.vector_search("How do similarity searches work?")
    
    # Print results
    print(f"Found {len(vector_results)} semantically similar chunks")

# Run the search example
asyncio.run(search_example())
```

### Using the REST API

```bash
# Start the API server
uvicorn main:app --reload

# Create a library
curl -X POST "http://localhost:8000/libraries" \
  -H "Content-Type: application/json" \
  -d '{"id":"tech-lib","name":"Technology Library","description":"Tech articles and guides"}'  

# Search for content
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"vector database","top_k":5,"indexer_type":"inverted"}'
```

## Performance Optimizations

### InvertedIndex Optimizations

The `InvertedIndex` implementation includes several key optimizations for performance and efficiency:

1. **Optimized `remove_chunk` Method**
   - Instead of scanning the entire index when removing a chunk, the method only processes terms that actually appear in the chunk being removed
   - This optimization significantly reduces computational overhead for large indexes with many terms
   - Time complexity improved from O(T) where T is the total number of terms in the index to O(C) where C is the number of terms in the chunk being removed

2. **Efficient Term Processing**
   - Terms are tokenized and normalized during indexing for consistent matching
   - Stop words can be filtered to reduce index size and improve search relevance
   - Term frequency information is maintained for relevance scoring

3. **Thread-Safe Operations**
   - All index operations use fine-grained locking to ensure thread safety
   - Read operations can proceed concurrently for high throughput
   - Write operations are properly synchronized to prevent data corruption

### ContentService Optimizations

The `ContentService` includes optimizations for efficient data management:

1. **Direct Object Access**
   - The `delete_library` method directly accesses documents from the library object
   - The `delete_document` method directly accesses chunks from the document object
   - These optimizations avoid scanning entire collections, improving performance for large datasets

2. **Asynchronous Operations**
   - All operations are implemented as async methods for non-blocking I/O
   - Background persistence tasks run concurrently with normal operations
   - Test mode allows disabling background tasks for reliable testing

### Custom Vector Search Implementation

The system includes a custom vector search implementation that doesn't rely on external services:

1. **EmbeddingService**: Generates vector embeddings for text
   - Uses TF-IDF with random projection for dimensionality reduction
   - Thread-safe implementation with RLock for model updates
   - Supports fitting on custom document corpora
   - Reproducible results with fixed random seeds
   - Configurable vector dimensions (default: 1536)
   - Optimized for performance with ~5,700 texts/second throughput

2. **SimilarityService**: Performs vector similarity search
   - Supports multiple distance metrics (cosine, euclidean, dot product)
   - Thread-safe operations for concurrent access
   - Includes metadata filtering capabilities
   - Optimized for performance with snapshot-based search
   - Sub-millisecond search latency after index building
   - Efficient filtering with minimal lock contention

3. **ContentService**: Manages libraries, documents, and chunks
   - Implements fine-grained locking for concurrent operations
   - Provides both text search and vector search capabilities
   - Thread-safe data access patterns for all operations
   - Separate locks for different resource types to maximize concurrency
   - Context managers for clean lock acquisition and release

### Thread Safety

All components implement robust concurrency control mechanisms:

- Fine-grained locking with separate locks for different data structures
- Minimizing critical sections by preparing data outside locks
- Creating snapshots to avoid holding locks during processing
- Defensive copying to prevent race conditions
- Validation to handle concurrent modifications
- Careful lock ordering to prevent deadlocks
- Lock-free operations where possible to maximize concurrency
- Thread-safe data structures for shared resources
- Atomic operations for critical updates
- Comprehensive load testing to validate thread safety

## Setup

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thread-safe-vector-search.git
cd thread-safe-vector-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Alternatively, use Docker to run the API server:
```bash
docker-compose up -d
```

5. Access the API documentation:
```
http://localhost:8000/docs
```

## Running Tests

### Unit Tests

To run all unit tests:

```bash
python -m unittest discover tests
```

To run specific test modules:

```bash
python -m unittest tests/services/test_vector_search.py
python -m unittest tests/services/test_indexers.py
python -m unittest tests/services/test_embedding_service.py
```

### Thread Safety Tests

To validate thread safety with concurrent operations:

```bash
python -m unittest tests/services/test_thread_safety.py
```

## Running Benchmarks

### Vector Search Benchmark

To benchmark the performance of vector search vs. traditional text search:

```bash
python benchmarks/vector_search_benchmark.py --chunks 1000 --size 100 --iterations 5
```

Options:
- `--chunks`: Number of chunks to create for testing (default: 1000)
- `--size`: Average number of words per chunk (default: 100)
- `--iterations`: Number of iterations per query (default: 5)

### Load Testing

To perform load testing with concurrent users:

```bash
python benchmarks/load_test.py --libraries 1 --documents 2 --chunks 10 --users 1,3,5,10 --duration 30
```

Options:
- `--libraries`: Number of libraries to create for testing
- `--documents`: Number of documents per library
- `--chunks`: Number of chunks per document
- `--users`: Comma-separated list of concurrent user counts
- `--duration`: Test duration in seconds

### Large-Scale Benchmark

To test performance with large vector collections:

```bash
python examples/large_scale_benchmark.py --vectors 50000 --dimension 1536
```

Options:
- `--vectors`: Number of vectors to test (default: 10000)
- `--dimension`: Vector dimension (default: 1536)

## Example Usage

### Using the ContentService Directly

```python
from services.content_service import ContentService
from models import Chunk, Document, Library
import asyncio
from datetime import datetime

async def main():
    # Initialize content service with custom vector search
    content_service = ContentService(
        indexer_type='suffix',  # Use suffix array index
        embedding_dimension=1536
    )
    
    # Create a library
    library = Library(
        id="test-library",
        name="Test Library",
        description="Test library for vector search",
        created_at=datetime.now()
    )
    await content_service.create_library(library)
    
    # Create a document
    document = Document(
        id="test-document",
        library_id="test-library",
        title="Test Document",
        content="This is a test document for vector search",
        created_at=datetime.now(),
        metadata={"type": "test"}
    )
    await content_service.create_document(document)
    
    # Create a chunk
    chunk = Chunk(
        id="test-chunk",
        document_id="test-document",
        text="This is a test chunk for vector similarity search",
        position=0,
        created_at=datetime.now(),
        metadata={"type": "test"}
    )
    await content_service.create_chunk(chunk)
    
    # Perform vector search
    results = await content_service.vector_search(
        query_text="similar vector search",
        top_k=5
    )
    print(results)
    
    # Perform text search
    results = await content_service.search(
        query="test chunk",
        indexer_type="suffix"
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the REST API

You can use the provided `api_demo.py` script to interact with the API:

```bash
python examples/api_demo.py
```

Or make direct HTTP requests:

```python
import requests
import json

# API endpoint
BASE_URL = "http://127.0.0.1:8000"

# Create a library
response = requests.post(
    f"{BASE_URL}/libraries",
    json={
        "id": "test-library",
        "name": "Test Library",
        "description": "Library for API demo"
    }
)
print(json.dumps(response.json(), indent=2))

# Create a document
response = requests.post(
    f"{BASE_URL}/documents",
    json={
        "id": "test-document",
        "title": "Test Document",
        "library_id": "test-library",
        "metadata": {
            "author": "API Demo",
            "category": "technical"
        }
    }
)
print(json.dumps(response.json(), indent=2))

# Create a chunk
response = requests.post(
    f"{BASE_URL}/chunks",
    json={
        "id": "test-chunk",
        "document_id": "test-document",
        "text": "This is a test chunk for vector similarity search",
        "position": 0,
        "metadata": {
            "type": "test"
        }
    }
)
print(json.dumps(response.json(), indent=2))

# Perform vector search
response = requests.get(
    f"{BASE_URL}/search/vector",
    params={
        "query": "similar vector search",
        "top_k": 5
    }
)
print(json.dumps(response.json(), indent=2))

# Perform text search
response = requests.get(
    f"{BASE_URL}/search/text",
    params={
        "query": "test chunk",
        "indexer_type": "suffix"
    }
)
print(json.dumps(response.json(), indent=2))
```

### Using the Python Client

The project includes a Python client for easier integration:

```python
from api.client import VectorSearchClient

# Create client
client = VectorSearchClient(base_url="http://127.0.0.1:8000")

# Create a library
library_id = client.create_library(
    id="test-library",
    name="Test Library",
    description="Library for client demo"
)

# Create a document
document_id = client.create_document(
    id="test-document",
    title="Test Document",
    library_id=library_id,
    metadata={"author": "Client Demo"}
)

# Create a chunk
chunk_id = client.create_chunk(
    id="test-chunk",
    document_id=document_id,
    text="This is a test chunk for vector similarity search",
    position=0,
    metadata={"type": "test"}
)

# Perform vector search
results = client.vector_search(
    query="similar vector search",
    top_k=5
)
print(results)

# Perform text search
results = client.text_search(
    query="test chunk",
    indexer_type="suffix"
)
print(results)
```

## Performance Characteristics

Based on extensive benchmarking, the system demonstrates the following performance characteristics:

### Vector Search Performance

- **Embedding Generation**: ~5,770 texts per second
- **Vector Search Latency**: ~0.0002 seconds (sub-millisecond) after index building
- **Text Search Latency**: ~0.19 seconds
- **Performance Improvement**: ~1,180x faster than traditional text search
- **Scaling**: Efficiently handles 50,000+ vectors with minimal performance degradation

### Concurrency Performance

- **Concurrent Users**: Successfully tested with up to 10 concurrent users
- **Operation Latencies Under Load**:
  - Vector search: ~0.016 seconds
  - Text search: ~0.018 seconds
  - Create operations: ~0.016 seconds
  - Delete operations: ~0.013 seconds
- **Success Rate**: >96% under heavy concurrent load

### Memory Usage

- **Vector Storage**: Efficient memory usage with numpy arrays
- **Index Overhead**: Minimal memory overhead for indexing structures
- **Scaling**: Linear memory scaling with vector count

## Persistence to Disk

The system implements a robust persistence mechanism to ensure data durability across container restarts:

### Persistence Architecture

1. **Snapshot-Based Persistence**:
   - Complete snapshots of the system state are taken at regular intervals
   - Each snapshot includes content store, embedding model, vectors, and indexer data
   - Atomic file operations ensure consistency even during crashes
   - Snapshots are stored in a configurable data directory

2. **Recovery Process**:
   - On startup, the system attempts to load the latest snapshot
   - If a snapshot is found, the system state is restored completely
   - If no snapshot is found, the system initializes with sample data
   - Thread-safe restoration ensures data integrity

3. **Performance Considerations**:
   - Background persistence minimizes impact on API performance
   - Separate locks for different data structures prevent contention
   - Incremental snapshots reduce disk I/O
   - Cleanup of old snapshots prevents disk space exhaustion

4. **Configuration Options**:
   - `DATA_DIR`: Directory to store persistence files (default: "./data")
   - `ENABLE_PERSISTENCE`: Whether to enable persistence (default: true)
   - `SNAPSHOT_INTERVAL`: Seconds between automatic snapshots (default: 300)

### Docker Integration

The Docker configuration includes volume mounting to ensure persistence across container restarts:

```yaml
volumes:
  - ./data:/app/data
```

This maps the host's `./data` directory to the container's `/app/data` directory, ensuring that all persistence files are stored on the host filesystem and survive container restarts.

## Technical Choices

### Why Custom Vector Implementation?

1. **Independence**: Eliminating external dependencies improves reliability and reduces costs
2. **Control**: Full control over implementation details and optimizations
3. **Thread Safety**: Custom implementation allowed for comprehensive thread safety design
4. **Performance**: Tailored optimizations for specific use cases
5. **Learning**: Deep understanding of vector search algorithms and data structures

### Why TF-IDF with Random Projection?

1. **Efficiency**: Fast computation without requiring GPU resources
2. **Simplicity**: Straightforward implementation with good performance
3. **Customization**: Easy to adjust for specific domains
4. **No Training**: Works without pre-training on large datasets

### Why Multiple Indexers?

1. **Flexibility**: Different search patterns require different indexing strategies
2. **Use Case Optimization**: Each indexer is optimized for specific query patterns
3. **Performance Tuning**: Allows selection of the most efficient indexer for each scenario

### Why Fine-Grained Locking?

1. **Concurrency**: Maximizes parallel operations
2. **Reduced Contention**: Minimizes lock waiting time
3. **Scalability**: Better performance with increasing concurrent users
4. **Resource Utilization**: Efficient use of multi-core systems
