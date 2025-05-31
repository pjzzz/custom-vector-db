import pytest
from unittest.mock import MagicMock
from models.chunk import Chunk
from models.document import Document
from models.library import Library
from services.content_service import ContentService
from services.vector_service import VectorService
from indexers import INDEXERS

@pytest.fixture
def mock_vector_service():
    """Mock VectorService for testing."""
    mock = MagicMock()
    mock.upsert.return_value = {"status": "success"}
    mock.delete.return_value = {"status": "success"}
    mock.get_embedding.return_value = [0.1] * 1536  # Mock embedding
    return mock

@pytest.fixture
def content_service(mock_vector_service):
    """ContentService instance with mocked VectorService."""
    return ContentService(vector_service=mock_vector_service, indexer_type='inverted')

@pytest.fixture
def test_chunks():
    """Test chunks fixture."""
    return [
        Chunk(
            id="chunk_1",
            text="Python is a great programming language",
            document_id="doc_1",
            position=1,
            metadata={"source": "tutorial"}
        ),
        Chunk(
            id="chunk_2",
            text="Java is also a popular programming language",
            document_id="doc_1",
            position=2,
            metadata={"source": "tutorial"}
        ),
        Chunk(
            id="chunk_3",
            text="Python has a large ecosystem of libraries",
            document_id="doc_2",
            position=1,
            metadata={"source": "documentation"}
        )
    ]

@pytest.mark.asyncio
async def test_search_with_inverted_index(content_service, test_chunks):
    """Test search functionality with inverted index."""
    # Create a document
    document = Document(
        id="doc_1",
        title="Programming Languages",
        library_id="lib_1"
    )
    
    # Add chunks to the service
    for chunk in test_chunks:
        await content_service.create_chunk(chunk)
    
    # Test search with inverted index
    results = await content_service.search("Python", indexer_type="inverted")
    assert len(results) == 2  # Should find both Python chunks
    assert all("Python" in result["text"] for result in results)

@pytest.mark.asyncio
async def test_search_with_trie_index(content_service, test_chunks):
    """Test search functionality with trie index."""
    # Create a document
    document = Document(
        id="doc_1",
        title="Programming Languages",
        library_id="lib_1"
    )
    
    # Add chunks to the service
    for chunk in test_chunks:
        await content_service.create_chunk(chunk)
    
    # Test search with trie index
    results = await content_service.search("pro", indexer_type="trie")  # Prefix search
    assert len(results) == 2  # Should find "programming" and "popular"
    assert all("pro" in result["text"] for result in results)

@pytest.mark.asyncio
async def test_search_with_suffix_index(content_service, test_chunks):
    """Test search functionality with suffix array index."""
    # Create a document
    document = Document(
        id="doc_1",
        title="Programming Languages",
        library_id="lib_1"
    )
    
    # Add chunks to the service
    for chunk in test_chunks:
        await content_service.create_chunk(chunk)
    
    # Test search with suffix array index
    results = await content_service.search("age", indexer_type="suffix")  # Suffix search
    assert len(results) == 2  # Should find "language" in both chunks
    assert all("age" in result["text"] for result in results)

@pytest.mark.asyncio
async def test_search_with_complete_content(content_service, test_chunks):
    """Test search functionality with complete content."""
    # Create a document
    document = Document(
        id="doc_1",
        title="Programming Languages",
        library_id="lib_1",
        content="This is a sample content for testing search functionality."
    )
    
    # Add chunks to the service
    for chunk in test_chunks:
        await content_service.create_chunk(chunk)
    
    # Test search with complete content
    results = await content_service.search("sample", indexer_type="complete")
    assert len(results) == 1  # Should find the document with the sample content
    assert results[0]["title"] == document.title

@pytest.mark.asyncio
async def test_search_with_complete_content_and_chunks(content_service, test_chunks):
    """Test search functionality with complete content and chunks."""
    # Create library
    library = Library(
        id="lib_1",
        name="Programming Library"
    )
    await content_service.create_library(library)
    
    # Create document
    document = Document(
        id="doc_1",
        title="Programming Languages",
        library_id="lib_1",
        content="This is a sample content for testing search functionality."
    )
    await content_service.create_document(document)
    
    # Add chunks to the service
    for chunk in test_chunks:
        await content_service.create_chunk(chunk)
    
    # Test search with complete content and chunks
    results = await content_service.search("Python sample", indexer_type="complete")
    assert len(results) == 2  # Should find the document with the sample content and the Python chunk
    assert any(result["title"] == document.title for result in results)
    assert any("Python" in result["text"] for result in results)
