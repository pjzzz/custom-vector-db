import pytest
from models.chunk import Chunk
from services.content_service import ContentService
from indexers import InvertedIndex, TrieIndex, SuffixArrayIndex


# Mock vector service fixture removed as it's no longer needed

@pytest.fixture
def content_service():
    """ContentService instance for testing."""
    return ContentService(indexer_type='inverted', test_mode=True)

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
async def test_inverted_index(test_chunks):
    """Test the InvertedIndex implementation."""
    # Create the index
    index = InvertedIndex()

    # Add chunks to the index
    for chunk in test_chunks:
        index.add_chunk(chunk)

    # Test exact word search
    results = index.search("Python")
    assert len(results) == 2
    assert all(chunk_id in ["chunk_1", "chunk_3"] for _, chunk_id, _ in results)

    # Test multiple word search
    results = index.search("programming language")
    assert len(results) == 2
    assert all(chunk_id in ["chunk_1", "chunk_2"] for _, chunk_id, _ in results)

@pytest.mark.asyncio
async def test_trie_index(test_chunks):
    """Test the TrieIndex implementation."""
    # Create the index
    index = TrieIndex()

    # Add chunks to the index
    for chunk in test_chunks:
        index.add_chunk(chunk)

    # Test prefix search
    results = index.search("pro")
    assert len(results) >= 2  # Should find "programming" in chunks

    # Test exact word search
    results = index.search("Python")
    assert len(results) == 2
    assert all(chunk_id in ["chunk_1", "chunk_3"] for _, chunk_id, _ in results)

@pytest.mark.asyncio
async def test_suffix_array_index(test_chunks):
    """Test the SuffixArrayIndex implementation."""
    # Create the index
    index = SuffixArrayIndex()

    # Add chunks to the index
    for chunk in test_chunks:
        index.add_chunk(chunk)

    # Test suffix search
    results = index.search("age")
    assert len(results) >= 2  # Should find "language" in chunks

    # Test substring search
    results = index.search("program")
    assert len(results) >= 2  # Should find "programming" in chunks
