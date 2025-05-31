import pytest
from unittest.mock import MagicMock
from models import Chunk, Document, Library
from services.content_service import ContentService
from services.vector_service import VectorService


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
def test_library():
    """Test library fixture."""
    return Library(
        id="lib_1",
        name="Test Library",
        description="Test description",
        metadata={"category": "test"}
    )


@pytest.fixture
def test_document():
    """Test document fixture."""
    return Document(
        id="doc_1",
        title="Test Document",
        library_id="lib_1",
        metadata={"author": "test"}
    )


@pytest.fixture
def test_chunk():
    """Test chunk fixture."""
    return Chunk(
        id="chunk_1",
        text="Test chunk text",
        document_id="doc_1",
        position=1,
        metadata={"source": "test"}
    )


@pytest.mark.asyncio
class TestContentService:
    async def test_create_library(self, content_service, test_library):
        """
        Test creating a new library.
        """
        result = await content_service.create_library(test_library)
        assert result["message"] == "Library created successfully"
        assert result["library"] == test_library.dict()

    async def test_get_library(self, content_service, test_library):
        """
        Test retrieving a library.
        """
        # First create the library
        await content_service.create_library(test_library)
        
        # Then get it
        result = await content_service.get_library(test_library.id)
        assert result["message"] == "Library retrieved successfully"
        assert result["library"] == test_library.model_dump()

    async def test_update_library(self, content_service, test_library):
        """
        Test updating a library.
        """
        # First create the library
        await content_service.create_library(test_library)
        
        # Update it
        updated_library = test_library.model_copy()
        updated_library.name = "Updated Library"
        result = await content_service.update_library(test_library.id, updated_library)
        
        assert result["message"] == "Library updated successfully"
        assert result["library"] == updated_library.model_dump()

    async def test_delete_library(self, content_service, test_library):
        """
        Test deleting a library.
        """
        # First create the library
        await content_service.create_library(test_library)
        
        # Delete it
        result = await content_service.delete_library(test_library.id)
        assert result["message"] == "Library deleted successfully"
        assert result["library_id"] == test_library.id

    async def test_create_document(self, content_service, test_document):
        """
        Test creating a new document.
        """
        # Create the library first
        library = Library(id="lib_1", name="Test Library")
        await content_service.create_library(library)
        
        result = await content_service.create_document(test_document)
        assert result["message"] == "Document created successfully"
        assert result["document"] == test_document.model_dump()

    async def test_get_document(self, content_service, test_document):
        """
        Test retrieving a document.
        """
        # Create the library and document first
        library = Library(id="lib_1", name="Test Library")
        await content_service.create_library(library)
        await content_service.create_document(test_document)
        
        result = await content_service.get_document(test_document.id)
        assert result["message"] == "Document retrieved successfully"
        assert result["document"] == test_document.model_dump()

    async def test_update_document(self, content_service, test_document):
        """
        Test updating a document.
        """
        # Create the library and document first
        library = Library(id="lib_1", name="Test Library")
        await content_service.create_library(library)
        await content_service.create_document(test_document)
        
        # Update the document
        updated_document = test_document.model_copy()
        updated_document.title = "Updated Title"
        result = await content_service.update_document(test_document.id, updated_document)
        
        assert result["message"] == "Document updated successfully"
        assert result["document"] == updated_document.model_dump()

    async def test_delete_document(self, content_service, test_document):
        """
        Test deleting a document.
        """
        # Create the library and document first
        library = Library(id="lib_1", name="Test Library")
        await content_service.create_library(library)
        await content_service.create_document(test_document)
        
        result = await content_service.delete_document(test_document.id)
        assert result["message"] == "Document deleted successfully"
        assert result["document_id"] == test_document.id

    async def test_create_chunk(self, content_service, test_chunk):
        """
        Test creating a new chunk.
        """
        # Create the library and document first
        library = Library(id="lib_1", name="Test Library")
        document = Document(id="doc_1", title="Test Document", library_id="lib_1")
        await content_service.create_library(library)
        await content_service.create_document(document)
        
        result = await content_service.create_chunk(test_chunk)
        assert result["message"] == "Chunk created successfully"
        assert result["chunk"] == test_chunk.model_dump()

    async def test_get_chunk(self, content_service, test_chunk):
        """
        Test retrieving a chunk.
        """
        # Create the library, document, and chunk first
        library = Library(id="lib_1", name="Test Library")
        document = Document(id="doc_1", title="Test Document", library_id="lib_1")
        await content_service.create_library(library)
        await content_service.create_document(document)
        await content_service.create_chunk(test_chunk)
        
        result = await content_service.get_chunk(test_chunk.id)
        assert result["message"] == "Chunk retrieved successfully"
        assert result["chunk"] == test_chunk.model_dump()

    async def test_update_chunk(self, content_service, test_chunk):
        """
        Test updating a chunk.
        """
        # Create the library, document, and chunk first
        library = Library(id="lib_1", name="Test Library")
        document = Document(id="doc_1", title="Test Document", library_id="lib_1")
        await content_service.create_library(library)
        await content_service.create_document(document)
        await content_service.create_chunk(test_chunk)
        
        # Update the chunk
        updated_chunk = test_chunk.model_copy()
        updated_chunk.text = "Updated text"
        result = await content_service.update_chunk(test_chunk.id, updated_chunk)
        
        assert result["message"] == "Chunk updated successfully"
        assert result["chunk"] == updated_chunk.model_dump()

    async def test_delete_chunk(self, content_service, test_chunk):
        """
        Test deleting a chunk.
        """
        # Create the library, document, and chunk first
        library = Library(id="lib_1", name="Test Library")
        document = Document(id="doc_1", title="Test Document", library_id="lib_1")
        await content_service.create_library(library)
        await content_service.create_document(document)
        await content_service.create_chunk(test_chunk)
        
        result = await content_service.delete_chunk(test_chunk.id)
        assert result["message"] == "Chunk deleted successfully"
        assert result["chunk_id"] == test_chunk.id
