"""
Test script for demonstrating persistence functionality.

This script:
1. Creates a library, document, and chunks
2. Triggers a snapshot to be saved to disk
3. Restarts the ContentService
4. Verifies that the data is loaded from disk
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.content_service import ContentService
from models import Library, Document, Chunk

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    # Create a temporary data directory for testing
    data_dir = "./test_persistence_data"
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Create a ContentService with persistence enabled
    logger.info("Creating ContentService with persistence enabled")
    content_service = ContentService(
        indexer_type="suffix",
        embedding_dimension=1536,
        data_dir=data_dir,
        enable_persistence=True,
        snapshot_interval=10  # Short interval for testing
    )

    # Step 2: Create sample data
    logger.info("Creating sample data")

    # Create a library
    library = Library(
        id="test-library",
        name="Test Library",
        description="Test library for persistence",
        created_at=datetime.now().isoformat()  # Convert to ISO format string
    )
    await content_service.create_library(library)

    # Create a document
    document = Document(
        id="test-document",
        library_id="test-library",
        title="Test Document",
        content="This is a test document for persistence",
        created_at=datetime.now().isoformat(),  # Convert to ISO format string
        metadata={"type": "test"}
    )
    await content_service.create_document(document)

    # Create chunks
    for i in range(5):
        chunk = Chunk(
            id=f"test-chunk-{i}",
            document_id="test-document",
            text=f"This is test chunk {i} for persistence testing",
            position=i,
            created_at=datetime.now().isoformat(),  # Convert to ISO format string
            metadata={"type": "test", "index": str(i)}  # Convert index to string
        )
        await content_service.create_chunk(chunk)

    # Step 3: Manually trigger a snapshot
    logger.info("Manually triggering a snapshot")
    await content_service._persist_changes()

    # Wait for the snapshot to complete
    logger.info("Waiting for snapshot to complete...")
    await asyncio.sleep(2)

    # Step 4: Verify data is in the first service
    libraries = await content_service.get_libraries()
    documents = await content_service.get_documents("test-library")
    chunks = []
    for doc in documents:
        doc_chunks = await content_service.get_chunks(doc["id"])
        chunks.extend(doc_chunks)

    logger.info(f"First service has {len(libraries)} libraries, {len(documents)} documents, and {len(chunks)} chunks")

    # Step 5: Create a new ContentService instance
    logger.info("Creating a new ContentService instance")
    new_content_service = ContentService(
        indexer_type="suffix",
        embedding_dimension=1536,
        data_dir=data_dir,
        enable_persistence=True,
        snapshot_interval=10
    )

    # Step 6: Load data from disk
    logger.info("Loading data from disk")
    loaded = await new_content_service.load_from_disk()

    if loaded:
        logger.info("Successfully loaded data from disk")
    else:
        logger.error("Failed to load data from disk")
        return

    # Step 7: Verify data is loaded in the new service
    new_libraries = await new_content_service.get_libraries()
    new_documents = await new_content_service.get_documents("test-library")
    new_chunks = []
    for doc in new_documents:
        doc_chunks = await new_content_service.get_chunks(doc["id"])
        new_chunks.extend(doc_chunks)

    logger.info(f"New service has {len(new_libraries)} libraries, {len(new_documents)} documents, and {len(new_chunks)} chunks")

    # Step 8: Verify data integrity
    if (len(libraries) == len(new_libraries) and
        len(documents) == len(new_documents) and
        len(chunks) == len(new_chunks)):
        logger.info("Data integrity verified - persistence is working correctly!")
    else:
        logger.error("Data integrity check failed - persistence is not working correctly")
        logger.error(f"Original: {len(libraries)} libraries, {len(documents)} documents, {len(chunks)} chunks")
        logger.error(f"Loaded: {len(new_libraries)} libraries, {len(new_documents)} documents, {len(new_chunks)} chunks")

    # Step 9: Test vector search functionality
    logger.info("Testing vector search functionality")
    vector_results = await new_content_service.vector_search(
        query_text="persistence testing",
        top_k=3
    )
    logger.info(f"Vector search returned {len(vector_results)} results")

    # Step 10: Test text search functionality
    logger.info("Testing text search functionality")
    text_results = await new_content_service.search(
        query="test chunk",
        indexer_type="suffix"
    )
    logger.info(f"Text search returned {len(text_results)} results")

    # Clean up test data directory
    if os.environ.get("KEEP_TEST_DATA", "false").lower() != "true":
        import shutil
        logger.info(f"Cleaning up test data directory: {data_dir}")
        shutil.rmtree(data_dir)

if __name__ == "__main__":
    asyncio.run(main())
