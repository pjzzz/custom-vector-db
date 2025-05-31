#!/usr/bin/env python3
"""
Simple Persistence Demo for Vector Search System

This script demonstrates the persistence capabilities of our custom vector search system
using a simplified approach that doesn't rely on the complex model structures.

The demo:
1. Creates a ContentService with persistence enabled
2. Adds sample data directly through the ContentService API
3. Performs vector and text searches
4. Shuts down the service
5. Creates a new ContentService instance
6. Loads the data from disk
7. Verifies that all data is correctly loaded
8. Performs the same searches to show functionality is preserved
"""

import asyncio
import os
import shutil
import time
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PERSISTENCE_DIR = "./demo_persistence_data"
SAMPLE_TEXTS = [
    "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
    "Natural language processing (NLP) is a field of computer science, artificial intelligence concerned with the interactions between computers and human language.",
    "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
    "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning."
]

async def run_demo():
    """Run the persistence demo."""
    # Clean up any existing persistence data
    if os.path.exists(PERSISTENCE_DIR):
        logger.info(f"Cleaning up existing persistence directory: {PERSISTENCE_DIR}")
        shutil.rmtree(PERSISTENCE_DIR)

    # Import here to avoid circular imports
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.content_service import ContentService
    from models import Library, Document, Chunk

    # Step 1: Create first ContentService instance with persistence
    logger.info("=== STEP 1: Creating first ContentService instance ===")
    first_service = ContentService(
        enable_persistence=True,
        data_dir=PERSISTENCE_DIR
    )

    # Step 2: Create sample data
    logger.info("=== STEP 2: Creating sample data ===")

    # Create a library
    library_id = str(uuid.uuid4())
    library = Library(
        id=library_id,
        name="AI Research",
        description="Research papers and articles about AI"
    )
    await first_service.create_library(library)
    logger.info(f"Created library with ID: {library_id}")

    # Create a document
    document_id = str(uuid.uuid4())
    document = Document(
        id=document_id,
        title="AI Topics",
        library_id=library_id,
        metadata={}
    )
    await first_service.create_document(document)
    logger.info(f"Created document with ID: {document_id}")

    # Create chunks
    chunk_ids = []
    for i, text in enumerate(SAMPLE_TEXTS):
        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            id=chunk_id,
            text=text,
            document_id=document_id,
            position=i,
            metadata={
                "topic": "AI",
                "subtopic": ["machine learning", "nlp", "computer vision", "reinforcement learning", "deep learning"][i],
                "importance": str(i + 1)  # Convert to string as metadata expects string values
            }
        )
        await first_service.create_chunk(chunk)
        chunk_ids.append(chunk_id)
        logger.info(f"Created chunk {i+1} with ID: {chunk_id}")

    # Step 3: Perform searches
    logger.info("=== STEP 3: Performing searches with first instance ===")

    # Vector search
    logger.info("Performing vector search for 'machine learning algorithms'...")
    vector_results = await first_service.vector_search(
        query_text="machine learning algorithms",
        top_k=3
    )
    logger.info(f"Vector search returned {len(vector_results)} results")
    for i, result in enumerate(vector_results):
        logger.info(f"  Result {i+1}: Score {result.get('score', 0):.4f}, Text: {result.get('text', '')[:50]}...")

    # Text search (if available)
    try:
        logger.info("Performing text search for 'learning'...")
        text_results = await first_service.search(
            query_text="learning",
            top_k=5
        )
        logger.info(f"Text search returned {len(text_results)} results")
        for i, result in enumerate(text_results):
            logger.info(f"  Result {i+1}: Text: {result.get('text', '')[:50]}...")
    except Exception as e:
        logger.warning(f"Text search not available: {str(e)}")

    # Step 4: Manually trigger a snapshot
    logger.info("=== STEP 4: Manually triggering a snapshot ===")
    await first_service.persistence_service.create_snapshot(first_service)

    # Step 5: Wait for snapshot to complete
    logger.info("=== STEP 5: Waiting for snapshot to complete ===")
    time.sleep(2)

    # Step 6: Simulate shutdown by creating a new ContentService
    logger.info("=== STEP 6: Simulating shutdown and restart ===")
    logger.info("Shutting down first ContentService instance...")
    # No explicit shutdown needed, just let it go out of scope

    # Step 7: Create a new ContentService instance
    logger.info("=== STEP 7: Creating new ContentService instance ===")
    new_service = ContentService(
        enable_persistence=True,
        data_dir=PERSISTENCE_DIR
    )

    # Step 8: Load data from disk
    logger.info("=== STEP 8: Loading data from disk ===")
    success = await new_service.persistence_service.load_latest_snapshot(new_service)
    if success:
        logger.info("Successfully loaded data from disk")
    else:
        logger.error("Failed to load data from disk")
        return

    # Step 9: Verify data integrity
    logger.info("=== STEP 9: Verifying data integrity ===")

    # Check if libraries were loaded
    libraries = new_service.content_store.get('libraries', {})
    logger.info(f"Loaded {len(libraries)} libraries")
    if library_id in libraries:
        logger.info(f"Found library with ID: {library_id}")
    else:
        logger.error(f"Library with ID {library_id} not found!")

    # Check if documents were loaded
    documents = new_service.content_store.get('documents', {})
    logger.info(f"Loaded {len(documents)} documents")
    if document_id in documents:
        logger.info(f"Found document with ID: {document_id}")
    else:
        logger.error(f"Document with ID {document_id} not found!")

    # Check if chunks were loaded
    chunks = new_service.content_store.get('chunks', {})
    logger.info(f"Loaded {len(chunks)} chunks")
    for chunk_id in chunk_ids:
        if chunk_id in chunks:
            logger.info(f"Found chunk with ID: {chunk_id}")
        else:
            logger.error(f"Chunk with ID {chunk_id} not found!")

    # Step 10: Perform the same searches with the new instance
    logger.info("=== STEP 10: Performing searches with new instance ===")

    # Vector search
    logger.info("Performing vector search for 'machine learning algorithms'...")
    vector_results = await new_service.vector_search(
        query_text="machine learning algorithms",
        top_k=3
    )
    logger.info(f"Vector search returned {len(vector_results)} results")
    for i, result in enumerate(vector_results):
        logger.info(f"  Result {i+1}: Score {result.get('score', 0):.4f}, Text: {result.get('text', '')[:50]}...")

    # Text search (if available)
    try:
        logger.info("Performing text search for 'learning'...")
        text_results = await new_service.search(
            query_text="learning",
            top_k=5
        )
        logger.info(f"Text search returned {len(text_results)} results")
        for i, result in enumerate(text_results):
            logger.info(f"  Result {i+1}: Text: {result.get('text', '')[:50]}...")
    except Exception as e:
        logger.warning(f"Text search not available: {str(e)}")

    # Step 11: Clean up
    logger.info("=== STEP 11: Cleaning up ===")
    if os.path.exists(PERSISTENCE_DIR):
        logger.info(f"Cleaning up persistence directory: {PERSISTENCE_DIR}")
        shutil.rmtree(PERSISTENCE_DIR)

    logger.info("=== Demo completed successfully! ===")

async def main():
    """Main function."""
    await run_demo()

if __name__ == "__main__":
    asyncio.run(main())
