#!/usr/bin/env python3
"""
Persistence Demo for Vector Search System

This script demonstrates the persistence capabilities of our custom vector search system.
It shows how data is saved and loaded across different instances of the ContentService.

The demo:
1. Creates a ContentService with persistence enabled
2. Adds sample data (libraries, documents, chunks)
3. Performs vector and text searches
4. Shuts down the service
5. Creates a new ContentService instance
6. Loads the data from disk
7. Verifies that all data is correctly loaded
8. Performs the same searches to show functionality is preserved

Run this script to see the persistence mechanism in action.
"""

import asyncio
import os
import shutil
import time
import logging
from typing import Dict, List, Optional

import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.content_service import ContentService
from models.library import Library
from models.document import Document
from models.chunk import Chunk

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

async def create_sample_data(content_service: ContentService) -> Dict[str, str]:
    """Create sample data in the content service."""
    logger.info("Creating sample data...")
    
    # Create a library
    library = Library(name="AI Research")
    library_id = await content_service.create_library(library)
    
    # Create a document
    document = Document(name="AI Topics", library_id=library_id)
    document_id = await content_service.create_document(document)
    
    # Create chunks
    chunk_ids = []
    for i, text in enumerate(SAMPLE_TEXTS):
        chunk = Chunk(
            text=text,
            document_id=document_id,
            metadata={
                "topic": "AI",
                "subtopic": ["machine learning", "nlp", "computer vision", "reinforcement learning", "deep learning"][i],
                "importance": i + 1
            }
        )
        chunk_id = await content_service.create_chunk(chunk)
        chunk_ids.append(chunk_id)
    
    logger.info(f"Created 1 library, 1 document, and {len(chunk_ids)} chunks")
    
    return {
        "library_id": library_id,
        "document_id": document_id,
        "chunk_ids": chunk_ids
    }

async def perform_searches(content_service: ContentService):
    """Perform sample searches to demonstrate functionality."""
    # Vector search
    logger.info("Performing vector search for 'machine learning algorithms'...")
    vector_results = await content_service.vector_search(
        query_text="machine learning algorithms",
        top_k=3
    )
    logger.info(f"Vector search returned {len(vector_results)} results")
    for i, result in enumerate(vector_results):
        logger.info(f"  Result {i+1}: Score {result['score']:.4f}, Text: {result['text'][:50]}...")
    
    # Text search
    logger.info("Performing text search for 'learning'...")
    text_results = await content_service.text_search(
        query_text="learning",
        top_k=5
    )
    logger.info(f"Text search returned {len(text_results)} results")
    for i, result in enumerate(text_results):
        logger.info(f"  Result {i+1}: Text: {result['text'][:50]}...")

async def verify_data_integrity(content_service: ContentService, ids: Dict[str, str]):
    """Verify that all data is correctly loaded."""
    logger.info("Verifying data integrity...")
    
    # Check library
    libraries = await content_service.list_libraries()
    if len(libraries) != 1 or libraries[0]['id'] != ids['library_id']:
        logger.error("Library data integrity check failed")
        return False
    
    # Check document
    documents = await content_service.list_documents(ids['library_id'])
    if len(documents) != 1 or documents[0]['id'] != ids['document_id']:
        logger.error("Document data integrity check failed")
        return False
    
    # Check chunks
    chunks = await content_service.list_chunks(ids['document_id'])
    if len(chunks) != len(ids['chunk_ids']):
        logger.error("Chunk data integrity check failed")
        return False
    
    chunk_ids_set = set(ids['chunk_ids'])
    loaded_chunk_ids = {chunk['id'] for chunk in chunks}
    if chunk_ids_set != loaded_chunk_ids:
        logger.error("Chunk IDs don't match")
        return False
    
    logger.info("Data integrity verified - all data loaded correctly!")
    return True

async def main():
    """Main function to run the demo."""
    # Clean up any existing persistence data
    if os.path.exists(PERSISTENCE_DIR):
        logger.info(f"Cleaning up existing persistence directory: {PERSISTENCE_DIR}")
        shutil.rmtree(PERSISTENCE_DIR)
    
    # Step 1: Create first ContentService instance with persistence
    logger.info("=== STEP 1: Creating first ContentService instance ===")
    first_service = ContentService(
        persistence_enabled=True,
        persistence_dir=PERSISTENCE_DIR
    )
    
    # Step 2: Create sample data
    logger.info("=== STEP 2: Creating sample data ===")
    ids = await create_sample_data(first_service)
    
    # Step 3: Perform searches
    logger.info("=== STEP 3: Performing searches with first instance ===")
    await perform_searches(first_service)
    
    # Step 4: Manually trigger a snapshot
    logger.info("=== STEP 4: Manually triggering a snapshot ===")
    await first_service.persistence_service.create_snapshot(first_service)
    
    # Step 5: Wait for snapshot to complete
    logger.info("=== STEP 5: Waiting for snapshot to complete ===")
    time.sleep(2)
    
    # Step 6: Print summary of first service
    libraries = await first_service.list_libraries()
    documents = await first_service.list_documents(ids['library_id'])
    chunks = await first_service.list_chunks(ids['document_id'])
    logger.info(f"First service has {len(libraries)} libraries, {len(documents)} documents, and {len(chunks)} chunks")
    
    # Step 7: Simulate shutdown by creating a new ContentService
    logger.info("=== STEP 6: Simulating shutdown and restart ===")
    logger.info("Shutting down first ContentService instance...")
    # No explicit shutdown needed, just let it go out of scope
    
    # Step 8: Create a new ContentService instance
    logger.info("=== STEP 7: Creating new ContentService instance ===")
    new_service = ContentService(
        persistence_enabled=True,
        persistence_dir=PERSISTENCE_DIR
    )
    
    # Step 9: Load data from disk
    logger.info("=== STEP 8: Loading data from disk ===")
    success = await new_service.persistence_service.load_latest_snapshot(new_service)
    if success:
        logger.info("Successfully loaded data from disk")
    else:
        logger.error("Failed to load data from disk")
        return
    
    # Step 10: Verify data integrity
    logger.info("=== STEP 9: Verifying data integrity ===")
    integrity_check = await verify_data_integrity(new_service, ids)
    if not integrity_check:
        logger.error("Data integrity check failed")
        return
    
    # Step 11: Perform the same searches with the new instance
    logger.info("=== STEP 10: Performing searches with new instance ===")
    await perform_searches(new_service)
    
    # Step 12: Clean up
    logger.info("=== STEP 11: Cleaning up ===")
    if os.path.exists(PERSISTENCE_DIR):
        logger.info(f"Cleaning up persistence directory: {PERSISTENCE_DIR}")
        shutil.rmtree(PERSISTENCE_DIR)
    
    logger.info("=== Demo completed successfully! ===")

if __name__ == "__main__":
    asyncio.run(main())
