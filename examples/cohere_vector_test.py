#!/usr/bin/env python3
"""
Test script to demonstrate vector search using Cohere embeddings.
This script uses the Cohere API to generate embeddings for sample text chunks,
then stores them in our custom vector store for similarity search.
"""

import sys
import os
import numpy as np
import time
import asyncio
import logging
from typing import List, Dict, Any
import uuid
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.cohere_embedding_service import CohereEmbeddingService
from services.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the Cohere API key
COHERE_API_KEY = "A1Fi5KBBNoekwBPIa833CBScs6Z2mHEtOXxr52KO"

# Sample text chunks for testing
SAMPLE_CHUNKS = [
    "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
    "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
    "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
    "Generative AI refers to artificial intelligence systems capable of generating text, images, or other media in response to prompts.",
    "Vector databases are specialized database systems designed to store and query high-dimensional vectors efficiently.",
    "Embeddings are dense vector representations of data that capture semantic meaning, allowing for similarity comparisons.",
    "Transformer models have revolutionized natural language processing by using self-attention mechanisms to process sequential data.",
    "Large language models (LLMs) are neural networks trained on vast amounts of text data to generate human-like text and perform various language tasks."
]

# Categories for the sample chunks
CATEGORIES = [
    "machine_learning", "machine_learning", "natural_language_processing", 
    "computer_vision", "machine_learning", "generative_ai", 
    "vector_databases", "embeddings", "transformers", "large_language_models"
]

async def generate_cohere_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Generate embeddings using Cohere API.
    
    Args:
        texts: List of text chunks to embed
        
    Returns:
        List of embedding vectors
    """
    logger.info(f"Generating embeddings for {len(texts)} text chunks using Cohere API")
    
    # Initialize the Cohere embedding service
    embedding_service = CohereEmbeddingService(
        api_key=COHERE_API_KEY,
        model="embed-english-v3.0",
        dimension=1536
    )
    
    # Generate embeddings in batches
    batch_size = 5
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        start_time = time.time()
        batch_embeddings = embedding_service.generate_embeddings(batch)
        elapsed = time.time() - start_time
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Generated embeddings for batch {i//batch_size + 1}: {elapsed:.2f}s")
    
    # Verify embedding dimensions
    for i, embedding in enumerate(all_embeddings):
        if len(embedding) != 1536:
            logger.warning(f"Embedding {i} has incorrect dimension: {len(embedding)}")
    
    return all_embeddings

async def populate_vector_store(texts: List[str], embeddings: List[np.ndarray], categories: List[str]) -> VectorStore:
    """
    Populate a vector store with text chunks and their embeddings.
    
    Args:
        texts: List of text chunks
        embeddings: List of embedding vectors for the chunks
        categories: List of categories for the chunks
        
    Returns:
        Populated VectorStore
    """
    logger.info("Populating vector store with embeddings")
    
    # Initialize the vector store with the actual dimension of Cohere embeddings
    embedding_dimension = 1024
    vector_store = VectorStore(dimension=embedding_dimension, use_index=True, n_trees=10)
    
    # Add vectors to the store
    for i, (text, embedding, category) in enumerate(zip(texts, embeddings, categories)):
        # Create a unique ID for the chunk
        chunk_id = f"chunk_{i+1}"
        
        # Create metadata
        metadata = {
            "text": text,
            "category": category,
            "length": len(text),
            "created_at": time.time()
        }
        
        # Add to vector store
        await vector_store.upsert(
            id=chunk_id,
            vector=embedding.tolist(),
            metadata=metadata
        )
        
    logger.info(f"Added {len(texts)} vectors to the store")
    return vector_store

async def run_similarity_search(vector_store: VectorStore, query_text: str, query_embedding: np.ndarray, top_k: int = 3) -> None:
    """
    Run a similarity search and display results.
    
    Args:
        vector_store: Vector store to search in
        query_text: Query text for display
        query_embedding: Query embedding vector
        top_k: Number of results to return
    """
    logger.info(f"Running similarity search for: '{query_text}'")
    
    # Search the vector store
    results = await vector_store.search(
        query_vector=query_embedding,
        top_k=top_k
    )
    
    # Display results
    print(f"\nQuery: '{query_text}'")
    print(f"Top {top_k} results:")
    print("-" * 80)
    
    for i, result in enumerate(results["matches"]):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['metadata']['text']}")
        print(f"   Category: {result['metadata']['category']}")
        print()
    
    print("-" * 80)

async def run_category_search(vector_store: VectorStore, category: str, top_k: int = 3) -> None:
    """
    Run a search filtered by category.
    
    Args:
        vector_store: Vector store to search in
        category: Category to filter by
        top_k: Number of results to return
    """
    logger.info(f"Running category search for: '{category}'")
    
    # Create a random query vector with the correct dimension
    embedding_dimension = 1024  # Cohere embeddings are 1024-dimensional
    
    # Use numpy's modern random Generator API with fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    query_vector = rng.standard_normal(embedding_dimension)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Search the vector store with a category filter
    results = await vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
        filter={"category": category}
    )
    
    # Display results
    print(f"\nCategory filter: '{category}'")
    print(f"Found {len(results['matches'])} results:")
    print("-" * 80)
    
    for i, result in enumerate(results["matches"]):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['metadata']['text']}")
        print(f"   Category: {result['metadata']['category']}")
        print()
    
    print("-" * 80)

async def main():
    """Main function to run the test."""
    # Generate embeddings for sample chunks
    embeddings = await generate_cohere_embeddings(SAMPLE_CHUNKS)
    
    # Populate the vector store
    vector_store = await populate_vector_store(SAMPLE_CHUNKS, embeddings, CATEGORIES)
    
    # Generate embeddings for some test queries
    test_queries = [
        "How do neural networks work?",
        "What are vector embeddings used for?",
        "Explain computer vision applications"
    ]
    query_embeddings = await generate_cohere_embeddings(test_queries)
    
    # Run similarity searches
    for query, embedding in zip(test_queries, query_embeddings):
        await run_similarity_search(vector_store, query, embedding)
    
    # Run category searches
    for category in ["machine_learning", "natural_language_processing", "embeddings"]:
        await run_category_search(vector_store, category)

if __name__ == "__main__":
    asyncio.run(main())
