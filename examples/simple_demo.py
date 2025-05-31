#!/usr/bin/env python3
"""
Simple demo script for interacting with the Vector Search API.
"""

import requests
import json
import time
import uuid

# API endpoint
BASE_URL = "http://127.0.0.1:8000"


def print_response(response, message="Response"):
    """Print formatted API response"""
    print(f"\n{message}:")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print("-" * 50)


def run_demo():
    """Run a simple demo of the vector search API"""
    # 1. Create a library
    print("\n=== Creating a library ===")
    library_id = f"lib_{uuid.uuid4().hex[:8]}"
    library_data = {
        "id": library_id,
        "name": "Demo Library",
        "description": "A demo library for testing the vector search API"
    }
    response = requests.post(f"{BASE_URL}/libraries", json=library_data)
    print_response(response, "Library creation")

    if response.status_code != 200:
        print("Failed to create library. Exiting demo.")
        return

    # 2. Create a document
    print("\n=== Creating a document ===")
    document_id = f"doc_{uuid.uuid4().hex[:8]}"
    document_data = {
        "id": document_id,
        "title": "Demo Document",
        "library_id": library_id,
        "metadata": {
            "author": "Demo User",
            "category": "technical"
        }
    }
    response = requests.post(f"{BASE_URL}/documents", json=document_data)
    print_response(response, "Document creation")

    if response.status_code != 200:
        print("Failed to create document. Exiting demo.")
        return

    # 3. Create chunks
    print("\n=== Creating chunks ===")
    sample_texts = [
        "Vector search is a technique used to find similar items in a dataset based on their vector representations.",
        "Embeddings are numerical representations of data that capture semantic meaning.",
        "Thread safety ensures that concurrent operations don't lead to race conditions or data corruption.",
        "Cosine similarity measures the cosine of the angle between two vectors, indicating their similarity.",
        "Dimensionality reduction techniques like random projection help manage high-dimensional data efficiently."
    ]

    chunk_ids = []
    for i, text in enumerate(sample_texts):
        chunk_data = {
            "id": f"chunk_{uuid.uuid4().hex[:8]}",
            "text": text,
            "document_id": document_id,
            "position": i,
            "metadata": {
                "source": "demo",
                "index": str(i)
            }
        }
        response = requests.post(f"{BASE_URL}/chunks", json=chunk_data)
        print_response(response, f"Chunk {i+1} creation")

        if response.status_code == 200:
            chunk_ids.append(response.json().get("id"))

    if not chunk_ids:
        print("Failed to create any chunks. Exiting demo.")
        return

    # Wait a moment for embeddings to be processed
    print("\nWaiting for embeddings to be processed...")
    time.sleep(2)

    # 4. Get statistics
    print("\n=== Getting statistics ===")
    response = requests.get(f"{BASE_URL}/stats")
    print_response(response, "Vector store statistics")

    # 5. Search for similar vectors
    print("\n=== Searching for similar vectors ===")
    search_queries = [
        "What is vector search?",
        "How do embeddings work?",
        "Tell me about thread safety"
    ]

    for query in search_queries:
        search_data = {
            "query": query,
            "top_k": 3,
            "filter": {"source": "demo"}
        }
        response = requests.post(f"{BASE_URL}/search", json=search_data)
        print_response(response, f"Search results for '{query}'")

    # 6. Get an embedding
    print("\n=== Getting an embedding ===")
    embedding_data = {
        "text": "This is a test embedding"
    }
    response = requests.post(f"{BASE_URL}/embedding", json=embedding_data)
    print_response(response, "Embedding")

    if response.status_code == 200:
        embedding = response.json().get("embedding")

        # 7. Upsert a vector directly
        print("\n=== Upserting a vector directly ===")
        vector_id = f"vector_{uuid.uuid4().hex[:8]}"
        upsert_data = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "source": "direct_upsert",
                "description": "A test vector"
            }
        }
        response = requests.post(f"{BASE_URL}/upsert", json=upsert_data)
        print_response(response, "Vector upsert")

        # 8. Search for the upserted vector
        print("\n=== Searching for the upserted vector ===")
        search_data = {
            "query": "This is a test embedding",
            "top_k": 1,
            "filter": {"source": "direct_upsert"}
        }
        response = requests.post(f"{BASE_URL}/search", json=search_data)
        print_response(response, "Search results for upserted vector")

        # 9. Delete the upserted vector
        print("\n=== Deleting the upserted vector ===")
        delete_data = {
            "ids": [vector_id]
        }
        response = requests.post(f"{BASE_URL}/delete", json=delete_data)
        print_response(response, "Vector deletion")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
