#!/usr/bin/env python3
"""
Demo script for interacting with the Vector Search API.
This demonstrates how to use the API for various operations.
"""


import requests
import json
import time


# API endpoint
BASE_URL = "http://127.0.0.1:8888"


def print_response(response, message="Response"):
    """Print formatted API response"""
    print(f"\n{message}:")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print("-" * 50)


def create_library():
    """Create a new library"""
    print("\n=== Creating a new library ===\n")

    url = f"{BASE_URL}/libraries"
    payload = {
        "id": f"lib_{int(time.time())}",
        "name": "Demo Library",
        "description": "Library for API demo"
    }

    response = requests.post(url, json=payload)
    print_response(response, "Library creation")

    if response.status_code == 200:
        return response.json()["library"]["id"]
    return None


def create_document(library_id):
    """Create a new document in the library"""
    print("\n=== Creating a new document ===\n")

    url = f"{BASE_URL}/documents"
    payload = {
        "id": f"doc_{int(time.time())}",
        "title": "Demo Document",
        "library_id": library_id,
        "metadata": {
            "author": "API Demo",
            "category": "technical"
        }
    }

    response = requests.post(url, json=payload)
    print_response(response, "Document creation")

    if response.status_code == 200:
        return response.json()["document"]["id"]
    return None


def create_chunks(document_id, texts):
    """Create chunks from the given texts"""
    print("\n=== Creating chunks ===\n")

    chunk_ids = []
    for i, text in enumerate(texts):
        url = f"{BASE_URL}/chunks"
        payload = {
            "id": f"chunk_{int(time.time())}_{i}",
            "text": text,
            "document_id": document_id,
            "position": i,
            "metadata": {
                "source": "api_demo",
                "index": str(i)
            }
        }

        response = requests.post(url, json=payload)
        print_response(response, f"Chunk {i+1} creation")

        if response.status_code == 200:
            chunk_ids.append(response.json()["chunk"]["id"])

    return chunk_ids


def search_text(query, indexer_type="suffix"):
    """Search for text using the specified indexer"""
    print(f"\n=== Text search for: '{query}' using {indexer_type} indexer ===\n")

    url = f"{BASE_URL}/search/text"
    payload = {
        "query": query,
        "indexer_type": indexer_type
    }

    response = requests.post(url, json=payload)
    print_response(response, "Text search results")

    return response.json() if response.status_code == 200 else None


def vector_search(query_text, top_k=5):
    """Search for vectors using semantic similarity"""
    print(f"\n=== Vector search for: '{query_text}' ===\n")

    url = f"{BASE_URL}/search/vector"
    payload = {
        "query_text": query_text,
        "top_k": top_k
    }

    response = requests.post(url, json=payload)
    print_response(response, "Vector search results")

    return response.json() if response.status_code == 200 else None


def delete_chunk(chunk_id):
    """Delete a chunk by ID"""
    print(f"\n=== Deleting chunk: {chunk_id} ===\n")

    url = f"{BASE_URL}/chunks/{chunk_id}"
    response = requests.delete(url)
    print_response(response, "Delete result")

    return response.status_code == 200


def get_stats():
    """Get statistics about the content service"""
    print("\n=== Getting system statistics ===\n")

    url = f"{BASE_URL}/stats"
    response = requests.get(url)
    print_response(response, "System statistics")

    return response.json() if response.status_code == 200 else None


def run_demo():
    """Run the complete API demo"""
    # Sample texts for the demo
    texts = [
        "Vector search is a technique used to find similar items in a dataset based on their vector representations.",
        "Embeddings are numerical representations of data that capture semantic meaning.",
        "Thread safety ensures that concurrent operations don't lead to race conditions or data corruption.",
        "Cosine similarity measures the cosine of the angle between two vectors, indicating their similarity.",
        "Dimensionality reduction techniques like random projection help manage high-dimensional data efficiently."
    ]

    # Create library and document
    library_id = create_library()
    if not library_id:
        print("Failed to create library. Exiting demo.")
        return

    document_id = create_document(library_id)
    if not document_id:
        print("Failed to create document. Exiting demo.")
        return

    # Create chunks
    chunk_ids = create_chunks(document_id, texts)
    if not chunk_ids:
        print("Failed to create chunks. Exiting demo.")
        return

    # Wait for embeddings to be processed
    print("\nWaiting for embeddings to be processed...")
    time.sleep(2)

    # Get statistics
    get_stats()

    # Search examples
    print("\n\n=== SEARCH EXAMPLES ===\n")

    search_text("search", indexer_type="suffix")

    # Vector search examples
    print("\n\n=== VECTOR SEARCH EXAMPLES ===\n")

    # Basic vector search
    vector_search("What is vector search?")

    # Vector search with higher top_k
    vector_search("similarity between vectors", top_k=3)

    # Delete operations
    print("\n\n=== DELETE OPERATIONS ===\n")

    # Delete a chunk
    if chunk_ids:
        delete_chunk(chunk_ids[0])

        # Verify deletion with search
        vector_search("vector search")

    print("\n\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()
