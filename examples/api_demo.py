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
    print(
        f"\n=== Text search for: '{query}' using {indexer_type} indexer ===\n"
    )

    url = f"{BASE_URL}/search/text"
    params = {
        "query": query,
        "indexer_type": indexer_type
    }

    response = requests.post(url, params=params)
    print_response(response, "Text search results")

    return response.json() if response.status_code == 200 else None


def vector_search(query_text, top_k=5):
    """Search for vectors using semantic similarity"""
    print(f"\n=== Vector search for: '{query_text}' ===\n")

    url = f"{BASE_URL}/search/vector"
    params = {
        "query_text": query_text,
        "top_k": top_k
    }

    response = requests.post(url, params=params)
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
    # Get initial statistics
    get_stats()
    # Sample texts for the demo - more diverse content for better differentiation
    texts = [
        # Technology category
        "Vector search is a technique used to find similar items in a dataset "
        "based on their vector representations. It enables semantic search capabilities.",
        "Embeddings are numerical representations of data that capture semantic meaning "
        "and allow machines to understand relationships between concepts.",
        "Thread safety ensures that concurrent operations don't lead to race "
        "conditions or data corruption in multi-threaded environments.",
        
        # Machine Learning category
        "Neural networks consist of layers of interconnected nodes that process "
        "and transform data to learn complex patterns and make predictions.",
        "Supervised learning algorithms learn from labeled training data to make "
        "predictions or decisions without being explicitly programmed to do so.",
        "Gradient descent is an optimization algorithm used to minimize the loss "
        "function by iteratively moving toward the steepest descent.",
        
        # Biology category
        "DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide "
        "chains that coil around each other to form a double helix.",
        "Photosynthesis is the process by which green plants and some other organisms "
        "use sunlight to synthesize nutrients from carbon dioxide and water.",
        "Cellular respiration is a set of metabolic reactions that take place in cells "
        "to convert biochemical energy from nutrients into ATP.",
        
        # History category
        "The Renaissance was a period in European history marking the transition "
        "from the Middle Ages to modernity and covering the 15th and 16th centuries.",
        "The Industrial Revolution was the transition to new manufacturing processes "
        "in Great Britain, continental Europe, and the United States.",
        "World War II was a global war that lasted from 1939 to 1945, involving "
        "the vast majority of the world's countries forming opposing alliances."
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

    # Technology category search
    print("\n--- Technology Category Search ---")
    vector_search("How does vector search work for finding similar items?")
    
    # Machine Learning category search
    print("\n--- Machine Learning Category Search ---")
    vector_search("Explain how neural networks process data", top_k=3)
    
    # Biology category search
    print("\n--- Biology Category Search ---")
    vector_search("Tell me about DNA structure and function", top_k=3)
    
    # History category search
    print("\n--- History Category Search ---")
    vector_search("What were the major events of World War II?", top_k=3)
    
    # Cross-category search
    print("\n--- Cross-Category Search ---")
    vector_search("How do algorithms process information?", top_k=5)

    # Delete operations
    print("\n\n=== DELETE OPERATIONS ===\n")

    # Delete a chunk
    if chunk_ids:
        delete_chunk(chunk_ids[0])

        # Verify deletion with search
        vector_search("vector search")

    # Get final statistics
    get_stats()

    print("\n\nDemo completed successfully!")



if __name__ == "__main__":
    run_demo()
