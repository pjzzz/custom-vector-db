#!/usr/bin/env python3
"""
Large-scale benchmark script to demonstrate the performance benefits of indexing
with larger datasets and more complex queries.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
import string
import sys
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import VectorStore
from services.vector_index import VectorIndex

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_random_vector(dim: int) -> np.ndarray:
    """Generate a random unit vector of specified dimension."""
    # Use numpy's modern random Generator API
    rng = np.random.default_rng()
    vec = rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def generate_random_id(length: int = 10) -> str:
    """Generate a random ID string."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_random_metadata(categories: List[str], sources: List[str]) -> Dict:
    """Generate random metadata for vectors."""
    return {
        "category": random.choice(categories),
        "source": random.choice(sources),
        "priority": random.randint(1, 5),
        "timestamp": time.time(),
        "tags": random.sample(["technical", "business", "science", "art", "history"], 
                             k=random.randint(1, 3))
    }


async def create_vector_store(vector_count: int, dimension: int = 1536) -> Tuple[VectorStore, VectorStore]:
    """
    Create and populate two vector stores with the same data.
    
    Args:
        vector_count: Number of vectors to generate
        dimension: Dimension of vectors
        
    Returns:
        Tuple of (indexed_store, non_indexed_store)
    """
    # Create vector stores
    indexed_store = VectorStore(dimension=dimension, use_index=True, n_trees=10)
    non_indexed_store = VectorStore(dimension=dimension, use_index=False)
    
    # Generate categories and sources for consistent metadata
    categories = ["technical", "business", "science", "art", "history", 
                 "sports", "politics", "entertainment", "health", "technology"]
    sources = ["document", "article", "book", "web", "video", 
              "podcast", "news", "blog", "social", "research"]
    
    # Generate random vectors
    logger.info(f"Generating {vector_count} random vectors of dimension {dimension}...")
    
    # Use batches for better performance with large datasets
    batch_size = 1000
    for i in tqdm(range(0, vector_count, batch_size)):
        batch = []
        batch_end = min(i + batch_size, vector_count)
        batch_size_actual = batch_end - i
        
        for j in range(batch_size_actual):
            vector_id = generate_random_id()
            vector = generate_random_vector(dimension)
            metadata = generate_random_metadata(categories, sources)
            batch.append({
                'id': vector_id,
                'values': vector.tolist(),
                'metadata': metadata
            })
        
        # Insert batch into both stores
        await indexed_store.bulk_upsert(batch)
        await non_indexed_store.bulk_upsert(batch)
    
    return indexed_store, non_indexed_store


async def benchmark_search_with_filters(
    indexed_store: VectorStore, 
    non_indexed_store: VectorStore,
    query_count: int = 10,
    top_k: int = 10,
    filter_complexity: str = "none"  # none, simple, complex
) -> Tuple[Dict, Dict]:
    """
    Benchmark search performance with and without indexing, using filters.
    
    Args:
        indexed_store: Vector store with indexing
        non_indexed_store: Vector store without indexing
        query_count: Number of queries to run
        top_k: Number of results to return per query
        filter_complexity: Type of filters to apply (none, simple, complex)
        
    Returns:
        Tuple of (indexed_results, non_indexed_results)
    """
    # Generate query vectors
    logger.info(f"Generating {query_count} query vectors...")
    query_vectors = [generate_random_vector(indexed_store.dimension) for _ in range(query_count)]
    
    # Define filters based on complexity
    filters = []
    if filter_complexity == "simple":
        filters = [
            {"category": "technical"},
            {"category": "business"},
            {"source": "document"},
            {"priority": 3},
            {"category": "science"}
        ]
    elif filter_complexity == "complex":
        filters = [
            {"category": "technical", "priority": 3},
            {"category": "business", "source": "article"},
            {"source": "document", "priority": 2},
            {"category": "science", "source": "research"},
            {"priority": 4, "source": "web"}
        ]
    else:
        filters = [None] * query_count
    
    # Ensure we have enough filters
    while len(filters) < query_count:
        filters.append(filters[0])
    
    # Benchmark indexed search
    logger.info(f"Benchmarking indexed search with {filter_complexity} filters...")
    indexed_times = []
    indexed_result_counts = []
    
    for i, (query_vector, filter_) in enumerate(zip(query_vectors, filters[:query_count])):
        start_time = time.time()
        result = await indexed_store.search(query_vector, top_k=top_k, filter=filter_)
        elapsed = time.time() - start_time
        indexed_times.append(elapsed)
        indexed_result_counts.append(len(result['matches']))
        logger.info(f"Indexed query {i+1}/{query_count}: {elapsed:.4f}s, {len(result['matches'])} results")
    
    # Benchmark non-indexed search
    logger.info(f"Benchmarking non-indexed search with {filter_complexity} filters...")
    non_indexed_times = []
    non_indexed_result_counts = []
    
    for i, (query_vector, filter_) in enumerate(zip(query_vectors, filters[:query_count])):
        start_time = time.time()
        result = await non_indexed_store.search(query_vector, top_k=top_k, filter=filter_)
        elapsed = time.time() - start_time
        non_indexed_times.append(elapsed)
        non_indexed_result_counts.append(len(result['matches']))
        logger.info(f"Non-indexed query {i+1}/{query_count}: {elapsed:.4f}s, {len(result['matches'])} results")
    
    # Calculate statistics
    indexed_avg = sum(indexed_times) / len(indexed_times)
    non_indexed_avg = sum(non_indexed_times) / len(non_indexed_times)
    speedup = non_indexed_avg / indexed_avg if indexed_avg > 0 else 0
    
    indexed_results = {
        "times": indexed_times,
        "avg_time": indexed_avg,
        "min_time": min(indexed_times),
        "max_time": max(indexed_times),
        "result_counts": indexed_result_counts
    }
    
    non_indexed_results = {
        "times": non_indexed_times,
        "avg_time": non_indexed_avg,
        "min_time": min(non_indexed_times),
        "max_time": max(non_indexed_times),
        "result_counts": non_indexed_result_counts
    }
    
    logger.info(f"Indexed search avg: {indexed_avg:.4f}s")
    logger.info(f"Non-indexed search avg: {non_indexed_avg:.4f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return indexed_results, non_indexed_results


def plot_results(indexed_results: Dict, non_indexed_results: Dict, 
                vector_count: int, filter_complexity: str) -> None:
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Bar chart of average times
    plt.subplot(2, 2, 1)
    methods = ['Indexed', 'Non-Indexed']
    avg_times = [indexed_results['avg_time'], non_indexed_results['avg_time']]
    bars = plt.bar(methods, avg_times)
    plt.ylabel('Average Time (s)')
    plt.title('Average Search Time Comparison')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}s',
                ha='center', va='bottom')
    
    # Plot 2: Box plot of time distributions
    plt.subplot(2, 2, 2)
    plt.boxplot([indexed_results['times'], non_indexed_results['times']], 
                labels=methods)
    plt.ylabel('Time (s)')
    plt.title('Search Time Distribution')
    
    # Plot 3: Line plot of individual query times
    plt.subplot(2, 1, 2)
    x = range(1, len(indexed_results['times']) + 1)
    plt.plot(x, indexed_results['times'], 'o-', label='Indexed')
    plt.plot(x, non_indexed_results['times'], 'o-', label='Non-Indexed')
    plt.xlabel('Query Number')
    plt.ylabel('Time (s)')
    plt.title('Individual Query Times')
    plt.legend()
    plt.grid(True)
    
    # Add speedup annotation
    speedup = non_indexed_results['avg_time'] / indexed_results['avg_time']
    plt.figtext(0.5, 0.01, 
                f'Speedup with indexing: {speedup:.2f}x faster\n'
                f'Vector count: {vector_count}, Filter: {filter_complexity}', 
                ha='center', fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename = f'benchmark_results_{vector_count}_{filter_complexity}.png'
    plt.savefig(filename)
    logger.info(f"Saved benchmark results to {filename}")


async def run_comprehensive_benchmark() -> None:
    """
    Run benchmarks with different vector counts and filter complexities.
    """
    vector_counts = [20000, 50000]
    filter_complexities = ["none", "simple", "complex"]
    
    for vector_count in vector_counts:
        # Create vector stores once for each vector count
        indexed_store, non_indexed_store = await create_vector_store(
            vector_count=vector_count,
            dimension=1536
        )
        
        # Test with different filter complexities
        for filter_complexity in filter_complexities:
            logger.info(f"Running benchmark with {vector_count} vectors and {filter_complexity} filters...")
            indexed_results, non_indexed_results = await benchmark_search_with_filters(
                indexed_store=indexed_store,
                non_indexed_store=non_indexed_store,
                query_count=5,
                top_k=10,
                filter_complexity=filter_complexity
            )
            
            # Plot results
            plot_results(
                indexed_results, 
                non_indexed_results, 
                vector_count, 
                filter_complexity
            )


async def main():
    """Main function to run benchmarks."""
    await run_comprehensive_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
