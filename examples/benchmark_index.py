#!/usr/bin/env python3
"""
Benchmark script to compare performance between indexed and non-indexed vector search.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
import string
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import VectorStore
from services.vector_index import VectorIndex
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_random_vector(dim: int) -> np.ndarray:
    """Generate a random unit vector of specified dimension."""
    vec = np.random.normal(0, 1, dim)
    return vec / np.linalg.norm(vec)


def generate_random_id(length: int = 10) -> str:
    """Generate a random ID string."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_random_metadata() -> Dict:
    """Generate random metadata for vectors."""
    categories = ["technical", "business", "science", "art", "history"]
    sources = ["document", "article", "book", "web", "video"]
    
    return {
        "category": random.choice(categories),
        "source": random.choice(sources),
        "priority": random.randint(1, 5),
        "timestamp": time.time()
    }


async def benchmark_search(
    vector_count: int, 
    dimension: int = 1536, 
    query_count: int = 10,
    top_k: int = 10
) -> Tuple[Dict, Dict]:
    """
    Benchmark search performance with and without indexing.
    
    Args:
        vector_count: Number of vectors to generate
        dimension: Dimension of vectors
        query_count: Number of queries to run
        top_k: Number of results to return per query
        
    Returns:
        Tuple of (indexed_results, non_indexed_results)
    """
    # Create vector stores
    indexed_store = VectorStore(dimension=dimension, use_index=True, n_trees=10)
    non_indexed_store = VectorStore(dimension=dimension, use_index=False)
    
    # Generate random vectors
    logger.info(f"Generating {vector_count} random vectors of dimension {dimension}...")
    vectors = []
    for _ in range(vector_count):
        vector_id = generate_random_id()
        vector = generate_random_vector(dimension)
        metadata = generate_random_metadata()
        vectors.append((vector_id, vector, metadata))
    
    # Insert vectors into both stores
    logger.info("Inserting vectors into stores...")
    for vector_id, vector, metadata in vectors:
        await indexed_store.upsert(vector_id, vector.tolist(), metadata)
        await non_indexed_store.upsert(vector_id, vector.tolist(), metadata)
    
    # Generate query vectors
    logger.info(f"Generating {query_count} query vectors...")
    query_vectors = [generate_random_vector(dimension) for _ in range(query_count)]
    
    # Benchmark indexed search
    logger.info("Benchmarking indexed search...")
    indexed_times = []
    for i, query_vector in enumerate(query_vectors):
        start_time = time.time()
        result = await indexed_store.search(query_vector, top_k=top_k)
        elapsed = time.time() - start_time
        indexed_times.append(elapsed)
        logger.info(f"Indexed query {i+1}/{query_count}: {elapsed:.4f}s, {len(result['matches'])} results")
    
    # Benchmark non-indexed search
    logger.info("Benchmarking non-indexed search...")
    non_indexed_times = []
    for i, query_vector in enumerate(query_vectors):
        start_time = time.time()
        result = await non_indexed_store.search(query_vector, top_k=top_k)
        elapsed = time.time() - start_time
        non_indexed_times.append(elapsed)
        logger.info(f"Non-indexed query {i+1}/{query_count}: {elapsed:.4f}s, {len(result['matches'])} results")
    
    # Calculate statistics
    indexed_avg = sum(indexed_times) / len(indexed_times)
    non_indexed_avg = sum(non_indexed_times) / len(non_indexed_times)
    speedup = non_indexed_avg / indexed_avg if indexed_avg > 0 else 0
    
    indexed_results = {
        "times": indexed_times,
        "avg_time": indexed_avg,
        "min_time": min(indexed_times),
        "max_time": max(indexed_times)
    }
    
    non_indexed_results = {
        "times": non_indexed_times,
        "avg_time": non_indexed_avg,
        "min_time": min(non_indexed_times),
        "max_time": max(non_indexed_times)
    }
    
    logger.info(f"Indexed search avg: {indexed_avg:.4f}s")
    logger.info(f"Non-indexed search avg: {non_indexed_avg:.4f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return indexed_results, non_indexed_results


def plot_results(indexed_results: Dict, non_indexed_results: Dict, vector_count: int) -> None:
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
    plt.figtext(0.5, 0.01, f'Speedup with indexing: {speedup:.2f}x faster\nVector count: {vector_count}', 
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('benchmark_results.png')
    logger.info("Saved benchmark results to benchmark_results.png")
    plt.show()


async def run_scaling_benchmark(
    dimensions: List[int] = [1536],
    vector_counts: List[int] = [1000, 5000, 10000, 20000],
    query_count: int = 5,
    top_k: int = 10
) -> None:
    """
    Run benchmarks with different vector counts to measure scaling.
    
    Args:
        dimensions: List of vector dimensions to test
        vector_counts: List of vector counts to test
        query_count: Number of queries per test
        top_k: Number of results to return per query
    """
    results = []
    
    for dimension in dimensions:
        dimension_results = []
        
        for vector_count in vector_counts:
            logger.info(f"Running benchmark with {vector_count} vectors of dimension {dimension}...")
            indexed_results, non_indexed_results = await benchmark_search(
                vector_count=vector_count,
                dimension=dimension,
                query_count=query_count,
                top_k=top_k
            )
            
            speedup = non_indexed_results['avg_time'] / indexed_results['avg_time']
            dimension_results.append({
                'vector_count': vector_count,
                'indexed_avg': indexed_results['avg_time'],
                'non_indexed_avg': non_indexed_results['avg_time'],
                'speedup': speedup
            })
            
            # Plot individual results
            plot_results(indexed_results, non_indexed_results, vector_count)
        
        results.append({
            'dimension': dimension,
            'results': dimension_results
        })
    
    # Plot scaling results
    plt.figure(figsize=(12, 8))
    
    for dim_result in results:
        dimension = dim_result['dimension']
        x = [r['vector_count'] for r in dim_result['results']]
        y_indexed = [r['indexed_avg'] for r in dim_result['results']]
        y_non_indexed = [r['non_indexed_avg'] for r in dim_result['results']]
        y_speedup = [r['speedup'] for r in dim_result['results']]
        
        plt.subplot(2, 1, 1)
        plt.plot(x, y_indexed, 'o-', label=f'Indexed (dim={dimension})')
        plt.plot(x, y_non_indexed, 's-', label=f'Non-Indexed (dim={dimension})')
        plt.xlabel('Vector Count')
        plt.ylabel('Average Search Time (s)')
        plt.title('Search Time vs Vector Count')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(x, y_speedup, 'D-', label=f'Dimension {dimension}')
        plt.xlabel('Vector Count')
        plt.ylabel('Speedup Factor')
        plt.title('Speedup vs Vector Count')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('scaling_benchmark_results.png')
    logger.info("Saved scaling benchmark results to scaling_benchmark_results.png")
    plt.show()


async def main():
    """Main function to run benchmarks."""
    # Run scaling benchmark with different vector counts
    await run_scaling_benchmark(
        dimensions=[1536],
        vector_counts=[1000, 5000, 10000],
        query_count=5,
        top_k=10
    )


if __name__ == "__main__":
    asyncio.run(main())
