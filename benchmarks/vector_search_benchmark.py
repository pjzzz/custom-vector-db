import sys
import os
import time
import asyncio
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.content_service import ContentService
from services.embedding_service import EmbeddingService
from services.similarity_service import SimilarityService
from models import Chunk, Document, Library

class VectorSearchBenchmark:
    """Benchmark the performance of vector search vs. traditional text search."""

    def __init__(self, num_chunks=1000, chunk_size=100, vector_size=256):
        """
        Initialize the benchmark.

        Args:
            num_chunks: Number of chunks to create for testing
            chunk_size: Average number of words per chunk
            vector_size: Dimension of the embedding vectors
        """
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.vector_size = vector_size

        # Initialize content service with custom vector search
        self.content_service = ContentService(
            indexer_type='suffix',  # Use suffix array index for benchmarking
            embedding_dimension=self.vector_size
        )

        # Sample vocabulary for generating test data
        self.vocabulary = [
            "vector", "search", "embedding", "similarity", "cosine",
            "euclidean", "distance", "index", "query", "result",
            "thread", "safe", "lock", "concurrent", "parallel",
            "suffix", "array", "trie", "inverted", "index",
            "machine", "learning", "artificial", "intelligence", "natural",
            "language", "processing", "semantic", "syntactic", "token",
            "data", "structure", "algorithm", "performance", "efficiency",
            "scalability", "robustness", "reliability", "maintainability", "extensibility",
            "database", "storage", "retrieval", "insertion", "deletion",
            "update", "transaction", "consistency", "atomicity", "isolation",
            "durability", "persistence", "cache", "memory", "disk",
            "network", "latency", "throughput", "bandwidth", "protocol",
            "security", "encryption", "authentication", "authorization", "integrity",
            "availability", "fault", "tolerance", "recovery", "backup",
            "restore", "checkpoint", "logging", "monitoring", "alerting",
            "visualization", "analytics", "metrics", "statistics", "probability",
            "distribution", "mean", "median", "mode", "variance",
            "standard", "deviation", "correlation", "regression", "classification",
            "clustering", "dimensionality", "reduction", "feature", "extraction",
            "selection", "normalization", "standardization", "transformation", "augmentation"
        ]

        # Test queries for benchmarking
        self.test_queries = [
            "vector similarity search with thread safety",
            "machine learning algorithms for natural language processing",
            "efficient data structures for information retrieval",
            "concurrent operations in distributed systems",
            "performance optimization techniques for search engines",
            "semantic similarity between documents using embeddings",
            "fault tolerance and reliability in database systems",
            "scalability challenges in vector search implementations",
            "dimensionality reduction for high-dimensional data",
            "feature extraction and selection for text classification"
        ]

        # Results storage
        self.results = {
            "vector_search": {
                "times": [],
                "result_counts": []
            },
            "text_search": {
                "times": [],
                "result_counts": []
            }
        }

    def generate_random_text(self, length=None):
        """Generate random text for testing."""
        if length is None:
            length = self.chunk_size
        return " ".join(random.choices(self.vocabulary, k=length))

    def generate_random_id(self, prefix="chunk", length=8):
        """Generate a random ID for testing."""
        random_part = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))
        return f"{prefix}-{random_part}"

    async def setup_test_data(self):
        """Set up test data for benchmarking."""
        print(f"Setting up test data with {self.num_chunks} chunks...")

        # Fit the embedding model on sample texts
        print("Fitting embedding model...")
        sample_texts = [self.generate_random_text() for _ in range(100)]
        self.content_service.embedding_service.fit(sample_texts)

        # Create a library
        library = Library(
            id="benchmark-library",
            name="Benchmark Library",
            description="Library for benchmarking vector search",
            created_at=datetime.now()
        )
        await self.content_service.create_library(library)

        # Create a document
        document = Document(
            id="benchmark-document",
            library_id="benchmark-library",
            title="Benchmark Document",
            content="This is a document for benchmarking vector search",
            created_at=datetime.now(),
            metadata={"type": "benchmark"}
        )
        await self.content_service.create_document(document)

        # Create chunks
        print("Creating chunks...")
        for i in tqdm(range(self.num_chunks)):
            chunk = Chunk(
                id=self.generate_random_id(),
                document_id="benchmark-document",
                text=self.generate_random_text(),
                position=i,
                created_at=datetime.now(),
                metadata={"index": str(i)}
            )
            await self.content_service.create_chunk(chunk)

    async def run_vector_search_benchmark(self, query, top_k=10):
        """Run a vector search benchmark for a single query."""
        start_time = time.time()
        results = await self.content_service.vector_search(query, top_k=top_k)
        end_time = time.time()

        return {
            "time": end_time - start_time,
            "result_count": len(results)
        }

    async def run_text_search_benchmark(self, query, indexer_type='suffix'):
        """Run a text search benchmark for a single query."""
        start_time = time.time()
        results = await self.content_service.search(query, indexer_type=indexer_type)
        end_time = time.time()

        return {
            "time": end_time - start_time,
            "result_count": len(results)
        }

    async def run_benchmarks(self, num_iterations=5):
        """Run all benchmarks."""
        print(f"Running benchmarks with {num_iterations} iterations per query...")

        for query in self.test_queries:
            print(f"\nBenchmarking query: '{query}'")

            # Run vector search benchmarks
            print("Running vector search...")
            for i in tqdm(range(num_iterations)):
                result = await self.run_vector_search_benchmark(query)
                self.results["vector_search"]["times"].append(result["time"])
                self.results["vector_search"]["result_counts"].append(result["result_count"])

            # Run text search benchmarks
            print("Running text search...")
            for i in tqdm(range(num_iterations)):
                result = await self.run_text_search_benchmark(query)
                self.results["text_search"]["times"].append(result["time"])
                self.results["text_search"]["result_counts"].append(result["result_count"])

    def analyze_results(self):
        """Analyze benchmark results."""
        print("\n=== Benchmark Results ===")

        # Calculate statistics
        vector_times = self.results["vector_search"]["times"]
        text_times = self.results["text_search"]["times"]

        vector_result_counts = self.results["vector_search"]["result_counts"]
        text_result_counts = self.results["text_search"]["result_counts"]

        # Time statistics
        vector_avg_time = statistics.mean(vector_times)
        vector_median_time = statistics.median(vector_times)
        vector_min_time = min(vector_times)
        vector_max_time = max(vector_times)

        text_avg_time = statistics.mean(text_times)
        text_median_time = statistics.median(text_times)
        text_min_time = min(text_times)
        text_max_time = max(text_times)

        # Result count statistics
        vector_avg_results = statistics.mean(vector_result_counts)
        text_avg_results = statistics.mean(text_result_counts)

        # Print statistics
        print("\nTime Statistics (seconds):")
        print(f"Vector Search: Avg={vector_avg_time:.6f}, Median={vector_median_time:.6f}, Min={vector_min_time:.6f}, Max={vector_max_time:.6f}")
        print(f"Text Search:   Avg={text_avg_time:.6f}, Median={text_median_time:.6f}, Min={text_min_time:.6f}, Max={text_max_time:.6f}")

        print("\nResult Count Statistics:")
        print(f"Vector Search: Avg={vector_avg_results:.2f}")
        print(f"Text Search:   Avg={text_avg_results:.2f}")

        # Calculate speedup or slowdown
        if text_avg_time > 0:
            relative_performance = vector_avg_time / text_avg_time
            if relative_performance < 1:
                print(f"\nVector search is {1/relative_performance:.2f}x faster than text search")
            else:
                print(f"\nVector search is {relative_performance:.2f}x slower than text search")

        # Generate plots
        self.generate_plots()

    def generate_plots(self):
        """Generate plots for benchmark results."""
        # Time comparison plot
        plt.figure(figsize=(12, 6))

        # Box plot for time comparison
        plt.subplot(1, 2, 1)
        plt.boxplot([self.results["vector_search"]["times"], self.results["text_search"]["times"]],
                   labels=["Vector Search", "Text Search"])
        plt.title("Search Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.grid(True, linestyle='--', alpha=0.7)

        # Histogram of times
        plt.subplot(1, 2, 2)
        plt.hist(self.results["vector_search"]["times"], alpha=0.5, label="Vector Search")
        plt.hist(self.results["text_search"]["times"], alpha=0.5, label="Text Search")
        plt.title("Distribution of Search Times")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        print("\nPlots saved to benchmark_results.png")

async def main():
    """Main function to run the benchmark."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Vector Search Benchmark")
    parser.add_argument("--chunks", type=int, default=1000, help="Number of chunks to create")
    parser.add_argument("--size", type=int, default=100, help="Average words per chunk")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per query")
    args = parser.parse_args()

    # Create and run benchmark
    benchmark = VectorSearchBenchmark(num_chunks=args.chunks, chunk_size=args.size)
    await benchmark.setup_test_data()
    await benchmark.run_benchmarks(num_iterations=args.iterations)
    benchmark.analyze_results()

if __name__ == "__main__":
    asyncio.run(main())
