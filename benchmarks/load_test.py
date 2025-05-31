import sys
import os
import time
import asyncio
import random
import uuid
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.client import VectorSearchClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorSearchLoadTest:
    """Load testing for the Vector Search API."""

    def __init__(self, base_url="http://localhost:8000", num_libraries=2,
                 num_documents=5, num_chunks=10):
        """
        Initialize the load test.

        Args:
            base_url: Base URL of the API
            num_libraries: Number of libraries to create
            num_documents: Number of documents per library
            num_chunks: Number of chunks per document
        """
        self.client = VectorSearchClient(base_url=base_url)
        self.num_libraries = num_libraries
        self.num_documents = num_documents
        self.num_chunks = num_chunks

        # Store created resources for cleanup
        self.libraries = []
        self.documents = []
        self.chunks = []

        # Sample vocabulary for generating test data
        self.vocabulary = [
            "vector", "search", "embedding", "similarity", "cosine",
            "euclidean", "distance", "index", "query", "result",
            "thread", "safe", "lock", "concurrent", "parallel",
            "suffix", "array", "trie", "inverted", "index",
            "machine", "learning", "artificial", "intelligence", "natural",
            "language", "processing", "semantic", "syntactic", "token",
            "data", "structure", "algorithm", "performance", "efficiency",
            "scalability", "robustness", "reliability", "maintainability", "extensibility"
        ]

        # Test queries
        self.queries = [
            "vector similarity search",
            "embedding techniques for text",
            "efficient data structures",
            "thread safety in concurrent systems",
            "content management and organization"
        ]

        # Results storage
        self.results = {
            "concurrent_users": [],
            "vector_search_latency": [],
            "text_search_latency": [],
            "create_latency": [],
            "delete_latency": [],
            "success_rate": []
        }

    def generate_random_text(self, length=50):
        """Generate random text for testing."""
        return " ".join(random.choices(self.vocabulary, k=length))

    def generate_random_id(self, prefix="test", length=8):
        """Generate a random ID for testing."""
        return f"{prefix}-{uuid.uuid4().hex[:length]}"

    async def setup_test_data(self):
        """Set up test data for load testing."""
        logger.info(f"Setting up test data with {self.num_libraries} libraries, "
                   f"{self.num_documents} documents per library, and "
                   f"{self.num_chunks} chunks per document...")

        # Create libraries
        for i in range(self.num_libraries):
            library_id = self.generate_random_id("lib")
            library = self.client.create_library(
                id=library_id,
                name=f"Test Library {i+1}",
                description=f"Library for load testing {i+1}"
            )
            self.libraries.append(library_id)

            # Create documents for this library
            for j in range(self.num_documents):
                document_id = self.generate_random_id("doc")
                document = self.client.create_document(
                    id=document_id,
                    library_id=library_id,
                    title=f"Test Document {j+1}",
                    content=f"This is test document {j+1} in library {i+1}",
                    metadata={"library_index": str(i), "document_index": str(j)}
                )
                self.documents.append(document_id)

                # Create chunks for this document
                for k in range(self.num_chunks):
                    chunk_id = self.generate_random_id("chunk")
                    chunk = self.client.create_chunk(
                        id=chunk_id,
                        document_id=document_id,
                        text=self.generate_random_text(),
                        position=k,
                        metadata={"library_index": str(i), "document_index": str(j), "chunk_index": str(k)}
                    )
                    self.chunks.append(chunk_id)

        total_chunks = len(self.chunks)
        logger.info(f"Created {total_chunks} chunks for load testing")

        # Wait for the embeddings to be processed
        logger.info("Waiting for embeddings to be processed...")
        time.sleep(2)

    async def cleanup_test_data(self):
        """Clean up test data after load testing."""
        logger.info("Cleaning up test data...")

        # Delete chunks
        for chunk_id in self.chunks:
            try:
                self.client.delete_chunk(chunk_id)
            except Exception as e:
                logger.warning(f"Failed to delete chunk {chunk_id}: {str(e)}")

        # Delete documents
        for document_id in self.documents:
            try:
                self.client.delete_document(document_id)
            except Exception as e:
                logger.warning(f"Failed to delete document {document_id}: {str(e)}")

        # Delete libraries
        for library_id in self.libraries:
            try:
                self.client.delete_library(library_id)
            except Exception as e:
                logger.warning(f"Failed to delete library {library_id}: {str(e)}")

        logger.info("Cleanup complete")

    def worker_vector_search(self):
        """Worker function for vector search."""
        query = random.choice(self.queries)
        start_time = time.time()
        success = False

        try:
            results = self.client.vector_search(
                query_text=query,
                top_k=5
            )
            success = True
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")

        end_time = time.time()
        latency = end_time - start_time

        return {"latency": latency, "success": success}

    def worker_text_search(self):
        """Worker function for text search."""
        query = random.choice(self.queries)
        indexer_type = random.choice(["suffix", "trie", "inverted"])
        start_time = time.time()
        success = False

        try:
            results = self.client.text_search(
                query=query,
                indexer_type=indexer_type
            )
            success = True
        except Exception as e:
            logger.error(f"Text search failed: {str(e)}")

        end_time = time.time()
        latency = end_time - start_time

        return {"latency": latency, "success": success}

    def worker_create_chunk(self):
        """Worker function for creating a chunk."""
        if not self.documents:
            return {"latency": 0, "success": False}

        document_id = random.choice(self.documents)
        chunk_id = self.generate_random_id("chunk")
        start_time = time.time()
        success = False

        try:
            chunk = self.client.create_chunk(
                id=chunk_id,
                document_id=document_id,
                text=self.generate_random_text(),
                position=random.randint(0, 100),
                metadata={"type": "load_test", "timestamp": datetime.now().isoformat()}
            )
            success = True
            self.chunks.append(chunk_id)
        except Exception as e:
            logger.error(f"Create chunk failed: {str(e)}")

        end_time = time.time()
        latency = end_time - start_time

        return {"latency": latency, "success": success}

    def worker_delete_chunk(self):
        """Worker function for deleting a chunk."""
        if not self.chunks:
            return {"latency": 0, "success": False}

        chunk_id = random.choice(self.chunks)
        start_time = time.time()
        success = False

        try:
            result = self.client.delete_chunk(chunk_id)
            success = True
            # Remove from our list
            if chunk_id in self.chunks:
                self.chunks.remove(chunk_id)
        except Exception as e:
            logger.error(f"Delete chunk failed: {str(e)}")

        end_time = time.time()
        latency = end_time - start_time

        return {"latency": latency, "success": success}

    async def run_load_test(self, concurrent_users, duration=30,
                           operation_mix=None):
        """
        Run a load test with the specified number of concurrent users.

        Args:
            concurrent_users: Number of concurrent users
            duration: Duration of the test in seconds
            operation_mix: Dict with operation mix percentages
                (vector_search, text_search, create, delete)
        """
        if operation_mix is None:
            operation_mix = {
                "vector_search": 40,
                "text_search": 40,
                "create": 10,
                "delete": 10
            }

        logger.info(f"Running load test with {concurrent_users} concurrent users "
                   f"for {duration} seconds")

        # Create a thread pool
        executor = ThreadPoolExecutor(max_workers=concurrent_users)

        # Results for this test
        vector_search_latencies = []
        text_search_latencies = []
        create_latencies = []
        delete_latencies = []
        success_count = 0
        total_count = 0

        # Run the test for the specified duration
        start_time = time.time()
        end_time = start_time + duration

        with tqdm(total=duration, desc=f"Load test progress") as pbar:
            while time.time() < end_time:
                # Submit tasks to the thread pool
                futures = []
                for _ in range(concurrent_users):
                    # Choose an operation based on the mix
                    operation = random.choices(
                        ["vector_search", "text_search", "create", "delete"],
                        weights=[
                            operation_mix["vector_search"],
                            operation_mix["text_search"],
                            operation_mix["create"],
                            operation_mix["delete"]
                        ]
                    )[0]

                    if operation == "vector_search":
                        future = executor.submit(self.worker_vector_search)
                    elif operation == "text_search":
                        future = executor.submit(self.worker_text_search)
                    elif operation == "create":
                        future = executor.submit(self.worker_create_chunk)
                    elif operation == "delete":
                        future = executor.submit(self.worker_delete_chunk)

                    futures.append((operation, future))

                # Wait for all tasks to complete
                for operation, future in futures:
                    result = future.result()
                    total_count += 1

                    if result["success"]:
                        success_count += 1

                        if operation == "vector_search":
                            vector_search_latencies.append(result["latency"])
                        elif operation == "text_search":
                            text_search_latencies.append(result["latency"])
                        elif operation == "create":
                            create_latencies.append(result["latency"])
                        elif operation == "delete":
                            delete_latencies.append(result["latency"])

                # Update progress bar
                elapsed = time.time() - start_time
                pbar.update(min(elapsed - pbar.n, duration))

                # Throttle to avoid overwhelming the server
                time.sleep(0.1)

        # Calculate statistics
        success_rate = success_count / total_count if total_count > 0 else 0

        vector_search_avg = np.mean(vector_search_latencies) if vector_search_latencies else 0
        text_search_avg = np.mean(text_search_latencies) if text_search_latencies else 0
        create_avg = np.mean(create_latencies) if create_latencies else 0
        delete_avg = np.mean(delete_latencies) if delete_latencies else 0

        # Store results
        self.results["concurrent_users"].append(concurrent_users)
        self.results["vector_search_latency"].append(vector_search_avg)
        self.results["text_search_latency"].append(text_search_avg)
        self.results["create_latency"].append(create_avg)
        self.results["delete_latency"].append(delete_avg)
        self.results["success_rate"].append(success_rate)

        # Log results
        logger.info(f"Load test results for {concurrent_users} concurrent users:")
        logger.info(f"  Vector search latency: {vector_search_avg:.4f} seconds")
        logger.info(f"  Text search latency: {text_search_avg:.4f} seconds")
        logger.info(f"  Create latency: {create_avg:.4f} seconds")
        logger.info(f"  Delete latency: {delete_avg:.4f} seconds")
        logger.info(f"  Success rate: {success_rate:.2%}")

        # Shutdown the thread pool
        executor.shutdown()

    def generate_report(self, output_file="load_test_results.png"):
        """
        Generate a report with the load test results.

        Args:
            output_file: Output file for the report
        """
        logger.info("Generating load test report...")

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot latency vs. concurrent users
        axs[0, 0].plot(self.results["concurrent_users"], self.results["vector_search_latency"],
                      marker='o', label="Vector Search")
        axs[0, 0].plot(self.results["concurrent_users"], self.results["text_search_latency"],
                      marker='s', label="Text Search")
        axs[0, 0].set_xlabel("Concurrent Users")
        axs[0, 0].set_ylabel("Latency (seconds)")
        axs[0, 0].set_title("Search Latency vs. Concurrent Users")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot create/delete latency vs. concurrent users
        axs[0, 1].plot(self.results["concurrent_users"], self.results["create_latency"],
                      marker='o', label="Create")
        axs[0, 1].plot(self.results["concurrent_users"], self.results["delete_latency"],
                      marker='s', label="Delete")
        axs[0, 1].set_xlabel("Concurrent Users")
        axs[0, 1].set_ylabel("Latency (seconds)")
        axs[0, 1].set_title("Create/Delete Latency vs. Concurrent Users")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot success rate vs. concurrent users
        axs[1, 0].plot(self.results["concurrent_users"], self.results["success_rate"],
                      marker='o', color='green')
        axs[1, 0].set_xlabel("Concurrent Users")
        axs[1, 0].set_ylabel("Success Rate")
        axs[1, 0].set_title("Success Rate vs. Concurrent Users")
        axs[1, 0].grid(True)

        # Plot latency comparison as bar chart
        if self.results["concurrent_users"]:
            max_users = max(self.results["concurrent_users"])
            max_users_idx = self.results["concurrent_users"].index(max_users)

            operations = ["Vector Search", "Text Search", "Create", "Delete"]
            latencies = [
                self.results["vector_search_latency"][max_users_idx],
                self.results["text_search_latency"][max_users_idx],
                self.results["create_latency"][max_users_idx],
                self.results["delete_latency"][max_users_idx]
            ]

            axs[1, 1].bar(operations, latencies, color=['blue', 'orange', 'green', 'red'])
            axs[1, 1].set_xlabel("Operation")
            axs[1, 1].set_ylabel("Latency (seconds)")
            axs[1, 1].set_title(f"Latency Comparison at {max_users} Concurrent Users")
            axs[1, 1].grid(True)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Report saved to {output_file}")


async def main():
    """Main function to run the load test."""
    parser = argparse.ArgumentParser(description="Vector Search API Load Test")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                      help="Base URL of the API")
    parser.add_argument("--libraries", type=int, default=2,
                      help="Number of libraries to create")
    parser.add_argument("--documents", type=int, default=3,
                      help="Number of documents per library")
    parser.add_argument("--chunks", type=int, default=5,
                      help="Number of chunks per document")
    parser.add_argument("--users", type=str, default="1,5,10,20",
                      help="Comma-separated list of concurrent users to test")
    parser.add_argument("--duration", type=int, default=30,
                      help="Duration of each test in seconds")
    parser.add_argument("--output", type=str, default="load_test_results.png",
                      help="Output file for the report")
    args = parser.parse_args()

    # Parse concurrent users
    concurrent_users = [int(u) for u in args.users.split(",")]

    # Create and run load test
    load_test = VectorSearchLoadTest(
        base_url=args.url,
        num_libraries=args.libraries,
        num_documents=args.documents,
        num_chunks=args.chunks
    )

    try:
        # Setup test data
        await load_test.setup_test_data()

        # Run load tests with different numbers of concurrent users
        for users in concurrent_users:
            await load_test.run_load_test(
                concurrent_users=users,
                duration=args.duration
            )

        # Generate report
        load_test.generate_report(output_file=args.output)
    finally:
        # Cleanup test data
        await load_test.cleanup_test_data()

if __name__ == "__main__":
    asyncio.run(main())
