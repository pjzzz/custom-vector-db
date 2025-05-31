# Vector Search Benchmarking and Load Testing

This directory contains tools for benchmarking and load testing our thread-safe vector content management system. These tools help validate the performance and thread safety of our implementation under various conditions.

## Available Tools

### 1. Vector Search Benchmark (`vector_search_benchmark.py`)

This tool compares the performance of our custom vector search implementation against traditional text search methods. It measures search times and result counts for different queries and indexers.

**Features:**
- Compares vector search vs. text search performance
- Tests all three indexing algorithms (SuffixArrayIndex, TrieIndex, InvertedIndex)
- Generates visualizations of performance metrics
- Provides detailed statistics on search times and result counts

**Usage:**
```bash
python vector_search_benchmark.py --chunks 1000 --size 100 --iterations 5
```

**Options:**
- `--chunks`: Number of chunks to create for testing (default: 1000)
- `--size`: Average number of words per chunk (default: 100)
- `--iterations`: Number of iterations per query (default: 5)

### 2. Load Testing Tool (`load_test.py`)

This tool simulates concurrent users accessing the vector search API to validate thread safety and measure performance under load. It tests various operations including vector search, text search, create, and delete.

**Features:**
- Simulates multiple concurrent users with configurable concurrency levels
- Tests a mix of operations (search, create, delete) with configurable ratios
- Measures latency and success rates under different loads
- Generates comprehensive performance reports with visualizations
- Validates thread safety by stressing concurrent access patterns

**Usage:**
```bash
python load_test.py --url http://localhost:8000 --libraries 2 --documents 3 --chunks 5 --users 1,5,10,20 --duration 30
```

**Options:**
- `--url`: Base URL of the API (default: http://localhost:8000)
- `--libraries`: Number of libraries to create for testing (default: 2)
- `--documents`: Number of documents per library (default: 3)
- `--chunks`: Number of chunks per document (default: 5)
- `--users`: Comma-separated list of concurrent users to test (default: 1,5,10,20)
- `--duration`: Duration of each test in seconds (default: 30)
- `--output`: Output file for the report (default: load_test_results.png)

## Interpreting Load Test Results

The load testing tool generates a report with four graphs:

1. **Search Latency vs. Concurrent Users**: Shows how vector search and text search latency scales with increasing concurrent users.
2. **Create/Delete Latency vs. Concurrent Users**: Shows how create and delete operation latency scales with increasing concurrent users.
3. **Success Rate vs. Concurrent Users**: Shows the percentage of successful operations as concurrent users increase.
4. **Latency Comparison**: Compares the latency of different operations at the maximum tested concurrency level.

### Key Metrics to Watch

- **Latency Scaling**: Ideally, latency should increase linearly or sub-linearly with concurrent users, indicating good scalability.
- **Success Rate**: Should remain close to 100% even under high concurrency, indicating robust thread safety.
- **Operation Comparison**: Vector search may have higher latency than text search due to more complex processing, but should still maintain acceptable performance.

### Thread Safety Indicators

The load test specifically validates our thread safety implementation by:

1. **Concurrent Modifications**: Testing simultaneous create and delete operations to stress locks
2. **Concurrent Reads**: Testing simultaneous search operations to validate snapshot-based search
3. **Mixed Workloads**: Testing a mix of reads and writes to validate lock acquisition patterns
4. **Success Rate**: Tracking operation success to detect race conditions or deadlocks

## Running in Docker

Both benchmarking tools can be run against the containerized API. Make sure the API is running with:

```bash
docker-compose up -d
```

Then run the benchmarking tools with the appropriate URL:

```bash
python benchmarks/load_test.py --url http://localhost:8000
```

## Dependencies

The benchmarking tools require the following dependencies:
- numpy
- matplotlib
- tqdm
- pandas (for the Jupyter notebook demo)

These are included in the project's requirements.txt file.
