#!/usr/bin/env python3
"""
Benchmark script to evaluate our custom embedding method:
Custom TF-IDF with random projection
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
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# No external API keys needed for our custom implementation


def load_test_data(num_samples: int = 100) -> List[str]:
    """
    Load test data from 20 newsgroups dataset.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of text samples
    """
    logger.info(f"Loading {num_samples} samples from 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Get a subset of the data
    indices = np.random.choice(len(newsgroups.data), min(num_samples, len(newsgroups.data)), replace=False)
    texts = [newsgroups.data[i] for i in indices]
    categories = [newsgroups.target_names[newsgroups.target[i]] for i in indices]
    
    # Clean up the texts a bit
    texts = [text.strip() for text in texts]
    texts = [text for text in texts if len(text) > 50]  # Filter out very short texts
    
    return texts[:num_samples], categories[:num_samples]


def generate_query_pairs(texts: List[str], categories: List[str], num_pairs: int = 10) -> List[Tuple[str, str, bool]]:
    """
    Generate pairs of texts for similarity testing.
    
    Args:
        texts: List of texts
        categories: List of categories for each text
        num_pairs: Number of pairs to generate
        
    Returns:
        List of (text1, text2, is_same_category) tuples
    """
    pairs = []
    
    # Group texts by category
    category_map = {}
    for text, category in zip(texts, categories):
        if category not in category_map:
            category_map[category] = []
        category_map[category].append(text)
    
    # Generate pairs from same category
    for _ in range(num_pairs // 2):
        # Pick a category with at least 2 texts
        valid_categories = [cat for cat, texts in category_map.items() if len(texts) >= 2]
        if not valid_categories:
            break
            
        category = random.choice(valid_categories)
        text1, text2 = random.sample(category_map[category], 2)
        pairs.append((text1, text2, True))
    
    # Generate pairs from different categories
    for _ in range(num_pairs - len(pairs)):
        cat1, cat2 = random.sample(list(category_map.keys()), 2)
        text1 = random.choice(category_map[cat1])
        text2 = random.choice(category_map[cat2])
        pairs.append((text1, text2, False))
    
    return pairs


def benchmark_embedding_services(texts: List[str], batch_size: int = 10) -> Dict:
    """
    Benchmark our custom embedding service.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for embedding generation
        
    Returns:
        Dictionary with benchmark results
    """
    # Initialize embedding service
    custom_service = EmbeddingService(vector_size=1536)
    
    # Fit the custom embedding service on the texts
    logger.info("Fitting custom embedding service...")
    start_time = time.time()
    custom_service.fit(texts)
    fit_time = time.time() - start_time
    logger.info(f"Fitted custom embedding service in {fit_time:.2f}s")
    
    # Benchmark custom embedding service
    logger.info("Benchmarking custom embedding service...")
    custom_times = []
    custom_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        start_time = time.time()
        batch_embeddings = [custom_service.get_embedding(text) for text in batch]
        end_time = time.time()
        custom_times.append(end_time - start_time)
        custom_embeddings.extend(batch_embeddings)
    
    custom_avg_time = np.mean(custom_times)
    custom_texts_per_second = batch_size / custom_avg_time
    
    logger.info(f"Custom: {custom_avg_time:.4f}s per batch, {custom_texts_per_second:.2f} texts/sec")
    
    return {
        "custom": {
            "times": custom_times,
            "avg_time": custom_avg_time,
            "texts_per_second": custom_texts_per_second,
            "embeddings": custom_embeddings
        }
    }


def evaluate_similarity_performance(
    pairs: List[Tuple[str, str, bool]], 
    custom_embeddings: Dict[str, np.ndarray]
) -> Dict:
    """
    Evaluate how well our custom embedding method captures semantic similarity.
    
    Args:
        pairs: List of (text1, text2, is_same_category) tuples
        custom_embeddings: Dictionary mapping text to custom embeddings
        
    Returns:
        Dictionary with evaluation results
    """
    custom_similarities = []
    ground_truth = []
    
    for text1, text2, is_same in pairs:
        # Custom embeddings similarity
        emb1 = custom_embeddings.get(text1)
        emb2 = custom_embeddings.get(text2)
        if emb1 is not None and emb2 is not None:
            sim = cosine_similarity([emb1], [emb2])[0][0]
            custom_similarities.append(sim)
            ground_truth.append(1 if is_same else 0)
    
    # Calculate accuracy using a threshold of 0.5
    custom_accuracy = calculate_accuracy(custom_similarities, ground_truth)
    
    logger.info(f"Custom accuracy: {custom_accuracy:.2f}")
    
    return {
        "custom": {
            "similarities": custom_similarities,
            "accuracy": custom_accuracy
        },
        "ground_truth": ground_truth
    }


def calculate_accuracy(similarities: List[float], ground_truth: List[int], threshold: float = 0.5) -> float:
    """
    Calculate accuracy of similarity predictions.
    
    Args:
        similarities: List of similarity scores
        ground_truth: List of ground truth labels (1 for same category, 0 for different)
        threshold: Similarity threshold for classification
        
    Returns:
        Accuracy score
    """
    predictions = [1 if sim >= threshold else 0 for sim in similarities]
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
    return correct / len(predictions) if predictions else 0


def plot_benchmark_results(results: Dict) -> None:
    """Plot benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Embedding generation time
    plt.subplot(2, 2, 1)
    methods = ['Custom (TF-IDF)']
    times = [results['custom']['avg_time']]
    plt.bar(methods, times)
    plt.ylabel('Average Time per Batch (s)')
    plt.title('Embedding Generation Time')
    
    # Plot 2: Throughput
    plt.subplot(2, 2, 2)
    throughputs = [results['custom']['texts_per_second']]
    plt.bar(methods, throughputs)
    plt.ylabel('Texts per Second')
    plt.title('Embedding Throughput')
    
    # Plot 3: Similarity distribution
    plt.subplot(2, 2, 3)
    eval_results = results.get('evaluation', {})
    if eval_results:
        df = pd.DataFrame({
            'Custom': eval_results['custom']['similarities'],
            'Same Category': eval_results['ground_truth']
        })
        
        same_cat = df[df['Same Category'] == 1]
        diff_cat = df[df['Same Category'] == 0]
        
        plt.boxplot([
            same_cat['Custom'], diff_cat['Custom']
        ], labels=['Same Category', 'Different Category'])
        plt.ylabel('Cosine Similarity')
        plt.title('Similarity Distribution by Category')
    
    # Plot 4: Accuracy
    plt.subplot(2, 2, 4)
    if eval_results:
        accuracies = [eval_results['custom']['accuracy']]
        plt.bar(methods, accuracies)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Similarity Classification Accuracy')
    
    plt.tight_layout()
    plt.savefig('embedding_benchmark_results.png')
    logger.info("Saved embedding benchmark results to embedding_benchmark_results.png")


def main():
    """Main function to run benchmarks."""
    # Load test data
    texts, categories = load_test_data(num_samples=50)
    
    # Generate query pairs for similarity testing
    pairs = generate_query_pairs(texts, categories, num_pairs=20)
    
    # Benchmark embedding services
    results = benchmark_embedding_services(texts, batch_size=5)
    
    # Create mapping from text to embedding for evaluation
    custom_embeddings = {text: emb for text, emb in zip(texts, results['custom']['embeddings'])}
    
    # Evaluate similarity performance
    eval_results = evaluate_similarity_performance(pairs, custom_embeddings)
    results['evaluation'] = eval_results
    
    # Plot results
    plot_benchmark_results(results)
    
    # Print summary
    logger.info("\n===== BENCHMARK SUMMARY =====")
    logger.info(f"Custom TF-IDF: {results['custom']['texts_per_second']:.2f} texts/sec, Accuracy: {eval_results['custom']['accuracy']:.2f}")


if __name__ == "__main__":
    main()
