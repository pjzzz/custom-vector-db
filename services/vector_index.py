"""
Vector indexing for fast approximate nearest neighbor search.
Inspired by Annoy (Approximate Nearest Neighbors Oh Yeah) but implemented from scratch.
"""

import numpy as np
import random
from typing import List, Dict, Optional, Set, Union, TypedDict
from threading import RLock
import logging
import time

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Thread-safe vector index for fast approximate nearest neighbor search.
    Uses random projection trees for indexing high-dimensional vectors.
    """

    def __init__(self, dimension: int, n_trees: int = 10, leaf_size: int = 40):
        """
        Initialize the vector index.

        Args:
            dimension: Dimension of the vectors to index
            n_trees: Number of trees to build for the index
            leaf_size: Maximum number of items in a leaf node
        """
        self.dimension = dimension
        self.n_trees = n_trees
        self.leaf_size = leaf_size

        # Initialize empty trees
        self.trees: List[Dict] = []

        # Map of vector id to vector data
        self.vectors: Dict[str, Dict] = {}

        # Locks for thread safety
        self._trees_lock = RLock()
        self._vectors_lock = RLock()

        # Build status
        self.is_built = False

        # Random seed for reproducibility
        self.random_seed = 42

        logger.info(f"Initialized VectorIndex with dimension={dimension}, n_trees={n_trees}, leaf_size={leaf_size}")

    def add_item(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """
        Add an item to the index.
        Thread-safe implementation.

        Args:
            id: Unique identifier for the vector
            vector: Vector to add
            metadata: Optional metadata associated with the vector
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")

        with self._vectors_lock:
            self.vectors[id] = {
                'id': id,
                'vector': vector,
                'metadata': metadata or {}
            }
            # Mark index as not built when new items are added
            self.is_built = False

        logger.debug(f"Added item {id} to index")

    def remove_item(self, id: str) -> bool:
        """
        Remove an item from the index.
        Thread-safe implementation.

        Args:
            id: Unique identifier for the vector to remove

        Returns:
            True if the item was removed, False if it wasn't found
        """
        with self._vectors_lock:
            if id in self.vectors:
                del self.vectors[id]
                # Mark index as not built when items are removed
                self.is_built = False
                logger.debug(f"Removed item {id} from index")
                return True
            return False

    def build(self, force: bool = False) -> None:
        """
        Build the index for fast searching.
        Thread-safe implementation.

        Args:
            force: If True, rebuild the index even if it's already built
        """
        with self._trees_lock:
            if self.is_built and not force:
                return

            start_time = time.time()

            # Reset trees
            self.trees = []

            # Get all vectors
            with self._vectors_lock:
                if not self.vectors:
                    logger.warning("No vectors to index")
                    return

                vector_items = list(self.vectors.values())

            # Skip building for small datasets
            if len(vector_items) <= 100:
                self.is_built = True
                return

            # Build trees in parallel for larger datasets
            if len(vector_items) > 1000 and self.n_trees > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(self.n_trees, 4)) as executor:
                    random.seed(self.random_seed)
                    futures = []
                    for i in range(self.n_trees):
                        tree_seed = self.random_seed + i
                        futures.append(executor.submit(
                            self._build_tree_with_seed, vector_items, 0, tree_seed))

                    for future in futures:
                        self.trees.append(future.result())
            else:
                # Build trees sequentially for smaller datasets
                random.seed(self.random_seed)
                for i in range(self.n_trees):
                    tree = self._build_tree(vector_items, 0)
                    self.trees.append(tree)

            self.is_built = True
            build_time = time.time() - start_time
            logger.info(f"Built index with {len(vector_items)} vectors in {build_time:.2f} seconds")

    def _build_tree(self, items: List[Dict], depth: int) -> Dict:
        """
        Recursively build a tree for the index.

        Args:
            items: List of vector items to index
            depth: Current depth in the tree

        Returns:
            Tree node
        """
        if len(items) <= self.leaf_size:
            return {
                'type': 'leaf',
                'items': items
            }

        # Choose two random items
        a, b = random.sample(items, 2)

        # Create a random hyperplane
        normal = a['vector'] - b['vector']

        # Normalize the normal vector
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        # Split items based on the hyperplane
        left_items = []
        right_items = []

        for item in items:
            # Project the item onto the normal vector
            projection = np.dot(item['vector'], normal)

            if projection <= 0:
                left_items.append(item)
            else:
                right_items.append(item)

        # Handle edge cases where all items end up on one side
        if not left_items or not right_items:
            # Just split the items in half
            mid = len(items) // 2
            left_items = items[:mid]
            right_items = items[mid:]

        # Build left and right subtrees
        left_tree = self._build_tree(left_items, depth + 1)
        right_tree = self._build_tree(right_items, depth + 1)

        return {
            'type': 'node',
            'normal': normal,
            'left': left_tree,
            'right': right_tree
        }

    def search(self, query_vector: np.ndarray, k: int = 10, search_k: int = -1) -> List[Dict]:
        """
        Search for the k nearest neighbors of the query vector.
        Thread-safe implementation.

        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            search_k: Number of nodes to inspect during search (-1 means unlimited)

        Returns:
            List of nearest neighbors with their distances and metadata
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}")

        # If there are no vectors, return empty results
        with self._vectors_lock:
            vector_count = len(self.vectors)
            if vector_count == 0:
                return []

            # For small datasets, linear search is faster
            if vector_count <= 100 or vector_count <= k * 2:
                return self._linear_search(query_vector, k)

        # Build the index if it's not built yet
        if not self.is_built:
            # Only log this once
            if not hasattr(self, '_logged_build'):
                logger.info("Index not built, building now")
                self._logged_build = True
            self.build()

        # If there are no trees, return empty results
        if not self.trees:
            logger.warning("No trees in index")
            return self._linear_search(query_vector, k)

        # Create a snapshot of the trees for thread safety
        with self._trees_lock:
            trees_snapshot = self.trees.copy()

        # Calculate how many nodes to inspect per tree
        if search_k == -1:
            search_k = max(k * 10, k * self.n_trees)

        # Search each tree and collect candidates
        candidates = set()
        for tree in trees_snapshot:
            tree_candidates = self._search_tree(tree, query_vector, max(search_k // self.n_trees, k))
            candidates.update(tree_candidates)

        # If we didn't find enough candidates, fall back to linear search
        if len(candidates) < k:
            return self._linear_search(query_vector, k)

        # Calculate distances for all candidates
        results = []
        with self._vectors_lock:
            for id in candidates:
                if id in self.vectors:
                    item = self.vectors[id]
                    distance = self._calculate_distance(query_vector, item['vector'])
                    results.append({
                        'id': id,
                        'score': 1.0 - distance,  # Convert distance to similarity score
                        'metadata': item['metadata']
                    })

        # Sort by similarity (higher is better)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def _search_tree(self, node: Dict, query_vector: np.ndarray, n_nodes: int) -> List[str]:
        """
        Search a tree for nearest neighbors.

        Args:
            node: Tree node
            query_vector: Query vector
            n_nodes: Number of nodes to inspect

        Returns:
            List of candidate vector IDs
        """
        if node['type'] == 'leaf':
            return [item['id'] for item in node['items']]

        # Project the query vector onto the normal vector
        projection = np.dot(query_vector, node['normal'])

        # Determine which side to search first
        if projection <= 0:
            first, second = node['left'], node['right']
        else:
            first, second = node['right'], node['left']

        # Search the first side
        candidates = self._search_tree(first, query_vector, n_nodes)

        # If we haven't found enough candidates, search the second side
        if len(candidates) < n_nodes:
            candidates.extend(self._search_tree(second, query_vector, n_nodes - len(candidates)))

        return candidates

    def _linear_search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        """
        Perform a linear search through all vectors.
        Used as a fallback when the index is not effective.

        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return

        Returns:
            List of nearest neighbors with their distances and metadata
        """
        results = []
        with self._vectors_lock:
            for id, item in self.vectors.items():
                distance = self._calculate_distance(query_vector, item['vector'])
                results.append({
                    'id': id,
                    'score': 1.0 - distance,  # Convert distance to similarity score
                    'metadata': item['metadata']
                })

        # Sort by similarity (higher is better)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the distance between two vectors.
        Uses cosine distance (1 - cosine similarity).

        Args:
            a: First vector
            b: Second vector

        Returns:
            Distance between the vectors (0 to 2, where 0 is identical)
        """
        # Use a faster dot product calculation for normalized vectors
        dot_product = np.dot(a, b)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 1.0  # Maximum distance for zero vectors

        # Calculate cosine similarity
        similarity = dot_product / (a_norm * b_norm)

        # Convert to distance (1 - similarity)
        distance = 1.0 - similarity

        return max(0.0, min(2.0, distance))  # Clamp to [0, 2]

    def _build_tree_with_seed(self, items: List[Dict], depth: int, seed: int) -> Dict:
        """
        Build a tree with a specific random seed.
        Used for parallel tree building.

        Args:
            items: List of vector items to index
            depth: Current depth in the tree
            seed: Random seed for this tree

        Returns:
            Tree node
        """
        random.seed(seed)
        return self._build_tree(items, depth)

    def get_stats(self) -> Dict:
        """
        Get statistics about the index.
        Thread-safe implementation.

        Returns:
            Dictionary with index statistics
        """
        with self._vectors_lock:
            vector_count = len(self.vectors)

        with self._trees_lock:
            tree_count = len(self.trees)

        return {
            'vector_count': vector_count,
            'tree_count': tree_count,
            'dimension': self.dimension,
            'is_built': self.is_built,
            'leaf_size': self.leaf_size
        }
