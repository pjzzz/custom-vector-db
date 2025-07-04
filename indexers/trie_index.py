from typing import Dict, List, Set, Tuple, Any
from models.chunk import Chunk
from collections import defaultdict
import threading
from copy import deepcopy
import logging
from .base_indexer import BaseIndexer

logger = logging.getLogger(__name__)


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.documents: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.is_end_of_word = False


class TrieIndex(BaseIndexer):
    """
    A Trie-based index that supports prefix matching.

    Space Complexity: O(T) where T is the total number of characters in all terms
    Time Complexity:
    - Build: O(T)
    - Search: O(P + k) where P is prefix length and k is number of results

    This index is particularly useful for autocomplete and prefix-based queries.
    It's more space-efficient than inverted index for prefix matching.
    """

    def __init__(self):
        self.root = TrieNode()
        self.chunk_map: Dict[str, Chunk] = {}
        # Add thread safety with locks
        self._trie_lock = threading.RLock()  # Reentrant lock for trie operations
        self._chunk_lock = threading.RLock()  # Reentrant lock for chunk map operations

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the index."""
        # Store chunk in map with lock protection
        with self._chunk_lock:
            self.chunk_map[chunk.id] = chunk

        # Process words outside the lock
        words = chunk.text.lower().split()
        word_data = []

        for word in words:
            word_data.append((word, chunk.document_id, chunk.id, chunk.position))

        # Update trie with lock protection
        with self._trie_lock:
            for word, doc_id, chunk_id, position in word_data:
                current = self.root
                for char in word:
                    if char not in current.children:
                        current.children[char] = TrieNode()
                    current = current.children[char]
                    current.documents[doc_id].append((chunk_id, position))
                current.is_end_of_word = True

    def search(self, prefix: str) -> List[Tuple[str, str, int]]:
        """
        Search for chunks containing words starting with the given prefix.
        Returns list of (document_id, chunk_id, position) tuples.
        """
        prefix = prefix.lower()

        # Navigate to the prefix node with lock protection
        with self._trie_lock:
            current = self.root
            for char in prefix:
                if char not in current.children:
                    return []  # Prefix not found
                current = current.children[char]

            # Create a snapshot of the trie structure to avoid holding the lock
            # during the recursive traversal
            node_snapshot = self._create_node_snapshot(current)

        # Collect results from the snapshot without holding the lock
        results = []
        self._collect_results_from_snapshot(node_snapshot, results)

        return results

    def _create_node_snapshot(self, node: TrieNode) -> Dict:
        """Create a snapshot of a node and its relevant data."""
        snapshot = {
            'documents': {},
            'children': {}
        }

        # Copy document references
        for doc_id, positions in node.documents.items():
            snapshot['documents'][doc_id] = positions.copy()

        # Copy child nodes recursively
        for char, child in node.children.items():
            snapshot['children'][char] = self._create_node_snapshot(child)

        return snapshot

    def _collect_results_from_snapshot(self, node_snapshot: Dict, results: List[Tuple[str, str, int]]):
        """Helper method to collect all documents from a node snapshot."""
        # Process documents in this node
        for doc_id, positions in node_snapshot['documents'].items():
            for chunk_id, pos in positions:
                results.append((doc_id, chunk_id, pos))

        # Process child nodes recursively
        for child_snapshot in node_snapshot['children'].values():
            self._collect_results_from_snapshot(child_snapshot, results)

    def _collect_results(self, node: TrieNode, results: List[Tuple[str, str, int]]):
        """Helper method to collect all documents from a node and its children.
        """
        for doc_id, positions in node.documents.items():
            results.extend([(doc_id, chunk_id, pos) for chunk_id, pos in positions])

        for child in node.children.values():
            self._collect_results(child, results)

    def remove_chunk(self, chunk_id: str) -> None:
        """
        Remove a chunk from the index.
        """
        # Check if the chunk exists and get its text for processing
        chunk_text = None
        with self._chunk_lock:
            if chunk_id not in self.chunk_map:
                return  # Nothing to remove

            # Get the text before removing from map
            chunk_text = self.chunk_map[chunk_id].text.lower()
            # Remove from chunk map
            del self.chunk_map[chunk_id]

        # Process the words to remove from trie
        words = chunk_text.split()

        # Remove chunk references from trie with lock protection
        with self._trie_lock:
            for word in words:
                self._remove_word_chunk_reference(self.root, word, 0, chunk_id)

    def _remove_word_chunk_reference(self, node: TrieNode, word: str, index: int, chunk_id: str) -> bool:
        """
        Helper method to remove chunk references from a word path in the trie.
        Returns True if the node should be deleted (no more references).
        """
        # If we've processed all characters in the word
        if index == len(word):
            # Remove chunk references from all document entries
            for doc_id in list(node.documents.keys()):
                # Filter out positions for this chunk
                node.documents[doc_id] = [pos for pos in node.documents[doc_id]
                                        if pos[0] != chunk_id]
                # If no more positions for this document, remove the document entry
                if not node.documents[doc_id]:
                    del node.documents[doc_id]

            # Return True if this node has no documents and no children
            return not node.documents and not node.children

        # If we haven't reached the end of the word yet
        char = word[index]
        if char in node.children:
            # Recursively process the next character
            should_delete = self._remove_word_chunk_reference(
                node.children[char], word, index + 1, chunk_id
            )

            # If the child should be deleted, remove it
            if should_delete:
                del node.children[char]

        # Remove chunk references from this node too
        for doc_id in list(node.documents.keys()):
            node.documents[doc_id] = [pos for pos in node.documents[doc_id]
                                    if pos[0] != chunk_id]
            if not node.documents[doc_id]:
                del node.documents[doc_id]

        # Return True if this node has no documents and no children
        return not node.documents and not node.children
        
    def get_serializable_data(self) -> Dict[str, Any]:
        """Get serializable data for persistence.

        Returns:
            Dict containing serializable data
        """
        with self._trie_lock, self._chunk_lock:
            # Convert chunk objects to dictionaries
            chunk_dict = {}
            for chunk_id, chunk in self.chunk_map.items():
                chunk_dict[chunk_id] = chunk.model_dump()

            # Serialize the trie structure
            # This is a simplified approach - for a real implementation,
            # you might want a more efficient serialization of the trie
            def serialize_node(node):
                return {
                    "documents": {doc_id: positions.copy() for doc_id, positions in node.documents.items()},
                    "is_end_of_word": node.is_end_of_word,
                    "children": {char: serialize_node(child) for char, child in node.children.items()}
                }

            return {
                "root": serialize_node(self.root),
                "chunk_map": chunk_dict
            }

    def load_serializable_data(self, data: Dict[str, Any]) -> None:
        """Load serializable data from persistence.

        Args:
            data: Dict containing serializable data
        """
        from models import Chunk  # Import here to avoid circular imports

        with self._trie_lock, self._chunk_lock:
            # Deserialize the trie structure
            def deserialize_node(node_data):
                node = TrieNode()
                node.documents = defaultdict(list)
                for doc_id, positions in node_data["documents"].items():
                    node.documents[doc_id] = positions.copy()
                node.is_end_of_word = node_data["is_end_of_word"]
                node.children = {char: deserialize_node(child_data) 
                                for char, child_data in node_data["children"].items()}
                return node

            self.root = deserialize_node(data["root"])

            # Convert dictionaries back to Chunk objects
            self.chunk_map = {}
            for chunk_id, chunk_data in data["chunk_map"].items():
                self.chunk_map[chunk_id] = Chunk(**chunk_data)

            logger.info(f"Loaded trie index with {len(self.chunk_map)} chunks")
