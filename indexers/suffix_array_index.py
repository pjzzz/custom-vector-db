from typing import Dict, List, Set, Tuple
from models.chunk import Chunk
import bisect
import threading
import logging

logger = logging.getLogger(__name__)


class SuffixArrayIndex:
    """
    A suffix array index that supports substring matching.

    Space Complexity: O(T) where T is the total length of all text
    Time Complexity:
    - Build: O(T log T)
    - Search: O(P log T + k) where P is pattern length and k is number of results

    This index is particularly useful for substring searches and fuzzy matching.
    It's more powerful than the other two but has higher build time complexity.
    """

    def __init__(self):
        self.suffix_array: List[Tuple[str, int, str, int]] = []  # (suffix, position, document_id, chunk_id)
        self.chunk_map: Dict[str, Chunk] = {}
        # Add thread safety with locks
        self._suffix_lock = threading.RLock()  # Reentrant lock for suffix array operations
        self._chunk_lock = threading.RLock()   # Reentrant lock for chunk map operations

    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the index with thread safety."""
        # Use locks to ensure thread safety
        with self._chunk_lock:
            self.chunk_map[chunk.id] = chunk

        text = chunk.text.lower()
        suffixes = []

        # Prepare all suffixes of the text
        for i in range(len(text)):
            suffix = text[i:]
            suffixes.append((suffix, i, chunk.document_id, chunk.id))

        # Add all suffixes to the array with lock protection
        with self._suffix_lock:
            for suffix_tuple in suffixes:
                bisect.insort(self.suffix_array, suffix_tuple)

    def search(self, query: str) -> List[Tuple[str, str, int]]:
        """
        Search for chunks containing the query as a substring.
        Returns list of (document_id, chunk_id, position) tuples.
        Thread-safe implementation.
        """
        query = query.lower()
        n = len(query)

        # Find the range of suffixes that contain the query with lock protection
        with self._suffix_lock:
            # Create a snapshot of the suffix array for thread safety
            start = bisect.bisect_left(self.suffix_array, (query, 0, '', ''))
            end = bisect.bisect_right(self.suffix_array, (query + '\uffff', float('inf'), '', ''))
            # Make a copy of the relevant portion to avoid holding the lock too long
            suffix_matches = self.suffix_array[start:end].copy() if start < end else []

        results = []
        # Access chunk data with lock protection
        with self._chunk_lock:
            for _, pos, doc_id, chunk_id in suffix_matches:
                if chunk_id in self.chunk_map:
                    # Check if the suffix actually contains the query
                    chunk_text = self.chunk_map[chunk_id].text.lower()
                    if pos + n <= len(chunk_text) and chunk_text[pos:pos+n] == query:
                        results.append((doc_id, chunk_id, pos))

        return results

    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the index.
        Thread-safe implementation.

        Args:
            chunk_id: ID of the chunk to remove

        Returns:
            True if the chunk was removed, False otherwise
        """
        # First remove the chunk from the chunk map
        with self._chunk_lock:
            if chunk_id not in self.chunk_map:
                return False
            del self.chunk_map[chunk_id]

        # Then remove all suffixes for this chunk from the suffix array
        with self._suffix_lock:
            # Filter out suffixes for this chunk
            self.suffix_array = [s for s in self.suffix_array if s[3] != chunk_id]

        return True

    def get_serializable_data(self):
        """Get serializable data for persistence.
        Thread-safe implementation.

        Returns:
            Dict containing serializable data
        """
        with self._suffix_lock, self._chunk_lock:
            # Convert chunk objects to dictionaries
            chunk_dict = {}
            for chunk_id, chunk in self.chunk_map.items():
                chunk_dict[chunk_id] = chunk.model_dump()

            return {
                "suffix_array": self.suffix_array,
                "chunk_map": chunk_dict
            }

    def load_serializable_data(self, data):
        """Load serializable data from persistence.
        Thread-safe implementation.

        Args:
            data: Dict containing serializable data
        """
        from models import Chunk  # Import here to avoid circular imports

        with self._suffix_lock, self._chunk_lock:
            self.suffix_array = data["suffix_array"]

            # Convert dictionaries back to Chunk objects
            self.chunk_map = {}
            for chunk_id, chunk_data in data["chunk_map"].items():
                self.chunk_map[chunk_id] = Chunk(**chunk_data)

            logger.info(f"Loaded suffix array index with {len(self.suffix_array)} suffixes and {len(self.chunk_map)} chunks")
