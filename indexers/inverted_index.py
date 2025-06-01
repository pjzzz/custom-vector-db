from typing import Dict, List, Set, Tuple
from models.chunk import Chunk
import re
from collections import defaultdict
import threading
from copy import deepcopy
import logging
from .base_indexer import BaseIndexer

logger = logging.getLogger(__name__)


class InvertedIndex(BaseIndexer):
    """
    A simple inverted index that maps words to their locations in documents.

    Space Complexity: O(T) where T is the total number of terms across all documents
    Time Complexity:
    - Build: O(T)
    - Search: O(Q + k) where Q is query size and k is number of results

    This is a good baseline index that's simple to implement and understand.
    It's particularly effective for exact word matching and boolean queries.
    """

    def __init__(self):
        self.index: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(dict)
        self.chunk_map: Dict[str, Chunk] = {}
        # Add thread safety with locks
        self._index_lock = threading.RLock()  # Reentrant lock for index operations
        self._chunk_lock = threading.RLock()  # Reentrant lock for chunk map operations

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on whitespace and remove punctuation."""
        return re.findall(r'\w+', text.lower())

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the index."""
        # Store chunk in map with lock protection
        with self._chunk_lock:
            self.chunk_map[chunk.id] = chunk

        # Process tokens outside the lock
        tokens = self.tokenize(chunk.text)
        token_data = {}

        # Prepare token data outside the lock
        for token in tokens:
            if token not in token_data:
                token_data[token] = []
            token_data[token].append((chunk.document_id, chunk.id, chunk.position))

        # Update index with lock protection
        with self._index_lock:
            for token, entries in token_data.items():
                for doc_id, chunk_id, position in entries:
                    if doc_id not in self.index[token]:
                        self.index[token][doc_id] = []
                    self.index[token][doc_id].append((chunk_id, position))

    def search(self, query: str) -> List[Tuple[str, str, int]]:
        """
        Search for chunks containing all query terms.
        Returns list of (document_id, chunk_id, position) tuples.
        """
        tokens = self.tokenize(query)
        if not tokens:
            return []

        # Get a snapshot of the index
        index_snapshot = self._get_index_snapshot(tokens)
        if not index_snapshot:
            return []  # One or more tokens not found

        # Find matching documents and chunks
        matching_docs = self._find_matching_docs(index_snapshot, tokens)
        chunk_results = self._find_matching_chunks(index_snapshot, tokens, matching_docs)

        return list(chunk_results)

    def _get_index_snapshot(self, tokens: List[str]) -> Dict:
        """Create a snapshot of the index for the given tokens."""
        with self._index_lock:
            # Check if all tokens exist in the index
            for token in tokens:
                if token not in self.index:
                    return {}

            # Create a deep copy of the relevant parts of the index
            index_snapshot = {}
            for token in tokens:
                index_snapshot[token] = deepcopy(self.index[token])

        return index_snapshot

    def _find_matching_docs(self, index_snapshot: Dict, tokens: List[str]) -> Set[str]:
        """Find documents that contain all the tokens."""
        # Start with documents containing the first token
        matching_docs = set(index_snapshot[tokens[0]].keys())

        # Intersect with documents containing other tokens
        for token in tokens[1:]:
            matching_docs &= set(index_snapshot[token].keys())

        return matching_docs

    def _find_matching_chunks(self, index_snapshot: Dict, tokens: List[str],
                             matching_docs: Set[str]) -> Set[Tuple[str, str, int]]:
        """Find chunks that contain all the tokens in the matching documents."""
        chunk_results = set()

        for doc_id in matching_docs:
            # Get chunks from first token
            matching_chunks = self._get_chunks_for_token(index_snapshot, tokens[0], doc_id)

            # Intersect with chunks from other tokens
            for token in tokens[1:]:
                current_chunks = self._get_chunks_for_token(index_snapshot, token, doc_id)
                matching_chunks &= current_chunks

            # Add verified chunks to results
            self._add_verified_chunks(matching_chunks, doc_id, chunk_results)

        return chunk_results

    def _get_chunks_for_token(self, index_snapshot: Dict, token: str, doc_id: str) -> Set[str]:
        """Get the set of chunk IDs for a token in a document."""
        chunks = set()
        for chunk_id, _ in index_snapshot[token][doc_id]:
            chunks.add(chunk_id)
        return chunks

    def _add_verified_chunks(self, matching_chunks: Set[str], doc_id: str,
                            chunk_results: Set[Tuple[str, str, int]]):
        """Add chunks to results after verifying they still exist."""
        for chunk_id in matching_chunks:
            # Verify the chunk still exists in the chunk map
            with self._chunk_lock:
                if chunk_id in self.chunk_map:
                    chunk_results.add((doc_id, chunk_id, 0))  # Position 0 as placeholder

    def remove_chunk(self, chunk_id: str) -> None:
        """
        Remove a chunk from the index.

        Args:
            chunk_id: ID of the chunk to remove
        """
        # First get the chunk and its document ID before removing it
        chunk_to_remove = None
        doc_id = None
        
        with self._chunk_lock:
            if chunk_id not in self.chunk_map:
                return  # Nothing to remove
                
            # Get the chunk and its document ID before removing it
            chunk_to_remove = self.chunk_map[chunk_id]
            doc_id = chunk_to_remove.document_id
            
            # Remove the chunk from the chunk map
            del self.chunk_map[chunk_id]
        
        # If we couldn't get the chunk or document ID, we can't proceed efficiently
        if not chunk_to_remove or not doc_id:
            logger.warning(f"Could not get chunk data for chunk_id {chunk_id}")
            return
            
        # Tokenize the chunk to get the terms we need to update
        terms_to_update = set(self.tokenize(chunk_to_remove.text))
        
        # Then remove all references to the chunk from the index, but only for relevant terms
        with self._index_lock:
            # Only process terms that are in the chunk and in the index
            for term in terms_to_update:
                if term not in self.index:
                    continue
                    
                # If this document exists in the posting list for this term
                if doc_id in self.index[term]:
                    # Filter out positions for this chunk
                    self.index[term][doc_id] = [
                        pos for pos in self.index[term][doc_id]
                        if pos[0] != chunk_id
                    ]

                    # If no positions left for this document, remove the document
                    if not self.index[term][doc_id]:
                        del self.index[term][doc_id]

                # If no documents left for this term, remove the term
                if term in self.index and not self.index[term]:
                    del self.index[term]

        return

    def get_serializable_data(self) -> Dict[str, Any]:
        """Get serializable data for persistence.

        Returns:
            Dict containing serializable data
        """
        with self._index_lock, self._chunk_lock:
            # Convert defaultdict to regular dict for serialization
            index_dict = {}
            for term, docs in self.index.items():
                index_dict[term] = {}
                for doc_id, positions in docs.items():
                    index_dict[term][doc_id] = positions

            # Convert chunk objects to dictionaries
            chunk_dict = {}
            for chunk_id, chunk in self.chunk_map.items():
                chunk_dict[chunk_id] = chunk.model_dump()

            return {
                "index": index_dict,
                "chunk_map": chunk_dict
            }

    def load_serializable_data(self, data: Dict[str, Any]) -> None:
        """Load serializable data from persistence.

        Args:
            data: Dict containing serializable data
        """
        from models import Chunk  # Import here to avoid circular imports

        with self._index_lock, self._chunk_lock:
            # Convert dict back to defaultdict
            self.index = defaultdict(dict)
            for term, docs in data["index"].items():
                for doc_id, positions in docs.items():
                    self.index[term][doc_id] = positions

            # Convert dictionaries back to Chunk objects
            self.chunk_map = {}
            for chunk_id, chunk_data in data["chunk_map"].items():
                self.chunk_map[chunk_id] = Chunk(**chunk_data)

            logger.info(f"Loaded inverted index with {len(self.index)} terms and {len(self.chunk_map)} chunks")
