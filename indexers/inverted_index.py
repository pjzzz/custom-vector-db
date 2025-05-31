from typing import Dict, List, Set, Tuple
from models.chunk import Chunk
import re
from collections import defaultdict
import threading
from copy import deepcopy

class InvertedIndex:
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
    
    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the index with thread safety."""
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
        Thread-safe implementation.
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
        """Create a thread-safe snapshot of the index for the given tokens."""
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
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the index with thread safety.
        Returns True if the chunk was found and removed, False otherwise.
        """
        # Check if the chunk exists and get its text for processing
        chunk_text = None
        with self._chunk_lock:
            if chunk_id not in self.chunk_map:
                return False
            
            # Get the text before removing from map
            chunk_text = self.chunk_map[chunk_id].text
            doc_id = self.chunk_map[chunk_id].document_id
            # Remove from chunk map
            del self.chunk_map[chunk_id]
        
        # Process the tokens to remove from index
        tokens = self.tokenize(chunk_text)
        
        # Remove chunk references from index with lock protection
        with self._index_lock:
            for token in tokens:
                if token in self.index and doc_id in self.index[token]:
                    # Filter out positions for this chunk
                    self.index[token][doc_id] = [
                        pos for pos in self.index[token][doc_id] 
                        if pos[0] != chunk_id
                    ]
                    
                    # If no more positions for this document, remove the document entry
                    if not self.index[token][doc_id]:
                        del self.index[token][doc_id]
                    
                    # If no more documents for this token, remove the token entry
                    if not self.index[token]:
                        del self.index[token]
        
        return True
