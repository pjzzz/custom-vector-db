from .inverted_index import InvertedIndex
from .trie_index import TrieIndex
from .suffix_array_index import SuffixArrayIndex

__all__ = ['InvertedIndex', 'TrieIndex', 'SuffixArrayIndex']

INDEXERS = {
    'inverted': InvertedIndex,
    'trie': TrieIndex,
    'suffix': SuffixArrayIndex
}

INDEXER_DESCRIPTIONS = {
    'inverted': """
    Inverted Index:
    - Space: O(T) where T is total number of terms
    - Build: O(T)
    - Search: O(Q + k) where Q is query size and k is number of results
    - Best for: Exact word matching, boolean queries
    - Strengths: Simple, efficient for exact matches
    - Weaknesses: No prefix or substring matching
    """,
    'trie': """
    Trie Index:
    - Space: O(T) where T is total number of characters
    - Build: O(T)
    - Search: O(P + k) where P is prefix length and k is number of results
    - Best for: Prefix matching, autocomplete
    - Strengths: Efficient prefix matching, space-efficient for prefixes
    - Weaknesses: No suffix matching
    """,
    'suffix': """
    Suffix Array Index:
    - Space: O(T) where T is total length of all text
    - Build: O(T log T)
    - Search: O(P log T + k) where P is pattern length and k is number of results
    - Best for: Substring matching, fuzzy matching
    - Strengths: Powerful substring matching, supports fuzzy queries
    - Weaknesses: Higher build time complexity
    """
}
