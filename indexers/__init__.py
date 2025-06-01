from .inverted_index import InvertedIndex
from .trie_index import TrieIndex
from .suffix_array_index import SuffixArrayIndex
from .indexer_factory import IndexerFactory

__all__ = ['InvertedIndex', 'TrieIndex', 'SuffixArrayIndex', 'IndexerFactory']

# For backward compatibility, but use IndexerFactory.create() for new code
INDEXERS = {
    'inverted': InvertedIndex,
    'trie': TrieIndex,
    'suffix': SuffixArrayIndex
}

# For backward compatibility, but use IndexerFactory.get_description() for new code
INDEXER_DESCRIPTIONS = {
    'inverted': IndexerFactory.get_description('inverted'),
    'trie': IndexerFactory.get_description('trie'),
    'suffix': IndexerFactory.get_description('suffix')
}
