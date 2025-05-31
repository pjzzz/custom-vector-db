import numpy as np
from typing import List, Dict, Optional, Union
import re
import logging
import threading
from collections import Counter

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    A custom embedding service that generates vector embeddings for text
    without relying on external services.

    This implementation uses a simple but effective TF-IDF based approach
    combined with dimensionality reduction to create meaningful embeddings.

    Thread-safe implementation with optimized embedding generation.
    """

    def __init__(self, vector_size: int = 256, min_word_freq: int = 2):
        """
        Initialize the embedding service.

        Args:
            vector_size: Dimension of the embedding vectors
            min_word_freq: Minimum frequency for a word to be included in the vocabulary
        """
        self.vector_size = vector_size
        self.min_word_freq = min_word_freq

        # Vocabulary and IDF values
        self.vocabulary: Dict[str, int] = {}  # word -> index
        self.idf: Dict[str, float] = {}      # word -> idf value
        self.doc_count = 0

        # Random projection matrix for dimensionality reduction
        self.projection_matrix = None

        # Thread safety
        self._model_lock = threading.RLock()  # Reentrant lock for model updates

        logger.info(f"EmbeddingService initialized with vector_size={vector_size}")

    def fit(self, texts: List[str]):
        """
        Fit the embedding model on a corpus of texts.
        Thread-safe implementation.

        Args:
            texts: List of text documents to fit the model on
        """
        with self._model_lock:
            # Count word frequencies across all documents
            word_counts = Counter()
            doc_word_counts = []

            for text in texts:
                tokens = self._tokenize(text)
                doc_words = set(tokens)  # Unique words in this document
                doc_word_counts.append(doc_words)
                word_counts.update(doc_words)  # Count each word once per document

            # Build vocabulary with words that appear at least min_word_freq times
            self.vocabulary = {}
            index = 0
            for word, count in word_counts.items():
                if count >= self.min_word_freq:
                    self.vocabulary[word] = index
                    index += 1

            # Calculate IDF values
            self.doc_count = len(texts)
            self.idf = {}

            for word, idx in self.vocabulary.items():
                # Count documents containing this word
                doc_freq = sum(1 for doc_words in doc_word_counts if word in doc_words)
                # Calculate IDF: log(N/df)
                self.idf[word] = np.log(self.doc_count / (1 + doc_freq))

            # Create random projection matrix for dimensionality reduction
            vocab_size = len(self.vocabulary)
            if vocab_size > 0:
                # Initialize with small random values for stability using the new Generator API
                # Use a fixed seed for reproducibility
                rng = np.random.default_rng(seed=42)
                self.projection_matrix = rng.normal(
                    0, 0.1, (vocab_size, self.vector_size)
                )

                logger.info(f"Embedding model fitted with vocabulary size: {vocab_size}")
            else:
                logger.warning("No words met the minimum frequency threshold")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        Thread-safe implementation.

        Args:
            text: The text to generate an embedding for

        Returns:
            Embedding vector as a list of floats
        """
        with self._model_lock:
            # Check if model is fitted
            if not self.vocabulary or self.projection_matrix is None:
                # Return a random vector if model is not fitted
                logger.warning("Embedding model not fitted, returning random vector")
                # Use the new Generator API with a fixed seed for reproducibility
                rng = np.random.default_rng(seed=42)
                return list(rng.normal(0, 0.1, self.vector_size))

            # Generate TF-IDF vector
            tokens = self._tokenize(text)
            tf_vector = self._calculate_tf(tokens)

            # Apply dimensionality reduction
            embedding = np.zeros(self.vector_size)
            if len(tf_vector) > 0:
                # Project the sparse TF-IDF vector to the lower-dimensional space
                for word, tf in tf_vector.items():
                    if word in self.vocabulary and word in self.idf:
                        idx = self.vocabulary[word]
                        tfidf = tf * self.idf[word]
                        embedding += tfidf * self.projection_matrix[idx]

            # Normalize the embedding vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.
        Thread-safe implementation.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        return [self.get_embedding(text) for text in texts]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: The text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate term frequency for tokens.

        Args:
            tokens: List of tokens

        Returns:
            Dict mapping tokens to their term frequencies
        """
        if not tokens:
            return {}

        # Count token frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        # Calculate term frequency
        return {word: count / total_tokens for word, count in token_counts.items()}

    def get_model_info(self) -> Dict:
        """
        Get information about the embedding model.
        Thread-safe implementation.

        Returns:
            Dict with model information
        """
        with self._model_lock:
            return {
                "vector_size": self.vector_size,
                "vocabulary_size": len(self.vocabulary),
                "document_count": self.doc_count,
                "is_fitted": self.projection_matrix is not None
            }

    def get_model_data(self):
        """
        Get the model data for persistence.
        Thread-safe implementation.

        Returns:
            Dict containing the model data
        """
        with self._model_lock:
            model_data = {
                "vector_size": self.vector_size,
                "min_word_freq": self.min_word_freq,
                "vocabulary": self.vocabulary,
                "idf": self.idf,
                "doc_count": self.doc_count,
            }

            # Convert projection matrix to list if it exists
            if self.projection_matrix is not None:
                model_data["projection_matrix"] = self.projection_matrix.tolist()
            else:
                model_data["projection_matrix"] = None

            return model_data

    def load_model_data(self, model_data):
        """
        Load the model data from persistence.
        Thread-safe implementation.

        Args:
            model_data: Dict containing the model data
        """
        with self._model_lock:
            self.vector_size = model_data["vector_size"]
            self.min_word_freq = model_data["min_word_freq"]
            self.vocabulary = model_data["vocabulary"]
            self.idf = model_data["idf"]
            self.doc_count = model_data["doc_count"]

            # Convert projection matrix back to numpy array if it exists
            if model_data["projection_matrix"] is not None:
                self.projection_matrix = np.array(model_data["projection_matrix"])
            else:
                self.projection_matrix = None

            logger.info(f"Loaded embedding model with {len(self.vocabulary)} words in vocabulary")
