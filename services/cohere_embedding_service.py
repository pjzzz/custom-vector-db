import cohere
import numpy as np
from typing import List, Dict, Optional, Union
import logging
import threading

logger = logging.getLogger(__name__)


class CohereEmbeddingService:
    """
    An embedding service that uses Cohere's API to generate embeddings.
    
    This implementation provides high-quality embeddings through Cohere's
    pre-trained models, offering better semantic understanding than the
    custom TF-IDF implementation.
    
    Thread-safe implementation with optimized embedding generation.
    """

    def __init__(self, api_key: str, model: str = "embed-english-v3.0", vector_size: int = 1024):
        """
        Initialize the Cohere embedding service.
        
        Args:
            api_key: Cohere API key
            model: Cohere embedding model to use
            vector_size: Dimension of the embedding vectors (depends on the model)
        """
        self.api_key = api_key
        self.model = model
        self.vector_size = vector_size
        
        # Initialize Cohere client
        self.client = cohere.Client(api_key)
        
        # Thread safety
        self._model_lock = threading.RLock()  # Reentrant lock for model updates
        
        logger.info(f"CohereEmbeddingService initialized with model={model}")
    
    def fit(self, texts: List[str]):
        """
        No-op method to maintain compatibility with the custom embedding service.
        Cohere's models are already pre-trained, so no fitting is required.
        
        Args:
            texts: List of text documents (ignored)
        """
        logger.info("Cohere embedding models are pre-trained, no fitting required")
        return
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text using Cohere's API.
        Thread-safe implementation.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        with self._model_lock:
            try:
                # Call Cohere API to get embeddings
                response = self.client.embed(
                    texts=[text],
                    model=self.model,
                    input_type="search_query"
                )
                
                # Extract the embedding vector
                embedding = response.embeddings[0]
                
                # Normalize the embedding vector
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding_array = embedding_array / norm
                
                return embedding_array.tolist()
            except Exception as e:
                logger.error(f"Error getting embedding from Cohere: {str(e)}")
                # Return a zero vector as fallback
                return [0.0] * self.vector_size
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts using Cohere's API.
        Thread-safe implementation.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        with self._model_lock:
            try:
                if not texts:
                    return []
                
                # Call Cohere API to get embeddings for multiple texts
                response = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type="search_query"
                )
                
                # Extract the embedding vectors and normalize them
                embeddings = []
                for emb in response.embeddings:
                    emb_array = np.array(emb)
                    norm = np.linalg.norm(emb_array)
                    if norm > 0:
                        emb_array = emb_array / norm
                    embeddings.append(emb_array.tolist())
                
                return embeddings
            except Exception as e:
                logger.error(f"Error getting embeddings from Cohere: {str(e)}")
                # Return zero vectors as fallback
                return [[0.0] * self.vector_size for _ in range(len(texts))]
    
    def get_model_info(self) -> Dict:
        """
        Get information about the embedding model.
        Thread-safe implementation.
        
        Returns:
            Dict with model information
        """
        with self._model_lock:
            return {
                "provider": "cohere",
                "model": self.model,
                "vector_size": self.vector_size,
                "is_fitted": True  # Always true for pre-trained models
            }
    
    def get_model_data(self):
        """
        Get the model data for persistence.
        Thread-safe implementation.
        
        Returns:
            Dict containing the model data
        """
        with self._model_lock:
            return {
                "provider": "cohere",
                "model": self.model,
                "vector_size": self.vector_size,
                "api_key": "***"  # Don't store the actual API key
            }
    
    def load_model_data(self, model_data):
        """
        Load the model data from persistence.
        Thread-safe implementation.
        
        Args:
            model_data: Dict containing the model data
        """
        with self._model_lock:
            # Only update the model and vector size, not the API key
            if "model" in model_data:
                self.model = model_data["model"]
            if "vector_size" in model_data:
                self.vector_size = model_data["vector_size"]
            
            logger.info(f"Loaded Cohere embedding model configuration: {self.model}")
