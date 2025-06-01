import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Literal

load_dotenv()


class Settings(BaseSettings):
    # Index settings
    INDEX_NAME: str = "stack-ai-index"
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 100
    BATCH_SIZE: int = 100
    INDEXER_TYPE: str = "inverted"  # Default indexer type - can be 'inverted', 'trie', or 'suffix'
    
    # Embedding settings
    EMBEDDING_PROVIDER: Literal["custom", "cohere"] = "cohere"  # Use 'custom' or 'cohere'
    EMBEDDING_DIMENSION: int = 1536  # For custom embedding service
    COHERE_API_KEY: str = "A1Fi5KBBNoekwBPIa833CBScs6Z2mHEtOXxr52KO"  # Cohere API key
    COHERE_MODEL: str = "embed-english-v3.0"  # Cohere embedding model
    COHERE_DIMENSION: int = 1024  # Dimension for Cohere embeddings


settings = Settings()
