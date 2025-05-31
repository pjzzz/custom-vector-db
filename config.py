import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    INDEX_NAME: str = "stack-ai-index"
    EMBEDDING_DIMENSION: int = 1536
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 100
    BATCH_SIZE: int = 100


settings = Settings()
