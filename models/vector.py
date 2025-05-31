from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class UpsertRequest(BaseModel):
    """Request model for upserting vectors."""

    id: str = Field(..., description="Unique identifier for the vector")
    values: List[float] = Field(..., description="Vector values")
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Optional metadata associated with the vector"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "vector_123",
                "values": [0.1, 0.2, 0.3],
                "metadata": {
                    "source": "document_1",
                    "category": "technical"
                }
            }
        }

class DeleteRequest(BaseModel):
    """Request model for deleting vectors."""

    ids: List[str] = Field(..., description="List of vector IDs to delete")

    class Config:
        json_schema_extra = {
            "example": {
                "ids": ["vector_123", "vector_456"]
            }
        }

class TextEmbeddingRequest(BaseModel):
    """Request model for generating text embeddings."""

    text: str = Field(..., description="Text to generate embedding for")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text for embedding."
            }
        }
