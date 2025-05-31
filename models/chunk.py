from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict
from .config import EXAMPLE_TIMESTAMP

class Chunk(BaseModel):
    """Represents a text chunk with metadata and vector embedding."""

    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    document_id: str = Field(..., description="ID of the document this chunk belongs to")
    position: int = Field(..., description="Position of this chunk in the document")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of chunk creation"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123",
                "text": "This is a sample text chunk.",
                "document_id": "doc_123",
                "position": 1,
                "created_at": EXAMPLE_TIMESTAMP,
                "metadata": {
                    "source": "chapter_1",
                    "page": "1"
                }
            }
        }
