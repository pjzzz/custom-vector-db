from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict
from .config import EXAMPLE_TIMESTAMP
from .chunk import Chunk

class Document(BaseModel):
    """Represents a document containing multiple chunks."""
    
    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    library_id: str = Field(..., description="ID of the library this document belongs to")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of document creation"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of last update"
    )
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="List of chunks in this document"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata about the document"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_123",
                "title": "Sample Document",
                "library_id": "lib_123",
                "created_at": EXAMPLE_TIMESTAMP,
                "updated_at": EXAMPLE_TIMESTAMP,
                "chunks": [
                    {
                        "id": "chunk_123",
                        "text": "This is a sample text chunk.",
                        "document_id": "doc_123",
                        "position": 1
                    }
                ],
                "metadata": {
                    "author": "John Doe",
                    "category": "technical"
                }
            }
        }
