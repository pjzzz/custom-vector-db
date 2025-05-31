from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict
from .config import EXAMPLE_TIMESTAMP
from .document import Document

class Library(BaseModel):
    """Represents a collection of documents."""
    
    id: str = Field(..., description="Unique identifier for the library")
    name: str = Field(..., description="Name of the library")
    description: Optional[str] = Field(
        None, description="Optional description of the library"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of library creation"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of last update"
    )
    documents: List[Document] = Field(
        default_factory=list,
        description="List of documents in this library"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata about the library"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "lib_123",
                "name": "Technical Documentation",
                "description": "Library containing technical documentation",
                "created_at": EXAMPLE_TIMESTAMP,
                "updated_at": EXAMPLE_TIMESTAMP,
                "documents": [
                    {
                        "id": "doc_123",
                        "title": "Sample Document",
                        "library_id": "lib_123"
                    }
                ],
                "metadata": {
                    "access_level": "public",
                    "category": "technical"
                }
            }
        }
