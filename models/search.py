from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class SearchRequest(BaseModel):
    """Request model for vector search."""
    
    query: str = Field(..., description="The text query to search for")
    top_k: int = Field(10, description="Number of results to return")
    filter: Optional[Dict[str, str]] = Field(
        None, 
        description="Optional metadata filter for search results"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is vector search?",
                "top_k": 5,
                "filter": {
                    "category": "technical"
                }
            }
        }

class SearchResponse(BaseModel):
    """Response model for vector search results."""
    
    matches: List[Dict] = Field(..., description="List of matching results")
    query_vector: List[float] = Field(..., description="Vector representation of the query")
    top_k: int = Field(..., description="Number of results returned")
    
    class Config:
        json_schema_extra = {
            "example": {
                "matches": [
                    {
                        "id": "chunk_123",
                        "score": 0.92,
                        "metadata": {
                            "source": "chapter_1",
                            "page": "1"
                        }
                    }
                ],
                "query_vector": [0.1, 0.2, 0.3],
                "top_k": 5
            }
        }
