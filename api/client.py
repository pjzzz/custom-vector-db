import requests
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VectorSearchClient:
    """
    Client for interacting with the Vector Content Management API.
    
    This client provides methods for all API endpoints and handles
    the HTTP requests and response parsing.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client with the base URL of the API.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            data: Request data (for POST requests)
            params: Query parameters
            
        Returns:
            Response data as a dictionary
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"        
        try:
            if method == "GET":
                response = requests.get(url, params=params)
            elif method == "POST":
                response = requests.post(url, json=data, params=params)
            elif method == "DELETE":
                response = requests.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    # Health check
    def health_check(self) -> Dict:
        """Check if the API is running."""
        return self._request("GET", "/health")
    
    # Library methods
    def create_library(self, id: str, name: str, description: str) -> Dict:
        """
        Create a new library.
        
        Args:
            id: Library ID
            name: Library name
            description: Library description
            
        Returns:
            Response data
        """
        data = {
            "id": id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        return self._request("POST", "/libraries", data)
    
    def get_library(self, library_id: str) -> Dict:
        """
        Get a library by ID.
        
        Args:
            library_id: Library ID
            
        Returns:
            Library data
        """
        return self._request("GET", f"/libraries/{library_id}")
    
    def delete_library(self, library_id: str) -> Dict:
        """
        Delete a library by ID.
        
        Args:
            library_id: Library ID
            
        Returns:
            Response data
        """
        return self._request("DELETE", f"/libraries/{library_id}")
    
    # Document methods
    def create_document(self, id: str, library_id: str, title: str, 
                       content: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a new document.
        
        Args:
            id: Document ID
            library_id: Library ID
            title: Document title
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Response data
        """
        data = {
            "id": id,
            "library_id": library_id,
            "title": title,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        return self._request("POST", "/documents", data)
    
    def get_document(self, document_id: str) -> Dict:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data
        """
        return self._request("GET", f"/documents/{document_id}")
    
    def delete_document(self, document_id: str) -> Dict:
        """
        Delete a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Response data
        """
        return self._request("DELETE", f"/documents/{document_id}")
    
    # Chunk methods
    def create_chunk(self, id: str, document_id: str, text: str, 
                    position: int, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a new chunk.
        
        Args:
            id: Chunk ID
            document_id: Document ID
            text: Chunk text
            position: Chunk position
            metadata: Optional metadata
            
        Returns:
            Response data
        """
        data = {
            "id": id,
            "document_id": document_id,
            "text": text,
            "position": position,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        return self._request("POST", "/chunks", data)
    
    def get_chunk(self, chunk_id: str) -> Dict:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk data
        """
        return self._request("GET", f"/chunks/{chunk_id}")
    
    def delete_chunk(self, chunk_id: str) -> Dict:
        """
        Delete a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Response data
        """
        return self._request("DELETE", f"/chunks/{chunk_id}")
    
    # Search methods
    def text_search(self, query: str, indexer_type: Optional[str] = None) -> List[Dict]:
        """
        Search for chunks using text search.
        
        Args:
            query: Search query
            indexer_type: Optional indexer type (suffix, trie, inverted)
            
        Returns:
            List of matching chunks
        """
        params = {
            "query": query
        }
        if indexer_type:
            params["indexer_type"] = indexer_type
            
        return self._request("POST", "/search/text", params=params)
    
    def vector_search(self, query_text: str, top_k: int = 10, 
                     filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for chunks using vector similarity search.
        
        Args:
            query_text: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of matching chunks with similarity scores
        """
        params = {
            "query_text": query_text,
            "top_k": top_k
        }
        if filter_dict:
            params["filter_dict"] = json.dumps(filter_dict)
            
        return self._request("POST", "/search/vector", params=params)
    
    # Utility methods
    def get_stats(self) -> Dict:
        """
        Get statistics about the content service.
        
        Returns:
            Statistics data
        """
        return self._request("GET", "/stats")


# Example usage
if __name__ == "__main__":
    import uuid
    
    # Create client
    client = VectorSearchClient()
    
    try:
        # Check health
        health = client.health_check()
        print(f"API Health: {health}")
        
        # Create library
        library_id = f"lib-{uuid.uuid4().hex[:8]}"
        library = client.create_library(
            id=library_id,
            name="Test Library",
            description="Library created by client library"
        )
        print(f"Created library: {library}")
        
        # Create document
        document_id = f"doc-{uuid.uuid4().hex[:8]}"
        document = client.create_document(
            id=document_id,
            library_id=library_id,
            title="Test Document",
            content="This is a test document created by the client library",
            metadata={"source": "client_library"}
        )
        print(f"Created document: {document}")
        
        # Create chunks
        chunks = []
        for i in range(3):
            chunk_id = f"chunk-{uuid.uuid4().hex[:8]}"
            chunk = client.create_chunk(
                id=chunk_id,
                document_id=document_id,
                text=f"This is test chunk {i+1} for vector similarity search",
                position=i,
                metadata={"index": i}
            )
            chunks.append(chunk)
            print(f"Created chunk {i+1}: {chunk}")
        
        # Vector search
        vector_results = client.vector_search(
            query_text="similar vector search",
            top_k=5
        )
        print(f"Vector search results: {vector_results}")
        
        # Text search
        text_results = client.text_search(
            query="test chunk",
            indexer_type="suffix"
        )
        print(f"Text search results: {text_results}")
        
        # Get stats
        stats = client.get_stats()
        print(f"API Stats: {stats}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
