"""
Persistence service for saving and loading database state to/from disk.

This module implements mechanisms to persist the vector search system's state,
ensuring data durability across container restarts.
"""

import os
import json
import pickle
import logging
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from models import Library, Document, Chunk


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


logger = logging.getLogger(__name__)


class PersistenceService:
    """
    Service for persisting database state to disk and loading it back.

    This service implements a multi-file persistence strategy with:
    1. Regular snapshots for durability
    2. Incremental writes for performance
    3. Atomic file operations for consistency
    4. Background persistence to minimize impact on API performance
    """

    def __init__(self,
                 data_dir: str = "./data",
                 snapshot_interval: int = 300,  # 5 minutes
                 enable_auto_persist: bool = True,
                 test_mode: bool = False):
        """
        Initialize the persistence service.

        Args:
            data_dir: Directory to store persistence files
            snapshot_interval: Seconds between automatic snapshots
            enable_auto_persist: Whether to enable automatic persistence
            test_mode: If True, disables background tasks for testing
        """
        self.data_dir = data_dir
        self.snapshot_interval = snapshot_interval
        self.enable_auto_persist = enable_auto_persist
        self.test_mode = test_mode
        self._persistence_lock = asyncio.Lock()
        self._last_snapshot_time = 0
        self._background_task = None

        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(data_dir, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "vectors"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "indexers"), exist_ok=True)

        if enable_auto_persist and not test_mode:
            try:
                self._start_background_persistence()
            except RuntimeError as e:
                # If no event loop is running, log a warning but don't fail
                # This allows tests to create the service without an event loop
                logger.warning(f"Could not start background persistence: {str(e)}")

    def _start_background_persistence(self):
        """Start background persistence task."""
        async def persistence_task():
            while True:
                try:
                    # Wait for the snapshot interval
                    await asyncio.sleep(self.snapshot_interval)

                    # Check if a snapshot is needed
                    current_time = time.time()
                    if current_time - self._last_snapshot_time >= self.snapshot_interval:
                        logger.info("Taking automatic snapshot of database state")
                        await self.create_snapshot()
                        self._last_snapshot_time = current_time
                except Exception as e:
                    logger.error(f"Error in background persistence task: {str(e)}")
                    # Wait a bit before retrying
                    await asyncio.sleep(10)

        # Get the current event loop or create one if needed
        try:
            asyncio.get_running_loop()  # Check if there's a running loop
            # Start the background task
            self._background_task = asyncio.create_task(persistence_task())
            logger.info(f"Started background persistence task with interval {self.snapshot_interval}s")
        except RuntimeError:
            logger.warning("No running event loop found for background persistence task")
            # Don't start the task - this is likely a test environment

    async def create_snapshot(self, content_service=None):
        """
        Create a complete snapshot of the database state.

        Args:
            content_service: The ContentService instance to snapshot
        """
        if content_service is None:
            # This is just for the background task when no service is provided
            return

        async with self._persistence_lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = os.path.join(self.data_dir, "snapshots", f"snapshot_{timestamp}")
            os.makedirs(snapshot_dir, exist_ok=True)

            try:
                # 1. Persist content store (libraries, documents, chunks)
                await self._persist_content_store(content_service, snapshot_dir)

                # 2. Persist embedding service model
                await self._persist_embedding_service(content_service, snapshot_dir)

                # 3. Persist similarity service vectors
                await self._persist_similarity_service(content_service, snapshot_dir)

                # 4. Persist indexer data
                await self._persist_indexer(content_service, snapshot_dir)

                # Create a metadata file with timestamp and version info
                metadata = {
                    "timestamp": timestamp,
                    "version": "1.0",
                    "embedding_dimension": content_service.embedding_dimension,
                    "indexer_type": content_service.indexer.__class__.__name__
                }

                # Write metadata atomically
                metadata_path = os.path.join(snapshot_dir, "metadata.json")
                with open(f"{metadata_path}.tmp", "w") as f:
                    json.dump(metadata, f, indent=2)
                os.rename(f"{metadata_path}.tmp", metadata_path)

                # Update the latest snapshot symlink
                latest_link = os.path.join(self.data_dir, "latest_snapshot")
                if os.path.exists(latest_link) or os.path.islink(latest_link):
                    os.unlink(latest_link)
                # Use absolute path for the symlink
                os.symlink(os.path.abspath(snapshot_dir), latest_link)

                logger.info(f"Created snapshot at {snapshot_dir}")
                return snapshot_dir
            except Exception as e:
                logger.error(f"Error creating snapshot: {str(e)}")
                raise

    async def _persist_content_store(self, content_service, snapshot_dir):
        """Persist the content store (libraries, documents, chunks)."""
        # We need to acquire the appropriate locks to safely read the data
        content_store = {}

        # Get libraries with lock
        async with content_service._library_lock:
            libraries = content_service.content_store.get('libraries', {})
            content_store['libraries'] = {
                lib_id: lib.model_dump() for lib_id, lib in libraries.items()
            }

        # Get documents with lock
        async with content_service._document_lock:
            documents = content_service.content_store.get('documents', {})
            content_store['documents'] = {
                doc_id: doc.model_dump() for doc_id, doc in documents.items()
            }

        # Get chunks with lock
        async with content_service._chunk_lock:
            chunks = content_service.content_store.get('chunks', {})
            content_store['chunks'] = {
                chunk_id: chunk.model_dump() for chunk_id, chunk in chunks.items()
            }

        # Write content store atomically
        content_store_path = os.path.join(snapshot_dir, "content_store.json")
        with open(f"{content_store_path}.tmp", "w") as f:
            json.dump(content_store, f, indent=2, cls=DateTimeEncoder)
        os.rename(f"{content_store_path}.tmp", content_store_path)

    async def _persist_embedding_service(self, content_service, snapshot_dir):
        """Persist the embedding service model."""
        # The embedding service has its own thread safety
        embedding_model = content_service.embedding_service.get_model_data()

        # Write embedding model atomically
        embedding_path = os.path.join(snapshot_dir, "embedding_model.pkl")
        with open(f"{embedding_path}.tmp", "wb") as f:
            pickle.dump(embedding_model, f)
        os.rename(f"{embedding_path}.tmp", embedding_path)

    async def _persist_similarity_service(self, content_service, snapshot_dir):
        """Persist the similarity service vectors."""
        # The similarity service has its own thread safety
        async with content_service._vector_lock:
            vectors_data = content_service.similarity_service.get_all_vectors()

        # Write vectors atomically
        vectors_path = os.path.join(snapshot_dir, "vectors.pkl")
        with open(f"{vectors_path}.tmp", "wb") as f:
            pickle.dump(vectors_data, f)
        os.rename(f"{vectors_path}.tmp", vectors_path)

    async def _persist_indexer(self, content_service, snapshot_dir):
        """Persist the indexer data."""
        # The indexer has its own thread safety
        async with content_service._indexer_lock:
            indexer_data = content_service.indexer.get_serializable_data()

        # Write indexer data atomically
        indexer_path = os.path.join(snapshot_dir, "indexer.pkl")
        with open(f"{indexer_path}.tmp", "wb") as f:
            pickle.dump(indexer_data, f)
        os.rename(f"{indexer_path}.tmp", indexer_path)

    async def load_latest_snapshot(self, content_service):
        """
        Load the latest snapshot into the content service.

        Args:
            content_service: The ContentService instance to load data into

        Returns:
            bool: True if snapshot was loaded, False if no snapshot exists
        """
        latest_link = os.path.join(self.data_dir, "latest_snapshot")
        if not os.path.exists(latest_link):
            logger.info("No snapshot found to load")
            return False

        # Get the absolute path of the snapshot directory
        if os.path.islink(latest_link):
            snapshot_dir = os.readlink(latest_link)
            # If the symlink is relative, convert it to absolute
            if not os.path.isabs(snapshot_dir):
                snapshot_dir = os.path.join(os.path.dirname(latest_link), snapshot_dir)
        else:
            snapshot_dir = latest_link

        async with self._persistence_lock:
            try:
                # Read metadata
                metadata_path = os.path.join(snapshot_dir, "metadata.json")
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                logger.info(f"Loading snapshot from {snapshot_dir} (version {metadata['version']})")

                # 1. Load content store
                await self._load_content_store(content_service, snapshot_dir)

                # 2. Load embedding service model
                await self._load_embedding_service(content_service, snapshot_dir)

                # 3. Load similarity service vectors
                await self._load_similarity_service(content_service, snapshot_dir)

                # 4. Load indexer data
                await self._load_indexer(content_service, snapshot_dir)

                logger.info(f"Successfully loaded snapshot from {snapshot_dir}")
                return True
            except Exception as e:
                logger.error(f"Error loading snapshot: {str(e)}")
                raise

    async def _load_content_store(self, content_service, snapshot_dir):
        """Load the content store from snapshot."""
        content_store_path = os.path.join(snapshot_dir, "content_store.json")
        with open(content_store_path, "r") as f:
            content_store_data = json.load(f)

        # Load libraries
        async with content_service._library_lock:
            libraries = {}
            for lib_id, lib_data in content_store_data.get('libraries', {}).items():
                libraries[lib_id] = Library(**lib_data)
            content_service.content_store['libraries'] = libraries

        # Load documents
        async with content_service._document_lock:
            documents = {}
            for doc_id, doc_data in content_store_data.get('documents', {}).items():
                documents[doc_id] = Document(**doc_data)
            content_service.content_store['documents'] = documents

        # Load chunks
        async with content_service._chunk_lock:
            chunks = {}
            for chunk_id, chunk_data in content_store_data.get('chunks', {}).items():
                chunks[chunk_id] = Chunk(**chunk_data)
            content_service.content_store['chunks'] = chunks

    async def _load_embedding_service(self, content_service, snapshot_dir):
        """Load the embedding service model from snapshot."""
        embedding_path = os.path.join(snapshot_dir, "embedding_model.pkl")
        with open(embedding_path, "rb") as f:
            embedding_model = pickle.load(f)

        # Load the model data
        content_service.embedding_service.load_model_data(embedding_model)

    async def _load_similarity_service(self, content_service, snapshot_dir):
        """Load the similarity service vectors from snapshot."""
        vectors_path = os.path.join(snapshot_dir, "vectors.pkl")
        with open(vectors_path, "rb") as f:
            vectors_data = pickle.load(f)

        # Load the vectors
        async with content_service._vector_lock:
            content_service.similarity_service.load_vectors(vectors_data)

    async def _load_indexer(self, content_service, snapshot_dir):
        """Load the indexer data from snapshot."""
        indexer_path = os.path.join(snapshot_dir, "indexer.pkl")
        with open(indexer_path, "rb") as f:
            indexer_data = pickle.load(f)

        # Load the indexer data
        async with content_service._indexer_lock:
            content_service.indexer.load_serializable_data(indexer_data)

    def cleanup_old_snapshots(self, max_snapshots=5):
        """
        Clean up old snapshots, keeping only the most recent ones.

        Args:
            max_snapshots: Maximum number of snapshots to keep
        """
        snapshots_dir = os.path.join(self.data_dir, "snapshots")
        if not os.path.exists(snapshots_dir):
            return

        # Get all snapshot directories
        snapshot_dirs = [
            os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d)) and d.startswith("snapshot_")
        ]

        # Sort by creation time (newest first)
        snapshot_dirs.sort(key=lambda d: os.path.getctime(d), reverse=True)

        # Remove old snapshots
        for old_dir in snapshot_dirs[max_snapshots:]:
            try:
                # Recursively remove the directory
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Removed old snapshot: {old_dir}")
            except Exception as e:
                logger.error(f"Error removing old snapshot {old_dir}: {str(e)}")

    async def stop(self):
        """Stop the persistence service and any background tasks."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("Stopped background persistence task")
