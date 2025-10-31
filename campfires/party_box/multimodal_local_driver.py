"""
Enhanced local driver with multimodal support and metadata extraction.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import mimetypes
import aiofiles
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from campfires.party_box.local_driver import LocalDriver
from .metadata_extractor import MetadataExtractor, ThumbnailGenerator


logger = logging.getLogger(__name__)


class MultimodalLocalDriver(LocalDriver):
    """
    A local filesystem driver for multimodal assets, extending LocalDriver.
    Handles metadata extraction, thumbnail generation, indexing, and deduplication.
    """

    def __init__(self, base_path: str = "./multimodal_party_box", config: Optional[Dict[str, Any]] = None, ollama_client=None):
        super().__init__(base_path, config, ollama_client)
        self.metadata_dir = self.base_path / "metadata"
        self.thumbnails_dir = self.base_path / "thumbnails"
        self.index_dir = self.base_path / "indexes"

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        self.enable_thumbnails = config.get("enable_thumbnails", True) if config else True
        self.enable_deduplication = config.get("enable_deduplication", True) if config else True
        self.metadata_cache_size = config.get("metadata_cache_size", 1000) if config else 1000

        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._inverted_index: Dict[str, List[str]] = {}
        self._fingerprint_index: Dict[str, str] = {}

        self._load_indexes()

    async def put(self, content: Union[Path, str, bytes], content_type: str, filename: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a multimodal asset.

        Args:
            content: The asset content, can be a Path object, a string path, or bytes.
            content_type: The MIME type of the content (e.g., 'image/png', 'audio/mpeg').
            filename: Optional original filename. If not provided, a name will be derived or defaulted.
            metadata: Optional dictionary of additional metadata to store with the asset.

        Returns:
            The content hash of the stored asset.
        """
        content_bytes: bytes
        local_driver_key: str

        if isinstance(content, Path):
            async with aiofiles.open(content, 'rb') as f:
                content_bytes = await f.read()
            local_driver_key = filename if filename else content.name
        elif isinstance(content, str):
            content_path = Path(content)
            async with aiofiles.open(content_path, 'rb') as f:
                content_bytes = await f.read()
            local_driver_key = filename if filename else content_path.name
        else:  # content is bytes
            content_bytes = content
            local_driver_key = filename if filename else f"asset.{mimetypes.guess_extension(content_type) or 'bin'}"

        content_hash = self.generate_hash(content_bytes)

        # Pass the content_hash as the key to the super.put method, ensuring it has an extension
        # The local_driver_key is used for determining the subdirectory and original filename in LocalDriver.put
        await super().put(local_driver_key, content_bytes)

        # Extract and merge metadata
        extracted_metadata = await self._extract_and_merge_metadata(content_bytes, content_type, metadata)
        extracted_metadata['content_hash'] = content_hash
        extracted_metadata['stored_timestamp'] = datetime.utcnow().isoformat()

        try:
            # Deduplication logic
            if self.enable_deduplication:
                fingerprint = MetadataExtractor.generate_content_fingerprint(extracted_metadata)
                if fingerprint in self._fingerprint_index:
                    existing_hash = self._fingerprint_index[fingerprint]
                    logger.info(f"Duplicate content detected. Existing hash: {existing_hash}")
                    # Update metadata to reference existing content
                    extracted_metadata['duplicate_of'] = existing_hash
                    extracted_metadata['is_duplicate'] = True
                else:
                    self._fingerprint_index[fingerprint] = content_hash
                    extracted_metadata['content_fingerprint'] = fingerprint
                    extracted_metadata['is_duplicate'] = False

            # Store metadata
            await self._store_metadata(content_hash, extracted_metadata)

            # Generate thumbnail if enabled and supported
            if self.enable_thumbnails:
                await self._generate_and_store_thumbnail(content_hash, content_bytes, extracted_metadata)

            # Update indexes
            await self._update_indexes(content_hash, extracted_metadata)

            # Cache metadata
            if len(self._metadata_cache) < self.metadata_cache_size:
                self._metadata_cache[content_hash] = extracted_metadata

        except Exception as e:
            logger.error(f"Error processing multimodal content {content_hash}: {e}")
            # Store basic metadata as fallback
            basic_metadata = {
                'content_hash': content_hash,
                'error': str(e),
                'stored_timestamp': datetime.utcnow().isoformat()
            }
            await self._store_metadata(content_hash, basic_metadata)

        return content_hash

    async def get(self, content_hash: str) -> Path:
        """
        Retrieve the file path for a stored asset.

        Args:
            content_hash: The content hash of the asset.

        Returns:
            A Path object pointing to the asset file.

        Raises:
            FileNotFoundError: If the asset is not found.
        """
        metadata = self.get_metadata(content_hash)
        if metadata and 'file_path' in metadata:
            file_path = Path(metadata['file_path'])
            if file_path.exists():
                return file_path

        # Fallback: search all subdirectories if metadata is missing or file_path is invalid
        for subdir in ["images", "audio", "documents", "other"]:
            subdir_path = self.base_path / subdir
            for file_in_subdir in subdir_path.glob(f"{content_hash}.*"):
                if file_in_subdir.is_file():
                    return file_in_subdir

        raise FileNotFoundError(f"Asset with hash {content_hash} not found.")

    async def get_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata for a given content hash.
        """
        if content_hash in self._metadata_cache:
            return self._metadata_cache[content_hash]

        metadata = await self.get_metadata(content_hash)
        metadata_path = self.metadata_dir / f"{content_hash}.json"
        if metadata_path.exists():
            try:
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
                    if len(self._metadata_cache) < self.metadata_cache_size:
                        self._metadata_cache[content_hash] = metadata
                    return metadata
            except Exception as e:
                logger.error(f"Error loading metadata for {content_hash}: {e}")
        return None

    async def exists(self, content_hash: str) -> bool:
        """
        Check if an asset with the given content hash exists.
        """
        try:
            await self.get(content_hash)
            return True
        except FileNotFoundError:
            return False

    async def delete(self, content_hash: str):
        """
        Delete an asset and its associated metadata and thumbnail.
        """
        metadata = await self.get_metadata(content_hash)
        if not metadata:
            logger.info(f"Attempted to delete non-existent asset: {content_hash}")
            return

        # Delete the actual content file
        file_path = Path(metadata['file_path'])
        if file_path.exists():
            file_path.unlink()

        # Delete metadata file
        metadata_path = self.metadata_dir / f"{content_hash}.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Delete thumbnail file
        # TODO: Handle different thumbnail sizes/formats
        thumbnail_path = self.thumbnails_dir / f"{content_hash}_200x200.jpg" # Assuming default size
        if thumbnail_path.exists():
            thumbnail_path.unlink()

        # Remove from indexes and cache
        self._remove_from_indexes(content_hash, metadata)
        self._metadata_cache.pop(content_hash, None)

        logger.info(f"Deleted asset and its metadata/thumbnail for hash: {content_hash}")

    def _remove_from_indexes(self, content_hash: str, metadata: Dict[str, Any]):
        """
        Remove asset from inverted index and fingerprint index.
        """
        # Remove from inverted index
        for key, values in self._inverted_index.items():
            if content_hash in values:
                values.remove(content_hash)

        # Remove from fingerprint index if not a duplicate
        if not metadata.get('is_duplicate', False):
            fingerprint = metadata.get('content_fingerprint')
            if fingerprint and fingerprint in self._fingerprint_index and self._fingerprint_index[fingerprint] == content_hash:
                self._fingerprint_index.pop(fingerprint)

    async def _extract_and_merge_metadata(self, content_bytes: bytes, content_type: str, user_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts metadata from content and merges with user-provided metadata.
        """
        extracted_metadata = await MetadataExtractor.extract_metadata(content_bytes, content_type)
        if user_metadata:
            extracted_metadata.update(user_metadata)
        return extracted_metadata

    async def _store_metadata(self, content_hash: str, metadata: Dict[str, Any]):
        """
        Stores metadata to a JSON file.
        """
        metadata_path = self.metadata_dir / f"{content_hash}.json"
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=4))

    async def _generate_and_store_thumbnail(self, content_hash: str, content_bytes: bytes, metadata: Dict[str, Any]):
        """
        Generates and stores a thumbnail for image/video assets.
        """
        content_type = metadata.get('content_type', '')
        if content_type.startswith('image/') or content_type.startswith('video/') or content_type == 'application/pdf':
            try:
                thumbnail_data = await ThumbnailGenerator.generate_image_thumbnail(content_bytes, content_type)
                if thumbnail_data:
                    thumbnail_path = self.thumbnails_dir / f"{content_hash}_200x200.jpg"
                    async with aiofiles.open(thumbnail_path, 'wb') as f:
                        await f.write(thumbnail_data)
            except Exception as e:
                logger.error(f"Error generating thumbnail for {content_hash}: {e}")

    async def _update_indexes(self, content_hash: str, metadata: Dict[str, Any]):
        """
        Update inverted index with new asset metadata.
        """
        # Index content type
        content_type = metadata.get('content_type')
        if content_type:
            self._inverted_index.setdefault('content_type:' + content_type, []).append(content_hash)

        # Index tags
        tags = metadata.get('tags', [])
        for tag in tags:
            self._inverted_index.setdefault('tag:' + tag, []).append(content_hash)

        # Index other searchable metadata fields (e.g., 'title', 'description')
        for field in ['title', 'description']:
            value = metadata.get(field)
            if value and isinstance(value, str):
                self._inverted_index.setdefault(f'{field}:{value.lower()}', []).append(content_hash)

        await self._save_indexes()

    async def _load_indexes(self):
        """
        Load inverted index and fingerprint index from disk.
        """
        inverted_index_path = self.index_dir / "inverted_index.json"
        fingerprint_index_path = self.index_dir / "fingerprint_index.json"

        if inverted_index_path.exists():
            try:
                async with aiofiles.open(inverted_index_path, 'r', encoding='utf-8') as f:
                    self._inverted_index = json.loads(await f.read())
            except Exception as e:
                logger.error(f"Error loading inverted index: {e}")

        if fingerprint_index_path.exists():
            try:
                async with aiofiles.open(fingerprint_index_path, 'r', encoding='utf-8') as f:
                    self._fingerprint_index = json.loads(await f.read())
            except Exception as e:
                logger.error(f"Error loading fingerprint index: {e}")

    async def _save_indexes(self):
        """
        Save inverted index and fingerprint index to disk.
        """
        inverted_index_path = self.index_dir / "inverted_index.json"
        fingerprint_index_path = self.index_dir / "fingerprint_index.json"
        try:
            async with aiofiles.open(inverted_index_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self._inverted_index, indent=4))
            async with aiofiles.open(fingerprint_index_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self._fingerprint_index, indent=4))
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")

    def generate_hash(self, content: bytes) -> str:
        """
        Generate a SHA256 hash for the given content.
        """
        import hashlib
        return hashlib.sha256(content).hexdigest()

    def __del__(self):
        """
        Ensure indexes are saved on object deletion.
        """
        # Destructors cannot be async, so we can't await _save_indexes here.
        # Consider alternative strategies for graceful shutdown if index saving is critical.
        pass


class MultimodalAssetManager:
    """
    Manages multimodal assets using a MultimodalLocalDriver.
    """

    def __init__(self, driver: MultimodalLocalDriver):
        self.driver = driver

    async def add_asset(self, content: Union[Path, str, bytes], content_type: str = None, filename: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Adds a new asset to the system.

        Args:
            content: The asset content (Path, string path, or bytes).
            content_type: The MIME type of the content. Will be guessed if not provided.
            filename: Optional original filename.
            metadata: Optional additional metadata.

        Returns:
            The content hash of the added asset.
        """
        if not content_type:
            if isinstance(content, Path):
                content_type = mimetypes.guess_type(content.name)[0] or 'application/octet-stream'
            elif isinstance(content, str):
                content_type = mimetypes.guess_type(content)[0] or 'application/octet-stream'
            else:
                content_type = 'application/octet-stream' # Default for bytes if no filename hint

        return await self.driver.put(content, content_type, filename, metadata)

    async def get_asset(self, content_hash: str) -> Path:
        """
        Retrieves an asset by its content hash.

        Args:
            content_hash: The content hash of the asset.

        Returns:
            Path to the asset file.
        """
        return self.driver.get(content_hash)

    async def delete_asset(self, content_hash: str):
        """
        Deletes an asset by its content hash.
        """
        self.driver.delete(content_hash)

    def get_asset_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata for an asset.
        """
        return self.driver.get_metadata(content_hash)

    def search_assets(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Searches for assets based on various criteria.
        """
        return self.driver.search_assets(**kwargs)

    def get_content_stats(self) -> Dict[str, Any]:
        """
        Gets statistics about stored content.
        """
        return self.driver.get_content_stats()

    def cleanup_orphaned_metadata(self) -> int:
        """
        Cleans up metadata files that no longer have corresponding asset files.
        """
        return self.driver.cleanup_orphaned_metadata()