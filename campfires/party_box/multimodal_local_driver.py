"""
Enhanced local driver with multimodal support and metadata extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime

from .local_driver import LocalDriver
from .metadata_extractor import MetadataExtractor, ThumbnailGenerator

logger = logging.getLogger(__name__)


class MultimodalLocalDriver(LocalDriver):
    """
    Enhanced local driver with multimodal support and metadata extraction.
    
    Extends LocalDriver to provide:
    - Automatic metadata extraction for all content types
    - Thumbnail generation for supported formats
    - Enhanced asset organization with metadata indexing
    - Content deduplication based on fingerprints
    - Rich asset search and filtering capabilities
    """
    
    def __init__(self, base_path: str, enable_thumbnails: bool = True, 
                 enable_deduplication: bool = True, metadata_cache_size: int = 1000):
        """
        Initialize multimodal local driver.
        
        Args:
            base_path: Base directory for asset storage
            enable_thumbnails: Whether to generate thumbnails
            enable_deduplication: Whether to enable content deduplication
            metadata_cache_size: Size of metadata cache
        """
        super().__init__(base_path)
        
        self.enable_thumbnails = enable_thumbnails
        self.enable_deduplication = enable_deduplication
        self.metadata_cache_size = metadata_cache_size
        
        # Additional directories for multimodal support
        self.metadata_dir = self.base_path / "metadata"
        self.thumbnails_dir = self.base_path / "thumbnails"
        self.index_dir = self.base_path / "indexes"
        
        # Create additional directories
        for directory in [self.metadata_dir, self.thumbnails_dir, self.index_dir]:
            directory.mkdir(exist_ok=True)
        
        # In-memory caches
        self._metadata_cache = {}
        self._fingerprint_index = {}
        
        # Load existing indexes
        self._load_indexes()
    
    def put(self, content: bytes, filename: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Store content with enhanced multimodal support.
        
        Args:
            content: File content as bytes
            filename: Optional filename for metadata extraction
            metadata: Optional additional metadata
            
        Returns:
            Content hash
        """
        # Get basic hash and store content
        content_hash = super().put(content, filename)
        
        try:
            # Extract metadata
            file_path = Path(filename) if filename else Path(f"unknown.{content_hash[:8]}")
            extracted_metadata = MetadataExtractor.extract_metadata(file_path, content)
            
            # Merge with provided metadata
            if metadata:
                extracted_metadata.update(metadata)
            
            # Add storage information
            extracted_metadata.update({
                'content_hash': content_hash,
                'storage_path': str(self._get_asset_path(content_hash)),
                'stored_timestamp': datetime.utcnow().isoformat(),
                'driver_version': '2.0.0'
            })
            
            # Check for duplicates if enabled
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
            self._store_metadata(content_hash, extracted_metadata)
            
            # Generate thumbnail if enabled and supported
            if self.enable_thumbnails:
                self._generate_and_store_thumbnail(content_hash, content, extracted_metadata)
            
            # Update indexes
            self._update_indexes(content_hash, extracted_metadata)
            
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
            self._store_metadata(content_hash, basic_metadata)
        
        return content_hash
    
    def get_metadata(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an asset.
        
        Args:
            content_hash: Content hash
            
        Returns:
            Metadata dictionary or None if not found
        """
        # Check cache first
        if content_hash in self._metadata_cache:
            return self._metadata_cache[content_hash]
        
        # Load from disk
        metadata_path = self.metadata_dir / f"{content_hash}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Cache if space available
                if len(self._metadata_cache) < self.metadata_cache_size:
                    self._metadata_cache[content_hash] = metadata
                
                return metadata
            except Exception as e:
                logger.error(f"Error loading metadata for {content_hash}: {e}")
        
        return None
    
    def get_thumbnail(self, content_hash: str, size: Tuple[int, int] = None) -> Optional[bytes]:
        """
        Get thumbnail for an asset.
        
        Args:
            content_hash: Content hash
            size: Optional thumbnail size (width, height)
            
        Returns:
            Thumbnail data as bytes or None if not available
        """
        size_str = f"{size[0]}x{size[1]}" if size else "200x200"
        thumbnail_path = self.thumbnails_dir / f"{content_hash}_{size_str}.jpg"
        
        if thumbnail_path.exists():
            try:
                with open(thumbnail_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error loading thumbnail for {content_hash}: {e}")
        
        # Try to generate thumbnail if not exists
        if self.enable_thumbnails:
            return self._generate_thumbnail_on_demand(content_hash, size)
        
        return None
    
    def search_assets(self, query: Dict[str, Any] = None, content_type: str = None,
                     tags: List[str] = None, date_range: Tuple[str, str] = None,
                     size_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """
        Search assets based on metadata criteria.
        
        Args:
            query: General query parameters
            content_type: Filter by content type (image, audio, video, document)
            tags: Filter by tags
            date_range: Filter by date range (start_date, end_date)
            size_range: Filter by file size range (min_size, max_size)
            
        Returns:
            List of matching asset metadata
        """
        results = []
        
        # Get all asset hashes
        asset_hashes = self.list_assets()
        
        for content_hash in asset_hashes:
            metadata = self.get_metadata(content_hash)
            if not metadata:
                continue
            
            # Apply filters
            if content_type and metadata.get('content_type') != content_type:
                continue
            
            if tags:
                asset_tags = metadata.get('tags', [])
                if not any(tag in asset_tags for tag in tags):
                    continue
            
            if date_range:
                stored_time = metadata.get('stored_timestamp')
                if stored_time:
                    if stored_time < date_range[0] or stored_time > date_range[1]:
                        continue
            
            if size_range:
                file_size = metadata.get('file_size', 0)
                if file_size < size_range[0] or file_size > size_range[1]:
                    continue
            
            # Apply general query
            if query:
                match = True
                for key, value in query.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(metadata)
        
        return results
    
    def get_content_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about stored content.
        
        Returns:
            Statistics dictionary
        """
        stats = super().get_storage_info()
        
        # Enhanced stats
        content_types = {}
        total_metadata_size = 0
        total_thumbnail_size = 0
        duplicate_count = 0
        
        asset_hashes = self.list_assets()
        
        for content_hash in asset_hashes:
            metadata = self.get_metadata(content_hash)
            if metadata:
                # Count by content type
                content_type = metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Check for duplicates
                if metadata.get('is_duplicate', False):
                    duplicate_count += 1
        
        # Calculate metadata and thumbnail sizes
        if self.metadata_dir.exists():
            total_metadata_size = sum(f.stat().st_size for f in self.metadata_dir.rglob('*.json'))
        
        if self.thumbnails_dir.exists():
            total_thumbnail_size = sum(f.stat().st_size for f in self.thumbnails_dir.rglob('*.jpg'))
        
        stats.update({
            'content_types': content_types,
            'total_metadata_size': total_metadata_size,
            'total_thumbnail_size': total_thumbnail_size,
            'duplicate_count': duplicate_count,
            'deduplication_enabled': self.enable_deduplication,
            'thumbnails_enabled': self.enable_thumbnails,
            'metadata_cache_size': len(self._metadata_cache),
            'fingerprint_index_size': len(self._fingerprint_index)
        })
        
        return stats
    
    def cleanup_orphaned_metadata(self) -> int:
        """
        Clean up metadata files for assets that no longer exist.
        
        Returns:
            Number of orphaned metadata files removed
        """
        removed_count = 0
        
        if not self.metadata_dir.exists():
            return removed_count
        
        # Get existing asset hashes
        existing_hashes = set(self.list_assets())
        
        # Check metadata files
        for metadata_file in self.metadata_dir.glob('*.json'):
            content_hash = metadata_file.stem
            if content_hash not in existing_hashes:
                try:
                    metadata_file.unlink()
                    removed_count += 1
                    
                    # Also remove from cache
                    self._metadata_cache.pop(content_hash, None)
                    
                    # Remove associated thumbnail
                    for thumb_file in self.thumbnails_dir.glob(f"{content_hash}_*.jpg"):
                        thumb_file.unlink()
                        
                except Exception as e:
                    logger.error(f"Error removing orphaned metadata {content_hash}: {e}")
        
        return removed_count
    
    def _store_metadata(self, content_hash: str, metadata: Dict[str, Any]):
        """Store metadata to disk."""
        metadata_path = self.metadata_dir / f"{content_hash}.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error storing metadata for {content_hash}: {e}")
    
    def _generate_and_store_thumbnail(self, content_hash: str, content: bytes, 
                                    metadata: Dict[str, Any]):
        """Generate and store thumbnail for content."""
        content_type = metadata.get('content_type')
        if not content_type:
            return
        
        # Check if thumbnail generation is supported
        file_extension = metadata.get('extension', '')
        if not ThumbnailGenerator.can_generate_thumbnail(content_type, file_extension):
            return
        
        try:
            # Generate thumbnail
            thumbnail_data = None
            
            if content_type == 'image':
                thumbnail_data = ThumbnailGenerator.generate_image_thumbnail(
                    Path(f"temp{file_extension}"), content
                )
            
            if thumbnail_data:
                # Store thumbnail
                thumbnail_path = self.thumbnails_dir / f"{content_hash}_200x200.jpg"
                with open(thumbnail_path, 'wb') as f:
                    f.write(thumbnail_data)
                
                # Update metadata
                metadata['has_thumbnail'] = True
                metadata['thumbnail_size'] = len(thumbnail_data)
            
        except Exception as e:
            logger.error(f"Error generating thumbnail for {content_hash}: {e}")
    
    def _generate_thumbnail_on_demand(self, content_hash: str, size: Tuple[int, int] = None) -> Optional[bytes]:
        """Generate thumbnail on demand if not exists."""
        # Get original content
        content = self.get_bytes(content_hash)
        if not content:
            return None
        
        # Get metadata
        metadata = self.get_metadata(content_hash)
        if not metadata:
            return None
        
        content_type = metadata.get('content_type')
        file_extension = metadata.get('extension', '')
        
        if not ThumbnailGenerator.can_generate_thumbnail(content_type, file_extension):
            return None
        
        try:
            size = size or (200, 200)
            
            if content_type == 'image':
                thumbnail_data = ThumbnailGenerator.generate_image_thumbnail(
                    Path(f"temp{file_extension}"), content, size
                )
                
                if thumbnail_data:
                    # Store for future use
                    size_str = f"{size[0]}x{size[1]}"
                    thumbnail_path = self.thumbnails_dir / f"{content_hash}_{size_str}.jpg"
                    with open(thumbnail_path, 'wb') as f:
                        f.write(thumbnail_data)
                    
                    return thumbnail_data
            
        except Exception as e:
            logger.error(f"Error generating thumbnail on demand for {content_hash}: {e}")
        
        return None
    
    def _update_indexes(self, content_hash: str, metadata: Dict[str, Any]):
        """Update search indexes."""
        # This is a simple implementation
        # In a production system, you might use a proper search engine like Elasticsearch
        
        try:
            # Content type index
            content_type = metadata.get('content_type', 'unknown')
            content_type_index_path = self.index_dir / f"content_type_{content_type}.json"
            
            # Load existing index
            if content_type_index_path.exists():
                with open(content_type_index_path, 'r') as f:
                    index = json.load(f)
            else:
                index = []
            
            # Add to index if not already present
            if content_hash not in index:
                index.append(content_hash)
                
                # Save updated index
                with open(content_type_index_path, 'w') as f:
                    json.dump(index, f)
            
        except Exception as e:
            logger.error(f"Error updating indexes for {content_hash}: {e}")
    
    def _load_indexes(self):
        """Load existing indexes into memory."""
        try:
            # Load fingerprint index
            fingerprint_index_path = self.index_dir / "fingerprints.json"
            if fingerprint_index_path.exists():
                with open(fingerprint_index_path, 'r') as f:
                    self._fingerprint_index = json.load(f)
            
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
    
    def _save_indexes(self):
        """Save indexes to disk."""
        try:
            # Save fingerprint index
            fingerprint_index_path = self.index_dir / "fingerprints.json"
            with open(fingerprint_index_path, 'w') as f:
                json.dump(self._fingerprint_index, f)
            
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
    
    def __del__(self):
        """Cleanup when driver is destroyed."""
        try:
            self._save_indexes()
        except Exception:
            pass


class MultimodalAssetManager:
    """
    High-level asset manager for multimodal content.
    """
    
    def __init__(self, driver: MultimodalLocalDriver):
        """
        Initialize asset manager.
        
        Args:
            driver: Multimodal driver instance
        """
        self.driver = driver
    
    def add_asset(self, content: bytes, filename: str = None, 
                 tags: List[str] = None, description: str = None,
                 custom_metadata: Dict[str, Any] = None) -> str:
        """
        Add an asset with enhanced metadata.
        
        Args:
            content: Asset content as bytes
            filename: Optional filename
            tags: Optional tags for categorization
            description: Optional description
            custom_metadata: Optional custom metadata
            
        Returns:
            Content hash
        """
        metadata = custom_metadata or {}
        
        if tags:
            metadata['tags'] = tags
        if description:
            metadata['description'] = description
        
        return self.driver.put(content, filename, metadata)
    
    def get_asset_info(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive asset information.
        
        Args:
            content_hash: Content hash
            
        Returns:
            Asset information including metadata and availability
        """
        if not self.driver.exists(content_hash):
            return None
        
        metadata = self.driver.get_metadata(content_hash)
        if not metadata:
            return None
        
        info = {
            'content_hash': content_hash,
            'exists': True,
            'has_thumbnail': self.driver.get_thumbnail(content_hash) is not None,
            'metadata': metadata
        }
        
        return info
    
    def find_similar_assets(self, content_hash: str, threshold: float = 0.8) -> List[str]:
        """
        Find assets similar to the given one.
        
        Args:
            content_hash: Reference content hash
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar asset hashes
        """
        # This is a placeholder implementation
        # In a real system, you might use perceptual hashing for images,
        # audio fingerprinting for audio, etc.
        
        reference_metadata = self.driver.get_metadata(content_hash)
        if not reference_metadata:
            return []
        
        similar_assets = []
        content_type = reference_metadata.get('content_type')
        
        # Search assets of the same type
        candidates = self.driver.search_assets({'content_type': content_type})
        
        for candidate in candidates:
            candidate_hash = candidate.get('content_hash')
            if candidate_hash == content_hash:
                continue
            
            # Simple similarity based on metadata
            similarity = self._calculate_metadata_similarity(reference_metadata, candidate)
            if similarity >= threshold:
                similar_assets.append(candidate_hash)
        
        return similar_assets
    
    def _calculate_metadata_similarity(self, metadata1: Dict[str, Any], 
                                     metadata2: Dict[str, Any]) -> float:
        """Calculate similarity between two metadata objects."""
        # Simple implementation - could be much more sophisticated
        
        content_type1 = metadata1.get('content_type')
        content_type2 = metadata2.get('content_type')
        
        if content_type1 != content_type2:
            return 0.0
        
        similarity_score = 0.0
        total_factors = 0
        
        if content_type1 == 'image':
            # Compare image dimensions
            w1, h1 = metadata1.get('width', 0), metadata1.get('height', 0)
            w2, h2 = metadata2.get('width', 0), metadata2.get('height', 0)
            
            if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
                aspect_ratio1 = w1 / h1
                aspect_ratio2 = w2 / h2
                aspect_similarity = 1.0 - abs(aspect_ratio1 - aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
                similarity_score += aspect_similarity
                total_factors += 1
        
        elif content_type1 == 'audio':
            # Compare audio duration
            duration1 = metadata1.get('duration', 0)
            duration2 = metadata2.get('duration', 0)
            
            if duration1 > 0 and duration2 > 0:
                duration_similarity = 1.0 - abs(duration1 - duration2) / max(duration1, duration2)
                similarity_score += duration_similarity
                total_factors += 1
        
        # Compare file sizes
        size1 = metadata1.get('file_size', 0)
        size2 = metadata2.get('file_size', 0)
        
        if size1 > 0 and size2 > 0:
            size_similarity = 1.0 - abs(size1 - size2) / max(size1, size2)
            similarity_score += size_similarity
            total_factors += 1
        
        return similarity_score / total_factors if total_factors > 0 else 0.0