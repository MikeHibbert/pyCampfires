"""
Tests for multimodal local driver functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from campfires.party_box.multimodal_local_driver import (
    MultimodalLocalDriver,
    MultimodalAssetManager
)
from campfires.party_box.metadata_extractor import ContentMetadata, ImageMetadata


class TestMultimodalLocalDriver:
    """Test MultimodalLocalDriver class."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def driver(self, temp_storage_dir):
        """Create a MultimodalLocalDriver instance."""
        return MultimodalLocalDriver(storage_path=temp_storage_dir)
    
    @pytest.fixture
    def mock_image_data(self):
        """Create mock image data."""
        return b"fake_image_data"
    
    def test_driver_initialization(self, driver, temp_storage_dir):
        """Test driver initialization."""
        assert driver.storage_path == temp_storage_dir
        assert driver.metadata_extractor is not None
        assert os.path.exists(os.path.join(temp_storage_dir, "metadata"))
    
    @patch('campfires.party_box.multimodal_local_driver.MetadataExtractor')
    def test_put_asset_with_metadata(self, mock_extractor_class, driver, mock_image_data):
        """Test putting an asset with metadata extraction."""
        # Mock metadata extractor
        mock_extractor = Mock()
        mock_metadata = ImageMetadata(
            width=1920,
            height=1080,
            format="JPEG",
            file_path="test.jpg",
            file_size=len(mock_image_data),
            mime_type="image/jpeg",
            content_hash="abc123"
        )
        mock_extractor.extract_metadata.return_value = mock_metadata
        mock_extractor_class.return_value = mock_extractor
        
        # Create new driver to use mocked extractor
        driver = MultimodalLocalDriver(storage_path=driver.storage_path)
        
        asset_hash = driver.put(mock_image_data, filename="test.jpg")
        
        assert asset_hash is not None
        assert len(asset_hash) == 64  # SHA-256 hex length
        
        # Verify metadata was extracted and stored
        mock_extractor.extract_metadata.assert_called_once()
    
    def test_get_asset_metadata_existing(self, driver):
        """Test getting metadata for existing asset."""
        # First put an asset
        test_data = b"test_data"
        asset_hash = driver.put(test_data, filename="test.txt")
        
        # Mock metadata file
        metadata_file = os.path.join(driver.storage_path, "metadata", f"{asset_hash}.json")
        mock_metadata = {
            "file_path": "test.txt",
            "file_size": len(test_data),
            "mime_type": "text/plain",
            "content_hash": asset_hash
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(mock_metadata, f)
        
        metadata = driver.get_asset_metadata(asset_hash)
        
        assert metadata is not None
        assert metadata["file_path"] == "test.txt"
        assert metadata["file_size"] == len(test_data)
    
    def test_get_asset_metadata_nonexistent(self, driver):
        """Test getting metadata for non-existent asset."""
        metadata = driver.get_asset_metadata("nonexistent_hash")
        assert metadata is None
    
    def test_search_assets_by_type(self, driver):
        """Test searching assets by content type."""
        # Mock some metadata files
        metadata_dir = os.path.join(driver.storage_path, "metadata")
        
        # Image asset
        image_metadata = {
            "content_type": "image",
            "mime_type": "image/jpeg",
            "file_path": "image.jpg"
        }
        with open(os.path.join(metadata_dir, "hash1.json"), 'w') as f:
            json.dump(image_metadata, f)
        
        # Audio asset
        audio_metadata = {
            "content_type": "audio",
            "mime_type": "audio/mp3",
            "file_path": "audio.mp3"
        }
        with open(os.path.join(metadata_dir, "hash2.json"), 'w') as f:
            json.dump(audio_metadata, f)
        
        # Search for images
        results = driver.search_assets(content_type="image")
        
        assert len(results) == 1
        assert results[0]["asset_hash"] == "hash1"
        assert results[0]["metadata"]["content_type"] == "image"
    
    def test_search_assets_by_mime_type(self, driver):
        """Test searching assets by MIME type."""
        metadata_dir = os.path.join(driver.storage_path, "metadata")
        
        # JPEG image
        jpeg_metadata = {
            "content_type": "image",
            "mime_type": "image/jpeg",
            "file_path": "image.jpg"
        }
        with open(os.path.join(metadata_dir, "hash1.json"), 'w') as f:
            json.dump(jpeg_metadata, f)
        
        # PNG image
        png_metadata = {
            "content_type": "image",
            "mime_type": "image/png",
            "file_path": "image.png"
        }
        with open(os.path.join(metadata_dir, "hash2.json"), 'w') as f:
            json.dump(png_metadata, f)
        
        # Search for JPEG images
        results = driver.search_assets(mime_type="image/jpeg")
        
        assert len(results) == 1
        assert results[0]["metadata"]["mime_type"] == "image/jpeg"
    
    def test_search_assets_by_size_range(self, driver):
        """Test searching assets by file size range."""
        metadata_dir = os.path.join(driver.storage_path, "metadata")
        
        # Small file
        small_metadata = {
            "file_size": 1000,
            "file_path": "small.txt"
        }
        with open(os.path.join(metadata_dir, "hash1.json"), 'w') as f:
            json.dump(small_metadata, f)
        
        # Large file
        large_metadata = {
            "file_size": 10000,
            "file_path": "large.txt"
        }
        with open(os.path.join(metadata_dir, "hash2.json"), 'w') as f:
            json.dump(large_metadata, f)
        
        # Search for files between 500 and 5000 bytes
        results = driver.search_assets(min_size=500, max_size=5000)
        
        assert len(results) == 1
        assert results[0]["metadata"]["file_size"] == 1000
    
    def test_get_assets_by_type(self, driver):
        """Test getting assets grouped by content type."""
        metadata_dir = os.path.join(driver.storage_path, "metadata")
        
        # Create metadata for different types
        types_data = [
            ("hash1", {"content_type": "image", "file_path": "img1.jpg"}),
            ("hash2", {"content_type": "image", "file_path": "img2.png"}),
            ("hash3", {"content_type": "audio", "file_path": "audio1.mp3"}),
            ("hash4", {"content_type": "video", "file_path": "video1.mp4"})
        ]
        
        for hash_val, metadata in types_data:
            with open(os.path.join(metadata_dir, f"{hash_val}.json"), 'w') as f:
                json.dump(metadata, f)
        
        assets_by_type = driver.get_assets_by_type()
        
        assert "image" in assets_by_type
        assert "audio" in assets_by_type
        assert "video" in assets_by_type
        assert len(assets_by_type["image"]) == 2
        assert len(assets_by_type["audio"]) == 1
        assert len(assets_by_type["video"]) == 1
    
    def test_get_storage_stats(self, driver):
        """Test getting storage statistics."""
        metadata_dir = os.path.join(driver.storage_path, "metadata")
        
        # Create some metadata files
        metadata_files = [
            ("hash1", {"content_type": "image", "file_size": 1000}),
            ("hash2", {"content_type": "image", "file_size": 2000}),
            ("hash3", {"content_type": "audio", "file_size": 5000})
        ]
        
        for hash_val, metadata in metadata_files:
            with open(os.path.join(metadata_dir, f"{hash_val}.json"), 'w') as f:
                json.dump(metadata, f)
        
        stats = driver.get_storage_stats()
        
        assert stats["total_assets"] == 3
        assert stats["total_size"] == 8000
        assert "by_type" in stats
        assert stats["by_type"]["image"]["count"] == 2
        assert stats["by_type"]["image"]["total_size"] == 3000
        assert stats["by_type"]["audio"]["count"] == 1
        assert stats["by_type"]["audio"]["total_size"] == 5000
    
    def test_find_duplicates(self, driver):
        """Test finding duplicate assets."""
        metadata_dir = os.path.join(driver.storage_path, "metadata")
        
        # Create metadata with same content hash (duplicates)
        duplicate_metadata = [
            ("hash1", {"content_hash": "abc123", "file_path": "file1.txt"}),
            ("hash2", {"content_hash": "abc123", "file_path": "file2.txt"}),
            ("hash3", {"content_hash": "def456", "file_path": "file3.txt"})
        ]
        
        for hash_val, metadata in duplicate_metadata:
            with open(os.path.join(metadata_dir, f"{hash_val}.json"), 'w') as f:
                json.dump(metadata, f)
        
        duplicates = driver.find_duplicates()
        
        assert len(duplicates) == 1  # One group of duplicates
        assert len(duplicates[0]) == 2  # Two files with same content hash
        assert all(item["metadata"]["content_hash"] == "abc123" for item in duplicates[0])
    
    def test_delete_asset_with_metadata(self, driver):
        """Test deleting an asset and its metadata."""
        # Put an asset first
        test_data = b"test_data"
        asset_hash = driver.put(test_data, filename="test.txt")
        
        # Verify asset and metadata exist
        assert driver.exists(asset_hash)
        metadata_file = os.path.join(driver.storage_path, "metadata", f"{asset_hash}.json")
        
        # Create metadata file manually for this test
        with open(metadata_file, 'w') as f:
            json.dump({"test": "metadata"}, f)
        
        assert os.path.exists(metadata_file)
        
        # Delete the asset
        result = driver.delete(asset_hash)
        
        assert result is True
        assert not driver.exists(asset_hash)
        assert not os.path.exists(metadata_file)


class TestMultimodalAssetManager:
    """Test MultimodalAssetManager class."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def manager(self, temp_storage_dir):
        """Create a MultimodalAssetManager instance."""
        return MultimodalAssetManager(storage_path=temp_storage_dir)
    
    def test_manager_initialization(self, manager, temp_storage_dir):
        """Test manager initialization."""
        assert manager.driver is not None
        assert manager.driver.storage_path == temp_storage_dir
    
    def test_add_asset_from_file(self, manager):
        """Test adding an asset from file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_file = f.name
        
        try:
            asset_hash = manager.add_asset_from_file(temp_file)
            
            assert asset_hash is not None
            assert len(asset_hash) == 64
            assert manager.driver.exists(asset_hash)
        finally:
            os.unlink(temp_file)
    
    def test_add_asset_from_bytes(self, manager):
        """Test adding an asset from bytes."""
        test_data = b"test content"
        asset_hash = manager.add_asset_from_bytes(test_data, filename="test.txt")
        
        assert asset_hash is not None
        assert manager.driver.exists(asset_hash)
    
    def test_get_asset_info(self, manager):
        """Test getting asset information."""
        # Add an asset first
        test_data = b"test content"
        asset_hash = manager.add_asset_from_bytes(test_data, filename="test.txt")
        
        # Mock metadata
        metadata_file = os.path.join(manager.driver.storage_path, "metadata", f"{asset_hash}.json")
        mock_metadata = {
            "file_path": "test.txt",
            "file_size": len(test_data),
            "mime_type": "text/plain"
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(mock_metadata, f)
        
        info = manager.get_asset_info(asset_hash)
        
        assert info is not None
        assert info["exists"] is True
        assert info["metadata"]["file_path"] == "test.txt"
    
    def test_get_asset_info_nonexistent(self, manager):
        """Test getting info for non-existent asset."""
        info = manager.get_asset_info("nonexistent_hash")
        
        assert info["exists"] is False
        assert info["metadata"] is None
    
    def test_search_by_content_type(self, manager):
        """Test searching assets by content type."""
        # Mock the driver's search method
        manager.driver.search_assets = Mock(return_value=[
            {"asset_hash": "hash1", "metadata": {"content_type": "image"}}
        ])
        
        results = manager.search_by_content_type("image")
        
        assert len(results) == 1
        assert results[0]["asset_hash"] == "hash1"
        manager.driver.search_assets.assert_called_once_with(content_type="image")
    
    def test_get_storage_summary(self, manager):
        """Test getting storage summary."""
        # Mock the driver's get_storage_stats method
        mock_stats = {
            "total_assets": 10,
            "total_size": 50000,
            "by_type": {
                "image": {"count": 5, "total_size": 30000},
                "audio": {"count": 3, "total_size": 15000},
                "other": {"count": 2, "total_size": 5000}
            }
        }
        manager.driver.get_storage_stats = Mock(return_value=mock_stats)
        
        summary = manager.get_storage_summary()
        
        assert summary["total_assets"] == 10
        assert summary["total_size"] == 50000
        assert "by_type" in summary
        manager.driver.get_storage_stats.assert_called_once()
    
    def test_cleanup_duplicates(self, manager):
        """Test cleaning up duplicate assets."""
        # Mock the driver's find_duplicates method
        mock_duplicates = [
            [
                {"asset_hash": "hash1", "metadata": {"file_path": "file1.txt"}},
                {"asset_hash": "hash2", "metadata": {"file_path": "file2.txt"}}
            ]
        ]
        manager.driver.find_duplicates = Mock(return_value=mock_duplicates)
        manager.driver.delete = Mock(return_value=True)
        
        removed_count = manager.cleanup_duplicates()
        
        assert removed_count == 1  # Should remove one duplicate
        manager.driver.find_duplicates.assert_called_once()
        manager.driver.delete.assert_called_once_with("hash2")  # Should keep first, remove second
    
    def test_remove_asset(self, manager):
        """Test removing an asset."""
        # Mock the driver's delete method
        manager.driver.delete = Mock(return_value=True)
        
        result = manager.remove_asset("test_hash")
        
        assert result is True
        manager.driver.delete.assert_called_once_with("test_hash")


if __name__ == "__main__":
    pytest.main([__file__])