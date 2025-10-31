import pytest
import asyncio
from pathlib import Path
import tempfile
import os
import json
from unittest.mock import AsyncMock, patch

from campfires.party_box.multimodal_local_driver import MultimodalLocalDriver
from campfires.party_box.metadata_extractor import MetadataExtractor, ThumbnailGenerator

# Mock logger to prevent actual logging during tests
class MockLogger:
    def error(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass

# Replace the actual logger with the mock logger
MultimodalLocalDriver.logger = MockLogger()

@pytest.fixture
def temp_storage_dir(tmp_path):
    return tmp_path / "test_storage"

@pytest.fixture
def driver(temp_storage_dir):
    return MultimodalLocalDriver(str(temp_storage_dir))

@pytest.fixture
def asset_manager(driver):
    from campfires.party_box.multimodal_local_driver import MultimodalAssetManager
    return MultimodalAssetManager(driver)

@pytest.mark.asyncio
class TestMultimodalLocalDriver:

    async def test_init(self, temp_storage_dir):
        driver = MultimodalLocalDriver(str(temp_storage_dir))
        assert driver.base_path == temp_storage_dir
        assert driver.metadata_dir.exists()
        assert driver.thumbnails_dir.exists()
        assert driver.index_dir.exists()

    async def test_store_asset(self, driver):
        asset_data = b"test_content"
        content_hash = await driver.put(asset_data, "text/plain")
        assert await driver.exists(content_hash)
        retrieved_path = await driver.get(content_hash)
        assert retrieved_path.exists()
        async with aiofiles.open(retrieved_path, 'rb') as f:
            retrieved_data = await f.read()
        assert retrieved_data == asset_data

    async def test_get_asset(self, driver):
        asset_data = b"test_content_get"
        content_hash = await driver.put(asset_data, "text/plain")
        retrieved_path = await driver.get(content_hash)
        assert retrieved_path.exists()
        async with aiofiles.open(retrieved_path, 'rb') as f:
            retrieved_data = await f.read()
        assert retrieved_data == asset_data

    async def test_retrieve_nonexistent_asset(self, driver):
        with pytest.raises(FileNotFoundError):
            await driver.get("nonexistent_id")

    async def test_delete_asset(self, driver):
        asset_data = b"test_content_delete"
        content_hash = await driver.put(asset_data, "text/plain")
        assert await driver.exists(content_hash)
        await driver.delete(content_hash)
        assert not await driver.exists(content_hash)

    async def test_delete_nonexistent_asset(self, driver):
        # Deleting a nonexistent asset should not raise an error
        await driver.delete("nonexistent_id")
        assert not await driver.exists("nonexistent_id")

    async def test_asset_manager_store_asset(self, asset_manager):
        asset_data = b"asset_manager_content"
        filename = "test.txt"
        content_hash = await asset_manager.add_asset(asset_data, filename=filename)
        assert await asset_manager.driver.exists(content_hash)
        retrieved_path = await asset_manager.driver.get(content_hash)
        assert retrieved_path.exists()
        async with aiofiles.open(retrieved_path, 'rb') as f:
            retrieved_data = await f.read()
        assert retrieved_data == asset_data

    async def test_asset_manager_retrieve_asset(self, asset_manager):
        asset_data = b"asset_manager_retrieve_content"
        filename = "retrieve.txt"
        content_hash = await asset_manager.add_asset(asset_data, filename=filename)
        retrieved_path = await asset_manager.driver.get(content_hash)
        assert retrieved_path.exists()
        async with aiofiles.open(retrieved_path, 'rb') as f:
            retrieved_data = await f.read()
        assert retrieved_data == asset_data

    async def test_asset_manager_delete_asset(self, asset_manager):
        asset_data = b"asset_manager_delete_content"
        filename = "delete.txt"
        content_hash = await asset_manager.add_asset(asset_data, filename=filename)
        assert await asset_manager.driver.exists(content_hash)
        await asset_manager.driver.delete(content_hash)
        assert not await asset_manager.driver.exists(content_hash)

    async def test_metadata_extraction(self, driver):
        image_data = b"fake_image_data"
        content_hash = await driver.put(image_data, "image/png", metadata={"user_key": "user_value"})
        metadata = await driver.get_metadata(content_hash)
        assert metadata is not None
        assert metadata['content_type'] == "image/png"
        assert metadata['user_key'] == "user_value"
        assert 'file_size' in metadata
        assert 'stored_timestamp' in metadata

    async def test_thumbnail_generation(self, driver):
        # Mock thumbnail generation to avoid actual image processing
        with patch.object(ThumbnailGenerator, 'generate_image_thumbnail', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = b"fake_thumbnail_data"
            driver.enable_thumbnails = True
            image_data = b"fake_image_data_for_thumbnail"
            content_hash = await driver.put(image_data, "image/png")
            thumbnail_data = await driver.get_thumbnail(content_hash)
            assert thumbnail_data == b"fake_thumbnail_data"
            mock_generate.assert_called_once()

    async def test_deduplication(self, driver):
        driver.enable_deduplication = True
        asset_data1 = b"duplicate_content"
        content_hash1 = await driver.put(asset_data1, "text/plain")
        content_hash2 = await driver.put(b"duplicate_content", "text/plain")

        metadata1 = await driver.get_metadata(content_hash1)
        metadata2 = await driver.get_metadata(content_hash2)

        assert metadata1['is_duplicate'] == False
        assert metadata2['is_duplicate'] == True
        assert metadata2['duplicate_of'] == content_hash1

    async def test_search_assets(self, driver):
        await driver.put(b"image1", "image/jpeg", metadata={"tag": "nature"})
        await driver.put(b"image2", "image/png", metadata={"tag": "city"})
        await driver.put(b"audio1", "audio/mpeg", metadata={"tag": "nature"})

        results = await driver.search_assets(content_type="image/jpeg")
        assert len(results) == 1
        assert results[0]['content_type'] == "image/jpeg"

        results = await driver.search_assets(tags=["nature"])
        assert len(results) == 2

    async def test_get_content_stats(self, driver):
        await driver.put(b"image1", "image/jpeg")
        await driver.put(b"audio1", "audio/mpeg")

        stats = await driver.get_content_stats()
        assert stats['total_assets'] == 2
        assert stats['content_types']['image/jpeg'] == 1
        assert stats['content_types']['audio/mpeg'] == 1

    async def test_cleanup_orphaned_metadata(self, driver):
        # Create a dummy asset and then delete its content file to orphan metadata
        asset_data = b"orphan_content"
        content_hash = await driver.put(asset_data, "text/plain")
        
        # Manually delete the content file, leaving metadata behind
        content_path = driver.base_path / content_hash[0:2] / content_hash
        content_path.unlink()

        # Create a dummy metadata file that doesn't correspond to any asset
        orphan_metadata_hash = "nonexistent_hash"
        orphan_metadata_path = driver.metadata_dir / f"{orphan_metadata_hash}.json"
        async with aiofiles.open(orphan_metadata_path, 'w') as f:
            await f.write(json.dumps({"content_hash": orphan_metadata_hash}))

        removed_count = await driver.cleanup_orphaned_metadata()
        assert removed_count == 1  # Only the manually created orphan should be removed
        assert not orphan_metadata_path.exists()
        assert await driver.get_metadata(content_hash) is not None # Original asset metadata should still exist

    async def test_generate_hash(self, driver):
        content = b"test_hash_content"
        expected_hash = "c6c762ea347bef3acfcf9dc7884276a7cd08f5eb35e07cb56652cbdfa7cd1de0"
        assert driver.generate_hash(content) == expected_hash