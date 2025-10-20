"""
Tests for metadata extraction functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from campfires.party_box.metadata_extractor import (
    ContentMetadata,
    ImageMetadata,
    AudioMetadata,
    VideoMetadata,
    DocumentMetadata,
    MetadataExtractor
)


class TestContentMetadata:
    """Test ContentMetadata base class."""
    
    def test_content_metadata_creation(self):
        """Test creating content metadata."""
        metadata = ContentMetadata(
            file_path="/test/file.txt",
            file_size=1024,
            mime_type="text/plain",
            content_hash="abc123",
            created_at="2024-01-01T00:00:00Z",
            modified_at="2024-01-01T00:00:00Z"
        )
        
        assert metadata.file_path == "/test/file.txt"
        assert metadata.file_size == 1024
        assert metadata.mime_type == "text/plain"
        assert metadata.content_hash == "abc123"
        assert metadata.created_at == "2024-01-01T00:00:00Z"
        assert metadata.modified_at == "2024-01-01T00:00:00Z"
    
    def test_content_metadata_optional_fields(self):
        """Test content metadata with optional fields."""
        metadata = ContentMetadata()
        
        assert metadata.file_path is None
        assert metadata.file_size is None
        assert metadata.mime_type is None
        assert metadata.content_hash is None
        assert metadata.created_at is None
        assert metadata.modified_at is None


class TestImageMetadata:
    """Test ImageMetadata class."""
    
    def test_image_metadata_creation(self):
        """Test creating image metadata."""
        metadata = ImageMetadata(
            width=1920,
            height=1080,
            format="JPEG",
            color_mode="RGB",
            has_transparency=False,
            exif_data={"Camera": "Canon"},
            quality_score=0.85
        )
        
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.format == "JPEG"
        assert metadata.color_mode == "RGB"
        assert metadata.has_transparency is False
        assert metadata.exif_data == {"Camera": "Canon"}
        assert metadata.quality_score == 0.85


class TestAudioMetadata:
    """Test AudioMetadata class."""
    
    def test_audio_metadata_creation(self):
        """Test creating audio metadata."""
        metadata = AudioMetadata(
            duration=120.5,
            bitrate=128000,
            sample_rate=44100,
            channels=2,
            format="MP3",
            codec="mp3"
        )
        
        assert metadata.duration == 120.5
        assert metadata.bitrate == 128000
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.format == "MP3"
        assert metadata.codec == "mp3"


class TestVideoMetadata:
    """Test VideoMetadata class."""
    
    def test_video_metadata_creation(self):
        """Test creating video metadata."""
        metadata = VideoMetadata(
            duration=300.0,
            width=1920,
            height=1080,
            fps=30.0,
            bitrate=5000000,
            codec="h264",
            has_audio=True
        )
        
        assert metadata.duration == 300.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.bitrate == 5000000
        assert metadata.codec == "h264"
        assert metadata.has_audio is True


class TestDocumentMetadata:
    """Test DocumentMetadata class."""
    
    def test_document_metadata_creation(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            page_count=10,
            word_count=5000,
            language="en",
            author="John Doe",
            title="Test Document",
            subject="Testing",
            creator="Test App"
        )
        
        assert metadata.page_count == 10
        assert metadata.word_count == 5000
        assert metadata.language == "en"
        assert metadata.author == "John Doe"
        assert metadata.title == "Test Document"
        assert metadata.subject == "Testing"
        assert metadata.creator == "Test App"


class TestMetadataExtractor:
    """Test MetadataExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a MetadataExtractor instance."""
        return MetadataExtractor()
    
    @pytest.fixture
    def mock_image_file(self):
        """Create a mock image file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake_jpeg_data")
            yield f.name
        os.unlink(f.name)
    
    def test_extractor_initialization(self, extractor):
        """Test MetadataExtractor initialization."""
        assert extractor is not None
    
    def test_get_basic_file_info(self, extractor, mock_image_file):
        """Test basic file info extraction."""
        info = extractor._get_basic_file_info(mock_image_file)
        
        assert info["file_path"] == mock_image_file
        assert info["file_size"] > 0
        assert "created_at" in info
        assert "modified_at" in info
    
    def test_calculate_content_hash(self, extractor, mock_image_file):
        """Test content hash calculation."""
        hash_value = extractor._calculate_content_hash(mock_image_file)
        
        assert hash_value is not None
        assert len(hash_value) == 64  # SHA-256 hex digest length
    
    def test_detect_mime_type_image(self, extractor, mock_image_file):
        """Test MIME type detection for images."""
        mime_type = extractor._detect_mime_type(mock_image_file)
        
        # Should detect as some kind of image or application/octet-stream
        assert mime_type is not None
    
    def test_classify_content_type_image(self, extractor):
        """Test content type classification for images."""
        content_type = extractor._classify_content_type("image/jpeg")
        assert content_type == "image"
    
    def test_classify_content_type_audio(self, extractor):
        """Test content type classification for audio."""
        content_type = extractor._classify_content_type("audio/mp3")
        assert content_type == "audio"
    
    def test_classify_content_type_video(self, extractor):
        """Test content type classification for video."""
        content_type = extractor._classify_content_type("video/mp4")
        assert content_type == "video"
    
    def test_classify_content_type_document(self, extractor):
        """Test content type classification for documents."""
        content_type = extractor._classify_content_type("application/pdf")
        assert content_type == "document"
    
    def test_classify_content_type_other(self, extractor):
        """Test content type classification for other types."""
        content_type = extractor._classify_content_type("application/octet-stream")
        assert content_type == "other"
    
    @patch('campfires.party_box.metadata_extractor.Image')
    def test_extract_image_metadata_success(self, mock_image, extractor, mock_image_file):
        """Test successful image metadata extraction."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (1920, 1080)
        mock_img.format = "JPEG"
        mock_img.mode = "RGB"
        mock_img.getexif.return_value = {"Camera": "Canon"}
        mock_image.open.return_value = mock_img
        
        metadata = extractor._extract_image_metadata(mock_image_file)
        
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.format == "JPEG"
        assert metadata.color_mode == "RGB"
        assert metadata.exif_data == {"Camera": "Canon"}
    
    @patch('campfires.party_box.metadata_extractor.Image')
    def test_extract_image_metadata_failure(self, mock_image, extractor, mock_image_file):
        """Test image metadata extraction failure."""
        mock_image.open.side_effect = Exception("Cannot open image")
        
        metadata = extractor._extract_image_metadata(mock_image_file)
        
        assert metadata.width is None
        assert metadata.height is None
        assert metadata.format is None
    
    @patch('campfires.party_box.metadata_extractor.mutagen')
    def test_extract_audio_metadata_success(self, mock_mutagen, extractor, mock_image_file):
        """Test successful audio metadata extraction."""
        # Mock mutagen file
        mock_file = Mock()
        mock_file.info.length = 120.5
        mock_file.info.bitrate = 128000
        mock_file.info.sample_rate = 44100
        mock_file.info.channels = 2
        mock_mutagen.File.return_value = mock_file
        
        metadata = extractor._extract_audio_metadata(mock_image_file)
        
        assert metadata.duration == 120.5
        assert metadata.bitrate == 128000
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
    
    @patch('campfires.party_box.metadata_extractor.mutagen')
    def test_extract_audio_metadata_failure(self, mock_mutagen, extractor, mock_image_file):
        """Test audio metadata extraction failure."""
        mock_mutagen.File.return_value = None
        
        metadata = extractor._extract_audio_metadata(mock_image_file)
        
        assert metadata.duration is None
        assert metadata.bitrate is None
    
    def test_extract_video_metadata(self, extractor, mock_image_file):
        """Test video metadata extraction (placeholder)."""
        metadata = extractor._extract_video_metadata(mock_image_file)
        
        # Should return empty metadata since we don't have video processing
        assert metadata.duration is None
        assert metadata.width is None
        assert metadata.height is None
    
    def test_extract_document_metadata(self, extractor, mock_image_file):
        """Test document metadata extraction (placeholder)."""
        metadata = extractor._extract_document_metadata(mock_image_file)
        
        # Should return empty metadata since we don't have document processing
        assert metadata.page_count is None
        assert metadata.word_count is None
    
    def test_estimate_image_quality_high(self, extractor):
        """Test high quality image estimation."""
        quality = extractor._estimate_image_quality(1920, 1080, "JPEG")
        assert quality >= 0.7  # Should be high quality
    
    def test_estimate_image_quality_low(self, extractor):
        """Test low quality image estimation."""
        quality = extractor._estimate_image_quality(320, 240, "JPEG")
        assert quality <= 0.5  # Should be low quality
    
    def test_extract_thumbnail_info_with_transparency(self, extractor):
        """Test thumbnail info extraction with transparency."""
        info = extractor._extract_thumbnail_info("RGBA", True)
        
        assert info["has_transparency"] is True
        assert info["supports_alpha"] is True
    
    def test_extract_thumbnail_info_without_transparency(self, extractor):
        """Test thumbnail info extraction without transparency."""
        info = extractor._extract_thumbnail_info("RGB", False)
        
        assert info["has_transparency"] is False
        assert info["supports_alpha"] is False
    
    def test_generate_content_fingerprint(self, extractor, mock_image_file):
        """Test content fingerprint generation."""
        fingerprint = extractor._generate_content_fingerprint(mock_image_file, "image")
        
        assert fingerprint is not None
        assert "content_hash" in fingerprint
        assert "file_size" in fingerprint
        assert "content_type" in fingerprint
        assert fingerprint["content_type"] == "image"
    
    @patch('campfires.party_box.metadata_extractor.Image')
    def test_extract_metadata_image_file(self, mock_image, extractor, mock_image_file):
        """Test complete metadata extraction for image file."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (1920, 1080)
        mock_img.format = "JPEG"
        mock_img.mode = "RGB"
        mock_img.getexif.return_value = {}
        mock_image.open.return_value = mock_img
        
        metadata = extractor.extract_metadata(mock_image_file)
        
        assert metadata is not None
        assert metadata.file_path == mock_image_file
        assert metadata.file_size > 0
        assert hasattr(metadata, 'width')  # Should be ImageMetadata
        assert metadata.width == 1920
        assert metadata.height == 1080
    
    def test_extract_metadata_nonexistent_file(self, extractor):
        """Test metadata extraction for non-existent file."""
        metadata = extractor.extract_metadata("nonexistent_file.jpg")
        
        assert metadata is None


if __name__ == "__main__":
    pytest.main([__file__])