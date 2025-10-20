"""
Tests for audio processing functionality.
"""

import pytest
import base64
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from campfires.core.audio_processor import AudioMetadata, AudioProcessor
from campfires.core.audio_utils import (
    AudioFormatDetector,
    AudioValidator,
    AudioConverter,
    AudioMetrics
)


class TestAudioMetadata:
    """Test AudioMetadata class."""
    
    def test_audio_metadata_creation(self):
        """Test creating audio metadata."""
        metadata = AudioMetadata(
            duration=120.5,
            bitrate=128000,
            sample_rate=44100,
            channels=2,
            format="mp3",
            file_size=1024000
        )
        
        assert metadata.duration == 120.5
        assert metadata.bitrate == 128000
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.format == "mp3"
        assert metadata.file_size == 1024000
    
    def test_audio_metadata_optional_fields(self):
        """Test audio metadata with optional fields."""
        metadata = AudioMetadata()
        
        assert metadata.duration is None
        assert metadata.bitrate is None
        assert metadata.sample_rate is None
        assert metadata.channels is None
        assert metadata.format is None
        assert metadata.file_size is None


class TestAudioProcessor:
    """Test AudioProcessor class."""
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake_mp3_data")
            yield f.name
        os.unlink(f.name)
    
    def test_audio_processor_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor()
        assert processor is not None
    
    @patch('campfires.core.audio_processor.mutagen')
    def test_extract_metadata_success(self, mock_mutagen, mock_audio_file):
        """Test successful metadata extraction."""
        # Mock mutagen file
        mock_file = Mock()
        mock_file.info.length = 120.5
        mock_file.info.bitrate = 128000
        mock_mutagen.File.return_value = mock_file
        
        processor = AudioProcessor()
        metadata = processor.extract_metadata(mock_audio_file)
        
        assert metadata.duration == 120.5
        assert metadata.bitrate == 128000
    
    @patch('campfires.core.audio_processor.mutagen')
    def test_extract_metadata_file_not_found(self, mock_mutagen):
        """Test metadata extraction with non-existent file."""
        mock_mutagen.File.return_value = None
        
        processor = AudioProcessor()
        metadata = processor.extract_metadata("nonexistent.mp3")
        
        assert metadata.duration is None
        assert metadata.bitrate is None
    
    @patch('campfires.core.audio_processor.pydub')
    def test_analyze_audio_content(self, mock_pydub, mock_audio_file):
        """Test audio content analysis."""
        # Mock pydub AudioSegment
        mock_audio = Mock()
        mock_audio.duration_seconds = 120.0
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.max_possible_amplitude = 32767
        mock_audio.max = 16383  # Half of max amplitude
        mock_pydub.AudioSegment.from_file.return_value = mock_audio
        
        processor = AudioProcessor()
        analysis = processor.analyze_audio_content(mock_audio_file)
        
        assert analysis["duration"] == 120.0
        assert analysis["sample_rate"] == 44100
        assert analysis["channels"] == 2
        assert "loudness_ratio" in analysis
    
    @patch('campfires.core.audio_processor.pydub')
    def test_convert_audio_format(self, mock_pydub, mock_audio_file):
        """Test audio format conversion."""
        mock_audio = Mock()
        mock_pydub.AudioSegment.from_file.return_value = mock_audio
        mock_audio.export.return_value = None
        
        processor = AudioProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            try:
                result = processor.convert_audio_format(
                    mock_audio_file, 
                    output_file.name, 
                    "wav"
                )
                assert result is True
                mock_audio.export.assert_called_once()
            finally:
                os.unlink(output_file.name)
    
    @patch('campfires.core.audio_processor.pydub')
    def test_extract_audio_segment(self, mock_pydub, mock_audio_file):
        """Test extracting audio segment."""
        mock_audio = Mock()
        mock_segment = Mock()
        mock_audio.__getitem__.return_value = mock_segment
        mock_pydub.AudioSegment.from_file.return_value = mock_audio
        mock_segment.export.return_value = None
        
        processor = AudioProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
            try:
                result = processor.extract_audio_segment(
                    mock_audio_file,
                    output_file.name,
                    start_time=10.0,
                    end_time=30.0
                )
                assert result is True
                mock_audio.__getitem__.assert_called_once_with(slice(10000, 30000))
            finally:
                os.unlink(output_file.name)
    
    @patch('campfires.core.audio_processor.pydub')
    def test_generate_waveform_data(self, mock_pydub, mock_audio_file):
        """Test waveform data generation."""
        mock_audio = Mock()
        mock_audio.get_array_of_samples.return_value = [100, 200, 150, 300]
        mock_audio.channels = 1
        mock_pydub.AudioSegment.from_file.return_value = mock_audio
        
        processor = AudioProcessor()
        waveform = processor.generate_waveform_data(mock_audio_file, samples=4)
        
        assert len(waveform) == 4
        assert all(isinstance(x, (int, float)) for x in waveform)


class TestAudioFormatDetector:
    """Test AudioFormatDetector class."""
    
    def test_detect_format_from_bytes_mp3(self):
        """Test MP3 format detection from bytes."""
        mp3_header = b'\xff\xfb\x90\x00'  # MP3 header
        format_info = AudioFormatDetector.detect_format_from_bytes(mp3_header)
        
        assert format_info["format"] == "mp3"
        assert format_info["mime_type"] == "audio/mpeg"
    
    def test_detect_format_from_bytes_wav(self):
        """Test WAV format detection from bytes."""
        wav_header = b'RIFF\x00\x00\x00\x00WAVE'
        format_info = AudioFormatDetector.detect_format_from_bytes(wav_header)
        
        assert format_info["format"] == "wav"
        assert format_info["mime_type"] == "audio/wav"
    
    def test_detect_format_from_bytes_unknown(self):
        """Test unknown format detection."""
        unknown_data = b'\x00\x01\x02\x03'
        format_info = AudioFormatDetector.detect_format_from_bytes(unknown_data)
        
        assert format_info["format"] == "unknown"
        assert format_info["mime_type"] == "application/octet-stream"
    
    def test_detect_format_from_base64(self):
        """Test format detection from base64 data."""
        mp3_header = b'\xff\xfb\x90\x00'
        base64_data = base64.b64encode(mp3_header).decode()
        
        format_info = AudioFormatDetector.detect_format_from_base64(base64_data)
        assert format_info["format"] == "mp3"
    
    def test_detect_format_from_file(self):
        """Test format detection from file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b'\xff\xfb\x90\x00test_data')
            f.flush()
            
            try:
                format_info = AudioFormatDetector.detect_format_from_file(f.name)
                assert format_info["format"] == "mp3"
            finally:
                os.unlink(f.name)


class TestAudioValidator:
    """Test AudioValidator class."""
    
    def test_validate_audio_data_valid_mp3(self):
        """Test validation of valid MP3 data."""
        mp3_data = b'\xff\xfb\x90\x00' + b'a' * 1000  # Valid MP3 header + data
        
        result = AudioValidator.validate_audio_data(mp3_data)
        assert result["is_valid"] is True
        assert result["format"] == "mp3"
        assert result["size"] == 1004
    
    def test_validate_audio_data_too_large(self):
        """Test validation of oversized audio data."""
        large_data = b'a' * (50 * 1024 * 1024 + 1)  # > 50MB
        
        result = AudioValidator.validate_audio_data(large_data)
        assert result["is_valid"] is False
        assert "too large" in result["error"].lower()
    
    def test_validate_audio_data_invalid_format(self):
        """Test validation of invalid audio format."""
        invalid_data = b'\x00\x01\x02\x03'
        
        result = AudioValidator.validate_audio_data(invalid_data)
        assert result["is_valid"] is False
        assert "unsupported format" in result["error"].lower()
    
    def test_validate_audio_file(self):
        """Test validation of audio file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b'\xff\xfb\x90\x00test_data')
            f.flush()
            
            try:
                result = AudioValidator.validate_audio_file(f.name)
                assert result["is_valid"] is True
                assert result["format"] == "mp3"
            finally:
                os.unlink(f.name)


class TestAudioConverter:
    """Test AudioConverter class."""
    
    def test_bytes_to_base64(self):
        """Test converting bytes to base64."""
        audio_bytes = b'test_audio_data'
        base64_result = AudioConverter.bytes_to_base64(audio_bytes)
        
        expected = base64.b64encode(audio_bytes).decode()
        assert base64_result == expected
    
    def test_base64_to_bytes(self):
        """Test converting base64 to bytes."""
        audio_bytes = b'test_audio_data'
        base64_data = base64.b64encode(audio_bytes).decode()
        
        result = AudioConverter.base64_to_bytes(base64_data)
        assert result == audio_bytes
    
    def test_file_to_base64(self):
        """Test converting file to base64."""
        test_data = b'test_audio_file_data'
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_data)
            f.flush()
            
            try:
                base64_result = AudioConverter.file_to_base64(f.name)
                expected = base64.b64encode(test_data).decode()
                assert base64_result == expected
            finally:
                os.unlink(f.name)
    
    def test_base64_to_file(self):
        """Test converting base64 to file."""
        test_data = b'test_audio_data'
        base64_data = base64.b64encode(test_data).decode()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                AudioConverter.base64_to_file(base64_data, f.name)
                
                with open(f.name, 'rb') as read_file:
                    result = read_file.read()
                    assert result == test_data
            finally:
                os.unlink(f.name)


class TestAudioMetrics:
    """Test AudioMetrics class."""
    
    def test_calculate_compression_ratio(self):
        """Test compression ratio calculation."""
        original_size = 1000
        compressed_size = 200
        
        ratio = AudioMetrics.calculate_compression_ratio(original_size, compressed_size)
        assert ratio == 5.0  # 1000 / 200
    
    def test_estimate_bitrate(self):
        """Test bitrate estimation."""
        file_size = 1024000  # 1MB
        duration = 60.0  # 60 seconds
        
        bitrate = AudioMetrics.estimate_bitrate(file_size, duration)
        expected = (file_size * 8) / duration  # bits per second
        assert bitrate == expected
    
    def test_classify_audio_quality_high(self):
        """Test high quality audio classification."""
        quality = AudioMetrics.classify_audio_quality(320000, 44100)
        assert quality == "high"
    
    def test_classify_audio_quality_medium(self):
        """Test medium quality audio classification."""
        quality = AudioMetrics.classify_audio_quality(128000, 44100)
        assert quality == "medium"
    
    def test_classify_audio_quality_low(self):
        """Test low quality audio classification."""
        quality = AudioMetrics.classify_audio_quality(64000, 22050)
        assert quality == "low"


if __name__ == "__main__":
    pytest.main([__file__])