"""
Audio utilities and helper functions.
"""

import logging
import mimetypes
import base64
import io
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioFormatDetector:
    """
    Audio format detection utilities.
    """
    
    # MIME type mappings
    AUDIO_MIME_TYPES = {
        "audio/mpeg": ["mp3"],
        "audio/wav": ["wav"],
        "audio/x-wav": ["wav"],
        "audio/wave": ["wav"],
        "audio/mp4": ["m4a", "mp4"],
        "audio/x-m4a": ["m4a"],
        "audio/aac": ["aac"],
        "audio/ogg": ["ogg"],
        "audio/vorbis": ["ogg"],
        "audio/flac": ["flac"],
        "audio/x-flac": ["flac"],
        "audio/wma": ["wma"],
        "audio/x-ms-wma": ["wma"]
    }
    
    # File signature patterns (magic numbers)
    AUDIO_SIGNATURES = {
        b"ID3": "mp3",
        b"\xff\xfb": "mp3",
        b"\xff\xf3": "mp3",
        b"\xff\xf2": "mp3",
        b"RIFF": "wav",  # Will need additional check for WAVE
        b"fLaC": "flac",
        b"OggS": "ogg",
        b"\x00\x00\x00\x20ftypM4A": "m4a",
        b"ftypM4A": "m4a"
    }
    
    @classmethod
    def detect_format_from_bytes(cls, data: bytes) -> Optional[str]:
        """
        Detect audio format from byte data.
        
        Args:
            data: Audio file bytes
            
        Returns:
            Detected format or None
        """
        if not data:
            return None
        
        # Check magic numbers
        for signature, format_name in cls.AUDIO_SIGNATURES.items():
            if data.startswith(signature):
                # Special case for WAV files
                if format_name == "wav" and b"WAVE" in data[:20]:
                    return "wav"
                elif format_name == "wav":
                    continue  # Not a WAV file
                return format_name
        
        # Use python-magic if available
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_buffer(data, mime=True)
                return cls.mime_type_to_format(mime_type)
            except Exception as e:
                logger.debug(f"Magic detection failed: {e}")
        
        return None
    
    @classmethod
    def detect_format_from_file(cls, file_path: Union[str, Path]) -> Optional[str]:
        """
        Detect audio format from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Detected format or None
        """
        file_path = Path(file_path)
        
        # First try file extension
        extension = file_path.suffix.lower().lstrip('.')
        if extension in ["mp3", "wav", "m4a", "aac", "ogg", "flac", "wma"]:
            # Verify with content if possible
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(32)
                    detected = cls.detect_format_from_bytes(header)
                    if detected and detected == extension:
                        return extension
                    elif detected:
                        logger.warning(f"File extension {extension} doesn't match detected format {detected}")
                        return detected
                    else:
                        return extension  # Trust extension if detection fails
            except Exception:
                return extension
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return cls.mime_type_to_format(mime_type)
        
        # Try reading file content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
                return cls.detect_format_from_bytes(header)
        except Exception as e:
            logger.error(f"Error reading file for format detection: {e}")
            return None
    
    @classmethod
    def detect_format_from_base64(cls, base64_data: str) -> Optional[str]:
        """
        Detect audio format from base64 data.
        
        Args:
            base64_data: Base64 encoded audio data
            
        Returns:
            Detected format or None
        """
        try:
            # Handle data URLs
            if base64_data.startswith('data:'):
                # Extract MIME type from data URL
                if ';base64,' in base64_data:
                    mime_part = base64_data.split(';base64,')[0]
                    mime_type = mime_part.replace('data:', '')
                    format_from_mime = cls.mime_type_to_format(mime_type)
                    if format_from_mime:
                        return format_from_mime
                
                # Extract base64 part
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode and check magic numbers
            audio_bytes = base64.b64decode(base64_data[:100])  # Only decode first part for detection
            return cls.detect_format_from_bytes(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error detecting format from base64: {e}")
            return None
    
    @classmethod
    def mime_type_to_format(cls, mime_type: str) -> Optional[str]:
        """
        Convert MIME type to audio format.
        
        Args:
            mime_type: MIME type string
            
        Returns:
            Audio format or None
        """
        if not mime_type:
            return None
        
        mime_type = mime_type.lower()
        
        for mime, formats in cls.AUDIO_MIME_TYPES.items():
            if mime_type == mime:
                return formats[0]  # Return primary format
        
        return None
    
    @classmethod
    def format_to_mime_type(cls, format_name: str) -> Optional[str]:
        """
        Convert audio format to MIME type.
        
        Args:
            format_name: Audio format name
            
        Returns:
            MIME type or None
        """
        if not format_name:
            return None
        
        format_name = format_name.lower()
        
        for mime, formats in cls.AUDIO_MIME_TYPES.items():
            if format_name in formats:
                return mime
        
        return None


class AudioValidator:
    """
    Audio content validation utilities.
    """
    
    MIN_DURATION = 0.1  # Minimum duration in seconds
    MAX_DURATION = 7200  # Maximum duration in seconds (2 hours)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    SUPPORTED_SAMPLE_RATES = [8000, 11025, 16000, 22050, 44100, 48000, 96000, 192000]
    SUPPORTED_CHANNELS = [1, 2, 6, 8]  # Mono, Stereo, 5.1, 7.1
    
    @classmethod
    def validate_audio_data(cls, data: bytes, format_name: str = None) -> Dict[str, Any]:
        """
        Validate audio data.
        
        Args:
            data: Audio data bytes
            format_name: Optional format name for additional validation
            
        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check file size
        file_size = len(data)
        validation["info"]["file_size"] = file_size
        
        if file_size == 0:
            validation["valid"] = False
            validation["errors"].append("Empty audio data")
            return validation
        
        if file_size > cls.MAX_FILE_SIZE:
            validation["valid"] = False
            validation["errors"].append(f"File size {file_size} exceeds maximum {cls.MAX_FILE_SIZE}")
        
        # Detect format if not provided
        if not format_name:
            format_name = AudioFormatDetector.detect_format_from_bytes(data)
            if not format_name:
                validation["warnings"].append("Could not detect audio format")
            else:
                validation["info"]["detected_format"] = format_name
        
        # Format-specific validation
        if format_name:
            format_validation = cls._validate_format_specific(data, format_name)
            validation["errors"].extend(format_validation.get("errors", []))
            validation["warnings"].extend(format_validation.get("warnings", []))
            validation["info"].update(format_validation.get("info", {}))
        
        return validation
    
    @classmethod
    def validate_audio_file(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Validation results
        """
        file_path = Path(file_path)
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {"file_path": str(file_path)}
        }
        
        # Check if file exists
        if not file_path.exists():
            validation["valid"] = False
            validation["errors"].append(f"File does not exist: {file_path}")
            return validation
        
        # Check file size
        file_size = file_path.stat().st_size
        validation["info"]["file_size"] = file_size
        
        if file_size == 0:
            validation["valid"] = False
            validation["errors"].append("Empty audio file")
            return validation
        
        if file_size > cls.MAX_FILE_SIZE:
            validation["valid"] = False
            validation["errors"].append(f"File size {file_size} exceeds maximum {cls.MAX_FILE_SIZE}")
        
        # Read and validate content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB for validation
                
            format_name = AudioFormatDetector.detect_format_from_file(file_path)
            if format_name:
                validation["info"]["detected_format"] = format_name
            else:
                validation["warnings"].append("Could not detect audio format")
            
            # Basic header validation
            if format_name:
                format_validation = cls._validate_format_specific(header, format_name)
                validation["errors"].extend(format_validation.get("errors", []))
                validation["warnings"].extend(format_validation.get("warnings", []))
                validation["info"].update(format_validation.get("info", {}))
                
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Error reading file: {e}")
        
        return validation
    
    @classmethod
    def _validate_format_specific(cls, data: bytes, format_name: str) -> Dict[str, Any]:
        """
        Perform format-specific validation.
        
        Args:
            data: Audio data (at least header)
            format_name: Audio format name
            
        Returns:
            Format-specific validation results
        """
        validation = {"errors": [], "warnings": [], "info": {}}
        
        format_name = format_name.lower()
        
        if format_name == "mp3":
            validation.update(cls._validate_mp3(data))
        elif format_name == "wav":
            validation.update(cls._validate_wav(data))
        elif format_name == "flac":
            validation.update(cls._validate_flac(data))
        elif format_name == "ogg":
            validation.update(cls._validate_ogg(data))
        elif format_name in ["m4a", "aac"]:
            validation.update(cls._validate_m4a(data))
        
        return validation
    
    @classmethod
    def _validate_mp3(cls, data: bytes) -> Dict[str, Any]:
        """Validate MP3 format."""
        validation = {"errors": [], "warnings": [], "info": {}}
        
        if not data:
            validation["errors"].append("No data to validate")
            return validation
        
        # Check for ID3 tag or MP3 frame header
        if data.startswith(b"ID3"):
            validation["info"]["has_id3_tag"] = True
        elif data[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]:
            validation["info"]["has_mp3_frame"] = True
        else:
            validation["warnings"].append("No valid MP3 header found")
        
        return validation
    
    @classmethod
    def _validate_wav(cls, data: bytes) -> Dict[str, Any]:
        """Validate WAV format."""
        validation = {"errors": [], "warnings": [], "info": {}}
        
        if len(data) < 12:
            validation["errors"].append("Insufficient data for WAV validation")
            return validation
        
        # Check RIFF header
        if not data.startswith(b"RIFF"):
            validation["errors"].append("Invalid RIFF header")
            return validation
        
        # Check WAVE format
        if b"WAVE" not in data[:12]:
            validation["errors"].append("Invalid WAVE format identifier")
            return validation
        
        validation["info"]["valid_wav_header"] = True
        return validation
    
    @classmethod
    def _validate_flac(cls, data: bytes) -> Dict[str, Any]:
        """Validate FLAC format."""
        validation = {"errors": [], "warnings": [], "info": {}}
        
        if not data.startswith(b"fLaC"):
            validation["errors"].append("Invalid FLAC signature")
            return validation
        
        validation["info"]["valid_flac_header"] = True
        return validation
    
    @classmethod
    def _validate_ogg(cls, data: bytes) -> Dict[str, Any]:
        """Validate OGG format."""
        validation = {"errors": [], "warnings": [], "info": {}}
        
        if not data.startswith(b"OggS"):
            validation["errors"].append("Invalid OGG signature")
            return validation
        
        validation["info"]["valid_ogg_header"] = True
        return validation
    
    @classmethod
    def _validate_m4a(cls, data: bytes) -> Dict[str, Any]:
        """Validate M4A/AAC format."""
        validation = {"errors": [], "warnings": [], "info": {}}
        
        # M4A files can have various signatures
        if b"ftyp" in data[:20]:
            validation["info"]["has_ftyp_box"] = True
        else:
            validation["warnings"].append("No ftyp box found in M4A header")
        
        return validation


class AudioConverter:
    """
    Audio conversion utilities.
    """
    
    @staticmethod
    def bytes_to_base64(audio_bytes: bytes, mime_type: str = None) -> str:
        """
        Convert audio bytes to base64 data URL.
        
        Args:
            audio_bytes: Audio data as bytes
            mime_type: MIME type for data URL
            
        Returns:
            Base64 data URL string
        """
        base64_data = base64.b64encode(audio_bytes).decode('utf-8')
        
        if mime_type:
            return f"data:{mime_type};base64,{base64_data}"
        else:
            return base64_data
    
    @staticmethod
    def base64_to_bytes(base64_data: str) -> bytes:
        """
        Convert base64 data to bytes.
        
        Args:
            base64_data: Base64 string (with or without data URL prefix)
            
        Returns:
            Audio data as bytes
        """
        # Remove data URL prefix if present
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        return base64.b64decode(base64_data)
    
    @staticmethod
    def file_to_base64(file_path: Union[str, Path], include_mime: bool = True) -> str:
        """
        Convert audio file to base64.
        
        Args:
            file_path: Path to audio file
            include_mime: Whether to include MIME type in data URL
            
        Returns:
            Base64 string
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        if include_mime:
            format_name = AudioFormatDetector.detect_format_from_file(file_path)
            mime_type = AudioFormatDetector.format_to_mime_type(format_name) if format_name else None
            return AudioConverter.bytes_to_base64(audio_bytes, mime_type)
        else:
            return base64.b64encode(audio_bytes).decode('utf-8')
    
    @staticmethod
    def base64_to_file(base64_data: str, output_path: Union[str, Path]) -> None:
        """
        Save base64 audio data to file.
        
        Args:
            base64_data: Base64 audio data
            output_path: Output file path
        """
        audio_bytes = AudioConverter.base64_to_bytes(base64_data)
        
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)


class AudioMetrics:
    """
    Audio quality and performance metrics.
    """
    
    @staticmethod
    def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_size: Original file size in bytes
            compressed_size: Compressed file size in bytes
            
        Returns:
            Compression ratio
        """
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size
    
    @staticmethod
    def estimate_bitrate(file_size: int, duration: float) -> Optional[int]:
        """
        Estimate bitrate from file size and duration.
        
        Args:
            file_size: File size in bytes
            duration: Duration in seconds
            
        Returns:
            Estimated bitrate in kbps
        """
        if duration <= 0:
            return None
        
        # Convert to bits and calculate bitrate
        bits = file_size * 8
        bitrate_bps = bits / duration
        bitrate_kbps = int(bitrate_bps / 1000)
        
        return bitrate_kbps
    
    @staticmethod
    def classify_quality_by_bitrate(bitrate: int, format_name: str = None) -> str:
        """
        Classify audio quality based on bitrate.
        
        Args:
            bitrate: Bitrate in kbps
            format_name: Audio format for context
            
        Returns:
            Quality classification
        """
        if format_name and format_name.lower() in ["flac", "wav"]:
            return "lossless"
        
        if bitrate >= 320:
            return "excellent"
        elif bitrate >= 256:
            return "very_good"
        elif bitrate >= 192:
            return "good"
        elif bitrate >= 128:
            return "acceptable"
        elif bitrate >= 96:
            return "fair"
        else:
            return "poor"
    
    @staticmethod
    def get_format_characteristics(format_name: str) -> Dict[str, Any]:
        """
        Get characteristics of audio format.
        
        Args:
            format_name: Audio format name
            
        Returns:
            Format characteristics
        """
        characteristics = {
            "mp3": {
                "compression": "lossy",
                "typical_bitrates": [128, 192, 256, 320],
                "max_sample_rate": 48000,
                "max_channels": 2,
                "supports_metadata": True,
                "patent_free": False
            },
            "wav": {
                "compression": "none",
                "typical_bitrates": [1411],  # CD quality
                "max_sample_rate": 192000,
                "max_channels": 8,
                "supports_metadata": False,
                "patent_free": True
            },
            "flac": {
                "compression": "lossless",
                "typical_bitrates": [700, 1000],  # Variable
                "max_sample_rate": 655350,
                "max_channels": 8,
                "supports_metadata": True,
                "patent_free": True
            },
            "ogg": {
                "compression": "lossy",
                "typical_bitrates": [128, 192, 256],
                "max_sample_rate": 192000,
                "max_channels": 255,
                "supports_metadata": True,
                "patent_free": True
            },
            "m4a": {
                "compression": "lossy",
                "typical_bitrates": [128, 256, 320],
                "max_sample_rate": 96000,
                "max_channels": 48,
                "supports_metadata": True,
                "patent_free": False
            }
        }
        
        return characteristics.get(format_name.lower(), {
            "compression": "unknown",
            "typical_bitrates": [],
            "max_sample_rate": 0,
            "max_channels": 0,
            "supports_metadata": False,
            "patent_free": None
        })