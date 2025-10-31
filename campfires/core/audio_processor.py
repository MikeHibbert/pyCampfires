"""
Audio processing capabilities for Campfires.
"""

import logging
import tempfile
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import base64
import io

# Provide module-level names for patching in tests
try:
    import pydub  # type: ignore
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    pydub = None  # type: ignore
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    import mutagen  # type: ignore
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    mutagen = None  # type: ignore
    MUTAGEN_AVAILABLE = False
    MutagenFile = None

from .multimodal_torch import MultimodalContent, ContentType

logger = logging.getLogger(__name__)


class AudioMetadata:
    """
    Audio metadata container.
    """
    
    def __init__(
        self,
        duration: float = None,
        sample_rate: int = None,
        channels: int = None,
        bitrate: int = None,
        format: str = None,
        codec: str = None,
        title: str = None,
        artist: str = None,
        album: str = None,
        genre: str = None,
        year: int = None,
        track_number: int = None,
        file_size: int = None
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.bitrate = bitrate
        self.format = format
        self.codec = codec
        self.title = title
        self.artist = artist
        self.album = album
        self.genre = genre
        self.year = year
        self.track_number = track_number
        self.file_size = file_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bitrate": self.bitrate,
            "format": self.format,
            "codec": self.codec,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "year": self.year,
            "track_number": self.track_number,
            "file_size": self.file_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioMetadata":
        """Create from dictionary."""
        return cls(**data)


class AudioProcessor:
    """
    Audio processing and analysis capabilities.
    """
    
    SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "aac", "ogg", "flac", "wma"]
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available - audio processing capabilities limited")
        
        if not MUTAGEN_AVAILABLE:
            logger.warning("mutagen not available - metadata extraction limited")
        
        # Check for ffmpeg (required by pydub for many formats)
        if PYDUB_AVAILABLE and not which("ffmpeg"):
            logger.warning("ffmpeg not found - some audio formats may not be supported")
    
    def extract_metadata(self, audio_data: Union[str, bytes, Path]) -> AudioMetadata:
        """
        Extract metadata from audio file.
        
        Args:
            audio_data: Audio file path, bytes, or base64 string
            
        Returns:
            AudioMetadata object
        """
        try:
            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                file_path = Path(audio_data)
                if file_path.exists():
                    return self._extract_metadata_from_file(file_path)
                else:
                    # Assume it's base64 data
                    return self._extract_metadata_from_base64(str(audio_data))
            elif isinstance(audio_data, bytes):
                return self._extract_metadata_from_bytes(audio_data)
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
                
        except Exception as e:
            logger.error(f"Error extracting audio metadata: {e}")
            return AudioMetadata()
    
    def _extract_metadata_from_file(self, file_path: Path) -> AudioMetadata:
        """Extract metadata from audio file."""
        metadata = AudioMetadata()
        metadata.file_size = file_path.stat().st_size
        metadata.format = file_path.suffix.lower().lstrip('.')
        
        # Use mutagen for detailed metadata
        if mutagen is not None:
            try:
                audio_file = mutagen.File(file_path)
                if audio_file:
                    # Basic info
                    if hasattr(audio_file, 'info'):
                        info = audio_file.info
                        metadata.duration = getattr(info, 'length', None)
                        metadata.bitrate = getattr(info, 'bitrate', None)
                        metadata.sample_rate = getattr(info, 'sample_rate', None)
                        metadata.channels = getattr(info, 'channels', None)
                    
                    # Tags
                    if hasattr(audio_file, 'tags') and audio_file.tags:
                        tags = audio_file.tags
                        metadata.title = self._get_tag_value(tags, ['TIT2', 'TITLE', '\xa9nam'])
                        metadata.artist = self._get_tag_value(tags, ['TPE1', 'ARTIST', '\xa9ART'])
                        metadata.album = self._get_tag_value(tags, ['TALB', 'ALBUM', '\xa9alb'])
                        metadata.genre = self._get_tag_value(tags, ['TCON', 'GENRE', '\xa9gen'])
                        
                        # Year handling
                        year_value = self._get_tag_value(tags, ['TDRC', 'DATE', '\xa9day'])
                        if year_value:
                            try:
                                metadata.year = int(str(year_value)[:4])
                            except (ValueError, TypeError):
                                pass
                        
                        # Track number
                        track_value = self._get_tag_value(tags, ['TRCK', 'TRACKNUMBER', 'trkn'])
                        if track_value:
                            try:
                                # Handle "track/total" format
                                track_str = str(track_value).split('/')[0]
                                metadata.track_number = int(track_str)
                            except (ValueError, TypeError):
                                pass
                                
            except Exception as e:
                logger.warning(f"Error extracting metadata with mutagen: {e}")
        
        # Use pydub as fallback for basic info
        if pydub is not None and not metadata.duration:
            try:
                audio_segment = pydub.AudioSegment.from_file(file_path)
                # Prefer duration_seconds if available (used in tests)
                duration_sec = getattr(audio_segment, 'duration_seconds', None)
                metadata.duration = duration_sec if duration_sec is not None else (len(audio_segment) / 1000.0)
                metadata.channels = getattr(audio_segment, 'channels', None)
                metadata.sample_rate = getattr(audio_segment, 'frame_rate', None)
            except Exception as e:
                logger.warning(f"Error extracting metadata with pydub: {e}")
        
        return metadata
    
    def _extract_metadata_from_bytes(self, audio_bytes: bytes) -> AudioMetadata:
        """Extract metadata from audio bytes."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)
        
        try:
            metadata = self._extract_metadata_from_file(temp_path)
            metadata.file_size = len(audio_bytes)
            return metadata
        finally:
            # Clean up temporary file
            try:
                temp_path.unlink()
            except Exception:
                pass
    
    def _extract_metadata_from_base64(self, base64_data: str) -> AudioMetadata:
        """Extract metadata from base64 encoded audio."""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            
            audio_bytes = base64.b64decode(base64_data)
            return self._extract_metadata_from_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Error decoding base64 audio data: {e}")
            return AudioMetadata()
    
    def _get_tag_value(self, tags, tag_keys: List[str]) -> Optional[str]:
        """Get tag value from multiple possible keys."""
        for key in tag_keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif value:
                    return str(value)
        return None
    
    def analyze_audio_content(self, content: Union[str, Path, MultimodalContent]) -> Dict[str, Any]:
        """
        Analyze audio content and return comprehensive information.
        
        Args:
            content: File path or MultimodalContent with audio data
        
        Returns:
            Analysis results
        """
        # Support direct path input for tests
        if isinstance(content, (str, Path)):
            path = Path(content)
            # Basic info via pydub if available
            duration = sample_rate = channels = None
            loudness_ratio = None
            if pydub is not None:
                audio_segment = pydub.AudioSegment.from_file(path)
                # Prefer duration_seconds if provided by mock/tests
                duration = getattr(audio_segment, 'duration_seconds', None)
                if duration is None:
                    duration = len(audio_segment) / 1000.0
                sample_rate = audio_segment.frame_rate
                channels = audio_segment.channels
                # Approximate loudness ratio using max amplitude
                try:
                    max_possible = getattr(audio_segment, 'max_possible_amplitude', 32767)
                    current_max = getattr(audio_segment, 'max', None)
                    if current_max is not None and max_possible:
                        loudness_ratio = current_max / max_possible
                except Exception:
                    loudness_ratio = None
            metadata = self.extract_metadata(path)
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "loudness_ratio": loudness_ratio,
                "metadata": metadata.to_dict(),
            }
        
        # Original multimodal path
        if content.content_type != ContentType.AUDIO:
            raise ValueError("Content must be audio type")
        
        # Extract metadata
        metadata = self.extract_metadata(content.data)
        
        # Perform basic analysis
        analysis = {
            "metadata": metadata.to_dict(),
            "content_type": "audio",
            "analysis_timestamp": content.metadata.get("timestamp"),
            "source": content.metadata.get("source", "unknown")
        }
        
        # Add format-specific analysis
        if metadata.format:
            analysis["format_info"] = self._analyze_audio_format(metadata.format)
        
        # Add quality assessment
        analysis["quality_assessment"] = self._assess_audio_quality(metadata)
        
        # Add content classification
        analysis["content_classification"] = self._classify_audio_content(metadata)
        
        return analysis
    
    def _analyze_audio_format(self, format_name: str) -> Dict[str, Any]:
        """Analyze audio format characteristics."""
        format_info = {
            "mp3": {
                "type": "lossy",
                "compression": "high",
                "quality": "good",
                "compatibility": "excellent",
                "use_case": "general purpose, streaming"
            },
            "wav": {
                "type": "lossless",
                "compression": "none",
                "quality": "excellent",
                "compatibility": "good",
                "use_case": "professional audio, editing"
            },
            "flac": {
                "type": "lossless",
                "compression": "medium",
                "quality": "excellent",
                "compatibility": "good",
                "use_case": "archival, high-quality storage"
            },
            "m4a": {
                "type": "lossy",
                "compression": "high",
                "quality": "good",
                "compatibility": "good",
                "use_case": "Apple ecosystem, podcasts"
            },
            "ogg": {
                "type": "lossy",
                "compression": "high",
                "quality": "good",
                "compatibility": "limited",
                "use_case": "open source applications"
            }
        }
        
        return format_info.get(format_name.lower(), {
            "type": "unknown",
            "compression": "unknown",
            "quality": "unknown",
            "compatibility": "unknown",
            "use_case": "unknown"
        })
    
    def _assess_audio_quality(self, metadata: AudioMetadata) -> Dict[str, Any]:
        """Assess audio quality based on metadata."""
        quality = {"overall": "unknown", "factors": {}}
        
        # Sample rate assessment
        if metadata.sample_rate:
            if metadata.sample_rate >= 48000:
                quality["factors"]["sample_rate"] = "excellent"
            elif metadata.sample_rate >= 44100:
                quality["factors"]["sample_rate"] = "good"
            elif metadata.sample_rate >= 22050:
                quality["factors"]["sample_rate"] = "fair"
            else:
                quality["factors"]["sample_rate"] = "poor"
        
        # Bitrate assessment (for lossy formats)
        if metadata.bitrate:
            if metadata.bitrate >= 320:
                quality["factors"]["bitrate"] = "excellent"
            elif metadata.bitrate >= 256:
                quality["factors"]["bitrate"] = "good"
            elif metadata.bitrate >= 192:
                quality["factors"]["bitrate"] = "fair"
            elif metadata.bitrate >= 128:
                quality["factors"]["bitrate"] = "acceptable"
            else:
                quality["factors"]["bitrate"] = "poor"
        
        # Channels assessment
        if metadata.channels:
            if metadata.channels >= 6:
                quality["factors"]["channels"] = "surround"
            elif metadata.channels == 2:
                quality["factors"]["channels"] = "stereo"
            elif metadata.channels == 1:
                quality["factors"]["channels"] = "mono"
        
        # Overall assessment
        factor_scores = {"excellent": 4, "good": 3, "fair": 2, "acceptable": 1, "poor": 0}
        if quality["factors"]:
            avg_score = sum(factor_scores.get(v, 0) for v in quality["factors"].values()) / len(quality["factors"])
            if avg_score >= 3.5:
                quality["overall"] = "excellent"
            elif avg_score >= 2.5:
                quality["overall"] = "good"
            elif avg_score >= 1.5:
                quality["overall"] = "fair"
            else:
                quality["overall"] = "poor"
        
        return quality
    
    def _classify_audio_content(self, metadata: AudioMetadata) -> Dict[str, Any]:
        """Classify audio content based on metadata."""
        classification = {
            "type": "unknown",
            "confidence": 0.0,
            "indicators": []
        }
        
        # Duration-based classification
        if metadata.duration:
            if metadata.duration < 30:
                classification["indicators"].append("short_clip")
            elif metadata.duration < 300:  # 5 minutes
                classification["indicators"].append("song_or_speech")
            elif metadata.duration < 3600:  # 1 hour
                classification["indicators"].append("long_content")
            else:
                classification["indicators"].append("very_long_content")
        
        # Metadata-based classification
        if metadata.artist or metadata.album:
            classification["type"] = "music"
            classification["confidence"] = 0.8
            classification["indicators"].append("has_music_metadata")
        
        # Format-based hints
        if metadata.format in ["mp3", "m4a", "flac"]:
            if "music" not in classification["type"]:
                classification["indicators"].append("music_format")
        elif metadata.format in ["wav"]:
            classification["indicators"].append("professional_audio")
        
        return classification
    
    def convert_audio_format(
        self,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str,
    ) -> bool:
        """
        Convert audio to different format and write to output path.
        
        Args:
            source_path: Source audio file path
            output_path: Output file path
            target_format: Target format (mp3, wav, etc.)
        
        Returns:
            True on success
        """
        if pydub is None:
            raise RuntimeError("pydub is required for audio conversion")
        
        if target_format.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        try:
            audio_segment = pydub.AudioSegment.from_file(source_path)
            audio_segment.export(str(output_path), format=target_format.lower())
            return True
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            raise
    
    def extract_audio_segment(
        self,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float,
        end_time: float,
    ) -> bool:
        """
        Extract a single time segment and write to output path.
        """
        if pydub is None:
            raise RuntimeError("pydub is required for audio segmentation")
        try:
            audio_segment = pydub.AudioSegment.from_file(source_path)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment = audio_segment[start_ms:end_ms]
            segment.export(str(output_path), format="mp3")
            return True
        except Exception as e:
            logger.error(f"Error extracting audio segment: {e}")
            raise
    
    def generate_waveform_data(
        self,
        source_path: Union[str, Path],
        samples: int = 1000
    ) -> List[float]:
        """
        Generate a simplified waveform sample list for visualization.
        """
        if pydub is None:
            raise RuntimeError("pydub is required for waveform analysis")
        try:
            audio_segment = pydub.AudioSegment.from_file(source_path)
            array = audio_segment.get_array_of_samples()
            # Downsample to requested number of points
            step = max(1, len(array) // samples)
            return [float(array[i]) for i in range(0, len(array), step)][:samples]
        except Exception as e:
            logger.error(f"Error generating waveform data: {e}")
            raise