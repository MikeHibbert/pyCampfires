"""
Metadata extraction utilities for multimodal content.
"""

import logging
import mimetypes
import hashlib
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ContentMetadata:
    """Base metadata for any content type."""
    filename: str
    file_size: int
    content_type: str
    mime_type: Optional[str] = None
    content_hash: Optional[str] = None
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    extraction_timestamp: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageMetadata(ContentMetadata):
    """Metadata specific to image content."""
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    mode: Optional[str] = None
    has_transparency: Optional[bool] = None
    aspect_ratio: Optional[float] = None
    unique_colors: Optional[int] = None
    dominant_color: Optional[Union[tuple, str]] = None
    exif: Optional[Dict[str, Any]] = None
    date_taken: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    estimated_quality: Optional[str] = None


@dataclass
class AudioMetadata(ContentMetadata):
    """Metadata specific to audio content."""
    duration: Optional[float] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    mode: Optional[str] = None
    version: Optional[str] = None
    layer: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    quality: Optional[str] = None


@dataclass
class VideoMetadata(ContentMetadata):
    """Metadata specific to video content."""
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    aspect_ratio: Optional[float] = None
    bitrate: Optional[int] = None


@dataclass
class DocumentMetadata(ContentMetadata):
    """Metadata specific to document content."""
    character_count: Optional[int] = None
    word_count: Optional[int] = None
    line_count: Optional[int] = None
    header_count: Optional[int] = None
    encoding: Optional[str] = None
    format: Optional[str] = None
    page_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None


class MetadataExtractor:
    """
    Extract metadata from various file types.
    """
    
    # Content type mappings
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.wma'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.odt'}
    
    @classmethod
    def extract_metadata(cls, file_path: Union[str, Path], data: bytes = None) -> Dict[str, Any]:
        """
        Extract metadata from a file or data.
        
        Args:
            file_path: Path to the file
            data: Optional file data as bytes
            
        Returns:
            Dictionary containing extracted metadata
        """
        file_path = Path(file_path)
        
        metadata = {
            'filename': file_path.name,
            'extension': file_path.suffix.lower(),
            'extraction_timestamp': datetime.utcnow().isoformat(),
            'extractor_version': '1.0.0'
        }
        
        # Basic file information
        if file_path.exists():
            stat = file_path.stat()
            metadata.update({
                'file_size': stat.st_size,
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        elif data:
            metadata['file_size'] = len(data)
        
        # Content hash
        if data:
            metadata['content_hash'] = hashlib.sha256(data).hexdigest()
        elif file_path.exists():
            with open(file_path, 'rb') as f:
                content = f.read()
                metadata['content_hash'] = hashlib.sha256(content).hexdigest()
        
        # MIME type detection
        mime_type = cls._detect_mime_type(file_path, data)
        if mime_type:
            metadata['mime_type'] = mime_type
        
        # Content type classification
        content_type = cls._classify_content_type(file_path)
        metadata['content_type'] = content_type
        
        # Type-specific metadata extraction
        try:
            if content_type == 'image':
                metadata.update(cls._extract_image_metadata(file_path, data))
            elif content_type == 'audio':
                metadata.update(cls._extract_audio_metadata(file_path, data))
            elif content_type == 'video':
                metadata.update(cls._extract_video_metadata(file_path, data))
            elif content_type == 'document':
                metadata.update(cls._extract_document_metadata(file_path, data))
        except Exception as e:
            logger.warning(f"Error extracting {content_type} metadata: {e}")
            metadata['extraction_error'] = str(e)
        
        return metadata
    
    @classmethod
    def _detect_mime_type(cls, file_path: Path, data: bytes = None) -> Optional[str]:
        """Detect MIME type of file."""
        # Try python-magic first if available
        if MAGIC_AVAILABLE and data:
            try:
                return magic.from_buffer(data, mime=True)
            except Exception:
                pass
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type
    
    @classmethod
    def _classify_content_type(cls, file_path: Path) -> str:
        """Classify content type based on file extension."""
        ext = file_path.suffix.lower()
        
        if ext in cls.IMAGE_EXTENSIONS:
            return 'image'
        elif ext in cls.AUDIO_EXTENSIONS:
            return 'audio'
        elif ext in cls.VIDEO_EXTENSIONS:
            return 'video'
        elif ext in cls.DOCUMENT_EXTENSIONS:
            return 'document'
        else:
            return 'other'
    
    @classmethod
    def _extract_image_metadata(cls, file_path: Path, data: bytes = None) -> Dict[str, Any]:
        """Extract metadata from image files."""
        metadata = {}
        
        if not PIL_AVAILABLE:
            metadata['error'] = 'PIL not available for image metadata extraction'
            return metadata
        
        try:
            # Open image
            if data:
                from io import BytesIO
                image = Image.open(BytesIO(data))
            else:
                image = Image.open(file_path)
            
            # Basic image information
            metadata.update({
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            })
            
            # Calculate aspect ratio
            if image.height > 0:
                metadata['aspect_ratio'] = round(image.width / image.height, 3)
            
            # Color information
            if hasattr(image, 'getcolors'):
                try:
                    colors = image.getcolors(maxcolors=256*256*256)
                    if colors:
                        metadata['unique_colors'] = len(colors)
                        # Most common color
                        most_common = max(colors, key=lambda x: x[0])
                        metadata['dominant_color'] = most_common[1] if isinstance(most_common[1], (tuple, list)) else str(most_common[1])
                except Exception:
                    pass
            
            # EXIF data
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = {}
                exif = image._getexif()
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Convert bytes to string for JSON serialization
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            value = str(value)
                    
                    exif_data[tag] = value
                
                metadata['exif'] = exif_data
                
                # Extract common EXIF fields
                if 'DateTime' in exif_data:
                    metadata['date_taken'] = exif_data['DateTime']
                if 'Make' in exif_data:
                    metadata['camera_make'] = exif_data['Make']
                if 'Model' in exif_data:
                    metadata['camera_model'] = exif_data['Model']
                if 'Software' in exif_data:
                    metadata['software'] = exif_data['Software']
            
            # Image quality estimation
            metadata['estimated_quality'] = cls._estimate_image_quality(image)
            
        except Exception as e:
            metadata['error'] = f"Error extracting image metadata: {e}"
        
        return metadata
    
    @classmethod
    def _extract_audio_metadata(cls, file_path: Path, data: bytes = None) -> Dict[str, Any]:
        """Extract metadata from audio files."""
        metadata = {}
        
        if not MUTAGEN_AVAILABLE:
            metadata['error'] = 'Mutagen not available for audio metadata extraction'
            return metadata
        
        try:
            # Use file path for mutagen
            if file_path.exists():
                audio_file = MutagenFile(str(file_path))
            else:
                # For data-only extraction, we'd need to write to temp file
                metadata['error'] = 'Audio metadata extraction from bytes requires file path'
                return metadata
            
            if audio_file is None:
                metadata['error'] = 'Could not read audio file'
                return metadata
            
            # Basic audio information
            if hasattr(audio_file, 'info'):
                info = audio_file.info
                metadata.update({
                    'duration': getattr(info, 'length', 0),
                    'bitrate': getattr(info, 'bitrate', 0),
                    'sample_rate': getattr(info, 'sample_rate', 0),
                    'channels': getattr(info, 'channels', 0),
                })
                
                # Format-specific information
                if hasattr(info, 'mode'):
                    metadata['mode'] = info.mode
                if hasattr(info, 'version'):
                    metadata['version'] = info.version
                if hasattr(info, 'layer'):
                    metadata['layer'] = info.layer
            
            # Tags/metadata
            if audio_file.tags:
                tags = {}
                
                # Common tags
                tag_mapping = {
                    'TIT2': 'title',
                    'TPE1': 'artist',
                    'TALB': 'album',
                    'TDRC': 'year',
                    'TCON': 'genre',
                    'TPE2': 'album_artist',
                    'TRCK': 'track_number',
                    'TPOS': 'disc_number'
                }
                
                for tag_key, readable_key in tag_mapping.items():
                    if tag_key in audio_file.tags:
                        value = audio_file.tags[tag_key]
                        if hasattr(value, 'text'):
                            tags[readable_key] = str(value.text[0]) if value.text else ''
                        else:
                            tags[readable_key] = str(value)
                
                # Generic tag extraction for other formats
                for key, value in audio_file.tags.items():
                    if key not in tag_mapping:
                        if hasattr(value, 'text'):
                            tags[key] = str(value.text[0]) if value.text else ''
                        else:
                            tags[key] = str(value)
                
                metadata['tags'] = tags
            
            # Audio quality classification
            bitrate = metadata.get('bitrate', 0)
            if bitrate > 0:
                if bitrate >= 320:
                    metadata['quality'] = 'high'
                elif bitrate >= 192:
                    metadata['quality'] = 'medium'
                elif bitrate >= 128:
                    metadata['quality'] = 'standard'
                else:
                    metadata['quality'] = 'low'
            
        except Exception as e:
            metadata['error'] = f"Error extracting audio metadata: {e}"
        
        return metadata
    
    @classmethod
    def _extract_video_metadata(cls, file_path: Path, data: bytes = None) -> Dict[str, Any]:
        """Extract metadata from video files."""
        metadata = {}
        
        # Basic video metadata extraction without heavy dependencies
        # In a full implementation, you might use ffmpeg-python or opencv
        
        try:
            # For now, just basic file information
            metadata.update({
                'content_type': 'video',
                'note': 'Video metadata extraction requires additional dependencies (ffmpeg-python or opencv-python)'
            })
            
            # Could add ffmpeg-based extraction here if available
            # try:
            #     import ffmpeg
            #     probe = ffmpeg.probe(str(file_path))
            #     # Extract video stream info
            #     video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            #     if video_stream:
            #         metadata.update({
            #             'width': int(video_stream.get('width', 0)),
            #             'height': int(video_stream.get('height', 0)),
            #             'duration': float(video_stream.get('duration', 0)),
            #             'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            #             'codec': video_stream.get('codec_name'),
            #         })
            # except ImportError:
            #     pass
            
        except Exception as e:
            metadata['error'] = f"Error extracting video metadata: {e}"
        
        return metadata
    
    @classmethod
    def _extract_document_metadata(cls, file_path: Path, data: bytes = None) -> Dict[str, Any]:
        """Extract metadata from document files."""
        metadata = {}
        
        try:
            ext = file_path.suffix.lower()
            
            if ext == '.txt':
                # Text file analysis
                if data:
                    content = data.decode('utf-8', errors='ignore')
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                metadata.update({
                    'character_count': len(content),
                    'word_count': len(content.split()),
                    'line_count': len(content.splitlines()),
                    'encoding': 'utf-8'
                })
                
            elif ext == '.md':
                # Markdown file analysis
                if data:
                    content = data.decode('utf-8', errors='ignore')
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                lines = content.splitlines()
                headers = [line for line in lines if line.startswith('#')]
                
                metadata.update({
                    'character_count': len(content),
                    'word_count': len(content.split()),
                    'line_count': len(lines),
                    'header_count': len(headers),
                    'encoding': 'utf-8',
                    'format': 'markdown'
                })
                
            elif ext == '.pdf':
                # PDF metadata would require PyPDF2 or similar
                metadata.update({
                    'format': 'pdf',
                    'note': 'PDF metadata extraction requires additional dependencies (PyPDF2 or pdfplumber)'
                })
                
            else:
                metadata.update({
                    'format': ext.lstrip('.'),
                    'note': f'Metadata extraction for {ext} files not implemented'
                })
                
        except Exception as e:
            metadata['error'] = f"Error extracting document metadata: {e}"
        
        return metadata
    
    @classmethod
    def _estimate_image_quality(cls, image: 'Image.Image') -> str:
        """Estimate image quality based on various factors."""
        try:
            # Simple quality estimation based on resolution and format
            total_pixels = image.width * image.height
            
            if total_pixels >= 8000000:  # 8MP+
                quality = 'high'
            elif total_pixels >= 2000000:  # 2MP+
                quality = 'medium'
            elif total_pixels >= 500000:   # 0.5MP+
                quality = 'standard'
            else:
                quality = 'low'
            
            # Adjust based on format
            if image.format in ['PNG', 'TIFF']:
                # Lossless formats get a boost
                if quality == 'standard':
                    quality = 'medium'
                elif quality == 'medium':
                    quality = 'high'
            
            return quality
            
        except Exception:
            return 'unknown'
    
    @classmethod
    def extract_thumbnail_info(cls, file_path: Union[str, Path], data: bytes = None) -> Dict[str, Any]:
        """
        Extract information needed for thumbnail generation.
        
        Args:
            file_path: Path to the file
            data: Optional file data as bytes
            
        Returns:
            Dictionary with thumbnail generation info
        """
        file_path = Path(file_path)
        content_type = cls._classify_content_type(file_path)
        
        thumbnail_info = {
            'content_type': content_type,
            'can_generate_thumbnail': False,
            'recommended_size': (200, 200),
            'format': 'JPEG'
        }
        
        if content_type == 'image' and PIL_AVAILABLE:
            thumbnail_info.update({
                'can_generate_thumbnail': True,
                'method': 'PIL',
                'supported_formats': ['JPEG', 'PNG', 'WebP']
            })
        elif content_type == 'video':
            thumbnail_info.update({
                'can_generate_thumbnail': True,
                'method': 'ffmpeg',
                'note': 'Requires ffmpeg for video thumbnail extraction'
            })
        elif content_type == 'document' and file_path.suffix.lower() == '.pdf':
            thumbnail_info.update({
                'can_generate_thumbnail': True,
                'method': 'pdf2image',
                'note': 'Requires pdf2image for PDF thumbnail extraction'
            })
        
        return thumbnail_info
    
    @classmethod
    def generate_content_fingerprint(cls, metadata: Dict[str, Any]) -> str:
        """
        Generate a content fingerprint for deduplication.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Content fingerprint string
        """
        # Use content hash if available
        if 'content_hash' in metadata:
            return metadata['content_hash']
        
        # Fallback to metadata-based fingerprint
        fingerprint_data = {
            'file_size': metadata.get('file_size'),
            'content_type': metadata.get('content_type'),
        }
        
        # Add type-specific fingerprint data
        content_type = metadata.get('content_type')
        
        if content_type == 'image':
            fingerprint_data.update({
                'width': metadata.get('width'),
                'height': metadata.get('height'),
                'format': metadata.get('format')
            })
        elif content_type == 'audio':
            fingerprint_data.update({
                'duration': metadata.get('duration'),
                'bitrate': metadata.get('bitrate'),
                'sample_rate': metadata.get('sample_rate')
            })
        
        # Create hash from fingerprint data
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()


class ThumbnailGenerator:
    """
    Generate thumbnails for various content types.
    """
    
    DEFAULT_SIZE = (200, 200)
    DEFAULT_FORMAT = 'JPEG'
    
    @classmethod
    def generate_image_thumbnail(cls, file_path: Union[str, Path], data: bytes = None, 
                                size: tuple = None, format: str = None) -> Optional[bytes]:
        """
        Generate thumbnail for image files.
        
        Args:
            file_path: Path to image file
            data: Optional image data as bytes
            size: Thumbnail size tuple (width, height)
            format: Output format
            
        Returns:
            Thumbnail data as bytes or None if failed
        """
        if not PIL_AVAILABLE:
            return None
        
        size = size or cls.DEFAULT_SIZE
        format = format or cls.DEFAULT_FORMAT
        
        try:
            # Open image
            if data:
                from io import BytesIO
                image = Image.open(BytesIO(data))
            else:
                image = Image.open(file_path)
            
            # Convert to RGB if necessary (for JPEG output)
            if format == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Generate thumbnail
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            from io import BytesIO
            output = BytesIO()
            image.save(output, format=format, quality=85, optimize=True)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating image thumbnail: {e}")
            return None
    
    @classmethod
    def can_generate_thumbnail(cls, content_type: str, file_extension: str) -> bool:
        """
        Check if thumbnail generation is supported for content type.
        
        Args:
            content_type: Content type (image, video, document, etc.)
            file_extension: File extension
            
        Returns:
            True if thumbnail generation is supported
        """
        if content_type == 'image' and PIL_AVAILABLE:
            return True
        
        # Could add video and PDF support here
        # if content_type == 'video' and FFMPEG_AVAILABLE:
        #     return True
        # if content_type == 'document' and file_extension == '.pdf' and PDF2IMAGE_AVAILABLE:
        #     return True
        
        return False