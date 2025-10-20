"""
Multimodal torch data structures for passing messages with multiple content types.
"""

import time
import base64
from enum import Enum
from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field
from .torch import Torch


class ContentType(Enum):
    """Supported content types for multimodal messages."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class MultimodalContent(BaseModel):
    """
    Represents a single piece of content in a multimodal message.
    """
    
    content_type: ContentType = Field(
        description="Type of content (text, image, audio, video, document)"
    )
    
    data: Union[str, bytes] = Field(
        description="Content data - text string or binary data"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Content-specific metadata (dimensions, duration, format, etc.)"
    )
    
    asset_hash: Optional[str] = Field(
        default=None,
        description="Hash reference to Party Box asset if data is stored there"
    )
    
    mime_type: Optional[str] = Field(
        default=None,
        description="MIME type of the content (e.g., 'image/jpeg', 'audio/mp3')"
    )
    
    encoding: Optional[str] = Field(
        default=None,
        description="Encoding format for binary data (e.g., 'base64')"
    )
    
    def get_data_as_string(self) -> str:
        """
        Get content data as string, handling encoding if necessary.
        
        Returns:
            String representation of the data
        """
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, bytes):
            if self.encoding == "base64":
                return base64.b64encode(self.data).decode('utf-8')
            else:
                # Try to decode as UTF-8, fallback to base64
                try:
                    return self.data.decode('utf-8')
                except UnicodeDecodeError:
                    return base64.b64encode(self.data).decode('utf-8')
        else:
            return str(self.data)
    
    def get_data_as_bytes(self) -> bytes:
        """
        Get content data as bytes, handling decoding if necessary.
        
        Returns:
            Bytes representation of the data
        """
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            if self.encoding == "base64":
                return base64.b64decode(self.data)
            else:
                return self.data.encode('utf-8')
        else:
            return str(self.data).encode('utf-8')
    
    def is_binary(self) -> bool:
        """Check if this content represents binary data."""
        return self.content_type in [ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO]
    
    def get_size(self) -> int:
        """Get the size of the content in bytes."""
        if isinstance(self.data, bytes):
            return len(self.data)
        elif isinstance(self.data, str):
            return len(self.data.encode('utf-8'))
        else:
            return len(str(self.data).encode('utf-8'))


class MultimodalTorch(BaseModel):
    """
    A multimodal message carrying multiple types of content between campfires.
    
    Extends the concept of Torch to support text, images, audio, video, and documents
    in a single message while maintaining backward compatibility.
    """
    
    contents: List[MultimodalContent] = Field(
        description="List of content pieces in this multimodal message"
    )
    
    primary_claim: str = Field(
        description="Main text description or claim about the multimodal content"
    )
    
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the overall message (0.0 to 1.0)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata and context for the entire message"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when torch was created"
    )
    
    source_campfire: str = Field(
        description="Name of the campfire that created this torch"
    )
    
    channel: str = Field(
        description="MCP channel this torch is being sent to"
    )
    
    torch_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this torch"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.torch_id is None:
            # Generate a simple torch ID based on timestamp and source
            self.torch_id = f"{self.source_campfire}_{int(self.timestamp * 1000)}_mm"
    
    def get_text_contents(self) -> List[MultimodalContent]:
        """Get all text content pieces."""
        return [c for c in self.contents if c.content_type == ContentType.TEXT]
    
    def get_image_contents(self) -> List[MultimodalContent]:
        """Get all image content pieces."""
        return [c for c in self.contents if c.content_type == ContentType.IMAGE]
    
    def get_audio_contents(self) -> List[MultimodalContent]:
        """Get all audio content pieces."""
        return [c for c in self.contents if c.content_type == ContentType.AUDIO]
    
    def get_video_contents(self) -> List[MultimodalContent]:
        """Get all video content pieces."""
        return [c for c in self.contents if c.content_type == ContentType.VIDEO]
    
    def get_document_contents(self) -> List[MultimodalContent]:
        """Get all document content pieces."""
        return [c for c in self.contents if c.content_type == ContentType.DOCUMENT]
    
    def get_content_types(self) -> List[ContentType]:
        """Get list of unique content types in this torch."""
        return list(set(c.content_type for c in self.contents))
    
    def has_content_type(self, content_type: ContentType) -> bool:
        """Check if torch contains content of specified type."""
        return content_type in self.get_content_types()
    
    def is_multimodal(self) -> bool:
        """Check if torch contains more than just text content."""
        content_types = self.get_content_types()
        return len(content_types) > 1 or (len(content_types) == 1 and ContentType.TEXT not in content_types)
    
    def get_total_size(self) -> int:
        """Get total size of all content in bytes."""
        return sum(content.get_size() for content in self.contents)
    
    def add_content(self, content: MultimodalContent) -> None:
        """Add a new content piece to the torch."""
        self.contents.append(content)
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method to add text content."""
        content = MultimodalContent(
            content_type=ContentType.TEXT,
            data=text,
            metadata=metadata or {},
            mime_type="text/plain"
        )
        self.add_content(content)
    
    def add_image(self, image_data: Union[str, bytes], mime_type: str = "image/jpeg", 
                  asset_hash: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method to add image content."""
        content = MultimodalContent(
            content_type=ContentType.IMAGE,
            data=image_data,
            metadata=metadata or {},
            asset_hash=asset_hash,
            mime_type=mime_type,
            encoding="base64" if isinstance(image_data, bytes) else None
        )
        self.add_content(content)
    
    def add_audio(self, audio_data: Union[str, bytes], mime_type: str = "audio/mp3",
                  asset_hash: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method to add audio content."""
        content = MultimodalContent(
            content_type=ContentType.AUDIO,
            data=audio_data,
            metadata=metadata or {},
            asset_hash=asset_hash,
            mime_type=mime_type,
            encoding="base64" if isinstance(audio_data, bytes) else None
        )
        self.add_content(content)
    
    def to_mcp_message(self) -> Dict[str, Any]:
        """
        Convert multimodal torch to MCP message format.
        
        Returns:
            Dict containing the torch data in MCP format
        """
        return {
            "type": "multimodal_torch",
            "id": self.torch_id,
            "channel": self.channel,
            "source": self.source_campfire,
            "timestamp": self.timestamp,
            "payload": {
                "primary_claim": self.primary_claim,
                "confidence": self.confidence,
                "metadata": self.metadata,
                "contents": [
                    {
                        "content_type": content.content_type.value,
                        "data": content.get_data_as_string(),
                        "metadata": content.metadata,
                        "asset_hash": content.asset_hash,
                        "mime_type": content.mime_type,
                        "encoding": content.encoding
                    }
                    for content in self.contents
                ]
            }
        }
    
    @classmethod
    def from_mcp_message(cls, message: Dict[str, Any]) -> "MultimodalTorch":
        """
        Create a MultimodalTorch from an MCP message.
        
        Args:
            message: MCP message dictionary
            
        Returns:
            MultimodalTorch instance
        """
        payload = message.get("payload", {})
        
        contents = []
        for content_data in payload.get("contents", []):
            content = MultimodalContent(
                content_type=ContentType(content_data.get("content_type", "text")),
                data=content_data.get("data", ""),
                metadata=content_data.get("metadata", {}),
                asset_hash=content_data.get("asset_hash"),
                mime_type=content_data.get("mime_type"),
                encoding=content_data.get("encoding")
            )
            contents.append(content)
        
        return cls(
            contents=contents,
            primary_claim=payload.get("primary_claim", ""),
            confidence=payload.get("confidence", 1.0),
            metadata=payload.get("metadata", {}),
            timestamp=message.get("timestamp", time.time()),
            source_campfire=message.get("source", ""),
            channel=message.get("channel", ""),
            torch_id=message.get("id")
        )
    
    def to_legacy_torch(self) -> Torch:
        """
        Convert to legacy Torch format for backward compatibility.
        
        Returns:
            Torch instance with text content and asset references
        """
        # Combine all text content
        text_parts = [content.data for content in self.get_text_contents() if isinstance(content.data, str)]
        combined_text = self.primary_claim
        if text_parts:
            combined_text += "\n" + "\n".join(text_parts)
        
        # Use first asset hash if available
        asset_path = None
        for content in self.contents:
            if content.asset_hash:
                asset_path = f"./party_box/{content.asset_hash}"
                break
        
        # Combine metadata
        combined_metadata = dict(self.metadata)
        combined_metadata["multimodal"] = True
        combined_metadata["content_types"] = [ct.value for ct in self.get_content_types()]
        combined_metadata["total_contents"] = len(self.contents)
        
        return Torch(
            claim=combined_text,
            path=asset_path,
            confidence=self.confidence,
            metadata=combined_metadata,
            timestamp=self.timestamp,
            source_campfire=self.source_campfire,
            channel=self.channel,
            torch_id=self.torch_id
        )
    
    @classmethod
    def from_legacy_torch(cls, torch: Torch, content_type: ContentType = ContentType.TEXT) -> "MultimodalTorch":
        """
        Create MultimodalTorch from legacy Torch.
        
        Args:
            torch: Legacy Torch instance
            content_type: Content type to assign to the torch claim
            
        Returns:
            MultimodalTorch instance
        """
        contents = [
            MultimodalContent(
                content_type=content_type,
                data=torch.claim,
                metadata={},
                mime_type="text/plain" if content_type == ContentType.TEXT else None
            )
        ]
        
        return cls(
            contents=contents,
            primary_claim=torch.claim,
            confidence=torch.confidence,
            metadata=torch.metadata,
            timestamp=torch.timestamp,
            source_campfire=torch.source_campfire,
            channel=torch.channel,
            torch_id=torch.torch_id
        )
    
    def __str__(self) -> str:
        content_summary = ", ".join([f"{ct.value}({len([c for c in self.contents if c.content_type == ct])})" 
                                   for ct in self.get_content_types()])
        return f"MultimodalTorch(claim='{self.primary_claim[:50]}...', contents=[{content_summary}])"
    
    def __repr__(self) -> str:
        return (f"MultimodalTorch(primary_claim='{self.primary_claim}', "
                f"contents={len(self.contents)}, confidence={self.confidence}, "
                f"source='{self.source_campfire}')")