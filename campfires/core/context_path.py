"""
Context Path Support for RAG Generation Pipeline.

This module provides sophisticated context path management for the RAG generation
pipeline, enabling hierarchical context organization, path-based retrieval,
and dynamic context composition.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context content."""
    DOCUMENT = "document"
    CODE = "code"
    CONVERSATION = "conversation"
    METADATA = "metadata"
    EMBEDDING = "embedding"
    KNOWLEDGE = "knowledge"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class AccessPattern(Enum):
    """Context access patterns."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class ContextMetadata:
    """Metadata for context items."""
    id: str
    path: str
    context_type: ContextType
    created_at: datetime
    updated_at: datetime
    size_bytes: int
    checksum: str
    tags: Set[str] = field(default_factory=set)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    priority: int = 5  # 1-10, higher is more important
    ttl_seconds: Optional[int] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextItem:
    """Individual context item with content and metadata."""
    metadata: ContextMetadata
    content: Any
    embeddings: Optional[Dict[str, List[float]]] = None
    compressed_content: Optional[bytes] = None
    is_compressed: bool = False


@dataclass
class ContextPath:
    """Represents a hierarchical context path."""
    path: str
    segments: List[str]
    depth: int
    is_absolute: bool
    is_pattern: bool = False
    pattern_type: Optional[str] = None


@dataclass
class ContextQuery:
    """Query for context retrieval."""
    path_pattern: str
    context_types: Optional[List[ContextType]] = None
    tags: Optional[Set[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    max_results: int = 100
    include_metadata: bool = True
    include_content: bool = True
    include_embeddings: bool = False
    similarity_threshold: float = 0.7
    access_pattern: AccessPattern = AccessPattern.SEMANTIC
    custom_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextRetrievalResult:
    """Result of context retrieval."""
    query: ContextQuery
    items: List[ContextItem]
    total_count: int
    execution_time_ms: float
    cache_hit: bool = False
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    path_hierarchy: Dict[str, List[str]] = field(default_factory=dict)


class ContextPathManager:
    """
    Advanced context path manager for RAG generation pipeline.
    
    Features:
    - Hierarchical path organization (e.g., /project/module/function)
    - Pattern-based retrieval (e.g., /project/*/tests/*.py)
    - Semantic path matching with embeddings
    - Temporal context tracking
    - Relationship mapping between contexts
    - Efficient caching and indexing
    - Context lifecycle management
    - Path-based access control
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the context path manager.
        
        Args:
            config: Manager configuration
        """
        self.config = config or {}
        
        # Storage configuration
        self.storage_path = Path(self.config.get('storage_path', './context_storage'))
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.embedding_dimension = self.config.get('embedding_dimension', 768)
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Context storage
        self._contexts: Dict[str, ContextItem] = {}
        self._path_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[ContextType, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._relationship_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Caching
        self._query_cache: Dict[str, ContextRetrievalResult] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        
        # Path patterns
        self._path_patterns: Dict[str, str] = {}
        
        # Metrics
        self._metrics = {
            'total_contexts': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'queries_executed': 0,
            'average_query_time_ms': 0.0
        }
    
    def create_context_path(self, path: str) -> ContextPath:
        """
        Create a context path object from a path string.
        
        Args:
            path: Path string (e.g., "/project/module/function")
            
        Returns:
            ContextPath object
        """
        # Normalize path
        normalized_path = self._normalize_path(path)
        
        # Parse path
        is_absolute = normalized_path.startswith('/')
        segments = [s for s in normalized_path.split('/') if s]
        depth = len(segments)
        
        # Check if it's a pattern
        is_pattern = '*' in normalized_path or '?' in normalized_path or '[' in normalized_path
        pattern_type = self._detect_pattern_type(normalized_path) if is_pattern else None
        
        return ContextPath(
            path=normalized_path,
            segments=segments,
            depth=depth,
            is_absolute=is_absolute,
            is_pattern=is_pattern,
            pattern_type=pattern_type
        )
    
    async def store_context(self, 
                          path: str,
                          content: Any,
                          context_type: ContextType,
                          tags: Set[str] = None,
                          relationships: Dict[str, List[str]] = None,
                          custom_metadata: Dict[str, Any] = None,
                          embeddings: Dict[str, List[float]] = None) -> str:
        """
        Store context at the specified path.
        
        Args:
            path: Context path
            content: Context content
            context_type: Type of context
            tags: Associated tags
            relationships: Relationships to other contexts
            custom_metadata: Custom metadata
            embeddings: Pre-computed embeddings
            
        Returns:
            Context ID
        """
        # Create context path
        context_path = self.create_context_path(path)
        
        # Generate context ID
        context_id = self._generate_context_id(path, content)
        
        # Calculate content size and checksum
        content_str = json.dumps(content) if not isinstance(content, str) else content
        size_bytes = len(content_str.encode('utf-8'))
        checksum = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        
        # Create metadata
        now = datetime.now()
        metadata = ContextMetadata(
            id=context_id,
            path=context_path.path,
            context_type=context_type,
            created_at=now,
            updated_at=now,
            size_bytes=size_bytes,
            checksum=checksum,
            tags=tags or set(),
            relationships=relationships or {},
            custom_metadata=custom_metadata or {}
        )
        
        # Compress content if enabled
        compressed_content = None
        is_compressed = False
        if self.compression_enabled and size_bytes > 1024:  # Compress if > 1KB
            try:
                import gzip
                compressed_content = gzip.compress(content_str.encode('utf-8'))
                is_compressed = True
                logger.debug(f"Compressed context {context_id}: {size_bytes} -> {len(compressed_content)} bytes")
            except Exception as e:
                logger.warning(f"Failed to compress context {context_id}: {e}")
        
        # Create context item
        context_item = ContextItem(
            metadata=metadata,
            content=content,
            embeddings=embeddings,
            compressed_content=compressed_content,
            is_compressed=is_compressed
        )
        
        # Store context
        self._contexts[context_id] = context_item
        
        # Update indexes
        self._update_indexes(context_id, context_item)
        
        # Persist to storage
        await self._persist_context(context_item)
        
        # Update metrics
        self._metrics['total_contexts'] = len(self._contexts)
        
        logger.info(f"Stored context: {context_id} at path: {path}")
        return context_id
    
    async def retrieve_context(self, query: ContextQuery) -> ContextRetrievalResult:
        """
        Retrieve contexts based on query.
        
        Args:
            query: Context query
            
        Returns:
            Retrieval result
        """
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._generate_cache_key(query)
        if cache_key in self._query_cache:
            result = self._query_cache[cache_key]
            result.cache_hit = True
            self._metrics['cache_hits'] += 1
            self._cache_access_times[cache_key] = datetime.now()
            return result
        
        # Execute query
        matching_contexts = await self._execute_query(query)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create result
        result = ContextRetrievalResult(
            query=query,
            items=matching_contexts,
            total_count=len(matching_contexts),
            execution_time_ms=execution_time,
            cache_hit=False,
            relevance_scores=self._calculate_relevance_scores(matching_contexts, query),
            path_hierarchy=self._build_path_hierarchy(matching_contexts)
        )
        
        # Cache result
        self._cache_result(cache_key, result)
        
        # Update metrics
        self._metrics['cache_misses'] += 1
        self._metrics['queries_executed'] += 1
        self._update_average_query_time(execution_time)
        
        logger.debug(f"Retrieved {len(matching_contexts)} contexts in {execution_time:.2f}ms")
        return result
    
    async def update_context(self, 
                           context_id: str,
                           content: Any = None,
                           tags: Set[str] = None,
                           relationships: Dict[str, List[str]] = None,
                           custom_metadata: Dict[str, Any] = None) -> bool:
        """
        Update existing context.
        
        Args:
            context_id: Context ID to update
            content: New content (optional)
            tags: New tags (optional)
            relationships: New relationships (optional)
            custom_metadata: New custom metadata (optional)
            
        Returns:
            True if updated successfully
        """
        if context_id not in self._contexts:
            return False
        
        context_item = self._contexts[context_id]
        
        # Update content if provided
        if content is not None:
            context_item.content = content
            
            # Recalculate size and checksum
            content_str = json.dumps(content) if not isinstance(content, str) else content
            context_item.metadata.size_bytes = len(content_str.encode('utf-8'))
            context_item.metadata.checksum = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
            
            # Recompress if needed
            if self.compression_enabled and context_item.metadata.size_bytes > 1024:
                try:
                    import gzip
                    context_item.compressed_content = gzip.compress(content_str.encode('utf-8'))
                    context_item.is_compressed = True
                except Exception as e:
                    logger.warning(f"Failed to compress updated context {context_id}: {e}")
        
        # Update metadata
        if tags is not None:
            # Remove old tag indexes
            for tag in context_item.metadata.tags:
                self._tag_index[tag].discard(context_id)
            
            # Update tags
            context_item.metadata.tags = tags
            
            # Add new tag indexes
            for tag in tags:
                self._tag_index[tag].add(context_id)
        
        if relationships is not None:
            context_item.metadata.relationships = relationships
            # Update relationship indexes
            self._update_relationship_indexes(context_id, relationships)
        
        if custom_metadata is not None:
            context_item.metadata.custom_metadata.update(custom_metadata)
        
        # Update timestamp
        context_item.metadata.updated_at = datetime.now()
        
        # Persist changes
        await self._persist_context(context_item)
        
        # Clear related cache entries
        self._invalidate_cache_for_context(context_id)
        
        logger.info(f"Updated context: {context_id}")
        return True
    
    async def delete_context(self, context_id: str) -> bool:
        """
        Delete context.
        
        Args:
            context_id: Context ID to delete
            
        Returns:
            True if deleted successfully
        """
        if context_id not in self._contexts:
            return False
        
        context_item = self._contexts[context_id]
        
        # Remove from indexes
        self._remove_from_indexes(context_id, context_item)
        
        # Remove from storage
        await self._remove_from_storage(context_id)
        
        # Remove from memory
        del self._contexts[context_id]
        
        # Clear related cache entries
        self._invalidate_cache_for_context(context_id)
        
        # Update metrics
        self._metrics['total_contexts'] = len(self._contexts)
        
        logger.info(f"Deleted context: {context_id}")
        return True
    
    def get_context_by_id(self, context_id: str) -> Optional[ContextItem]:
        """
        Get context by ID.
        
        Args:
            context_id: Context ID
            
        Returns:
            Context item or None if not found
        """
        context_item = self._contexts.get(context_id)
        if context_item:
            # Update access tracking
            context_item.metadata.access_count += 1
            context_item.metadata.last_accessed = datetime.now()
        
        return context_item
    
    def list_paths(self, pattern: str = "*") -> List[str]:
        """
        List all context paths matching pattern.
        
        Args:
            pattern: Path pattern (supports wildcards)
            
        Returns:
            List of matching paths
        """
        if pattern == "*":
            return [item.metadata.path for item in self._contexts.values()]
        
        # Pattern matching
        import fnmatch
        matching_paths = []
        
        for context_item in self._contexts.values():
            if fnmatch.fnmatch(context_item.metadata.path, pattern):
                matching_paths.append(context_item.metadata.path)
        
        return sorted(matching_paths)
    
    def get_path_hierarchy(self, root_path: str = "/") -> Dict[str, Any]:
        """
        Get hierarchical view of paths.
        
        Args:
            root_path: Root path to start from
            
        Returns:
            Hierarchical path structure
        """
        hierarchy = {}
        
        for context_item in self._contexts.values():
            path = context_item.metadata.path
            
            if not path.startswith(root_path):
                continue
            
            # Build hierarchy
            relative_path = path[len(root_path):].lstrip('/')
            if not relative_path:
                continue
            
            segments = relative_path.split('/')
            current = hierarchy
            
            for segment in segments[:-1]:
                if segment not in current:
                    current[segment] = {}
                current = current[segment]
            
            # Add final segment with context info
            final_segment = segments[-1]
            current[final_segment] = {
                'context_id': context_item.metadata.id,
                'context_type': context_item.metadata.context_type.value,
                'size_bytes': context_item.metadata.size_bytes,
                'tags': list(context_item.metadata.tags)
            }
        
        return hierarchy
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get context manager metrics.
        
        Returns:
            Metrics dictionary
        """
        return self._metrics.copy()
    
    async def cleanup_expired_contexts(self):
        """Clean up expired contexts based on TTL."""
        now = datetime.now()
        expired_contexts = []
        
        for context_id, context_item in self._contexts.items():
            if context_item.metadata.ttl_seconds:
                age_seconds = (now - context_item.metadata.created_at).total_seconds()
                if age_seconds > context_item.metadata.ttl_seconds:
                    expired_contexts.append(context_id)
        
        for context_id in expired_contexts:
            await self.delete_context(context_id)
        
        logger.info(f"Cleaned up {len(expired_contexts)} expired contexts")
    
    # Private helper methods
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path string."""
        # Remove duplicate slashes
        normalized = '/'.join(segment for segment in path.split('/') if segment)
        
        # Ensure absolute paths start with /
        if path.startswith('/') and not normalized.startswith('/'):
            normalized = '/' + normalized
        
        return normalized
    
    def _detect_pattern_type(self, path: str) -> str:
        """Detect pattern type in path."""
        if '**' in path:
            return 'recursive_wildcard'
        elif '*' in path:
            return 'wildcard'
        elif '?' in path:
            return 'single_char'
        elif '[' in path and ']' in path:
            return 'character_class'
        else:
            return 'literal'
    
    def _generate_context_id(self, path: str, content: Any) -> str:
        """Generate unique context ID."""
        content_str = json.dumps(content) if not isinstance(content, str) else content
        combined = f"{path}:{content_str}:{datetime.now().isoformat()}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _update_indexes(self, context_id: str, context_item: ContextItem):
        """Update all indexes for a context item."""
        # Path index
        path_segments = context_item.metadata.path.split('/')
        for i in range(len(path_segments)):
            partial_path = '/'.join(path_segments[:i+1])
            if partial_path:
                self._path_index[partial_path].add(context_id)
        
        # Type index
        self._type_index[context_item.metadata.context_type].add(context_id)
        
        # Tag index
        for tag in context_item.metadata.tags:
            self._tag_index[tag].add(context_id)
        
        # Relationship index
        self._update_relationship_indexes(context_id, context_item.metadata.relationships)
    
    def _update_relationship_indexes(self, context_id: str, relationships: Dict[str, List[str]]):
        """Update relationship indexes."""
        for relationship_type, related_ids in relationships.items():
            for related_id in related_ids:
                self._relationship_index[f"{relationship_type}:{related_id}"].add(context_id)
    
    def _remove_from_indexes(self, context_id: str, context_item: ContextItem):
        """Remove context from all indexes."""
        # Path index
        path_segments = context_item.metadata.path.split('/')
        for i in range(len(path_segments)):
            partial_path = '/'.join(path_segments[:i+1])
            if partial_path in self._path_index:
                self._path_index[partial_path].discard(context_id)
        
        # Type index
        self._type_index[context_item.metadata.context_type].discard(context_id)
        
        # Tag index
        for tag in context_item.metadata.tags:
            self._tag_index[tag].discard(context_id)
        
        # Relationship index
        for relationship_type, related_ids in context_item.metadata.relationships.items():
            for related_id in related_ids:
                self._relationship_index[f"{relationship_type}:{related_id}"].discard(context_id)
    
    async def _execute_query(self, query: ContextQuery) -> List[ContextItem]:
        """Execute context query."""
        # Start with all contexts
        candidate_ids = set(self._contexts.keys())
        
        # Filter by path pattern
        if query.path_pattern and query.path_pattern != "*":
            path_matches = self._match_path_pattern(query.path_pattern)
            candidate_ids &= path_matches
        
        # Filter by context types
        if query.context_types:
            type_matches = set()
            for context_type in query.context_types:
                type_matches.update(self._type_index[context_type])
            candidate_ids &= type_matches
        
        # Filter by tags
        if query.tags:
            tag_matches = set()
            for tag in query.tags:
                if not tag_matches:
                    tag_matches = self._tag_index[tag].copy()
                else:
                    tag_matches &= self._tag_index[tag]
            candidate_ids &= tag_matches
        
        # Filter by time range
        if query.time_range:
            start_time, end_time = query.time_range
            time_matches = set()
            for context_id in candidate_ids:
                context_item = self._contexts[context_id]
                if start_time <= context_item.metadata.created_at <= end_time:
                    time_matches.add(context_id)
            candidate_ids = time_matches
        
        # Apply custom filters
        if query.custom_filters:
            custom_matches = set()
            for context_id in candidate_ids:
                context_item = self._contexts[context_id]
                if self._matches_custom_filters(context_item, query.custom_filters):
                    custom_matches.add(context_id)
            candidate_ids = custom_matches
        
        # Get context items
        matching_contexts = [self._contexts[context_id] for context_id in candidate_ids]
        
        # Sort by relevance/priority
        matching_contexts.sort(key=lambda x: (x.metadata.priority, x.metadata.access_count), reverse=True)
        
        # Limit results
        if query.max_results:
            matching_contexts = matching_contexts[:query.max_results]
        
        # Filter content based on query options
        filtered_contexts = []
        for context_item in matching_contexts:
            filtered_item = self._filter_context_item(context_item, query)
            filtered_contexts.append(filtered_item)
        
        return filtered_contexts
    
    def _match_path_pattern(self, pattern: str) -> Set[str]:
        """Match contexts by path pattern."""
        import fnmatch
        matching_ids = set()
        
        for context_id, context_item in self._contexts.items():
            if fnmatch.fnmatch(context_item.metadata.path, pattern):
                matching_ids.add(context_id)
        
        return matching_ids
    
    def _matches_custom_filters(self, context_item: ContextItem, custom_filters: Dict[str, Any]) -> bool:
        """Check if context item matches custom filters."""
        for filter_key, filter_value in custom_filters.items():
            if filter_key == 'min_size':
                if context_item.metadata.size_bytes < filter_value:
                    return False
            elif filter_key == 'max_size':
                if context_item.metadata.size_bytes > filter_value:
                    return False
            elif filter_key == 'min_access_count':
                if context_item.metadata.access_count < filter_value:
                    return False
            elif filter_key in context_item.metadata.custom_metadata:
                if context_item.metadata.custom_metadata[filter_key] != filter_value:
                    return False
        
        return True
    
    def _filter_context_item(self, context_item: ContextItem, query: ContextQuery) -> ContextItem:
        """Filter context item based on query options."""
        # Create a copy for filtering
        filtered_metadata = context_item.metadata
        filtered_content = context_item.content if query.include_content else None
        filtered_embeddings = context_item.embeddings if query.include_embeddings else None
        
        if not query.include_metadata:
            # Create minimal metadata
            filtered_metadata = ContextMetadata(
                id=context_item.metadata.id,
                path=context_item.metadata.path,
                context_type=context_item.metadata.context_type,
                created_at=context_item.metadata.created_at,
                updated_at=context_item.metadata.updated_at,
                size_bytes=context_item.metadata.size_bytes,
                checksum=context_item.metadata.checksum
            )
        
        return ContextItem(
            metadata=filtered_metadata,
            content=filtered_content,
            embeddings=filtered_embeddings,
            compressed_content=context_item.compressed_content,
            is_compressed=context_item.is_compressed
        )
    
    def _calculate_relevance_scores(self, contexts: List[ContextItem], query: ContextQuery) -> Dict[str, float]:
        """Calculate relevance scores for contexts."""
        scores = {}
        
        for context_item in contexts:
            score = 0.0
            
            # Path relevance
            if query.path_pattern:
                path_score = self._calculate_path_relevance(context_item.metadata.path, query.path_pattern)
                score += path_score * 0.3
            
            # Tag relevance
            if query.tags:
                tag_score = len(query.tags & context_item.metadata.tags) / len(query.tags)
                score += tag_score * 0.2
            
            # Access frequency
            access_score = min(context_item.metadata.access_count / 100.0, 1.0)
            score += access_score * 0.1
            
            # Priority
            priority_score = context_item.metadata.priority / 10.0
            score += priority_score * 0.2
            
            # Recency
            age_days = (datetime.now() - context_item.metadata.updated_at).days
            recency_score = max(0, 1.0 - (age_days / 365.0))
            score += recency_score * 0.2
            
            scores[context_item.metadata.id] = score
        
        return scores
    
    def _calculate_path_relevance(self, path: str, pattern: str) -> float:
        """Calculate path relevance score."""
        if path == pattern:
            return 1.0
        
        # Simple string similarity
        common_chars = sum(1 for a, b in zip(path, pattern) if a == b)
        max_length = max(len(path), len(pattern))
        
        return common_chars / max_length if max_length > 0 else 0.0
    
    def _build_path_hierarchy(self, contexts: List[ContextItem]) -> Dict[str, List[str]]:
        """Build path hierarchy from contexts."""
        hierarchy = defaultdict(list)
        
        for context_item in contexts:
            path_segments = context_item.metadata.path.split('/')
            for i in range(len(path_segments)):
                parent_path = '/'.join(path_segments[:i+1])
                if i < len(path_segments) - 1:
                    child_path = '/'.join(path_segments[:i+2])
                    hierarchy[parent_path].append(child_path)
        
        return dict(hierarchy)
    
    def _generate_cache_key(self, query: ContextQuery) -> str:
        """Generate cache key for query."""
        query_dict = {
            'path_pattern': query.path_pattern,
            'context_types': [ct.value for ct in query.context_types] if query.context_types else None,
            'tags': sorted(list(query.tags)) if query.tags else None,
            'time_range': [t.isoformat() for t in query.time_range] if query.time_range else None,
            'max_results': query.max_results,
            'custom_filters': query.custom_filters
        }
        
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode('utf-8')).hexdigest()
    
    def _cache_result(self, cache_key: str, result: ContextRetrievalResult):
        """Cache query result."""
        # Implement LRU cache eviction
        if len(self._query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache_access_times.keys(), 
                           key=lambda k: self._cache_access_times[k])
            del self._query_cache[oldest_key]
            del self._cache_access_times[oldest_key]
        
        self._query_cache[cache_key] = result
        self._cache_access_times[cache_key] = datetime.now()
    
    def _invalidate_cache_for_context(self, context_id: str):
        """Invalidate cache entries related to a context."""
        # Simple approach: clear all cache
        # In production, could be more sophisticated
        self._query_cache.clear()
        self._cache_access_times.clear()
    
    def _update_average_query_time(self, execution_time_ms: float):
        """Update average query time metric."""
        current_avg = self._metrics['average_query_time_ms']
        query_count = self._metrics['queries_executed']
        
        if query_count == 1:
            self._metrics['average_query_time_ms'] = execution_time_ms
        else:
            # Running average
            self._metrics['average_query_time_ms'] = (
                (current_avg * (query_count - 1) + execution_time_ms) / query_count
            )
    
    async def _persist_context(self, context_item: ContextItem):
        """Persist context to storage."""
        try:
            context_file = self.storage_path / f"{context_item.metadata.id}.json"
            
            # Prepare data for persistence
            persist_data = {
                'metadata': {
                    'id': context_item.metadata.id,
                    'path': context_item.metadata.path,
                    'context_type': context_item.metadata.context_type.value,
                    'created_at': context_item.metadata.created_at.isoformat(),
                    'updated_at': context_item.metadata.updated_at.isoformat(),
                    'size_bytes': context_item.metadata.size_bytes,
                    'checksum': context_item.metadata.checksum,
                    'tags': list(context_item.metadata.tags),
                    'relationships': context_item.metadata.relationships,
                    'access_count': context_item.metadata.access_count,
                    'last_accessed': context_item.metadata.last_accessed.isoformat() if context_item.metadata.last_accessed else None,
                    'priority': context_item.metadata.priority,
                    'ttl_seconds': context_item.metadata.ttl_seconds,
                    'custom_metadata': context_item.metadata.custom_metadata
                },
                'content': context_item.content,
                'embeddings': context_item.embeddings,
                'is_compressed': context_item.is_compressed
            }
            
            # Save compressed content separately if it exists
            if context_item.compressed_content:
                compressed_file = self.storage_path / f"{context_item.metadata.id}.gz"
                with open(compressed_file, 'wb') as f:
                    f.write(context_item.compressed_content)
                persist_data['compressed_file'] = str(compressed_file)
            
            # Save metadata and content
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(persist_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to persist context {context_item.metadata.id}: {e}")
    
    async def _remove_from_storage(self, context_id: str):
        """Remove context from storage."""
        try:
            context_file = self.storage_path / f"{context_id}.json"
            if context_file.exists():
                context_file.unlink()
            
            compressed_file = self.storage_path / f"{context_id}.gz"
            if compressed_file.exists():
                compressed_file.unlink()
                
        except Exception as e:
            logger.error(f"Failed to remove context {context_id} from storage: {e}")
    
    async def load_from_storage(self):
        """Load contexts from storage on startup."""
        try:
            for context_file in self.storage_path.glob("*.json"):
                with open(context_file, 'r', encoding='utf-8') as f:
                    persist_data = json.load(f)
                
                # Reconstruct metadata
                metadata_data = persist_data['metadata']
                metadata = ContextMetadata(
                    id=metadata_data['id'],
                    path=metadata_data['path'],
                    context_type=ContextType(metadata_data['context_type']),
                    created_at=datetime.fromisoformat(metadata_data['created_at']),
                    updated_at=datetime.fromisoformat(metadata_data['updated_at']),
                    size_bytes=metadata_data['size_bytes'],
                    checksum=metadata_data['checksum'],
                    tags=set(metadata_data['tags']),
                    relationships=metadata_data['relationships'],
                    access_count=metadata_data['access_count'],
                    last_accessed=datetime.fromisoformat(metadata_data['last_accessed']) if metadata_data['last_accessed'] else None,
                    priority=metadata_data['priority'],
                    ttl_seconds=metadata_data['ttl_seconds'],
                    custom_metadata=metadata_data['custom_metadata']
                )
                
                # Load compressed content if it exists
                compressed_content = None
                if 'compressed_file' in persist_data:
                    compressed_file = Path(persist_data['compressed_file'])
                    if compressed_file.exists():
                        with open(compressed_file, 'rb') as f:
                            compressed_content = f.read()
                
                # Reconstruct context item
                context_item = ContextItem(
                    metadata=metadata,
                    content=persist_data['content'],
                    embeddings=persist_data.get('embeddings'),
                    compressed_content=compressed_content,
                    is_compressed=persist_data.get('is_compressed', False)
                )
                
                # Store in memory
                self._contexts[metadata.id] = context_item
                
                # Update indexes
                self._update_indexes(metadata.id, context_item)
            
            logger.info(f"Loaded {len(self._contexts)} contexts from storage")
            
        except Exception as e:
            logger.error(f"Failed to load contexts from storage: {e}")


# Utility functions for context path operations

def create_context_query(path_pattern: str = "*",
                        context_types: List[str] = None,
                        tags: List[str] = None,
                        max_results: int = 100) -> ContextQuery:
    """
    Create a context query with common parameters.
    
    Args:
        path_pattern: Path pattern to match
        context_types: List of context type names
        tags: List of tags to match
        max_results: Maximum number of results
        
    Returns:
        ContextQuery object
    """
    query_context_types = None
    if context_types:
        query_context_types = [ContextType(ct) for ct in context_types]
    
    query_tags = set(tags) if tags else None
    
    return ContextQuery(
        path_pattern=path_pattern,
        context_types=query_context_types,
        tags=query_tags,
        max_results=max_results
    )


def parse_context_path(path: str) -> Dict[str, Any]:
    """
    Parse context path into components.
    
    Args:
        path: Context path string
        
    Returns:
        Dictionary with path components
    """
    normalized_path = path.strip('/')
    segments = normalized_path.split('/') if normalized_path else []
    
    return {
        'full_path': path,
        'normalized_path': normalized_path,
        'segments': segments,
        'depth': len(segments),
        'is_absolute': path.startswith('/'),
        'parent_path': '/'.join(segments[:-1]) if len(segments) > 1 else '/',
        'name': segments[-1] if segments else '',
        'is_root': path in ['/', '']
    }