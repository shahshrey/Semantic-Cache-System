"""
Semantic Cache System - A semantic cache for AI responses.

This package provides a semantic cache implementation for AI responses,
using embeddings to match semantically similar queries.
"""

__version__ = "0.1.0"

# Import key classes and functions for easier access
from semantic_cache_system.core.semantic_cache import SemanticCache
from semantic_cache_system.services.boardy import get_boardy_response, get_boardy_response_simple
