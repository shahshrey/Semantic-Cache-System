"""
Semantic cache implementation using ChromaDB and OpenAI embeddings.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import time
from typing import Dict, Any, Optional, Tuple

from semantic_cache_system.config.constants import (
    CHROMA_PERSIST_DIRECTORY, 
    COLLECTION_NAME, 
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_CACHE_SIZE, 
    DEFAULT_TTL,
    COLORS
)
from semantic_cache_system.utils.logger import logger

class SemanticCache:
    """
    A class-based implementation of a semantic cache using ChromaDB and OpenAI embeddings.
    Provides methods for adding to the cache, retrieving from the cache, and measuring semantic similarity.
    Includes cache size limits and time-based eviction policies.
    """
    
    def __init__(self, 
                 persist_directory: str = CHROMA_PERSIST_DIRECTORY,
                 collection_name: str = COLLECTION_NAME,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 max_cache_size: int = DEFAULT_CACHE_SIZE,
                 ttl: int = DEFAULT_TTL,
                 verbose: bool = False):
        """
        Initialize the semantic cache with ChromaDB backend.
        
        Args:
            persist_directory: Directory to persist the ChromaDB data
            collection_name: Name of the collection in ChromaDB
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
            max_cache_size: Maximum number of items to store in the cache
            ttl: Time-to-live for cache entries in seconds
            verbose: Whether to print detailed logs
        """
        try:
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.similarity_threshold = similarity_threshold
            self.max_cache_size = max_cache_size
            self.ttl = ttl
            self.verbose = verbose
            
            # Statistics tracking
            self.stats = {
                "hits": 0,
                "misses": 0,
                "adds": 0,
                "evictions": 0
            }
            
            # Set up OpenAI embeddings
            self.embeddings = OpenAIEmbeddings()
            
            # Initialize ChromaDB vector store
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            if self.verbose:
                print(f"{COLORS['CYAN']}Initialized SemanticCache with collection '{collection_name}'{COLORS['RESET']}")
        except Exception as e:
            logger.exception(f"Error initializing SemanticCache: {e}")
            raise
    
    def add_to_cache(self, query: str, response: str) -> bool:
        """
        Add a query-response pair to the cache.
        
        Args:
            query: The query string
            response: The response to cache
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            # Create metadata with timestamp for TTL
            metadata = {
                "timestamp": time.time(),
                "query": query
            }
            
            # Add to ChromaDB
            self.vector_store.add_texts(
                texts=[query],
                metadatas=[metadata],
                ids=[f"q_{int(time.time())}_{hash(query) % 10000}"],
                embeddings=None  # Let the embedding function handle this
            )
            
            # Store the response in the same document
            # In a real implementation, you might want to store large responses separately
            self.vector_store.add_texts(
                texts=[response],
                metadatas=[{**metadata, "is_response": True}],
                ids=[f"r_{int(time.time())}_{hash(query) % 10000}"],
                embeddings=None
            )
            
            # Check if we need to evict old entries
            self._enforce_cache_limits()
            
            # Update stats
            self.stats["adds"] += 1
            
            if self.verbose:
                print(f"{COLORS['BLUE']}Added to cache: {query[:50]}...{COLORS['RESET']}")
            return True
        except Exception as e:
            logger.exception(f"Error adding to cache: {e}")
            return False
    
    def get_from_cache(self, query: str, k: int = 3) -> Tuple[Optional[str], float, Optional[float]]:
        """
        Retrieve a response from the cache based on semantic similarity.
        
        Args:
            query: The query to look up
            k: Number of similar results to consider
            
        Returns:
            Tuple[Optional[str], float, Optional[float]]: The cached response if found, similarity score, and timestamp (None if not found)
        """
        try:
            # Perform similarity search in ChromaDB
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter results by similarity threshold and TTL
            current_time = time.time()
            valid_results = []
            
            for doc, score in results:
                # Convert score to similarity (ChromaDB returns distance)
                similarity = 1.0 - score
                
                # Check if the result is a query (not a response)
                if doc.metadata.get("is_response", False):
                    continue
                    
                # Check similarity threshold
                if similarity < self.similarity_threshold:
                    continue
                    
                # Check TTL
                timestamp = doc.metadata.get("timestamp", 0)
                if current_time - timestamp > self.ttl:
                    continue
                    
                valid_results.append((doc, similarity, timestamp))
            
            if not valid_results:
                # Update stats
                self.stats["misses"] += 1
                
                if self.verbose:
                    print(f"{COLORS['YELLOW']}Cache miss for: {query[:50]}...{COLORS['RESET']}")
                return None, 0.0, None
                
            # Get the most similar result
            best_match, similarity, timestamp = valid_results[0]
            original_query = best_match.metadata.get("query", "")
            
            # Find the corresponding response
            response_results = self.vector_store.similarity_search(
                query=original_query,
                k=5,
                filter={"is_response": True}
            )
            
            if not response_results:
                # Update stats
                self.stats["misses"] += 1
                
                if self.verbose:
                    print(f"{COLORS['YELLOW']}Found query but no response for: {original_query[:50]}...{COLORS['RESET']}")
                return None, 0.0, None
                
            response = response_results[0].page_content
            
            # Update stats
            self.stats["hits"] += 1
            
            if self.verbose:
                print(f"{COLORS['GREEN']}Cache hit for: {query[:50]}... (similarity: {similarity:.4f}){COLORS['RESET']}")
            return response, similarity, timestamp
        except Exception as e:
            logger.exception(f"Error retrieving from cache: {e}")
            return None, 0.0, None
    
    def _enforce_cache_limits(self) -> None:
        """
        Enforce cache size limits and evict old entries if necessary.
        """
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            if not all_docs or "ids" not in all_docs:
                return
            
            # Filter to only include query documents (not responses)
            query_docs = []
            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i]
                # Skip response documents
                if metadata.get("is_response", False):
                    continue
                timestamp = metadata.get("timestamp", 0)
                query = metadata.get("query", "")
                query_docs.append((doc_id, timestamp, query))
                
            # If we're under the limit, no need to evict
            if len(query_docs) <= self.max_cache_size:
                return
                
            # Sort by timestamp (oldest first)
            query_docs.sort(key=lambda x: x[1])
            
            # Calculate how many to remove
            to_remove = len(query_docs) - self.max_cache_size
            
            if to_remove <= 0:
                return
                
            # Get queries to remove
            queries_to_remove = []
            ids_to_remove = []
            
            for i in range(to_remove):
                doc_id, _, query = query_docs[i]
                ids_to_remove.append(doc_id)
                queries_to_remove.append(query)
            
            # Find and add corresponding response documents to remove
            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i]
                if metadata.get("is_response", False):
                    query = metadata.get("query", "")
                    if query in queries_to_remove:
                        ids_to_remove.append(doc_id)
            
            # Remove from ChromaDB
            if ids_to_remove:
                self.vector_store.delete(ids=ids_to_remove)
                
                # Update stats
                self.stats["evictions"] += to_remove
                
                if self.verbose:
                    print(f"{COLORS['MAGENTA']}Evicted {to_remove} old entries from cache{COLORS['RESET']}")
        except Exception as e:
            logger.exception(f"Error enforcing cache limits: {e}")
    
    def clear_cache(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            bool: True if successfully cleared, False otherwise
        """
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            if not all_docs or "ids" not in all_docs or not all_docs["ids"]:
                if self.verbose:
                    print(f"{COLORS['MAGENTA']}Cache already empty{COLORS['RESET']}")
                return True
                
            # Remove all documents
            self.vector_store.delete(ids=all_docs["ids"])
            
            if self.verbose:
                print(f"{COLORS['MAGENTA']}Cleared cache ({len(all_docs['ids'])} entries){COLORS['RESET']}")
            return True
        except Exception as e:
            logger.exception(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict[str, Any]: Statistics about the cache
        """
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            if not all_docs or "ids" not in all_docs:
                return {**self.stats, "size": 0, "oldest": None, "newest": None}
                
            # Calculate stats
            timestamps = []
            for metadata in all_docs["metadatas"]:
                timestamp = metadata.get("timestamp", 0)
                timestamps.append(timestamp)
                
            if not timestamps:
                return {**self.stats, "size": 0, "oldest": None, "newest": None}
                
            return {
                **self.stats,
                "size": len(all_docs["ids"]),
                "oldest": min(timestamps),
                "newest": max(timestamps)
            }
        except Exception as e:
            logger.exception(f"Error getting cache stats: {e}")
            return {**self.stats, "size": 0, "oldest": None, "newest": None, "error": str(e)} 