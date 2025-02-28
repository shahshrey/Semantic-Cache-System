from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import sys

# Set up logging
# Create a custom handler that writes INFO and below to stdout, WARNING and above to stderr
class CustomHandler(logging.Handler):
    def emit(self, record):
        pass  # We'll handle logging differently for cleaner test output

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(CustomHandler())

# Major constants
CHROMA_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "semantic_cache"
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimensions
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_CACHE_SIZE = 1000
DEFAULT_TTL = 86400  # 24 hours in seconds

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "BG_RED": "\033[41m",
    "BG_GREEN": "\033[42m",
    "BG_YELLOW": "\033[43m",
    "BG_BLUE": "\033[44m"
}

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
    
    def get_from_cache(self, query: str, k: int = 3) -> Tuple[Optional[str], float]:
        """
        Retrieve a response from the cache based on semantic similarity.
        
        Args:
            query: The query to look up
            k: Number of similar results to consider
            
        Returns:
            Tuple[Optional[str], float]: The cached response if found and similarity score, (None, 0.0) otherwise
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
                    
                valid_results.append((doc, similarity))
            
            if not valid_results:
                # Update stats
                self.stats["misses"] += 1
                
                if self.verbose:
                    print(f"{COLORS['YELLOW']}Cache miss for: {query[:50]}...{COLORS['RESET']}")
                return None, 0.0
                
            # Get the most similar result
            best_match, similarity = valid_results[0]
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
                return None, 0.0
                
            response = response_results[0].page_content
            
            # Update stats
            self.stats["hits"] += 1
            
            if self.verbose:
                print(f"{COLORS['GREEN']}Cache hit for: {query[:50]}... (similarity: {similarity:.4f}){COLORS['RESET']}")
            return response, similarity
        except Exception as e:
            logger.exception(f"Error retrieving from cache: {e}")
            return None, 0.0
    
    def _enforce_cache_limits(self) -> None:
        """
        Enforce cache size limits and evict old entries if necessary.
        """
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            if not all_docs or "ids" not in all_docs:
                return
                
            # If we're under the limit, no need to evict
            if len(all_docs["ids"]) <= self.max_cache_size:
                return
                
            # Get timestamps for all documents
            timestamps = []
            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i]
                timestamp = metadata.get("timestamp", 0)
                timestamps.append((doc_id, timestamp))
                
            # Sort by timestamp (oldest first)
            timestamps.sort(key=lambda x: x[1])
            
            # Calculate how many to remove
            to_remove = len(timestamps) - self.max_cache_size
            
            if to_remove <= 0:
                return
                
            # Get IDs to remove
            ids_to_remove = [item[0] for item in timestamps[:to_remove]]
            
            # Remove from ChromaDB
            self.vector_store.delete(ids=ids_to_remove)
            
            # Update stats
            self.stats["evictions"] += to_remove
            
            if self.verbose:
                print(f"{COLORS['MAGENTA']}Evicted {len(ids_to_remove)} old entries from cache{COLORS['RESET']}")
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


def mock_llm_call(query: str) -> str:
    """
    This function simulates an LLM call.
    In a real implementation, this would call an actual LLM API.
    
    Args:
        query: The query to send to the LLM
        
    Returns:
        str: The LLM response
    """
    # Simulate some processing time
    time.sleep(0.5)
    return f"Mock LLM response for: {query}"


def get_boardy_response(query: str, semantic_cache: SemanticCache) -> Tuple[str, bool, float]:
    """
    Get a response from Boardy, using the semantic cache when possible.
    
    Args:
        query: The user's query
        semantic_cache: The semantic cache instance
        
    Returns:
        Tuple[str, bool, float]: The response, a boolean indicating if it was from cache, and similarity score
    """
    try:
        # Check the semantic cache for similar queries
        cached_response, similarity = semantic_cache.get_from_cache(query)
        
        if cached_response:
            # Cache hit: return cached response
            return cached_response, True, similarity
        
        # Cache miss: call the LLM
        llm_response = mock_llm_call(query)
        
        # Cache the new response
        semantic_cache.add_to_cache(query, llm_response)
        
        return llm_response, False, 0.0
    except Exception as e:
        logger.exception(f"Error getting Boardy response: {e}")
        return f"Error: {str(e)}", False, 0.0


def print_header(text: str) -> None:
    """Print a formatted header."""
    width = 80
    print("\n" + "=" * width)
    print(f"{COLORS['BOLD']}{COLORS['WHITE']}{text.center(width)}{COLORS['RESET']}")
    print("=" * width)


def print_test_result(query: str, response: str, from_cache: bool, expected_cached: bool, similarity: float = 0.0) -> None:
    """Print a formatted test result."""
    status = "✓" if from_cache == expected_cached else "✗"
    status_color = COLORS['GREEN'] if from_cache == expected_cached else COLORS['RED']
    cache_status = f"{COLORS['GREEN']}CACHE HIT{COLORS['RESET']}" if from_cache else f"{COLORS['YELLOW']}CACHE MISS{COLORS['RESET']}"
    
    print(f"\n{status_color}{COLORS['BOLD']}{status}{COLORS['RESET']} Query: '{COLORS['CYAN']}{query}{COLORS['RESET']}'")
    print(f"  {cache_status}")
    
    if from_cache:
        print(f"  Similarity: {COLORS['MAGENTA']}{similarity:.4f}{COLORS['RESET']}")
    
    print(f"  Response: {COLORS['WHITE']}{response[:50]}...{COLORS['RESET']}")


def print_test_summary(test_name: str, results: List[bool]) -> None:
    """Print a summary of test results."""
    success = all(results)
    total = len(results)
    passed = sum(results)
    
    status_color = COLORS['GREEN'] if success else COLORS['RED']
    status = "PASSED" if success else "FAILED"
    
    print(f"\n{status_color}{COLORS['BOLD']}{status}{COLORS['RESET']} {test_name}: {passed}/{total} tests passed")


def print_cache_stats(cache: SemanticCache) -> None:
    """Print cache statistics."""
    stats = cache.get_cache_stats()
    
    print_header("CACHE STATISTICS")
    
    print(f"{COLORS['BOLD']}Cache Size:{COLORS['RESET']} {stats['size']} items")
    print(f"{COLORS['BOLD']}Cache Hits:{COLORS['RESET']} {COLORS['GREEN']}{stats['hits']}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}Cache Misses:{COLORS['RESET']} {COLORS['YELLOW']}{stats['misses']}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}Items Added:{COLORS['RESET']} {COLORS['BLUE']}{stats['adds']}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}Items Evicted:{COLORS['RESET']} {COLORS['MAGENTA']}{stats['evictions']}{COLORS['RESET']}")
    
    if stats['hits'] + stats['misses'] > 0:
        hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) * 100
        print(f"{COLORS['BOLD']}Hit Rate:{COLORS['RESET']} {COLORS['CYAN']}{hit_rate:.2f}%{COLORS['RESET']}")


if __name__ == "__main__":
    # Check if terminal supports colors
    if not sys.stdout.isatty():
        # Disable colors if not in a terminal
        for key in COLORS:
            COLORS[key] = ""
    
    print_header("BOARDY AI SEMANTIC CACHE TEST")
    
    # Initialize the semantic cache
    cache = SemanticCache(verbose=False)
    
    # Clear the cache before starting tests
    cache.clear_cache()
    
    # Test cases from the assignment
    test_cases = [
        ("Exact Match Test", [
            ("What's the weather in New York?", False),
            ("What's the weather in New York?", True)
        ]),
        ("Semantic Similarity Test", [
            ("What's the capital of France?", False),
            ("What is the capital city of France?", True)
        ]),
        ("Cache Miss Test", [
            ("What's the population of Tokyo?", False),
            ("What's the weather in London?", False)
        ]),
        ("Basic Persistence Test", [
            ("What's the largest planet?", False),
            ("Who wrote Romeo and Juliet?", False),
            ("What's the largest planet?", True)
        ]),
        ("Complex Semantic Similarity Test", [
            ("What are the health benefits of eating apples?", False),
            ("How do apples contribute to a healthy diet?", True)
        ])
    ]
    
    # Run test cases
    all_test_results = []
    
    for test_name, queries in test_cases:
        print_header(test_name)
        test_results = []
        
        for query, expected_cached in queries:
            response, from_cache, similarity = get_boardy_response(query, cache)
            test_result = from_cache == expected_cached
            test_results.append(test_result)
            all_test_results.append(test_result)
            
            print_test_result(query, response, from_cache, expected_cached, similarity)
        
        print_test_summary(test_name, test_results)
    
    # Print overall summary
    print_header("OVERALL TEST SUMMARY")
    total = len(all_test_results)
    passed = sum(all_test_results)
    success = passed == total
    
    status_color = COLORS['GREEN'] if success else COLORS['RED']
    status = "PASSED" if success else "FAILED"
    
    print(f"\n{status_color}{COLORS['BOLD']}{status}{COLORS['RESET']} {passed}/{total} tests passed ({passed/total*100:.2f}%)")
    
    # Print cache statistics
    print_cache_stats(cache)
