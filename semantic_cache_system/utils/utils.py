"""
Utility functions for formatting and printing test results.
"""

import sys
import time
from typing import List, Optional, TYPE_CHECKING, Any, Dict

from semantic_cache_system.config.constants import COLORS

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from semantic_cache_system.core.semantic_cache import SemanticCache

def print_header(text: str) -> None:
    """
    Print a formatted header.
    
    Args:
        text: The header text to print
    """
    width = 80
    print("\n" + "=" * width)
    print(f"{COLORS['BOLD']}{COLORS['WHITE']}{text.center(width)}{COLORS['RESET']}")
    print("=" * width)


def print_test_result(query: str, response: str, from_cache: bool, expected_cached: bool, similarity: float = 0.0, timestamp: Optional[float] = None) -> None:
    """
    Print a formatted test result.
    
    Args:
        query: The query that was tested
        response: The response that was received
        from_cache: Whether the response came from cache
        expected_cached: Whether the response was expected to come from cache
        similarity: The similarity score (if from cache)
        timestamp: The timestamp of the cached response (if from cache)
    """
    status = "✓" if from_cache == expected_cached else "✗"
    status_color = COLORS['GREEN'] if from_cache == expected_cached else COLORS['RED']
    cache_status = f"{COLORS['GREEN']}CACHE HIT{COLORS['RESET']}" if from_cache else f"{COLORS['YELLOW']}CACHE MISS{COLORS['RESET']}"
    
    print(f"\n{status_color}{COLORS['BOLD']}{status}{COLORS['RESET']} Query: '{COLORS['CYAN']}{query}{COLORS['RESET']}'")
    print(f"  {cache_status}")
    
    if from_cache:
        print(f"  Similarity: {COLORS['MAGENTA']}{similarity:.4f}{COLORS['RESET']}")
        if timestamp:
            age = time.time() - timestamp
            print(f"  Age: {COLORS['BLUE']}{age:.2f}s{COLORS['RESET']}")
    
    print(f"  Response: {COLORS['WHITE']}{response[:50]}...{COLORS['RESET']}")


def print_test_summary(test_name: str, results: List[bool]) -> None:
    """
    Print a summary of test results.
    
    Args:
        test_name: The name of the test
        results: A list of boolean results (True for pass, False for fail)
    """
    success = all(results)
    total = len(results)
    passed = sum(results)
    
    status_color = COLORS['GREEN'] if success else COLORS['RED']
    status = "PASSED" if success else "FAILED"
    
    print(f"\n{status_color}{COLORS['BOLD']}{status}{COLORS['RESET']} {test_name}: {passed}/{total} tests passed")


def print_cache_stats(cache: "SemanticCache") -> None:
    """
    Print cache statistics.
    
    Args:
        cache: The semantic cache instance
    """
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


def setup_colors() -> None:
    """
    Set up colors for terminal output.
    Disables colors if not in a terminal.
    """
    if not sys.stdout.isatty():
        # Disable colors if not in a terminal
        for key in COLORS:
            COLORS[key] = "" 