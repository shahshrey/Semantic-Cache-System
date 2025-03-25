"""Test cases for the semantic cache system."""

from typing import List, Tuple

from semantic_cache_system.config.constants import COLORS
from semantic_cache_system.core.semantic_cache import SemanticCache
from semantic_cache_system.services.ai import get_ai_response
from semantic_cache_system.utils.utils import (
    print_cache_stats,
    print_header,
    print_test_result,
    print_test_summary,
    setup_colors,
)

# Test cases from the assignment
TEST_CASES = [
    (
        "Exact Match Test",
        [
            ("What's the weather in New York?", False),
            ("What's the weather in New York?", True),
        ],
    ),
    (
        "Semantic Similarity Test",
        [
            ("What's the capital of France?", False),
            ("What is the capital city of France?", True),
        ],
    ),
    (
        "Cache Miss Test",
        [
            ("What's the population of Tokyo?", False),
            ("What's the weather in London?", False),
        ],
    ),
    (
        "Basic Persistence Test",
        [
            ("What's the largest planet?", False),
            ("Who wrote Romeo and Juliet?", False),
            ("What's the largest planet?", True),
        ],
    ),
    (
        "Simple Eviction Test",
        [
            ("What is the capital of France?", False),
            ("What is the largest planet in our solar system?", False),
            ("Who wrote the novel Pride and Prejudice?", False),
            ("What is the boiling point of water?", False),
            (
                "What is the capital of France?",
                False,
            ),  # Should be evicted if cache size is 3
        ],
    ),
    (
        "Complex Semantic Similarity Test",
        [
            ("What are the health benefits of eating apples?", False),
            ("How do apples contribute to a healthy diet?", True),
        ],
    ),
    (
        "Time-Sensitive Queries Test",
        [
            ("What's the current time in London?", False),
            (
                "What time is it in London now?",
                False,
            ),  # Should miss due to time sensitivity
        ],
    ),
    (
        "Long and Complex Queries Test",
        [
            (
                "What are the step-by-step instructions for baking a chocolate cake from scratch?",
                False,
            ),
            (
                "How do I make a homemade chocolate cake? Please provide detailed steps.",
                True,
            ),
        ],
    ),
    (
        "Special Characters and Multilingual Queries Test",
        [
            ("What's the meaning of 'こんにちは' in Japanese?", False),
            ("Translate 'こんにちは' from Japanese to English", True),
        ],
    ),
]


def run_tests() -> List[bool]:
    """
    Run all test cases.

    Returns:
        List[bool]: A list of boolean results (True for pass, False for fail)
    """
    # Initialize the semantic cache
    cache = SemanticCache(verbose=False)

    # Clear the cache before starting tests
    cache.clear_cache()

    # Run test cases
    all_test_results = []

    for test_name, queries in TEST_CASES:
        print_header(test_name)
        test_results = []

        # For the eviction test, use a smaller cache size
        if test_name == "Simple Eviction Test":
            # Create a new cache with size 3 for the eviction test
            small_cache = SemanticCache(max_cache_size=3, verbose=False)
            small_cache.clear_cache()
            test_cache = small_cache
        else:
            test_cache = cache

        for query, expected_cached in queries:
            # Get response from AI
            response, from_cache, similarity = get_ai_response(query, test_cache)

            # Get timestamp for the cached response if it's a cache hit
            timestamp = None
            if from_cache:
                # We need to get the timestamp from the cache
                _, _, timestamp = test_cache.get_from_cache(query)

            test_result = from_cache == expected_cached
            test_results.append(test_result)
            all_test_results.append(test_result)

            print_test_result(query, response, from_cache, expected_cached, similarity, timestamp)

        print_test_summary(test_name, test_results)

    # Print overall summary
    print_header("OVERALL TEST SUMMARY")
    total = len(all_test_results)
    passed = sum(all_test_results)
    success = passed == total

    status_color = "GREEN" if success else "RED"
    status = "PASSED" if success else "FAILED"

    print(
        f"\n{COLORS[status_color]}{COLORS['BOLD']}{status}{COLORS['RESET']} "
        f"{passed}/{total} tests passed ({passed/total*100:.2f}%)"
    )

    # Print cache statistics
    print_cache_stats(cache)

    return all_test_results


if __name__ == "__main__":
    setup_colors()
    print_header("AI AI SEMANTIC CACHE TEST")
    run_tests()
