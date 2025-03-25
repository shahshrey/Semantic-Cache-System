"""Main entry point for the semantic cache system."""

from semantic_cache_system.core.semantic_cache import SemanticCache
from semantic_cache_system.services.ai import get_ai_response_simple
from semantic_cache_system.tests.tests import run_tests
from semantic_cache_system.utils.utils import print_cache_stats, print_header, setup_colors


def run_example() -> None:
    """Run an example of using the semantic cache as in the assignment."""
    print_header("EXAMPLE USAGE (AS IN ASSIGNMENT)")

    # Create a new cache for the example
    example_cache = SemanticCache(verbose=True)
    example_cache.clear_cache()

    # First query
    print("\n# First query")
    response1 = get_ai_response_simple("What's the weather like in New York today?", example_cache)
    print(f"Response: {response1}")

    # Similar query
    print("\n# Similar query")
    response2 = get_ai_response_simple("How's the weather in NYC right now?", example_cache)
    print(f"Response: {response2}")

    # Different query
    print("\n# Different query")
    response3 = get_ai_response_simple("What's the capital of France?", example_cache)
    print(f"Response: {response3}")

    # Print cache statistics for the example
    print_cache_stats(example_cache)


def main():
    """Run the semantic cache system."""
    # Set up colors for terminal output
    setup_colors()

    print_header("AI AI SEMANTIC CACHE")

    # Run tests
    print_header("RUNNING TESTS")
    run_tests()

    # Run example
    run_example()


if __name__ == "__main__":
    main()
