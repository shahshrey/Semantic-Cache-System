"""Mock LLM functionality for testing the semantic cache."""

import time


def mock_llm_call(query: str) -> str:
    """Simulate an LLM call for testing purposes.

    In a real implementation, this would call an actual LLM API.

    Args:
        query: The query to send to the LLM

    Returns:
        str: The LLM response
    """
    # Simulate some processing time
    time.sleep(0.5)
    return f"Mock LLM response for: {query}"
