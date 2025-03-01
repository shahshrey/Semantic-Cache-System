# Boardy AI Engineering Challenge

**Abhi**  
**January 28, 2025 1:13 AM**  
*Not started*  
*Empty*

## Background
Boardy is an AI agent that makes matches in-network. To optimize performance and reduce costs, we want to implement a semantic caching scheme to decrease the number of LLM calls.

## Your Task
Design and implement a semantic caching system for Boardy, following these steps:

### Design and Implementation (25 minutes):
- Design a class-based approach for the semantic cache.
- Implement the design in Python, ensuring it can run and persist data on disk.

Your implementation should include:
- Methods for adding to the cache, retrieving from the cache, and measuring semantic similarity.
- A function that integrates the LLM call with the caching logic. This function should:
  - Check the cache for semantically similar queries
  - Make an LLM call if necessary 
  - Update the cache with new results
  - Return the appropriate response to the user
- Consider how to handle cache size limits and eviction policies.



## Evaluation Criteria
You'll be evaluated on:
- Breadth and depth of approaches considered
- Quality and efficiency of your Python implementation  
- Integration of caching logic with LLM calls
- Understanding of deployment and cloud infrastructure
- Ability to communicate technical concepts clearly
- Consideration of potential challenges and limitations

## Sample Usage Scenario
Here's an example of how your implementation might be used:
def mock_llm_call(query: str) -> str:
    # This function simulates an LLM call
    return f"Mock LLM response for: {query}"

def get_boardy_response(query: str) -> str:
    # TODO: Implement this function
    # It should:
    # 1. Check the semantic cache for similar queries
    # 2. If a similar query is found, return the cached response
    # 3. If no similar query is found, call the LLM and cache the result
    # 4. Return the response
    pass

# Usage example
semantic_cache = SemanticCache()  # You'll implement this class

# First query
response1 = get_boardy_response("What's the weather like in New York today?")
print(response1)  # Should print the LLM response and cache it

# Similar query
response2 = get_boardy_response("How's the weather in NYC right now?")
print(response2)  # Should return the cached response without calling the LLM

# Different query
response3 = get_boardy_response("What's the capital of France?")
print(response3)  # Should call the LLM and cache the new response



Your implementation should demonstrate:

1. Effective caching of semantically similar queries
2. Appropriate use of the LLM when the cache doesn't have a suitable response
3. Updating the cache with new responses
4. Persistence of the cache across multiple calls


test_cases = [
        ("Exact match", [
            ("What's the weather in New York?", False),
            ("What's the weather in New York?", True)
        ]),
        ("Simple semantic similarity", [
            ("What's the capital of France?", False),
            ("What is the capital city of France?", True)
        ]),
        ("Cache miss", [
            ("What's the population of Tokyo?", False),
            ("What's the weather in London?", False)
        ]),
        ("Basic persistence", [
            ("What's the largest planet?", False),
            ("Who wrote Romeo and Juliet?", False),
            ("What's the largest planet?", True)
        ]),
        ("Simple eviction (assuming cache size of 3)", [
            ("Item 1", False),
            ("Item 2", False),
            ("Item 3", False),
            ("Item 4", False),
            ("Item 1", False)  # Should be evicted
        ]),
        ("Complex semantic similarity", [
            ("What are the health benefits of eating apples?", False),
            ("How do apples contribute to a healthy diet?", True)
        ]),
        ("Time-sensitive queries", [
            ("What's the current time in London?", False),
            ("What time is it in London now?", False)  # Should miss due to time sensitivity
        ]),
        ("Long and complex queries", [
            ("What are the step-by-step instructions for baking a chocolate cake from scratch?", False),
            ("How do I make a homemade chocolate cake? Please provide detailed steps.", True)
        ]),
        ("Handling special characters and multilingual queries", [
            ("What's the meaning of 'こんにちは' in Japanese?", False),
            ("Translate 'こんにちは' from Japanese to English", True)
        ])
    ]