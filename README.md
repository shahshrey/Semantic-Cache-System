# Semantic Cache System

A semantic cache system for AI responses that uses embeddings to match semantically similar queries.

## Features

- Semantic similarity matching for cache hits
- Time-based cache eviction
- Size-based cache eviction
- Time-sensitive query detection
- Multilingual query support

## Installation

```bash
# Install with pip
pip install -e .
```

## Usage

```python
from semantic_cache_system.core.semantic_cache import SemanticCache
from semantic_cache_system.services.boardy import get_boardy_response_simple

# Create a cache instance
cache = SemanticCache(verbose=True)

# Get a response (will check cache first)
response = get_boardy_response_simple("What's the weather like in New York?", cache)
print(response)

# Similar query will hit the cache
response = get_boardy_response_simple("How's the weather in NYC?", cache)
print(response)
```

## Running the Example

```bash
# Run the example script
python -m semantic_cache_system.main
```

## Development

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. To set up the pre-commit hooks:

```bash
# Install pre-commit and set up hooks
python setup_hooks.py
```

The pre-commit hooks include:
- Code formatting with Black and isort
- Linting with Flake8
- Type checking with MyPy
- Various file checks (trailing whitespace, merge conflicts, etc.)

To manually run the hooks on all files:

```bash
pre-commit run --all-files
```

## Project Structure

```
semantic_cache_system/
├── __init__.py
├── main.py
├── core/
│   ├── __init__.py
│   └── semantic_cache.py
├── services/
│   ├── __init__.py
│   ├── boardy.py
│   └── mock_llm.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── utils.py
├── config/
│   ├── __init__.py
│   └── constants.py
└── tests/
    ├── __init__.py
    └── tests.py
```

## Dependencies

- langchain
- openai
- chromadb
- langchain_openai
- langchain_chroma
- meilisearch
- python-dotenv
- tiktoken
- pre-commit (development only)
