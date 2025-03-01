"""Constants and configuration for the semantic cache system."""

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
    "BG_BLUE": "\033[44m",
}

# Time-sensitive keywords for query detection
TIME_SENSITIVE_KEYWORDS = [
    "current time",
    "right now",
    "current weather",
    "today",
    "tonight",
    "this morning",
    "this afternoon",
    "this evening",
    "now",
    "current",
    "live",
    "real-time",
]
