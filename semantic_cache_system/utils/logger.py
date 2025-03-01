"""
Logging configuration for the semantic cache system.
"""

import logging


# Set up logging
# Create a custom handler that writes INFO and below to stdout, WARNING and above to stderr
class CustomHandler(logging.Handler):
    def emit(self, record):
        pass  # We'll handle logging differently for cleaner test output


# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.addHandler(CustomHandler())
