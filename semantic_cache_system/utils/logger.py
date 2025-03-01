"""
Logging configuration for the semantic cache system.
"""

import logging


# Set up logging
# Create a custom handler that writes INFO and below to stdout, WARNING and above to stderr
class CustomHandler(logging.Handler):
    """Custom logging handler that suppresses output for cleaner test results."""

    def emit(self, record):
        """Process a log record but don't emit it to any destination."""
        pass  # We'll handle logging differently for cleaner test output


# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.addHandler(CustomHandler())
