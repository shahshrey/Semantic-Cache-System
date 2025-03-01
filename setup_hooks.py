#!/usr/bin/env python3
"""Setup script for installing and updating pre-commit hooks."""

import logging
import os
import subprocess
import sys
from typing import List, Optional

# Constants
LOGGER_NAME = "setup_hooks"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOGGER_NAME)


def run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    """
    Run a shell command and handle exceptions.

    Args:
        command: List of command parts to execute
        cwd: Current working directory for the command

    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.exception(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.exception(f"Failed to run command: {e}")
        return False


def install_pre_commit() -> bool:
    """
    Install pre-commit package using UV.

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    return run_command(["uv", "pip", "install", "pre-commit"])


def install_hooks() -> bool:
    """
    Install pre-commit hooks into the git repository.

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    return run_command(["pre-commit", "install"])


def update_hooks() -> bool:
    """
    Update pre-commit hooks to the latest versions.

    Returns:
        bool: True if update succeeded, False otherwise
    """
    return run_command(["pre-commit", "autoupdate"])


def run_hooks() -> bool:
    """
    Run pre-commit hooks on all files.

    Returns:
        bool: True if hooks ran successfully, False otherwise
    """
    return run_command(["pre-commit", "run", "--all-files"])


def main() -> int:
    """
    Set up pre-commit hooks.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info("Setting up pre-commit hooks...")

    # Check if .pre-commit-config.yaml exists
    if not os.path.exists(".pre-commit-config.yaml"):
        logger.error(".pre-commit-config.yaml not found. Please create it first.")
        return 1

    # Install pre-commit
    if not install_pre_commit():
        logger.error("Failed to install pre-commit.")
        return 1

    # Install hooks
    if not install_hooks():
        logger.error("Failed to install pre-commit hooks.")
        return 1

    # Update hooks
    if not update_hooks():
        logger.error("Failed to update pre-commit hooks.")
        return 1

    # Run hooks
    logger.info("Running pre-commit hooks on all files...")
    run_hooks()

    logger.info("Pre-commit hooks setup completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
