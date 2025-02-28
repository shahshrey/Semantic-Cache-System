#!/bin/bash
# Installation script for the semantic cache system

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -sSf https://install.python-uv.org/install.sh | bash
    # Add UV to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with UV
echo "Installing dependencies with UV..."
uv pip install -r requirements.txt

# Install the package in development mode
echo "Installing the package in development mode..."
uv pip install -e .

echo "Installation complete! You can now run the semantic cache system with:"
echo "./run.py"
echo "or"
echo "python -m semantic_cache_system.main" 