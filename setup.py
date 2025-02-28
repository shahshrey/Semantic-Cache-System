"""
Setup script for the semantic_cache_system package.
"""

from setuptools import setup, find_packages

setup(
    name="semantic_cache_system",
    version="0.1.0",
    description="A semantic cache system for AI responses",
    author="Shrey Shah",
    author_email="sshreyv@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "openai",
        "python-dotenv",
        "langchain-community",
        "chromadb",
        "tiktoken",
        "langchain_openai",
        "langchain_chroma",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "semantic-cache=semantic_cache_system.main:main",
        ],
    },
) 