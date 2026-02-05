"""Setup for Agent Engineer package."""

from setuptools import setup, find_packages

setup(
    name="agent-engineer",
    version="1.0.0",
    description="Professional agent collaboration system based on LangGraph",
    author="Agent Engineer Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.0.20",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "server": [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent-engineer=backend.server:main",
        ],
    },
)
