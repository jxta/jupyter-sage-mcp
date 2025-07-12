#!/usr/bin/env python3
"""
Setup script for Jupyter MCP Server
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Model Context Protocol server for Jupyter Notebook integration"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "mcp>=0.1.0",
        "jupyter-client>=7.0.0",
        "jupyter-core>=4.7.0",
        "nbformat>=5.0.0",
        "nbconvert>=6.0.0",
        "tornado>=6.0.0",
        "traitlets>=5.0.0",
        "python-dateutil>=2.8.0",
        "pyzmq>=22.0.0",
    ]

setup(
    name="jupyter-mcp-server",
    version="0.1.0",
    author="Shigetoshi Yokoyama",
    author_email="yoko@nii.ac.jp",
    description="Model Context Protocol server for Jupyter Notebook integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jxta/jupyter-sage-mcp",
    project_urls={
        "Bug Tracker": "https://github.com/jxta/jupyter-sage-mcp/issues",
        "Documentation": "https://jupyter-mcp-server.readthedocs.io",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jupyter-mcp-server=jupyter_mcp_server.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "jupyter_mcp_server": ["config/*.json", "templates/*.ipynb"],
    },
    keywords=[
        "jupyter", "mcp", "model-context-protocol", "notebook", 
        "python", "sagemath", "julia", "ai", "assistant"
    ],
)
