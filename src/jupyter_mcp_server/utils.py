"""
Utility functions for Jupyter MCP Server
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('tornado').setLevel(logging.WARNING)
    logging.getLogger('jupyter_client').setLevel(logging.WARNING)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file"""
    
    # Default configuration
    default_config = {
        "server": {
            "host": "localhost",
            "port": 8000,
            "debug": False
        },
        "kernels": {
            "python3": {
                "display_name": "Python 3",
                "language": "python",
                "executable": "python"
            },
            "sagemath": {
                "display_name": "SageMath",
                "language": "python",
                "executable": "sage"
            },
            "julia": {
                "display_name": "Julia",
                "language": "julia",
                "executable": "julia"
            }
        },
        "notebooks": {
            "directory": "./notebooks",
            "auto_save": True,
            "backup": True
        },
        "logging": {
            "level": "INFO",
            "file": None
        }
    }
    
    if not config_path:
        return default_config
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logging.warning(f"Config file '{config_path}' not found, using defaults")
        return default_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        merged_config = merge_configs(default_config, user_config)
        
        logging.info(f"Loaded configuration from '{config_path}'")
        return merged_config
        
    except Exception as e:
        logging.error(f"Error loading config from '{config_path}': {e}")
        logging.info("Using default configuration")
        return default_config


def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config with default config"""
    
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Configuration saved to '{config_path}'")
        
    except Exception as e:
        logging.error(f"Error saving config to '{config_path}': {e}")
        raise


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for filesystem"""
    
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "untitled"
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def get_kernel_language_info(kernel_name: str) -> Dict[str, Any]:
    """Get language information for a kernel"""
    
    language_info = {
        'python3': {
            'name': 'python',
            'version': '3.x',
            'mimetype': 'text/x-python',
            'file_extension': '.py',
            'pygments_lexer': 'ipython3',
            'codemirror_mode': {'name': 'ipython', 'version': 3}
        },
        'sagemath': {
            'name': 'python',
            'version': 'SageMath',
            'mimetype': 'text/x-python',
            'file_extension': '.py',
            'pygments_lexer': 'ipython3',
            'codemirror_mode': {'name': 'ipython', 'version': 3}
        },
        'julia': {
            'name': 'julia',
            'version': '1.x',
            'mimetype': 'application/julia',
            'file_extension': '.jl',
            'pygments_lexer': 'julia',
            'codemirror_mode': 'julia'
        },
        'ir': {
            'name': 'R',
            'version': '4.x',
            'mimetype': 'text/x-r-source',
            'file_extension': '.r',
            'pygments_lexer': 'r',
            'codemirror_mode': 'r'
        }
    }
    
    return language_info.get(kernel_name, {
        'name': 'unknown',
        'version': 'unknown',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
        'pygments_lexer': 'text',
        'codemirror_mode': 'text'
    })


def create_empty_notebook(kernel_name: str = 'python3') -> Dict[str, Any]:
    """Create an empty notebook structure"""
    
    language_info = get_kernel_language_info(kernel_name)
    
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": kernel_name.title(),
                "language": language_info['name'],
                "name": kernel_name
            },
            "language_info": language_info
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def create_code_cell(source: str = '', execution_count: Optional[int] = None) -> Dict[str, Any]:
    """Create a code cell"""
    
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n') if source else []
    }


def create_markdown_cell(source: str = '') -> Dict[str, Any]:
    """Create a markdown cell"""
    
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n') if source else []
    }


def format_error_message(error: Exception) -> str:
    """Format error message for user display"""
    
    error_type = type(error).__name__
    error_message = str(error)
    
    # Add some helpful context for common errors
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error_message}"
    elif isinstance(error, PermissionError):
        return f"Permission denied: {error_message}"
    elif isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON format: {error_message}"
    else:
        return f"{error_type}: {error_message}"


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'hostname': platform.node()
    }
