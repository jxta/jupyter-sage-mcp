"""
Jupyter MCP Server - Model Context Protocol server for Jupyter integration
"""

__version__ = "0.1.0"
__author__ = "Shigetoshi Yokoyama"
__email__ = "yoko@nii.ac.jp"
__description__ = "Model Context Protocol server for Jupyter Notebook integration with support for multiple kernels"

from .server import JupyterMCPServer
from .kernel_manager import KernelManager
from .code_executor import CodeExecutor
from .notebook_manager import NotebookManager
from .utils import setup_logging, load_config

__all__ = [
    "JupyterMCPServer",
    "KernelManager", 
    "CodeExecutor",
    "NotebookManager",
    "setup_logging",
    "load_config"
]
