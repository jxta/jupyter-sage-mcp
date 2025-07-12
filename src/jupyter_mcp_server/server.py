#!/usr/bin/env python3
"""
Jupyter MCP Server - Model Context Protocol server for Jupyter integration
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .kernel_manager import KernelManager
from .code_executor import CodeExecutor
from .notebook_manager import NotebookManager
from .utils import setup_logging, load_config

# Configure logging
logger = logging.getLogger(__name__)


class JupyterMCPServer:
    """Main MCP server class for Jupyter integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.kernel_manager = KernelManager(self.config.get('kernels', {}))
        self.code_executor = CodeExecutor(self.kernel_manager)
        self.notebook_manager = NotebookManager(self.config.get('notebooks', {}))
        
        # Initialize MCP server
        self.server = Server("jupyter-mcp-server")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        # Tools
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="execute_code",
                    description="Execute code in a Jupyter kernel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to execute"},
                            "kernel": {"type": "string", "description": "Kernel name (python3, sagemath, julia, etc.)"},
                            "notebook_id": {"type": "string", "description": "Optional notebook ID"},
                        },
                        "required": ["code", "kernel"]
                    }
                ),
                types.Tool(
                    name="list_kernels",
                    description="List available Jupyter kernels",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="get_kernel_info",
                    description="Get information about a specific kernel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kernel": {"type": "string", "description": "Kernel name"}
                        },
                        "required": ["kernel"]
                    }
                ),
                types.Tool(
                    name="create_notebook",
                    description="Create a new Jupyter notebook",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Notebook name"},
                            "kernel": {"type": "string", "description": "Kernel to use"},
                            "directory": {"type": "string", "description": "Directory to save notebook"}
                        },
                        "required": ["name", "kernel"]
                    }
                ),
                types.Tool(
                    name="export_notebook",
                    description="Export notebook to various formats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "notebook_id": {"type": "string", "description": "Notebook ID"},
                            "format": {"type": "string", "description": "Export format (html, pdf, py, etc.)"},
                            "output_path": {"type": "string", "description": "Output file path"}
                        },
                        "required": ["notebook_id", "format"]
                    }
                ),
                types.Tool(
                    name="install_package",
                    description="Install a package in the specified kernel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "package": {"type": "string", "description": "Package name"},
                            "kernel": {"type": "string", "description": "Kernel name"},
                            "version": {"type": "string", "description": "Package version (optional)"}
                        },
                        "required": ["package", "kernel"]
                    }
                ),
                types.Tool(
                    name="list_notebooks",
                    description="List available notebooks",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Directory to search"}
                        },
                        "required": []
                    }
                ),
                types.Tool(
                    name="get_notebook_content",
                    description="Get the content of a notebook",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "notebook_id": {"type": "string", "description": "Notebook ID"}
                        },
                        "required": ["notebook_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            try:
                if name == "execute_code":
                    return await self._execute_code(arguments)
                elif name == "list_kernels":
                    return await self._list_kernels(arguments)
                elif name == "get_kernel_info":
                    return await self._get_kernel_info(arguments)
                elif name == "create_notebook":
                    return await self._create_notebook(arguments)
                elif name == "export_notebook":
                    return await self._export_notebook(arguments)
                elif name == "install_package":
                    return await self._install_package(arguments)
                elif name == "list_notebooks":
                    return await self._list_notebooks(arguments)
                elif name == "get_notebook_content":
                    return await self._get_notebook_content(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        
        # Resources
        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            """List available resources"""
            resources = []
            
            # Add notebook resources
            notebooks = await self.notebook_manager.list_notebooks()
            for notebook in notebooks:
                resources.append(
                    types.Resource(
                        uri=f"notebook://{notebook['id']}",
                        name=f"Notebook: {notebook['name']}",
                        description=f"Jupyter notebook: {notebook['path']}",
                        mimeType="application/x-ipynb+json"
                    )
                )
            
            # Add kernel resources
            kernels = await self.kernel_manager.list_kernels()
            for kernel in kernels:
                resources.append(
                    types.Resource(
                        uri=f"kernel://{kernel['name']}",
                        name=f"Kernel: {kernel['display_name']}",
                        description=f"Jupyter kernel: {kernel['language']}",
                        mimeType="application/json"
                    )
                )
            
            return resources
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource"""
            if uri.startswith("notebook://"):
                notebook_id = uri[11:]  # Remove "notebook://" prefix
                notebook = await self.notebook_manager.get_notebook(notebook_id)
                return json.dumps(notebook, indent=2)
            elif uri.startswith("kernel://"):
                kernel_name = uri[9:]  # Remove "kernel://" prefix
                kernel_info = await self.kernel_manager.get_kernel_info(kernel_name)
                return json.dumps(kernel_info, indent=2)
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
    
    async def _execute_code(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute code in a kernel"""
        code = arguments["code"]
        kernel_name = arguments["kernel"]
        notebook_id = arguments.get("notebook_id")
        
        result = await self.code_executor.execute_code(code, kernel_name, notebook_id)
        
        output_text = f"Code executed successfully in {kernel_name} kernel:\n\n"
        output_text += f"Input:\n{code}\n\n"
        
        if result.get("stdout"):
            output_text += f"Output:\n{result['stdout']}\n\n"
        
        if result.get("stderr"):
            output_text += f"Errors:\n{result['stderr']}\n\n"
        
        if result.get("display_data"):
            output_text += f"Display Data:\n{json.dumps(result['display_data'], indent=2)}\n\n"
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _list_kernels(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """List available kernels"""
        kernels = await self.kernel_manager.list_kernels()
        
        output_text = "Available Jupyter Kernels:\n\n"
        for kernel in kernels:
            output_text += f"• {kernel['display_name']} ({kernel['name']})\n"
            output_text += f"  Language: {kernel['language']}\n"
            output_text += f"  Status: {kernel.get('status', 'unknown')}\n\n"
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _get_kernel_info(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Get kernel information"""
        kernel_name = arguments["kernel"]
        kernel_info = await self.kernel_manager.get_kernel_info(kernel_name)
        
        output_text = f"Kernel Information: {kernel_name}\n\n"
        output_text += json.dumps(kernel_info, indent=2)
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _create_notebook(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Create a new notebook"""
        name = arguments["name"]
        kernel_name = arguments["kernel"]
        directory = arguments.get("directory")
        
        notebook = await self.notebook_manager.create_notebook(name, kernel_name, directory)
        
        output_text = f"Notebook created successfully:\n\n"
        output_text += f"Name: {notebook['name']}\n"
        output_text += f"ID: {notebook['id']}\n"
        output_text += f"Path: {notebook['path']}\n"
        output_text += f"Kernel: {kernel_name}\n"
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _export_notebook(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Export notebook"""
        notebook_id = arguments["notebook_id"]
        format = arguments["format"]
        output_path = arguments.get("output_path")
        
        result = await self.notebook_manager.export_notebook(notebook_id, format, output_path)
        
        output_text = f"Notebook exported successfully:\n\n"
        output_text += f"Format: {format}\n"
        output_text += f"Output Path: {result['output_path']}\n"
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _install_package(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Install a package in a kernel"""
        package = arguments["package"]
        kernel_name = arguments["kernel"]
        version = arguments.get("version")
        
        result = await self.code_executor.install_package(package, kernel_name, version)
        
        output_text = f"Package installation result:\n\n"
        output_text += f"Package: {package}\n"
        output_text += f"Kernel: {kernel_name}\n"
        output_text += f"Status: {result['status']}\n"
        
        if result.get("output"):
            output_text += f"Output:\n{result['output']}\n"
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _list_notebooks(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """List available notebooks"""
        directory = arguments.get("directory")
        notebooks = await self.notebook_manager.list_notebooks(directory)
        
        output_text = "Available Notebooks:\n\n"
        for notebook in notebooks:
            output_text += f"• {notebook['name']} ({notebook['id']})\n"
            output_text += f"  Path: {notebook['path']}\n"
            output_text += f"  Kernel: {notebook.get('kernel', 'unknown')}\n"
            output_text += f"  Modified: {notebook.get('modified', 'unknown')}\n\n"
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def _get_notebook_content(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Get notebook content"""
        notebook_id = arguments["notebook_id"]
        notebook = await self.notebook_manager.get_notebook(notebook_id)
        
        output_text = f"Notebook Content: {notebook['name']}\n\n"
        output_text += json.dumps(notebook, indent=2)
        
        return [types.TextContent(type="text", text=output_text)]
    
    async def start(self):
        """Start the MCP server"""
        logger.info("Starting Jupyter MCP Server")
        await self.kernel_manager.initialize()
        await self.notebook_manager.initialize()
        
        # Start the server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


async def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    
    # Get config path from environment or command line
    config_path = os.getenv("JUPYTER_MCP_CONFIG", "config.json")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and start server
    server = JupyterMCPServer(config_path)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
