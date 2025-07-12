#!/usr/bin/env python3
"""
Jupyter MCP Server for Claude Desktop
Allows Claude Desktop to execute code in Jupyter kernels
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

try:
    import httpx
    import websockets
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp", "httpx", "websockets"])
    import httpx
    import websockets
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Jupyter configuration from environment
JUPYTER_URL = os.getenv("JUPYTER_URL", "http://localhost:8888")
JUPYTER_TOKEN = os.getenv("JUPYTER_TOKEN", "")

class JupyterMCPServer:
    def __init__(self):
        self.server = Server("jupyter-executor")
        self.kernels: Dict[str, str] = {}  # kernel_name -> kernel_id
        self.websockets: Dict[str, Any] = {}
        self.jupyter_client = None
        
    async def initialize_jupyter_client(self):
        """Initialize HTTP client for Jupyter API"""
        headers = {}
        if JUPYTER_TOKEN:
            headers["Authorization"] = f"token {JUPYTER_TOKEN}"
        
        self.jupyter_client = httpx.AsyncClient(
            base_url=JUPYTER_URL,
            headers=headers,
            timeout=30.0
        )
        
    async def create_kernel(self, kernel_name: str = "python3") -> str:
        """Create a new Jupyter kernel"""
        if not self.jupyter_client:
            await self.initialize_jupyter_client()
            
        try:
            response = await self.jupyter_client.post("/api/kernels", json={
                "name": kernel_name
            })
            response.raise_for_status()
            kernel_data = response.json()
            kernel_id = kernel_data["id"]
            
            self.kernels[kernel_name] = kernel_id
            logger.info(f"Created {kernel_name} kernel: {kernel_id}")
            return kernel_id
            
        except Exception as e:
            logger.error(f"Failed to create kernel: {e}")
            raise
    
    async def execute_code(self, kernel_id: str, code: str) -> Dict[str, Any]:
        """Execute code in specified kernel via WebSocket"""
        try:
            # Create WebSocket URL
            ws_url = JUPYTER_URL.replace("http://", "ws://").replace("https://", "wss://")
            ws_url += f"/api/kernels/{kernel_id}/channels"
            
            if JUPYTER_TOKEN:
                ws_url += f"?token={JUPYTER_TOKEN}"
            
            # Connect to kernel WebSocket
            async with websockets.connect(ws_url) as websocket:
                # Create execution message
                msg_id = str(uuid.uuid4())
                execute_msg = {
                    "header": {
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                        "version": "5.3"
                    },
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": True,
                        "user_expressions": {},
                        "allow_stdin": False
                    }
                }
                
                # Send execution request
                await websocket.send(json.dumps(execute_msg))
                
                # Collect results
                outputs = []
                execution_count = None
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        msg_data = json.loads(message)
                        
                        parent_msg_id = msg_data.get("parent_header", {}).get("msg_id")
                        if parent_msg_id != msg_id:
                            continue
                            
                        msg_type = msg_data.get("msg_type")
                        
                        if msg_type == "execute_reply":
                            execution_count = msg_data["content"].get("execution_count")
                            status = msg_data["content"]["status"]
                            if status == "error":
                                outputs.append({
                                    "type": "error",
                                    "name": msg_data["content"]["ename"],
                                    "value": msg_data["content"]["evalue"],
                                    "traceback": msg_data["content"]["traceback"]
                                })
                            break
                            
                        elif msg_type == "stream":
                            outputs.append({
                                "type": "stream",
                                "name": msg_data["content"]["name"],
                                "text": msg_data["content"]["text"]
                            })
                            
                        elif msg_type == "execute_result":
                            execution_count = msg_data["content"]["execution_count"]
                            outputs.append({
                                "type": "execute_result",
                                "data": msg_data["content"]["data"],
                                "execution_count": execution_count
                            })
                            
                        elif msg_type == "display_data":
                            outputs.append({
                                "type": "display_data",
                                "data": msg_data["content"]["data"]
                            })
                            
                    except asyncio.TimeoutError:
                        logger.warning("Execution timeout")
                        break
                        
                return {
                    "status": "success",
                    "execution_count": execution_count,
                    "outputs": outputs
                }
                
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "outputs": []
            }

    def setup_handlers(self):
        """Setup MCP tool handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="create_python_kernel",
                    description="Create a new Python kernel for code execution",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="create_sagemath_kernel",
                    description="Create a new SageMath kernel for mathematical computations",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="execute_python",
                    description="Execute Python code in a Python kernel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="execute_sagemath",
                    description="Execute SageMath code in a SageMath kernel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "SageMath code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="list_kernels",
                    description="List all active kernels",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution"""
            
            if name == "create_python_kernel":
                try:
                    kernel_id = await self.create_kernel("python3")
                    return [types.TextContent(
                        type="text",
                        text=f"✅ Created Python kernel: {kernel_id}\nYou can now execute Python code!"
                    )]
                except Exception as e:
                    return [types.TextContent(
                        type="text",
                        text=f"❌ Failed to create Python kernel: {str(e)}"
                    )]
            
            elif name == "create_sagemath_kernel":
                try:
                    kernel_id = await self.create_kernel("sagemath")
                    return [types.TextContent(
                        type="text",
                        text=f"✅ Created SageMath kernel: {kernel_id}\nYou can now execute SageMath code!"
                    )]
                except Exception as e:
                    return [types.TextContent(
                        type="text",
                        text=f"❌ Failed to create SageMath kernel: {str(e)}"
                    )]
            
            elif name == "execute_python":
                code = arguments.get("code", "")
                if not code:
                    return [types.TextContent(
                        type="text",
                        text="❌ No code provided"
                    )]
                
                # Ensure Python kernel exists
                if "python3" not in self.kernels:
                    await self.create_kernel("python3")
                
                kernel_id = self.kernels["python3"]
                result = await self.execute_code(kernel_id, code)
                
                if result["status"] == "success":
                    output_text = f"🐍 Python Code Executed (#{result.get('execution_count', '?')}):\n\n"
                    output_text += f"```python\n{code}\n```\n\n"
                    
                    if result["outputs"]:
                        output_text += "📤 Output:\n"
                        for output in result["outputs"]:
                            if output["type"] == "stream":
                                output_text += f"{output['text']}"
                            elif output["type"] == "execute_result":
                                data = output["data"]
                                if "text/plain" in data:
                                    output_text += f"Result: {data['text/plain']}\n"
                            elif output["type"] == "error":
                                output_text += f"❌ {output['name']}: {output['value']}\n"
                    else:
                        output_text += "✅ Code executed successfully (no output)\n"
                else:
                    output_text = f"❌ Execution failed: {result.get('error', 'Unknown error')}"
                
                return [types.TextContent(type="text", text=output_text)]
            
            elif name == "execute_sagemath":
                code = arguments.get("code", "")
                if not code:
                    return [types.TextContent(
                        type="text",
                        text="❌ No code provided"
                    )]
                
                # Ensure SageMath kernel exists
                if "sagemath" not in self.kernels:
                    await self.create_kernel("sagemath")
                
                kernel_id = self.kernels["sagemath"]
                result = await self.execute_code(kernel_id, code)
                
                if result["status"] == "success":
                    output_text = f"🧮 SageMath Code Executed (#{result.get('execution_count', '?')}):\n\n"
                    output_text += f"```sage\n{code}\n```\n\n"
                    
                    if result["outputs"]:
                        output_text += "📤 Output:\n"
                        for output in result["outputs"]:
                            if output["type"] == "stream":
                                output_text += f"{output['text']}"
                            elif output["type"] == "execute_result":
                                data = output["data"]
                                if "text/plain" in data:
                                    output_text += f"Result: {data['text/plain']}\n"
                            elif output["type"] == "error":
                                output_text += f"❌ {output['name']}: {output['value']}\n"
                    else:
                        output_text += "✅ Code executed successfully (no output)\n"
                else:
                    output_text = f"❌ Execution failed: {result.get('error', 'Unknown error')}"
                
                return [types.TextContent(type="text", text=output_text)]
            
            elif name == "list_kernels":
                if not self.kernels:
                    return [types.TextContent(
                        type="text",
                        text="📋 No active kernels. Create a kernel first using create_python_kernel or create_sagemath_kernel."
                    )]
                
                kernel_list = "📋 Active Kernels:\n"
                for kernel_name, kernel_id in self.kernels.items():
                    kernel_list += f"  • {kernel_name}: {kernel_id}\n"
                
                return [types.TextContent(type="text", text=kernel_list)]
            
            else:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Unknown tool: {name}"
                )]

async def main():
    """Main server function"""
    # Get Jupyter URL from command line or environment
    if len(sys.argv) > 1:
        jupyter_url = sys.argv[1]
        os.environ["JUPYTER_URL"] = jupyter_url
    
    if len(sys.argv) > 2:
        jupyter_token = sys.argv[2]
        os.environ["JUPYTER_TOKEN"] = jupyter_token
    
    logger.info(f"Starting Jupyter MCP Server")
    logger.info(f"Jupyter URL: {JUPYTER_URL}")
    logger.info(f"Token configured: {'Yes' if JUPYTER_TOKEN else 'No'}")
    
    # Create server instance
    mcp_server = JupyterMCPServer()
    mcp_server.setup_handlers()
    
    # Initialize Jupyter client
    await mcp_server.initialize_jupyter_client()
    
    # Run server via stdio
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(
            read_stream,
            write_stream,
            InitializeResult(
                protocolVersion="2024-11-05",
                capabilities=mcp_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
