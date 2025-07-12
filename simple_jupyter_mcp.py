#!/usr/bin/env python3
"""
Simple Jupyter MCP Server - No dependencies version
Allows Claude Desktop to execute code in Jupyter kernels
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple MCP protocol implementation
class SimpleMCPServer:
    def __init__(self):
        self.tools = [
            {
                "name": "create_python_kernel",
                "description": "Create a new Python kernel for code execution",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "execute_python",
                "description": "Execute Python code",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "create_sagemath_kernel",
                "description": "Create a new SageMath kernel",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "execute_sagemath",
                "description": "Execute SageMath code",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "SageMath code to execute"
                        }
                    },
                    "required": ["code"]
                }
            }
        ]
        self.kernels = {}
        self.jupyter_url = os.getenv("JUPYTER_URL", "http://localhost:8888")
        self.jupyter_token = os.getenv("JUPYTER_TOKEN", "")
        
    async def handle_request(self, request):
        """Handle MCP requests"""
        method = request.get("method")
        
        if method == "initialize":
            return {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "jupyter-executor",
                        "version": "1.0.0"
                    }
                }
            }
        elif method == "tools/list":
            return {
                "result": {
                    "tools": self.tools
                }
            }
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            
            result = await self.call_tool(name, arguments)
            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
        
        return {"error": {"code": -32601, "message": "Method not found"}}
    
    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute tool calls"""
        try:
            if name == "create_python_kernel":
                return await self.create_kernel("python3")
            elif name == "create_sagemath_kernel":
                return await self.create_kernel("sagemath")
            elif name == "execute_python":
                code = arguments.get("code", "")
                return await self.execute_code("python3", code)
            elif name == "execute_sagemath":
                code = arguments.get("code", "")
                return await self.execute_code("sagemath", code)
            else:
                return f"❌ Unknown tool: {name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    async def create_kernel(self, kernel_name: str) -> str:
        """Create a Jupyter kernel"""
        try:
            import httpx
            
            headers = {}
            if self.jupyter_token:
                headers["Authorization"] = f"token {self.jupyter_token}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.jupyter_url}/api/kernels",
                    json={"name": kernel_name},
                    headers=headers
                )
                
                if response.status_code == 201:
                    kernel_data = response.json()
                    kernel_id = kernel_data["id"]
                    self.kernels[kernel_name] = kernel_id
                    return f"✅ Created {kernel_name} kernel: {kernel_id}"
                else:
                    return f"❌ Failed to create kernel: {response.text}"
                    
        except Exception as e:
            return f"❌ Failed to create kernel: {str(e)}"
    
    async def execute_code(self, kernel_name: str, code: str) -> str:
        """Execute code in a kernel"""
        if not code.strip():
            return "❌ No code provided"
        
        # Ensure kernel exists
        if kernel_name not in self.kernels:
            create_result = await self.create_kernel(kernel_name)
            if "❌" in create_result:
                return create_result
        
        kernel_id = self.kernels[kernel_name]
        
        try:
            import websockets
            
            # Create WebSocket URL
            ws_url = self.jupyter_url.replace("http://", "ws://").replace("https://", "wss://")
            ws_url += f"/api/kernels/{kernel_id}/channels"
            
            if self.jupyter_token:
                ws_url += f"?token={self.jupyter_token}"
            
            # Execute code via WebSocket
            async with websockets.connect(ws_url) as websocket:
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
                
                await websocket.send(json.dumps(execute_msg))
                
                # Collect outputs
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
                                outputs.append(f"❌ {msg_data['content']['ename']}: {msg_data['content']['evalue']}")
                            break
                            
                        elif msg_type == "stream":
                            outputs.append(msg_data["content"]["text"])
                            
                        elif msg_type == "execute_result":
                            data = msg_data["content"]["data"]
                            if "text/plain" in data:
                                outputs.append(f"Result: {data['text/plain']}")
                                
                    except asyncio.TimeoutError:
                        break
                
                # Format output
                emoji = "🐍" if kernel_name == "python3" else "🧮"
                result = f"{emoji} {kernel_name.title()} Code Executed (#{execution_count or '?'}):\n\n"
                result += f"```{kernel_name.replace('3', '')}\n{code}\n```\n\n"
                
                if outputs:
                    result += "📤 Output:\n" + "\n".join(outputs)
                else:
                    result += "✅ Code executed successfully (no output)"
                
                return result
                
        except Exception as e:
            return f"❌ Execution failed: {str(e)}"

async def main():
    """Main server function using stdio"""
    # Get configuration from command line
    if len(sys.argv) > 1:
        jupyter_url = sys.argv[1]
        os.environ["JUPYTER_URL"] = jupyter_url
    
    if len(sys.argv) > 2:
        jupyter_token = sys.argv[2]
        os.environ["JUPYTER_TOKEN"] = jupyter_token
    
    server = SimpleMCPServer()
    
    logger.info(f"Starting Simple Jupyter MCP Server")
    logger.info(f"Jupyter URL: {server.jupyter_url}")
    logger.info(f"Token configured: {'Yes' if server.jupyter_token else 'No'}")
    
    # Simple stdio protocol handler
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            try:
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                
                # Add request ID if present
                if "id" in request:
                    response["id"] = request["id"]
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                error_response = {
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            break

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import httpx
        import websockets
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "websockets"])
        import httpx
        import websockets
    
    asyncio.run(main())
