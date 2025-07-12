"""
SageMath対応MCP Helper - 完全版
SageMathとPythonカーネルの両方をサポートするMCPクライアント
mybinder.orgとNII解析基盤の両方で動作
"""

import asyncio
import websockets
import json
import requests
import os
import sys
import subprocess
from urllib.parse import urljoin
from IPython import get_ipython
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMCPClient:
    """SageMath対応MCPクライアント"""
    
    def __init__(self, base_url=None, token=None):
        self.base_url = base_url or os.environ.get('JUPYTER_URL', 'http://localhost:8888')
        self.token = token or os.environ.get('JUPYTER_TOKEN', 'binder_mcp_token')
        self.current_kernel = self._detect_kernel()
        self.session_id = self._generate_session_id()
        self.external_connections = {}
        
    def _detect_kernel(self):
        """現在のカーネルタイプを検出"""
        try:
            # SageMathカーネルかチェック
            import sage.all
            logger.info("🔬 SageMath kernel detected")
            return "sagemath"
        except ImportError:
            logger.info("🐍 Python kernel detected")
            return "python"
    
    def _generate_session_id(self):
        """セッションIDを生成"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_system_info(self):
        """システム情報を取得"""
        info = {
            "kernel_type": self.current_kernel,
            "session_id": self.session_id,
            "base_url": self.base_url,
            "python_version": sys.version,
            "available_packages": []
        }
        
        # 利用可能なパッケージチェック
        packages_to_check = [
            "sage", "numpy", "pandas", "matplotlib", 
            "jupyter_client", "websockets", "requests"
        ]
        
        for pkg in packages_to_check:
            try:
                __import__(pkg)
                info["available_packages"].append(pkg)
            except ImportError:
                pass
        
        return info
    
    def execute_code(self, code, kernel_type=None, capture_output=True):
        """コードを実行（カーネル自動判定）"""
        if kernel_type is None:
            kernel_type = self.current_kernel
            
        logger.info(f"🔧 Executing code in {kernel_type} kernel")
        
        try:
            if kernel_type == "sagemath":
                return self._execute_sage_code(code, capture_output)
            else:
                return self._execute_python_code(code, capture_output)
        except Exception as e:
            error_msg = f"Error in {kernel_type}: {e}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _execute_sage_code(self, code, capture_output=True):
        """SageMathコードの実行"""
        try:
            import sage.all as sage
            from io import StringIO
            import sys
            
            # 出力キャプチャの準備
            old_stdout = sys.stdout
            captured_output = StringIO() if capture_output else None
            
            if capture_output:
                sys.stdout = captured_output
            
            try:
                # Sageのpreparse機能を使用
                processed_code = sage.preparse(code)
                sage_globals = sage.__dict__.copy()
                local_vars = {}
                
                exec(processed_code, sage_globals, local_vars)
                
                output_text = captured_output.getvalue() if captured_output else ""
                result_value = None
                if local_vars:
                    last_var = list(local_vars.values())[-1]
                    if last_var is not None:
                        result_value = str(last_var)
                
                logger.info("✅ SageMath code executed successfully")
                return {
                    "status": "success",
                    "output": output_text,
                    "result": result_value,
                    "kernel": "sagemath"
                }
                
            finally:
                if capture_output:
                    sys.stdout = old_stdout
                
        except Exception as e:
            error_msg = f"Sage execution error: {e}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg, "kernel": "sagemath"}
    
    def _execute_python_code(self, code, capture_output=True):
        """Pythonコードの実行"""
        try:
            ipython = get_ipython()
            if ipython:
                result = ipython.run_cell(code)
                return {
                    "status": "success" if result.success else "error",
                    "result": str(result.result) if result.result else None,
                    "error": str(result.error_in_exec) if not result.success and result.error_in_exec else None,
                    "kernel": "python"
                }
            else:
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                captured_output = StringIO() if capture_output else None
                
                if capture_output:
                    sys.stdout = captured_output
                
                try:
                    exec(code)
                    output_text = captured_output.getvalue() if captured_output else ""
                    return {
                        "status": "success",
                        "output": output_text,
                        "kernel": "python"
                    }
                finally:
                    if capture_output:
                        sys.stdout = old_stdout
                        
        except Exception as e:
            error_msg = f"Python execution error: {e}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg, "kernel": "python"}
    
    def switch_kernel_context(self, kernel_type):
        """カーネルコンテキストの切り替え"""
        if kernel_type in ["python", "sagemath"]:
            self.current_kernel = kernel_type
            logger.info(f"🔄 Switched to {kernel_type} context")
            return {"status": "success", "message": f"Switched to {kernel_type} context"}
        else:
            error_msg = "Invalid kernel type. Use 'python' or 'sagemath'"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    async def connect_external_mcp(self, server_url, auth_token=None):
        """外部MCPサーバーへの接続"""
        try:
            headers = {}
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            websocket = await websockets.connect(server_url, extra_headers=headers)
            
            # MCP初期化メッセージ
            init_message = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "sage-mcp-client",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }
            
            await websocket.send(json.dumps(init_message))
            response = await websocket.recv()
            response_data = json.loads(response)
            
            if "error" not in response_data:
                connection_id = f"ext_{len(self.external_connections)}"
                self.external_connections[connection_id] = websocket
                logger.info(f"✅ Connected to external MCP: {server_url}")
                return {"status": "success", "connection_id": connection_id, "response": response_data}
            else:
                logger.error(f"MCP connection failed: {response_data.get('error')}")
                return {"status": "error", "error": response_data.get("error")}
                
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

def setup_sage_mcp():
    """SageMath対応MCP環境のセットアップ"""
    client = SageMCPClient()
    
    # グローバル変数として設定
    globals()['sage_mcp'] = client
    
    print("✅ SageMath MCP Client initialized!")
    print("🔬 Current kernel:", client.current_kernel)
    print("📊 System info:", client.get_system_info())
    
    return client

# 使用例
if __name__ == "__main__":
    mcp = setup_sage_mcp()
    
    # テスト実行
    test_result = mcp.execute_code("print('Hello from SageMath MCP!')")
    print("🧪 Test result:", test_result)
