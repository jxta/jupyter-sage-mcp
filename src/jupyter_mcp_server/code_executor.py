"""
Code Executor - Executes code in Jupyter kernels
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from jupyter_client import BlockingKernelClient

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Executes code in Jupyter kernels"""
    
    def __init__(self, kernel_manager):
        self.kernel_manager = kernel_manager
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute_code(self, code: str, kernel_name: str, 
                          notebook_id: Optional[str] = None, 
                          timeout: int = 30) -> Dict[str, Any]:
        """Execute code in the specified kernel"""
        try:
            # Get or start kernel
            km = await self.kernel_manager.get_kernel(kernel_name)
            
            # Create kernel client
            kc = km.client()
            
            # Execute code
            execution_id = str(uuid.uuid4())
            logger.info(f"Executing code in kernel '{kernel_name}' (execution_id: {execution_id})")
            
            # Send execution request
            msg_id = kc.execute(code, silent=False, store_history=True)
            
            # Collect results
            result = await self._collect_execution_results(kc, msg_id, timeout)
            
            # Store execution history
            execution_record = {
                'execution_id': execution_id,
                'kernel_name': kernel_name,
                'notebook_id': notebook_id,
                'code': code,
                'result': result,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            if notebook_id:
                if notebook_id not in self.execution_history:
                    self.execution_history[notebook_id] = []
                self.execution_history[notebook_id].append(execution_record)
            
            logger.info(f"Code execution completed (execution_id: {execution_id})")
            return result
            
        except Exception as e:
            logger.error(f"Error executing code in kernel '{kernel_name}': {e}")
            return {
                'status': 'error',
                'error': str(e),
                'stderr': str(e)
            }
    
    async def _collect_execution_results(self, kc: BlockingKernelClient, 
                                       msg_id: str, timeout: int) -> Dict[str, Any]:
        """Collect execution results from kernel"""
        result = {
            'status': 'ok',
            'stdout': '',
            'stderr': '',
            'display_data': [],
            'execution_count': None
        }
        
        # Collect messages
        start_time = asyncio.get_event_loop().time()
        
        while True:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > timeout:
                result['status'] = 'timeout'
                result['stderr'] = f"Execution timed out after {timeout} seconds"
                break
            
            try:
                # Get message with short timeout to avoid blocking
                msg = await asyncio.to_thread(kc.get_iopub_msg, timeout=1)
                
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue
                
                msg_type = msg['msg_type']
                content = msg['content']
                
                if msg_type == 'execute_result':
                    result['execution_count'] = content.get('execution_count')
                    if 'data' in content:
                        result['display_data'].append({
                            'type': 'execute_result',
                            'data': content['data'],
                            'metadata': content.get('metadata', {})
                        })
                
                elif msg_type == 'display_data':
                    if 'data' in content:
                        result['display_data'].append({
                            'type': 'display_data',
                            'data': content['data'],
                            'metadata': content.get('metadata', {})
                        })
                
                elif msg_type == 'stream':
                    stream_name = content.get('name', 'stdout')
                    text = content.get('text', '')
                    if stream_name == 'stdout':
                        result['stdout'] += text
                    elif stream_name == 'stderr':
                        result['stderr'] += text
                
                elif msg_type == 'error':
                    result['status'] = 'error'
                    result['error'] = content.get('ename', 'Unknown error')
                    result['stderr'] += '\n'.join(content.get('traceback', []))
                
                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    # Execution completed
                    break
                    
            except Exception as e:
                if "Empty queue" in str(e) or "timeout" in str(e).lower():
                    # No more messages, continue checking
                    await asyncio.sleep(0.1)
                    continue
                else:
                    logger.error(f"Error collecting execution results: {e}")
                    result['status'] = 'error'
                    result['stderr'] += f"\nError collecting results: {str(e)}"
                    break
        
        return result
    
    async def install_package(self, package: str, kernel_name: str, 
                            version: Optional[str] = None) -> Dict[str, Any]:
        """Install a package in the specified kernel"""
        try:
            # Determine installation command based on kernel
            spec = await self.kernel_manager.get_kernel_info(kernel_name)
            language = spec.get('language', '').lower()
            
            if language == 'python':
                if version:
                    install_code = f"import subprocess; subprocess.check_call(['pip', 'install', '{package}=={version}'])"
                else:
                    install_code = f"import subprocess; subprocess.check_call(['pip', 'install', '{package}'])"
            
            elif language == 'julia':
                if version:
                    install_code = f'using Pkg; Pkg.add(PackageSpec(name="{package}", version="{version}"))'
                else:
                    install_code = f'using Pkg; Pkg.add("{package}")'
            
            elif kernel_name == 'sagemath':
                # SageMath uses pip for Python packages
                if version:
                    install_code = f"import subprocess; subprocess.check_call(['pip', 'install', '{package}=={version}'])"
                else:
                    install_code = f"import subprocess; subprocess.check_call(['pip', 'install', '{package}'])"
            
            else:
                return {
                    'status': 'error',
                    'error': f"Package installation not supported for kernel '{kernel_name}'"
                }
            
            # Execute installation
            logger.info(f"Installing package '{package}' in kernel '{kernel_name}'")
            result = await self.execute_code(install_code, kernel_name, timeout=300)  # 5 minutes timeout
            
            if result['status'] == 'ok':
                return {
                    'status': 'success',
                    'package': package,
                    'version': version,
                    'kernel': kernel_name,
                    'output': result.get('stdout', '')
                }
            else:
                return {
                    'status': 'error',
                    'package': package,
                    'kernel': kernel_name,
                    'error': result.get('error', 'Installation failed'),
                    'output': result.get('stderr', '')
                }
                
        except Exception as e:
            logger.error(f"Error installing package '{package}' in kernel '{kernel_name}': {e}")
            return {
                'status': 'error',
                'package': package,
                'kernel': kernel_name,
                'error': str(e)
            }
    
    async def execute_cell_sequence(self, cells: List[str], kernel_name: str,
                                  notebook_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute a sequence of code cells"""
        results = []
        
        for i, cell_code in enumerate(cells):
            logger.info(f"Executing cell {i+1}/{len(cells)} in kernel '{kernel_name}'")
            
            result = await self.execute_code(cell_code, kernel_name, notebook_id)
            results.append({
                'cell_index': i,
                'code': cell_code,
                'result': result
            })
            
            # Stop execution if there's an error (unless explicitly configured otherwise)
            if result['status'] == 'error':
                logger.warning(f"Stopping cell sequence execution due to error in cell {i+1}")
                break
        
        return results
    
    async def get_execution_history(self, notebook_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a notebook"""
        return self.execution_history.get(notebook_id, [])
    
    async def clear_execution_history(self, notebook_id: str):
        """Clear execution history for a notebook"""
        if notebook_id in self.execution_history:
            del self.execution_history[notebook_id]
            logger.info(f"Cleared execution history for notebook '{notebook_id}'")
    
    async def interrupt_execution(self, kernel_name: str) -> Dict[str, Any]:
        """Interrupt execution in a kernel"""
        try:
            km = await self.kernel_manager.get_kernel(kernel_name)
            
            # Interrupt the kernel
            km.interrupt_kernel()
            
            logger.info(f"Interrupted execution in kernel '{kernel_name}'")
            return {
                'status': 'success',
                'message': f"Interrupted execution in kernel '{kernel_name}'"
            }
            
        except Exception as e:
            logger.error(f"Error interrupting kernel '{kernel_name}': {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_kernel_variables(self, kernel_name: str) -> Dict[str, Any]:
        """Get variables defined in the kernel namespace"""
        try:
            spec = await self.kernel_manager.get_kernel_info(kernel_name)
            language = spec.get('language', '').lower()
            
            if language == 'python' or kernel_name == 'sagemath':
                # Python/SageMath: use globals() and locals()
                code = """
import json
import sys
import types

def get_variables():
    variables = {}
    for name, obj in globals().items():
        if not name.startswith('_'):
            try:
                obj_type = type(obj).__name__
                if obj_type in ['int', 'float', 'str', 'bool', 'list', 'dict', 'tuple']:
                    variables[name] = {'type': obj_type, 'value': str(obj)[:100]}
                elif obj_type in ['function', 'builtin_function_or_method']:
                    variables[name] = {'type': 'function', 'value': f'<function {name}>'}
                elif obj_type == 'module':
                    variables[name] = {'type': 'module', 'value': f'<module {name}>'}
                elif obj_type == 'type':
                    variables[name] = {'type': 'class', 'value': f'<class {name}>'}
                else:
                    variables[name] = {'type': obj_type, 'value': f'<{obj_type} object>'}
            except:
                variables[name] = {'type': 'unknown', 'value': '<unable to inspect>'}
    return variables

print(json.dumps(get_variables()))
"""
            
            elif language == 'julia':
                # Julia: use varinfo()
                code = """
using JSON
import InteractiveUtils

function get_variables()
    vars = Dict()
    for name in names(Main)
        if !startswith(string(name), "_") && isdefined(Main, name)
            try
                obj = getfield(Main, name)
                obj_type = string(typeof(obj))
                if obj_type in ["Int64", "Float64", "String", "Bool"]
                    vars[string(name)] = Dict("type" => obj_type, "value" => string(obj)[1:min(100, length(string(obj)))])
                else
                    vars[string(name)] = Dict("type" => obj_type, "value" => "<$(obj_type) object>")
                end
            catch
                vars[string(name)] = Dict("type" => "unknown", "value" => "<unable to inspect>")
            end
        end
    end
    return vars
end

println(JSON.json(get_variables()))
"""
            
            else:
                return {
                    'status': 'error',
                    'error': f"Variable inspection not supported for kernel '{kernel_name}'"
                }
            
            # Execute code to get variables
            result = await self.execute_code(code, kernel_name)
            
            if result['status'] == 'ok' and result['stdout']:
                try:
                    variables = json.loads(result['stdout'].strip())
                    return {
                        'status': 'success',
                        'variables': variables,
                        'count': len(variables)
                    }
                except json.JSONDecodeError as e:
                    return {
                        'status': 'error',
                        'error': f"Failed to parse variables output: {e}"
                    }
            else:
                return {
                    'status': 'error',
                    'error': result.get('error', 'Failed to get variables'),
                    'stderr': result.get('stderr', '')
                }
                
        except Exception as e:
            logger.error(f"Error getting variables from kernel '{kernel_name}': {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
