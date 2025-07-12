"""
Kernel Manager - Manages Jupyter kernels for MCP server
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from jupyter_client import KernelManager as BaseKernelManager
from jupyter_client.kernelspec import KernelSpecManager

logger = logging.getLogger(__name__)


class KernelManager:
    """Manages Jupyter kernels for the MCP server"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.kernelspec_manager = KernelSpecManager()
        self.active_kernels: Dict[str, BaseKernelManager] = {}
        self.kernel_specs: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the kernel manager"""
        logger.info("Initializing KernelManager")
        await self._load_kernel_specs()
        logger.info(f"Loaded {len(self.kernel_specs)} kernel specifications")
    
    async def _load_kernel_specs(self):
        """Load available kernel specifications"""
        try:
            # Get kernel specs from jupyter
            specs = self.kernelspec_manager.get_all_specs()
            
            for name, spec in specs.items():
                self.kernel_specs[name] = {
                    'name': name,
                    'display_name': spec['spec'].get('display_name', name),
                    'language': spec['spec'].get('language', 'unknown'),
                    'argv': spec['spec'].get('argv', []),
                    'env': spec['spec'].get('env', {}),
                    'interrupt_mode': spec['spec'].get('interrupt_mode', 'signal'),
                    'resource_dir': spec['resource_dir']
                }
                
            logger.info(f"Available kernels: {list(self.kernel_specs.keys())}")
            
        except Exception as e:
            logger.error(f"Error loading kernel specs: {e}")
            # Fallback to basic Python kernel
            self.kernel_specs['python3'] = {
                'name': 'python3',
                'display_name': 'Python 3',
                'language': 'python',
                'argv': ['python', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'env': {},
                'interrupt_mode': 'signal'
            }
    
    async def list_kernels(self) -> List[Dict[str, Any]]:
        """List all available kernels"""
        kernels = []
        for name, spec in self.kernel_specs.items():
            # Check if kernel is currently active
            is_active = name in self.active_kernels
            status = 'running' if is_active else 'available'
            
            kernels.append({
                'name': name,
                'display_name': spec['display_name'],
                'language': spec['language'],
                'status': status,
                'spec': spec
            })
        
        return kernels
    
    async def get_kernel_info(self, kernel_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific kernel"""
        if kernel_name not in self.kernel_specs:
            raise ValueError(f"Kernel '{kernel_name}' not found")
        
        spec = self.kernel_specs[kernel_name]
        is_active = kernel_name in self.active_kernels
        
        info = {
            'name': kernel_name,
            'display_name': spec['display_name'],
            'language': spec['language'],
            'status': 'running' if is_active else 'available',
            'spec': spec
        }
        
        # Add runtime info if kernel is active
        if is_active:
            km = self.active_kernels[kernel_name]
            info.update({
                'has_kernel': km.has_kernel,
                'kernel_id': getattr(km, 'kernel_id', None),
                'connection_file': km.connection_file if hasattr(km, 'connection_file') else None
            })
        
        return info
    
    async def get_kernel(self, kernel_name: str) -> BaseKernelManager:
        """Get or start a kernel"""
        if kernel_name not in self.kernel_specs:
            raise ValueError(f"Kernel '{kernel_name}' not found")
        
        # Return existing kernel if already running
        if kernel_name in self.active_kernels:
            km = self.active_kernels[kernel_name]
            if km.has_kernel:
                logger.info(f"Using existing kernel '{kernel_name}'")
                return km
            else:
                # Kernel was stopped, remove from active list
                del self.active_kernels[kernel_name]
        
        # Start new kernel
        logger.info(f"Starting new kernel '{kernel_name}'")
        km = await self._start_kernel(kernel_name)
        self.active_kernels[kernel_name] = km
        
        return km
    
    async def _start_kernel(self, kernel_name: str) -> BaseKernelManager:
        """Start a new kernel"""
        try:
            spec = self.kernel_specs[kernel_name]
            
            # Create kernel manager
            km = BaseKernelManager(kernel_name=kernel_name)
            
            # Configure kernel environment
            env = os.environ.copy()
            env.update(spec.get('env', {}))
            
            # Apply any additional configuration
            if 'kernel_config' in self.config:
                kernel_config = self.config['kernel_config'].get(kernel_name, {})
                for key, value in kernel_config.items():
                    setattr(km, key, value)
            
            # Start the kernel
            await asyncio.to_thread(km.start_kernel, env=env)
            
            # Wait for kernel to be ready
            kc = km.client()
            await asyncio.to_thread(kc.wait_for_ready, timeout=30)
            
            logger.info(f"Successfully started kernel '{kernel_name}'")
            return km
            
        except Exception as e:
            logger.error(f"Failed to start kernel '{kernel_name}': {e}")
            raise
    
    async def stop_kernel(self, kernel_name: str) -> bool:
        """Stop a running kernel"""
        if kernel_name not in self.active_kernels:
            logger.warning(f"Kernel '{kernel_name}' is not running")
            return False
        
        try:
            km = self.active_kernels[kernel_name]
            await asyncio.to_thread(km.shutdown_kernel)
            del self.active_kernels[kernel_name]
            
            logger.info(f"Successfully stopped kernel '{kernel_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping kernel '{kernel_name}': {e}")
            return False
    
    async def restart_kernel(self, kernel_name: str) -> BaseKernelManager:
        """Restart a kernel"""
        logger.info(f"Restarting kernel '{kernel_name}'")
        
        # Stop existing kernel if running
        if kernel_name in self.active_kernels:
            await self.stop_kernel(kernel_name)
        
        # Start new kernel
        return await self.get_kernel(kernel_name)
    
    async def interrupt_kernel(self, kernel_name: str) -> bool:
        """Interrupt a running kernel"""
        if kernel_name not in self.active_kernels:
            logger.warning(f"Kernel '{kernel_name}' is not running")
            return False
        
        try:
            km = self.active_kernels[kernel_name]
            km.interrupt_kernel()
            
            logger.info(f"Successfully interrupted kernel '{kernel_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error interrupting kernel '{kernel_name}': {e}")
            return False
    
    async def get_kernel_status(self, kernel_name: str) -> Dict[str, Any]:
        """Get the status of a kernel"""
        if kernel_name not in self.kernel_specs:
            return {'status': 'unknown', 'error': f"Kernel '{kernel_name}' not found"}
        
        if kernel_name not in self.active_kernels:
            return {'status': 'stopped'}
        
        try:
            km = self.active_kernels[kernel_name]
            
            if not km.has_kernel:
                return {'status': 'stopped'}
            
            # Try to get kernel info to check if it's responsive
            kc = km.client()
            try:
                # Send a simple kernel info request with short timeout
                msg_id = kc.kernel_info()
                msg = await asyncio.to_thread(kc.get_shell_msg, timeout=5)
                
                if msg['msg_type'] == 'kernel_info_reply':
                    return {
                        'status': 'running',
                        'kernel_info': msg['content']
                    }
                else:
                    return {'status': 'unresponsive'}
                    
            except Exception:
                return {'status': 'unresponsive'}
                
        except Exception as e:
            logger.error(f"Error checking kernel status for '{kernel_name}': {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def cleanup(self):
        """Clean up all running kernels"""
        logger.info("Cleaning up KernelManager")
        
        for kernel_name in list(self.active_kernels.keys()):
            await self.stop_kernel(kernel_name)
        
        logger.info("KernelManager cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, 'active_kernels') and self.active_kernels:
            logger.warning("KernelManager destroyed with active kernels - cleanup may be incomplete")
