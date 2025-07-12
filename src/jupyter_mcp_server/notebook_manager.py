"""
Notebook Manager - Manages Jupyter notebooks
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import nbformat
from nbconvert import HTMLExporter, PDFExporter, PythonExporter

logger = logging.getLogger(__name__)


class NotebookManager:
    """Manages Jupyter notebooks"""
    
    def __init__(self, notebook_configs: Dict[str, Any]):
        self.notebook_configs = notebook_configs
        self.notebook_dir = Path(notebook_configs.get('directory', './notebooks'))
        self.auto_save = notebook_configs.get('auto_save', True)
        self.backup = notebook_configs.get('backup', True)
        self.notebooks: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize notebook manager"""
        logger.info("Initializing Notebook Manager")
        
        # Create notebook directory if it doesn't exist
        self.notebook_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing notebooks
        await self._load_existing_notebooks()
    
    async def _load_existing_notebooks(self):
        """Load existing notebooks from the notebook directory"""
        try:
            for notebook_path in self.notebook_dir.glob("*.ipynb"):
                notebook_id = str(uuid.uuid4())
                
                # Read notebook
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_content = json.load(f)
                
                # Extract metadata
                kernel_name = None
                if 'kernelspec' in notebook_content.get('metadata', {}):
                    kernel_name = notebook_content['metadata']['kernelspec'].get('name')
                
                self.notebooks[notebook_id] = {
                    'id': notebook_id,
                    'name': notebook_path.stem,
                    'path': str(notebook_path),
                    'kernel': kernel_name,
                    'content': notebook_content,
                    'modified': datetime.fromtimestamp(notebook_path.stat().st_mtime).isoformat(),
                    'created': datetime.fromtimestamp(notebook_path.stat().st_ctime).isoformat()
                }
            
            logger.info(f"Loaded {len(self.notebooks)} existing notebooks")
            
        except Exception as e:
            logger.error(f"Error loading existing notebooks: {e}")
    
    async def create_notebook(self, name: str, kernel_name: str, 
                            directory: Optional[str] = None) -> Dict[str, Any]:
        """Create a new notebook"""
        try:
            notebook_id = str(uuid.uuid4())
            
            # Determine save directory
            if directory:
                save_dir = Path(directory)
            else:
                save_dir = self.notebook_dir
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create notebook path
            notebook_path = save_dir / f"{name}.ipynb"
            
            # Create new notebook structure
            notebook_content = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": kernel_name,
                        "language": self._get_kernel_language(kernel_name),
                        "name": kernel_name
                    },
                    "language_info": {
                        "name": self._get_kernel_language(kernel_name)
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Save notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=2)
            
            # Store notebook info
            notebook_info = {
                'id': notebook_id,
                'name': name,
                'path': str(notebook_path),
                'kernel': kernel_name,
                'content': notebook_content,
                'modified': datetime.now().isoformat(),
                'created': datetime.now().isoformat()
            }
            
            self.notebooks[notebook_id] = notebook_info
            
            logger.info(f"Created notebook '{name}' with kernel '{kernel_name}'")
            return notebook_info
            
        except Exception as e:
            logger.error(f"Error creating notebook '{name}': {e}")
            raise
    
    def _get_kernel_language(self, kernel_name: str) -> str:
        """Get language for kernel"""
        language_map = {
            'python3': 'python',
            'python': 'python',
            'sagemath': 'python',
            'julia': 'julia',
            'julia-1.0': 'julia',
            'ir': 'r',
            'scala': 'scala'
        }
        return language_map.get(kernel_name, 'python')
    
    async def list_notebooks(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available notebooks"""
        notebooks = []
        
        for notebook_id, notebook_info in self.notebooks.items():
            if directory:
                # Filter by directory
                notebook_dir = Path(notebook_info['path']).parent
                if str(notebook_dir) != directory:
                    continue
            
            notebooks.append({
                'id': notebook_info['id'],
                'name': notebook_info['name'],
                'path': notebook_info['path'],
                'kernel': notebook_info['kernel'],
                'modified': notebook_info['modified'],
                'created': notebook_info['created']
            })
        
        return notebooks
    
    async def get_notebook(self, notebook_id: str) -> Dict[str, Any]:
        """Get notebook content"""
        if notebook_id not in self.notebooks:
            raise ValueError(f"Notebook '{notebook_id}' not found")
        
        notebook_info = self.notebooks[notebook_id]
        
        # Reload content from file to ensure it's up to date
        try:
            with open(notebook_info['path'], 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            notebook_info['content'] = content
            return notebook_info
            
        except Exception as e:
            logger.error(f"Error reading notebook '{notebook_id}': {e}")
            raise
    
    async def save_notebook(self, notebook_id: str, content: Dict[str, Any]) -> bool:
        """Save notebook content"""
        try:
            if notebook_id not in self.notebooks:
                raise ValueError(f"Notebook '{notebook_id}' not found")
            
            notebook_info = self.notebooks[notebook_id]
            notebook_path = Path(notebook_info['path'])
            
            # Create backup if enabled
            if self.backup:
                backup_path = notebook_path.with_suffix('.ipynb.bak')
                if notebook_path.exists():
                    import shutil
                    shutil.copy2(notebook_path, backup_path)
            
            # Save notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2)
            
            # Update stored info
            notebook_info['content'] = content
            notebook_info['modified'] = datetime.now().isoformat()
            
            logger.info(f"Saved notebook '{notebook_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error saving notebook '{notebook_id}': {e}")
            return False
    
    async def export_notebook(self, notebook_id: str, format: str, 
                            output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export notebook to various formats"""
        try:
            if notebook_id not in self.notebooks:
                raise ValueError(f"Notebook '{notebook_id}' not found")
            
            notebook_info = self.notebooks[notebook_id]
            notebook_path = Path(notebook_info['path'])
            
            # Read notebook with nbformat
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Determine output path
            if not output_path:
                if format == 'html':
                    output_path = notebook_path.with_suffix('.html')
                elif format == 'pdf':
                    output_path = notebook_path.with_suffix('.pdf')
                elif format == 'py' or format == 'python':
                    output_path = notebook_path.with_suffix('.py')
                else:
                    output_path = notebook_path.with_suffix(f'.{format}')
            else:
                output_path = Path(output_path)
            
            # Export based on format
            if format == 'html':
                exporter = HTMLExporter()
                (body, resources) = exporter.from_notebook_node(nb)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(body)
            
            elif format == 'pdf':
                exporter = PDFExporter()
                (body, resources) = exporter.from_notebook_node(nb)
                
                with open(output_path, 'wb') as f:
                    f.write(body)
            
            elif format in ['py', 'python']:
                exporter = PythonExporter()
                (body, resources) = exporter.from_notebook_node(nb)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(body)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported notebook '{notebook_id}' to {format} format")
            return {
                'status': 'success',
                'output_path': str(output_path),
                'format': format
            }
            
        except Exception as e:
            logger.error(f"Error exporting notebook '{notebook_id}': {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook"""
        try:
            if notebook_id not in self.notebooks:
                raise ValueError(f"Notebook '{notebook_id}' not found")
            
            notebook_info = self.notebooks[notebook_id]
            notebook_path = Path(notebook_info['path'])
            
            # Create backup before deletion if enabled
            if self.backup and notebook_path.exists():
                backup_path = notebook_path.with_suffix('.ipynb.deleted')
                import shutil
                shutil.copy2(notebook_path, backup_path)
            
            # Delete file
            if notebook_path.exists():
                notebook_path.unlink()
            
            # Remove from memory
            del self.notebooks[notebook_id]
            
            logger.info(f"Deleted notebook '{notebook_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting notebook '{notebook_id}': {e}")
            return False
    
    async def add_cell(self, notebook_id: str, cell_type: str = 'code', 
                      source: str = '', metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add a cell to notebook"""
        try:
            notebook = await self.get_notebook(notebook_id)
            
            # Create new cell
            new_cell = {
                "cell_type": cell_type,
                "metadata": metadata or {},
                "source": source.split('\n') if source else []
            }
            
            if cell_type == 'code':
                new_cell["execution_count"] = None
                new_cell["outputs"] = []
            
            # Add cell to notebook
            notebook['content']['cells'].append(new_cell)
            
            # Save if auto_save is enabled
            if self.auto_save:
                await self.save_notebook(notebook_id, notebook['content'])
            
            logger.info(f"Added {cell_type} cell to notebook '{notebook_id}'")
            return new_cell
            
        except Exception as e:
            logger.error(f"Error adding cell to notebook '{notebook_id}': {e}")
            raise
