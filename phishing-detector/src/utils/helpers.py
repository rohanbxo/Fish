"""
Helper utilities
"""

import os
import logging
import yaml
import json
from typing import Dict, Any


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_format: str = None
):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)


def get_project_root() -> str:
    """Get project root directory"""
    current_file = os.path.abspath(__file__)
    # Go up from src/utils/helpers.py to project root
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
